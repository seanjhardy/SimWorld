import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.networks.attention.utils import cosine_embedding
from modules.networks.classic.block import Block
from modules.networks.classic.utils import init_weights


@dataclass
class TransformerConfig:
    input_size: int = 900
    output_size: int = 900
    context_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    weight_decay: float = 0.0001,
    learning_rate: float = 0.001,
    betas: (float, float) = (0.9, 0.95),
    device_type: str = "cuda"


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.context_size is not None
        self.config = config

        self.model = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(config.n_embed, n_head=config.n_head,
                                   attn_drop=config.attn_dropout, drop=config.dropout,
                                   causal=True) for _ in range(config.n_layer)]),
            lm_head=nn.Linear(config.n_embed, config.output_size, bias=config.bias)
        ))
        self.to(config.device_type)
        self.optimizer = None

        # init all weights
        self.apply(init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.configure_optimizers(config)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _forward(self, idx):
        b, t, n = idx.size()
        assert t <= self.config.context_size, f"Cannot forward sequence of length {t}, " \
                                              f"block size is only {self.config.context_size}"

        # Compute positional embeddings and add it to sequence
        pos_emb = cosine_embedding(self.config.context_size, 40)
        pos_emb = torch.from_numpy(pos_emb).to(self.config.device_type).to(torch.float32)
        pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)
        padding = torch.zeros(self.config.n_embed - (self.config.input_size + 40))
        padding = padding.to(self.config.device_type).unsqueeze(0).expand(b, t, -1)
        x = torch.cat((idx, pos_emb, padding), dim=-1)

        for block in self.model.h:
            x = block(x)

        return x

    def forward(self, idx, targets=None):
        latent = self._forward(idx)

        if targets is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.model.lm_head(latent[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        else:
            # if we are given some desired targets also calculate the loss
            logits, loss = self.backward(latent, targets)

        return logits.cpu().detach().numpy()[0], loss

    def backward(self, latent, targets):
        logits = self.model.lm_head(latent)
        loss = F.l1_loss(logits, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.context_size
        self.config.context_size = block_size
        # self.model.wpe.weight = nn.Parameter(self.model.wpe.weight[:block_size])
        for block in self.model.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, config, log=False):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if log:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and config.device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas, **extra_args)
        if log:
            print(f"using fused AdamW: {use_fused}")

        self.optimizer = optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embed // cfg.n_head, cfg.context_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
