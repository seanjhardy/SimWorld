import os

import numpy as np
import torch

from dotmap import DotMap
from environments.fishTank.fishTank import FishTank
from modules.controller.controller import Controller
from contextlib import nullcontext

from modules.model.hrlblock import HRLBlock
from modules.model.transformer import TransformerConfig, Transformer

seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
profile = False  # use pytorch profiler, or just simple benchmarking?

# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# -----------------------------------------------------------------------------

class AgentController(Controller):
    training = False
    train_speed = 1000
    predicting = True
    save = False

    def __init__(self, input_size, output_size, config=None):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = 1
        self.block_size = 1024
        self.inputs = None
        self.predictions = None
        self.loss = 0

        self.n_embed = 400

        # model init
        self.transformerConfig = config if config is not None else TransformerConfig(
            input_size=input_size,
            block_size=self.block_size,  # context size
            n_layer=4, n_head=4, n_embd=self.n_embed,  # model params
            dropout=0.0,  # for determinism+
            bias=True,
            weight_decay=0.0001, # Optimizer settings
            learning_rate=0.001,
            betas=(0.9, 0.95),
            device_type=device_type
        )
        self.models = DotMap(
            world_model=Transformer(self.transformerConfig),
            rl_block_1=HRLBlock(self.n_embed, self.output_size),
        )

        torch.cuda.synchronize()

        self.load_model()
        self.reset()

    def run(self, input):

        # Shift everything in arr2 backwards by 1
        self.inputs = np.roll(self.inputs, shift=-1, axis=0)

        # Add arr1 to the last index of arr2
        self.inputs[-1] = input

        if AgentController.predicting:
            new_input = np.copy(self.predictions[1:self.block_size + 1])
            predicted_obs = FishTank.input.get_input(self.predictions[-1], "observation")
            new_input[-1] = FishTank.input.set_input(np.copy(input), "observation", predicted_obs)
            new_input = self.inputs[1:]
            new_input = torch.stack([torch.from_numpy(new_input.astype(np.float32))])
            new_input = new_input.pin_memory().to(device, non_blocking=True)
            with ctx:
                logits, loss = self.models.world_model(new_input, None)
            self.predictions = np.roll(self.predictions, shift=-1, axis=0)
            self.predictions[-1] = logits.cpu().detach().numpy()[0][-1]
            self.loss = np.mean(np.abs(self.predictions[-2] - self.inputs[-1]))
            return None  # FishTank.input.get_input(self.predictions[-1], "actions")
        else:
            self.predictions[-1] = input

        if AgentController.training and FishTank.time % AgentController.train_speed == 0:
            x, y = self.get_batch(self.inputs)
            self.train(x, y)

        return None

    def train(self, x, y):
        with ctx:
            logits, loss = self.models.world_model(x, y)
        self.predictions = np.roll(self.predictions, shift=-1, axis=0)
        self.predictions[-1] = logits.cpu().detach().numpy()[0][-1]
        self.models.world_model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.models.world_model.optimizer.step()
        lossf = loss.item()
        self.loss = lossf
        if FishTank.time % 10000 == 0 and FishTank.time != 0 and AgentController.save:
            self.save_model()
            print(f"loss: {self.loss:.4f}")

    def reset(self):
        self.inputs = np.zeros((self.block_size + 1, self.input_size), dtype=np.float32)
        self.predictions = np.zeros((self.block_size + 1, self.input_size), dtype=np.float32)

    def get_batch(self, data):
        x = torch.stack([torch.from_numpy((data[:self.block_size]).astype(np.float32))])
        y = torch.stack([torch.from_numpy((data[1:self.block_size + 1]).astype(np.float32))])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

    def model_name(self):
        return f"wm-a2c-hrl-{self.input_size}-v1"

    def save_model(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path_name = f"../../checkpoints/{self.model_name()}.pth"
        filename = os.path.join(dirname, path_name)

        save_dict = {}
        for name, model_obj in self.models.items():
            save_dict[f"{name}"] = model_obj.model.state_dict()
            save_dict[f"{name}_optimizer"] = model_obj.optimizer.state_dict()

        torch.save(save_dict, filename)

    def load_model(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path_name = f"../../checkpoints/{self.model_name()}.pth"
        filename = os.path.join(dirname, path_name)

        if not os.path.isfile(filename):
            return

        print("Loading saved weights")

        load_dict = torch.load(filename)
        self.models.world_model.model.load_state_dict(load_dict[f"transformer_state_dict"])
        self.models.world_model.model.optimizer.load_state_dict(load_dict[f"transformer_optimizer_state_dict"])
        print("loaded!")
        """for name, model_obj in self.models.items():
            if name in load_dict:
                model_obj.model.load_state_dict(load_dict[f"{name}"])
                model_obj.optimizer.load_state_dict(load_dict[f"{name}_optimizer"])"""
