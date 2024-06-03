import itertools
import math
import random
import time

import numpy as np
import pandas as pd
import torch

from environments.fishTank.fishTank import FishTank
from modules.controller.agentController import AgentController, device_type
from contextlib import nullcontext

from modules.networks.transformer import GPTConfig


def train(config, learning_rate):
    gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
    grad_clip = 0.0

    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 50  # how many steps to warm up for
    lr_decay_iters = 250  # should be ~= max_iters per Chinchilla
    iter_num = 0
    min_lr = 0.000001
    controller = AgentController(FishTank.inputType.get_size(), FishTank.output_size, config=config)
    block_size = config.block_size
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    running_mfu = -1.0

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # GET DATA

    def get_batch():
        name = f"../dataset/data-{random.randint(1, 7)}.npy"
        data = np.load(name, mmap_mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.float32)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.float32)) for i in ix])
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y

    t0 = time.time()
    X, Y = get_batch()
    avg_loss = []
    while iter_num < lr_decay_iters:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in controller.optimizer.param_groups:
            param_group['lr'] = lr

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = controller.model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch()
            # backward pass, with gradient scaling if training in fp16
            # loss.backward()
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(controller.optimizer)
            torch.nn.utils.clip_grad_norm_(controller.model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(controller.optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        controller.optimizer.zero_grad(set_to_none=True)
        dt = time.time() - t0
        t0 = time.time()
        if iter_num > lr_decay_iters - 20:
            avg_loss.append(loss.item())
        if iter_num % 100 == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if iter_num >= 5:  # let the training loop settle a bit
                mfu = controller.model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            #print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        iter_num += 1
    return np.sum(avg_loss)/len(avg_loss)

def search():
    hyperparameters = {
        "learning_rate": [0.001],
        "block_size": [64],
        "n_layer": [2, 8],
        "n_head": [4, 8],
        "n_embed": [400],
    }
    input_size = FishTank.inputType.get_size()
    searchGrid = itertools.product(*hyperparameters.values())
    size = 1
    for values in hyperparameters.values():
        size *= len(values)
    i = 0
    for lr, block_size, n_layer, n_head, n_embed in searchGrid:
        config = GPTConfig(
            input_size=input_size,
            block_size=block_size,  # how far back does the model look? i.e. context size
            n_layer=n_layer, n_head=n_head, n_embd=n_embed,  # size of the model
            dropout=0.0,  # for determinism+
            bias=True,
        )
        t = time.time()
        loss = train(config, lr)
        time_change = time.time() - t
        i += 1
        print(f"{i}/{size} lr: {lr}, block_size: {block_size}, n_layer: {n_layer}, n_head: {n_head}, n_embed: {n_embed}. LOSS: {loss}, dt: {time_change:.2f} LOSS/dt: {10000 * loss / time_change}")

def analyse(data):
    # Convert the data into a pandas DataFrame for easier analysis
    df = pd.DataFrame(data)

    # Calculate correlation coefficients between variables and loss
    correlation_coefficients = df.corr()['loss'].drop('loss')

    print("Correlation coefficients between variables and loss:")
    print(correlation_coefficients)

    # You can also perform linear regression analysis to understand the relationship more deeply
    # For example, using the statsmodels library:
    import statsmodels.api as sm

    # Add a constant term for the intercept
    X = sm.add_constant(df.drop(columns=['loss']))

    # Fit ordinary least squares regression model
    model = sm.OLS(df['loss'], X)
    results = model.fit()

    print("\nLinear regression results:")
    print(results.summary())


if __name__ == "__main__":
    search()
    data = [
        {"lr": 0.001, "block_size": 64, "n_layer": 2, "n_head": 4, "n_embed": 400, "loss": 0.02639, "params": 8.96},
        {"lr": 0.001, "block_size": 64, "n_layer": 2, "n_head": 4, "n_embed": 600, "loss": 0.02935, "params": 4.05},
        {"lr": 0.001, "block_size": 64, "n_layer": 8, "n_head": 8, "n_embed": 400, "loss": 0.02787, "params": 8.96},
        {"lr": 0.001, "block_size": 64, "n_layer": 8, "n_head": 8, "n_embed": 600, "loss": 0.02807, "params": 7.90},
    ]
    #analyse(data)
