import math
import random
import time

import numpy as np
import torch

from environments.fishTank.fishTank import FishTank
from modules.controller.agentController import AgentController, device_type
from contextlib import nullcontext


def train():
    gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
    grad_clip = 0.0

    learning_rate = 0.001
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 20  # how many steps to warm up for
    lr_decay_iters = 1000  # should be ~= max_iters per Chinchilla
    iter_num = 0
    min_lr = 0.00001
    controller = AgentController(FishTank.input.get_size(), FishTank.output_size)
    block_size = controller.block_size
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
    def get_batch(i):
        #Use incremental batches for first 1k iterations
        dataset_length = 500000
        env_change_interval = 5000
        if i < 0:
            interval = 100
            endIndex = (i + batch_size) * interval
            dataIndex = round(min(endIndex/500000, 6))
            name = f"../dataset/data-{dataIndex + 1}.npy"
            data = np.load(name, mmap_mode="r")
            ix = torch.arange(i, i + batch_size) * interval
            ix += block_size * (ix // (env_change_interval - block_size))
            ix = torch.minimum(ix, torch.tensor(500000))
            print(ix)
        else:
            name = f"../dataset/data-{random.randint(1, 7)}.npy"
            data = np.load(name, mmap_mode="r")
            # index fuckery so the range never lies on a multiple of 5,000 (when the environment changes)
            max_valid_start = dataset_length - block_size - (dataset_length % env_change_interval)
            ix = torch.randint(0, max_valid_start, (batch_size,))
            ix += block_size * (ix // (env_change_interval - block_size))
            ix %= 500000

        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.float32)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.float32)) for i in ix])
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y

    t0 = time.time()
    X, Y = get_batch(0)
    while True:
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
            if iter_num % 20 == 0:
                print("Y:\n", Y[0][-1], "\nLOGITS:\n", logits[0][-1])
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(iter_num)
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
        if iter_num % 1 == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if iter_num >= 5:  # let the training loop settle a bit
                mfu = controller.model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        iter_num += 1
        if iter_num % 100 == 0:
            controller.model.save(controller.optimizer)


if __name__ == "__main__":
    train()
