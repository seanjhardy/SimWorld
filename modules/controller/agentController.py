import math
import os
import random
import time

import numpy as np
import torch

from environments.fishTank.character import Character
from environments.fishTank.fishTank import FishTank
from modules.controller.controller import Controller
from contextlib import nullcontext

from modules.model.transformer import GPTConfig, Transformer

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
    predicting = False
    save = False

    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = 1
        self.block_size = 512
        self.inputs = None
        self.predictions = None
        self.loss = 0
        self.buffer = 1

        # model init
        self.gptconf = GPTConfig(
            input_size=input_size,
            block_size=self.block_size,  # how far back does the model look? i.e. context size
            n_layer=6, n_head=6, n_embd=300,  # size of the model
            dropout=0.0,  # for determinism+
            bias=True,
        )
        self.model = Transformer(self.gptconf)
        self.model.to(device)

        self.optimizer = self.model.configure_optimizers(weight_decay=0.0001, learning_rate=0.001, betas=(0.9, 0.95),
                                                         device_type=device_type)
        # self.model = torch.compile(self.model)  # pytorch 2.0
        self.dir = random.random() * math.pi * 2
        self.cur_dir = 0
        self.speedInput = 0.2
        torch.cuda.synchronize()
        self.model.load(self.optimizer)
        self.reset()

    def run(self, input):

        # Shift everything in arr2 backwards by 1
        self.inputs = np.roll(self.inputs, shift=-1, axis=0)

        # Add arr1 to the last index of arr2
        self.inputs[-1] = input

        if AgentController.predicting:
            new_input = np.copy(self.predictions[self.buffer:self.buffer + self.block_size])
            predicted_obs = FishTank.input.get_input(self.predictions[-1], "observation")
            new_input[-1] = FishTank.input.set_input(np.copy(input), "observation", predicted_obs)
            #new_input = self.inputs[1:]
            new_input = torch.stack([torch.from_numpy(new_input.astype(np.float32))])
            new_input = new_input.pin_memory().to(device, non_blocking=True)
            with ctx:
                logits, loss = self.model(new_input, None)
            self.predictions = np.roll(self.predictions, shift=-1, axis=0)
            self.predictions[-1] = logits.cpu().detach().numpy()[0][-1]
            return None#FishTank.input.get_input(self.predictions[-1], "actions")
        else:
            self.predictions[-1] = input

        if AgentController.training and FishTank.time % AgentController.train_speed == 0:
            X, Y = self.get_batch(self.inputs)
            self.train(X, Y)

        return None

    def train(self, X, Y):
        with ctx:
            logits, loss = self.model(X, Y)
        self.predictions[-1] = logits.cpu().detach().numpy()[0][-1]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        lossf = loss.item()
        self.loss = lossf
        if FishTank.time % 10000 == 0 and AgentController.save:
            self.model.save(self.optimizer)
            print(f"loss: {self.loss:.4f}")

    def reset(self):
        self.inputs = np.zeros((self.batch_size * self.block_size + self.buffer, self.input_size), dtype=np.float32)
        self.predictions = np.zeros((self.batch_size * self.block_size + self.buffer, self.input_size), dtype=np.float32)

    def get_batch(self, data):
        x = torch.stack([torch.from_numpy((data[:self.block_size]).astype(np.float32))])
        y = torch.stack([torch.from_numpy((data[self.buffer:self.buffer + self.block_size]).astype(np.float32))])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
