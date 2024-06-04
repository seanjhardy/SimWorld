# Function to visualize attention weights
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from environments.fishTank.fishTank import FishTank
from contextlib import nullcontext


def visualize_attention(attention_weights):
    plt.imshow(attention_weights[0], cmap='hot', interpolation='nearest')
    plt.xlabel('Query')
    plt.ylabel('Key')
    plt.title('Attention Heatmap')
    plt.colorbar()
    plt.show()


def visualise():
    from modules.controller.agentController import AgentController, device_type

    batch_size = 1  # if gradient_accumulation_steps > 1, this is the micro-batch size
    controller = AgentController(FishTank.inputType.get_size(), FishTank.output_size)
    block_size = controller.context_size
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def get_batch():
        name = f"../../dataset/data-{random.randint(1, 6)}.npy"
        data = np.load(name, mmap_mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.float32)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.float32)) for i in ix])
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y

    X, Y = get_batch()
    with ctx:
        logits, loss = controller.model(X, Y, True)
    print("Y:\n", Y[0][-1], "\nLOGITS:\n", logits[0][-1])
    # immediately async prefetch next batch while model is doing the forward pass on the GPU


if __name__ == "__main__":
    visualise()
