import math

import numpy as np
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import zoom
from modules.networks.transformer import LayerNorm, CausalSelfAttention, Block, MLP


class ActivationVisualizer:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, CausalSelfAttention)):
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def normalize_and_resize(self, activation, size):
        activation = activation / (activation.max() - activation.min())
        activation = activation - activation.mean() + 0.5
        if len(activation.shape) == 3:
            activation = activation[0][-1]
        else:
            activation = activation[0]
        activation = activation.view(1, 1, -1)

        activation_resized = F.interpolate(activation, size=size, mode='nearest')
        activation_resized = activation_resized.squeeze()
        return activation_resized.cpu().numpy()

    def visualize_activations(self, shape):
        size = 0
        pixels = shape[0] * shape[1]
        for name, activation in self.activations.items():
            size += activation[-1].size(0)
        item_per_pixel = pixels/size
        i = 0
        vis = np.full((shape[0] * shape[1]), 0.5)
        for name, activation in self.activations.items():
            layer_size = math.floor(item_per_pixel * activation[-1].size(0))
            if layer_size != 0:
                activation_resized = self.normalize_and_resize(activation, (layer_size,))
                vis[i: i + layer_size] = activation_resized
            i += layer_size
        vis = vis.reshape(shape)
        width_ratio = shape[0] / vis.shape[0]
        #vis = zoom(vis, (width_ratio, 1), order=1)
        return vis

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
