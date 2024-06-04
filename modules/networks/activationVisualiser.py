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
            if isinstance(layer, (nn.Linear, LayerNorm, CausalSelfAttention, MLP, Block)):
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def normalize_and_resize(self, activation, size=(100, 100)):
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)  # Normalize
        if len(activation.shape) == 3:
            activation = activation[0][-1]
        else:
            activation = activation[0]
        activation = activation.view(1, 1, 1, -1)

        activation_resized = F.interpolate(activation, size=size, mode='bilinear')
        activation_resized = activation_resized.squeeze()
        return activation_resized.cpu().numpy()

    def visualize_activations(self, shape):
        w = math.floor(shape[0] / len(self.activations.items()))
        i = 0
        vis = np.zeros((w * len(self.activations.items()), shape[1]))
        for name, activation in self.activations.items():
            activation_mean = activation
            activation_resized = self.normalize_and_resize(activation_mean, size=(w, shape[1]))
            vis[w*i:w*(i+1)] = activation_resized
            i += 1
        width_ratio = shape[0] / vis.shape[0]
        vis = zoom(vis, (width_ratio, 1), order=1)
        return vis

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
