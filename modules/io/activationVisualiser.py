import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modules.networks.bithtm import SpatialPooler


class ActivationVisualizer:
    def __init__(self, models):
        self.models = models
        self.activations = {}
        self.attention = {}
        self.last_activations = {}
        self.hooks = []
        self.active = False

    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                if not self.active:
                    return
                try:
                    output = output.detach().cpu().numpy()
                except:
                    output = output

                if isinstance(output, SpatialPooler.State):
                    self.activations[name] = output.get_bits()
                    return
                if "identity" in name:
                    self.activations[name] = output
                    return

                if len(output.shape) == 4:
                    output = np.sum(output[0], axis=0)
                elif len(output.shape) == 3:
                    output = output[0][-1]
                else:
                    output = output[-1]

                if "attn_dropout" in name:
                    self.attention[name] = output
                    return

                if "c_attn" in name:
                    q, k, v = np.split(output, 3, axis=-1)
                    self.activations[name + "q"] = q
                    self.activations[name + "k"] = k
                    self.activations[name + "v"] = v
                    return

                self.activations[name] = output
            return hook

        for model in self.models:
            for name, layer in model.named_modules():
                if not isinstance(layer, (nn.Linear, nn.Dropout, nn.Identity, SpatialPooler)):
                    continue
                if isinstance(layer, nn.Dropout) and "attn_dropout" not in name:
                    continue
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def reshape_aspect_ratio(self, arr, aspect_ratio):
        total_elements = np.prod(arr.shape)

        # Calculate the number of rows needed based on the aspect ratio
        cols = max(math.floor(np.sqrt(total_elements * aspect_ratio)), 1)
        rows = max(math.floor(total_elements / cols), 1)

        # Ensure we have enough elements to fill the reshaped tensor
        elements_needed = rows * cols

        if total_elements < elements_needed:
            new = elements_needed - total_elements
            arr = np.concatenate([arr, np.full(new, 0.5)])

        reshaped_tensor = arr[:elements_needed].reshape(rows, cols)

        return reshaped_tensor

    def normalize_and_resize(self, activation, size, axis=None):
        activation = activation / (activation.max() - activation.min() + 1e-8)
        activation = activation - activation.mean(axis) + 0.5
        # activation = np.clip(activation, 0.5, 1)

        aspect_ratio = size[1] / size[0]
        activation = self.reshape_aspect_ratio(activation, aspect_ratio)
        activation = activation.reshape(1, 1, activation.shape[0], activation.shape[1])

        activation_resized = F.interpolate(torch.from_numpy(activation).to(torch.float32), size=size, mode='nearest')
        activation_resized = activation_resized.squeeze()
        return activation_resized.numpy()

    def visualize_activations(self, shape=(500, 500)):
        if len(self.last_activations.keys()) == 0:
            self.last_activations = dict(self.activations)
            return np.full(shape, 0.5)
        num_activations = 0
        for name, activation in self.activations.items():
            num_activations += activation.shape[-1]
        w_per_activation = shape[0]/num_activations
        i = 0
        vis = np.full((shape[0], shape[1]), 0.5)
        for name, activation in self.activations.items():
            width = math.floor(w_per_activation * activation.shape[-1])
            if width == 0 or i + width > vis.shape[0]:
                continue
            activation_resized = self.normalize_and_resize(activation, (width, shape[1]))
            w = activation_resized.shape[0]
            vis[i: i + w] = activation_resized
            i += width
        self.last_activations = dict(self.activations)
        return vis

    def visualize_attention(self, layer, shape=(500, 500)):
        for name, attention in self.attention.items():
            if str(layer - 1) in name:
                vis = self.normalize_and_resize(attention[:100, :100], shape)
                vis = np.flip(vis, 0)
                vis = np.flip(vis, 1)
                return vis
        return np.full(shape, 0.4)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
