from torch import nn

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
            if isinstance(layer, (nn.Linear, nn.Conv2d, LayerNorm, CausalSelfAttention, MLP, Block)):
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_activations(self):
        return self.activations