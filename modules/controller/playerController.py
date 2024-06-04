import numpy as np
import pyglet
from pyglet.window import key

from modules.controller.controller import Controller


class PlayerController(Controller):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.keys_down = {key.W: False, key.S: False, key.A: False, key.D: False}
        viewer.window.on_key_press = self.key_press
        viewer.window.on_key_release = self.key_release

    def key_press(self, k, mod):
        if k in self.keys_down:
            self.keys_down[k] = True

    def key_release(self, k, mod):
        if k in self.keys_down:
            self.keys_down[k] = False

    def step(self, input, reward):
        actions = [0, 0.5, 0]

        actions[0] = int(self.keys_down[key.W]) * 0.5 - int(self.keys_down[key.S]) * 0.5
        actions[1] = 0.5 + int(self.keys_down[key.A]) * 0.5 + int(self.keys_down[key.D]) * -0.5
        # Return the control inputs
        return actions
