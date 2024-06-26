from pyglet.window import key
from gym.spaces import flatten
from modules.controller.controller import Controller


class PlayerController(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.viewer = env.viewer
        self.keys_down = {key.W: False, key.S: False, key.A: False, key.D: False}
        self.viewer.window.on_key_press = self.key_press
        self.viewer.window.on_key_release = self.key_release
        self.observation = None

    def key_press(self, k, mod):
        if k in self.keys_down:
            self.keys_down[k] = True

    def key_release(self, k, mod):
        if k in self.keys_down:
            self.keys_down[k] = False

    def step(self, input, reward):
        actions = [0, 0.5, 0]
        self.observation = flatten(self.env.observation_space, input)

        actions[0] = int(self.keys_down[key.W]) - int(self.keys_down[key.S])
        actions[1] = int(self.keys_down[key.A]) - int(self.keys_down[key.D])
        # Return the control inputs
        return actions
