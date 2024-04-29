from abc import ABC
import gym


class Environment(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, name):
        super().__init__()
        self.name = name
