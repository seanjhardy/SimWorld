from modules.controller.controller import Controller


class BaseController(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def step(self, observation: dict, reward: float):
        return self.env.random_policy()
