import os
import numpy as np
from modules.controller.controller import Controller


class SaveController(Controller):
    save_interval = 500000
    log_interval = 1000

    def __init__(self, input_size):
        super().__init__()
        self.data = None
        self.index = 0
        self.input_size = input_size
        self.clear()

    def step(self, input, reward):
        self.data[self.index % SaveController.save_interval] = input
        self.index += 1

        if self.index % SaveController.save_interval == 0 and self.index != 0:
            self.save()

        if self.index % SaveController.log_interval == 0:
            print("Iter: ", self.index)

        return None

    def save(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        pathName = f"../../dataset/data-{round(self.index / SaveController.save_interval)}"
        filename = os.path.join(dirname, pathName)

        np.save(filename, self.data)
        self.clear()

    def clear(self):
        self.data = np.zeros((SaveController.save_interval, self.input_size), dtype=np.float32)

    def reset(self):
        return
