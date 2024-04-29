from environments.fishTank.fishTank import FishTank
from environments.fishTank.character import Character
from modules.controller.saveController import SaveController

from modules.simulation.simulation import Simulation


def generateDataset():
    env = FishTank(400, 300)
    env.render()
    simulation = Simulation(env)

    ai = Character(300, 300)
    controller = SaveController(env.input.get_size())
    ai.add_controller(controller)
    simulation.env.add_character(ai)

    simulation.run()


if __name__ == "__main__":
    generateDataset()