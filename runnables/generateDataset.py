from environments.fishTank.fishTank import FishTank
from environments.fishTank.fish import Fish
from modules.controller.saveController import SaveController

from modules.simulation.simulation import Simulation


def generateDataset():
    env = FishTank(400, 300)
    env.render()
    simulation = Simulation(env)

    ai = Fish(300, 300)
    controller = SaveController(env.inputType.get_size())
    ai.add_controller(controller)
    simulation.env.add_character(ai)

    simulation.run()


if __name__ == "__main__":
    generateDataset()
