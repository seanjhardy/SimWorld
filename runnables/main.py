from environments.fishTank.fishTank import FishTank
from environments.fishTank.character import Character
from modules.controller.agentController import AgentController

from modules.simulation.simulation import Simulation


def main():
    env = FishTank(400, 300)
    env.render()
    simulation = Simulation(env)

    ai = Character(300, 300)
    controller = AgentController(env.input.get_size(), env.output_size)
    ai.add_controller(controller)
    simulation.env.add_character(ai)

    simulation.run()


if __name__ == "__main__":
    main()
