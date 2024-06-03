import gym

from environments.fishTank.fishTank import FishTank
from modules.controller.agentController import AgentController
from modules.simulation.simulation import Simulation


def main():
    env = FishTank()  # gym.make("FishTank")
    agent = AgentController(env)

    simulation = Simulation(env, agent)
    simulation.run()


if __name__ == "__main__":
    main()
