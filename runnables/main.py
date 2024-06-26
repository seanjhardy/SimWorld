import gym
import numpy as np

from environments.fishTank.fishTank import FishTank
from modules.controller.HRLController import HRLController
from modules.controller.HTMController import HTMController
from modules.controller.playerController import PlayerController
from modules.simulation.simulation import Simulation

np.set_printoptions(precision=4, suppress=True)


def main():
    env = FishTank()  # gym.make("FishTank")
    agent = HRLController(env)
    # agent = PlayerController(env)
    # agent = HTMController(env)

    simulation = Simulation(env, agent)
    simulation.run()


if __name__ == "__main__":
    main()
