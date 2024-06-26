import os

import numpy as np
import torch

from dotmap import DotMap
from gym.spaces import flatdim, flatten

from modules.controller.controller import Controller

from modules.io.encoders.RGBEncoder import encode_rgb, decode_rgb
from modules.io.encoders.ScalarEncoder import encode_scalar, encode_log
from modules.io.replayBuffer import ReplayBuffer
from modules.io.activationVisualiser import ActivationVisualizer
from modules.networks.bithtm import HierarchicalTemporalMemory


# -----------------------------------------------------------------------------
from modules.networks.bithtm.regularizations import LocalInhibition


class HTMController(Controller):
    def model_name(self):
        return f"htm-v1"

    def __init__(self, env):
        super().__init__()
        # Network size
        self.output_size = flatdim(env.action_space)
        self.env = env
        self.context_size = 10

        # Latent state representation size
        self.vision_size = flatdim(env.observation_space["vision"])
        self.vision_encoding_size = 20
        self.encoding_overlap = 0.1
        self.latent_vision_size = self.vision_size * self.vision_encoding_size
        self.dynamics_size = flatdim(env.observation_space["dynamics"]) * 20
        self.observation_size = self.latent_vision_size + self.dynamics_size

        # Memories and stored info for training
        self.observations = None
        self.actions = None
        self.rewards = None
        self.memories = None

        self.predictions = None
        self.reconstructions = None

        self.prediction_loss = 1
        self.reconstruction_loss = np.inf
        self.policy_loss = 0

        # Control variables for enabling training/predicting/saving weights
        self.train_interval = 1
        self.predicting = False
        self.dreaming = False
        self.save_interval = -1

        self.model = DotMap(
            neocortex=HierarchicalTemporalMemory(self.observation_size, 1024, 8),
        )

        self.visualizer = ActivationVisualizer(self.model.neocortex)
        self.visualizer.register_hooks()

        #self.load_model()
        self.reset()

    def step(self, observation: dict, reward: float):
        time = self.env.time

        # Internal curiosity reward for prediction errors (previous frame prediction error)
        # This guides the agent to explore more of the environment and improve its world model
        if self.prediction_loss != -1:
            reward += self.prediction_loss

        # Roll over memories
        self.observations = np.roll(self.observations, shift=-1, axis=0)
        self.latent = np.roll(self.latent, shift=-1, axis=0)
        self.actions = np.roll(self.actions, shift=-1, axis=0)
        self.rewards = np.roll(self.rewards, shift=-1, axis=0)
        if self.predicting or self.dreaming or \
                (time % self.train_interval == 0 and self.train_interval != -1):
            self.predictions = np.roll(self.predictions, shift=-1, axis=0)
            self.reconstructions = np.roll(self.reconstructions, shift=-1, axis=0)

        self.observations[-1] = flatten(self.env.observation_space, observation)

        observation["vision"] = encode_rgb(observation["vision"].flatten(), self.vision_encoding_size, self.encoding_overlap)

        observation["dynamics"]["collision_force"] = \
            encode_log(observation["dynamics"]["collision_force"][0], 20, 0.3)

        observation["dynamics"]["direction"] = \
            np.concatenate([
                encode_scalar(observation["dynamics"]["direction"][0], 0, 1, 20, 0.3),
                encode_scalar(observation["dynamics"]["direction"][1], 0, 1, 20, 0.3),
            ])

        observation["dynamics"]["velocity"] = \
            encode_scalar(observation["dynamics"]["velocity"][0], 0, 1, 20, 0.3)

        observation["dynamics"]["position"] = \
            np.concatenate([
                encode_scalar(observation["dynamics"]["position"][0], 0, 1, 20, 0.3),
                encode_scalar(observation["dynamics"]["position"][1], 0, 1, 20, 0.3),
            ])

        self.latent[-1] = flatten(self.env.observation_space, observation)
        self.rewards[-1] = reward

        # Generate random action
        actions = self.env.random_policy()
        self.actions[-1] = actions

        # Make predictions
        if self.predicting and not self.dreaming:
            self.predict()

        # Train the agent
        if time % self.train_interval == 0 and self.train_interval != -1:
            self.train_neocortex()

            # self.train_policy()

        # Save the model every save_interval iterations
        if self.save_interval != -1 and time % self.save_interval == 0:
            self.save_model()
            print(f"Saving {self.model_name()}. loss: {self.prediction_loss:.4f}")

        return self.actions[-1]

    def predict(self):
        return None

    def train_neocortex(self):
        inh = LocalInhibition(0.1)
        sp_state, predictions = self.model.neocortex.run(self.latent[-1])
        reconstruction = self.model.neocortex.inverse(sp_state.get_bits(), None,
                                                      self.latent[-1])
        prediction = self.model.neocortex.inverse(predictions, None)
        self.reconstructions[-1] = decode_rgb(reconstruction[:self.latent_vision_size], self.vision_encoding_size, self.encoding_overlap)
        self.predictions[-1] = decode_rgb(prediction[:self.latent_vision_size], self.vision_encoding_size, self.encoding_overlap)

    def reset(self):
        self.observations = np.zeros((self.context_size + 1, flatdim(self.env.observation_space)), dtype=np.float16)
        self.latent = np.zeros((self.context_size + 1, self.observation_size), dtype=np.float16)
        self.actions = np.zeros((self.context_size + 1, self.output_size), dtype=np.float16)
        self.rewards = np.zeros(self.context_size, dtype=np.float16)
        self.memories = ReplayBuffer(1024)

        self.predictions = np.zeros((self.context_size, self.vision_size), dtype=np.float16)
        self.reconstructions = np.zeros((self.context_size, self.vision_size), dtype=np.float16)

    def save_model(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path_name = f"../../checkpoints/{self.model_name()}.pth"
        filename = os.path.join(dirname, path_name)

        save_dict = {}
        for name, model_obj in self.model.items():
            try:
                save_dict[f"{name}"] = model_obj.model.state_dict()
                save_dict[f"{name}_optimizer"] = model_obj.optimizer.state_dict()
            except:
                save_dict[f"{name}"] = model_obj.state_dict()

        torch.save(save_dict, filename)

    def load_model(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path_name = f"../../checkpoints/{self.model_name()}.pth"
        filename = os.path.join(dirname, path_name)

        if not os.path.isfile(filename):
            return

        print(f"Loading saved model: {self.model_name()}")

        load_dict = torch.load(filename)
        for name, model_obj in self.model.items():
            if name in load_dict:
                try:
                    model_obj.model.load_state_dict(load_dict[f"{name}"])
                    model_obj.optimizer.load_state_dict(load_dict[f"{name}_optimizer"])
                except:
                    print("Failed to load:", name)

    def load_submodel(self, name, path, model_name, optimiser_name):
        dirname = os.path.dirname(os.path.realpath(__file__))
        load_dict = torch.load(os.path.join(dirname, path))
        self.model[name].model.load_state_dict(load_dict[model_name])
        self.model[name].optimizer.load_state_dict(load_dict[optimiser_name])
