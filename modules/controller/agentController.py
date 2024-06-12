import math
import os
import sys
import time

import numpy as np
import torch

from dotmap import DotMap
from gym.spaces import flatdim, flatten

from modules.controller.controller import Controller
from contextlib import nullcontext

from modules.io.replayBuffer import ReplayBuffer
from modules.networks.ACBlock import ACBlock
from modules.networks.Qtransformer import QTransformer
from modules.networks.activationVisualiser import ActivationVisualizer
from modules.networks.cvae import CVAE
from modules.networks.HRLblock import HRLBlock
from modules.networks.predictiveCodingNetwork import PredictiveCodingNetwork
from modules.networks.transformer import TransformerConfig
from torch import autograd

seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
fp16_precision = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# -----------------------------------------------------------------------------

class AgentController(Controller):
    def model_name(self):
        return f"cvae-wm-ac-v3"

    def __init__(self, env, config=None):
        super().__init__()
        # Network size
        self.output_size = flatdim(env.action_space)
        self.observation_size = flatdim(env.observation_space)
        self.env = env
        self.batch_size = 1
        self.context_size = 512

        # Latent state representation size
        self.latent_vis_size = 128
        self.vision_size = flatdim(env.observation_space["vision"])
        self.dynamics_size = flatdim(env.observation_space["dynamics"])
        self.latent_size = self.latent_vis_size + self.dynamics_size

        # Memories and stored info for training
        self.observations = None
        self.actions = None
        self.rewards = None
        self.memories = None

        self.latents = None
        self.predictions = None
        self.reconstructions = None
        self.q_values = None

        self.prediction_loss = 1
        self.reconstruction_loss = np.inf
        self.policy_loss = 0

        # Control variables for enabling training/predicting/saving weights
        self.train_interval = -1
        self.predicting = True
        self.dreaming = False
        self.save_interval = -1

        # Define model architecture
        self.transformerConfig = TransformerConfig(
            input_size=self.latent_size + self.output_size,
            output_size=self.latent_size,
            context_size=self.context_size,
            n_layer=6, n_head=6, n_embd=240,  # model params
            dropout=0.0, attn_dropout=0.0, bias=True,
            weight_decay=0.0001, learning_rate=0.001,  # Optimizer settings
            betas=(0.9, 0.95), device_type=device_type
        )

        self.model = DotMap(
            visual_cortex=CVAE(self.latent_vis_size, device=device),
            neocortex=QTransformer(config if config is not None else self.transformerConfig),
            rl_block_1=ACBlock(self.latent_size, self.output_size, 400, actor_only=True),
        )

        self.visualizer = ActivationVisualizer(self.model.neocortex)
        self.visualizer.register_hooks()

        torch.cuda.synchronize()

        self.load_model()
        self.model.rl_block_1 = ACBlock(self.latent_size, self.output_size, 400, actor_only=True)
        """self.load_submodel("visual_cortex",
                           "../../checkpoints/cvae-wm-ac-v3.pth",
                           "visual_cortex",
                           "visual_cortex_optimizer")"""
        self.reset()

    def step(self, observation: dict, reward: float):
        time = self.env.time

        # Internal curiosity reward for prediction errors (previous frame prediction error)
        # This guides the agent to explore more of the environment and improve its world model
        if self.prediction_loss != -1:
            reward += self.prediction_loss

        # Roll over memories
        self.observations = np.roll(self.observations, shift=-1, axis=0)
        self.actions = np.roll(self.actions, shift=-1, axis=0)
        self.rewards = np.roll(self.rewards, shift=-1, axis=0)
        self.latents = np.roll(self.latents, shift=-1, axis=0)
        if self.predicting or self.dreaming or \
            (time % self.train_interval == 0 and self.train_interval != -1):
            self.predictions = np.roll(self.predictions, shift=-1, axis=0)
            self.q_values = np.roll(self.q_values, shift=-1, axis=0)
            self.reconstructions = np.roll(self.reconstructions, shift=-1, axis=0)
        self.observations[-1] = flatten(self.env.observation_space, observation)
        self.rewards[-1] = reward
        # self.memories.add(self.latents[-2], self.q_values, self.actions, reward, self.latents[-1])

        # Generate random action
        actions = self.env.random_policy()
        self.actions[-1] = actions

        if time % self.env.reset_interval < self.context_size:
            return self.actions[-1]

        # Make predictions
        if self.predicting and not self.dreaming:
            with fp16_precision:
                self.predict()
            self.train_policy()

        if self.dreaming:
            self.dream()

        # Train the agent
        if time % self.train_interval == 0 and self.train_interval != -1:
            # Only start training the world model when our
            # latent space is expressive enough to be useful
            if self.reconstruction_loss > 0.001:
                self.visual_cortex(self.observations, "train")

            if self.reconstruction_loss < 0.002:
                latents, reconstructions = self.visual_cortex(self.observations, "forward")
                self.latents = np.concatenate([latents, self.observations[:, self.vision_size:]], axis=-1)
                self.reconstructions[-1] = reconstructions[-1]

                latent_predictions = self.train_neocortex(self.latents, self.actions, self.rewards)
                self.visual_cortex(latent_predictions[-1, :self.latent_vis_size], "decode")

            # if self.prediction_loss < 0.05:
            #    self.train_policy()

        # Save the model every save_interval iterations
        if self.save_interval != -1 and time % self.save_interval == 0:
            self.save_model()
            print(f"Saving {self.model_name()}. loss: {self.prediction_loss:.4f}")

        return self.actions[-1]

    def predict(self):
        # Produce latent representation (and reconstruction)
        latents, reconstructions = self.visual_cortex(self.observations, "forward")
        self.latents = np.concatenate([latents, self.observations[:, self.vision_size:]], axis=-1)
        self.reconstructions[-1] = reconstructions[-1]

        # Predict actions to take
        latent = torch.tensor(self.latents[-1])\
            .pin_memory().to(torch.float32, non_blocking=True).to(device)
        with nullcontext():
            actions = self.model.rl_block_1.forward(latent)
        self.actions[-1] = actions

        # Neocortex predicts next latent state using action taken
        neocortex_input = np.concatenate([self.latents, self.actions], axis=-1)
        neocortex_input = torch.from_numpy(neocortex_input[1:]).unsqueeze(0).to(device, non_blocking=True)
        latent_prediction, q_value, _ = self.model.neocortex(neocortex_input, None)
        self.q_values[-1] = q_value

        # Decode latent prediction back into visual representation
        self.visual_cortex(latent_prediction[-1, :self.latent_vis_size], "decode")
        self.predictions[-1, self.vision_size:] = latent_prediction[-1, self.latent_vis_size:]
        self.prediction_loss = np.mean(np.abs(self.predictions[-2] - self.observations[-1]))

    def dream(self):
        # Neocortex predicts next latent state using action taken
        neocortex_input = np.concatenate([self.latents[:-1], self.actions[:-1]], axis=-1)
        neocortex_input = torch.from_numpy(neocortex_input).unsqueeze(0).to(device)
        latent_prediction, q_value, _ = self.model.neocortex(neocortex_input, None)
        self.q_values[-1] = q_value
        # Overwrite latent with prediction
        self.latents[-1] = latent_prediction

        # Decode latent prediction back into visual representation
        self.visual_cortex(latent_prediction[-1, :self.latent_vis_size], "decode")

    def visual_cortex(self, data, mode="train"):  # Train, decode, encode
        if mode != "decode":
            x = data[..., :self.vision_size].reshape((-1, 3, 1, self.env.obs_pixels))
        else:
            x = data
        x = torch.from_numpy(x).pin_memory().to(torch.float32, non_blocking=True).to(device)

        if mode == "train":
            latents, reconstructions, loss = self.model.visual_cortex.backward(x)
            self.latents[:, :self.latent_vis_size] = latents
            self.latents[:, self.latent_vis_size:] = data[:, self.vision_size:]
            self.reconstructions[-1] = reconstructions[-1]
            self.reconstruction_loss = loss
        elif mode == "forward":
            with fp16_precision:
                latents, reconstructions, _, _ = self.model.visual_cortex.forward(x)
            return latents.cpu().detach().numpy(), \
                   reconstructions.cpu().detach().numpy().reshape(x.size(0), -1)
        elif mode == "encode":
            with fp16_precision:
                return self.model.visual_cortex.encode(x)
        elif mode == "decode":
            x = x.unsqueeze(0)
            with fp16_precision:
                reconstruction = self.model.visual_cortex.decode(x)
            self.predictions[-1, :self.vision_size] = reconstruction
        return None

    def train_neocortex(self, latents, actions, rewards):
        inputs = np.concatenate([latents, actions], axis=-1)
        inputs = torch.from_numpy(inputs).unsqueeze(0) \
            .pin_memory().to(torch.float32, non_blocking=True).to(device)
        targets_np = latents[1: self.context_size + 1]
        targets = torch.from_numpy(targets_np).unsqueeze(0) \
            .pin_memory().to(torch.float32, non_blocking=True).to(device)
        rewards = torch.from_numpy(rewards).unsqueeze(0) \
            .pin_memory().to(torch.float32, non_blocking=True).to(device)

        logits, q_values, loss = self.model.neocortex(inputs[:, :self.context_size, :],
                                                      targets,
                                                      rewards)
        self.q_values[-1] = q_values[-1]
        # Calculate loss of the final prediction
        self.prediction_loss = np.mean(np.abs(logits[-1] - targets_np[-1]))
        return logits

    def train_policy(self):
        # data = self.memories.sample(batch_size)
        # L1, Q1, R2, L2, Q2
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            data = (self.latents[:self.context_size],
                    self.actions[:self.context_size],
                    self.q_values[:self.context_size],
                    self.rewards,
                    self.latents[1:],
                    self.q_values[1:])
            data = tuple(torch.tensor(arr).pin_memory()
                         .to(torch.float32, non_blocking=True)
                         .to(device) for arr in data)
            self.model.rl_block_1.backward(data)

    def reset(self):
        self.observations = np.zeros((self.context_size + 1, self.observation_size), dtype=np.float16)
        self.actions = np.zeros((self.context_size + 1, self.output_size), dtype=np.float16)
        self.rewards = np.zeros(self.context_size, dtype=np.float16)
        self.memories = ReplayBuffer(1024)

        self.latents = np.zeros((self.context_size + 1, self.latent_size), dtype=np.float16)
        self.q_values = np.zeros(self.context_size + 1, dtype=np.float16)
        self.predictions = np.zeros((self.context_size, self.observation_size), dtype=np.float16)
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
