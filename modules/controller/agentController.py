import os

import numpy as np
import torch

from dotmap import DotMap
from gym.spaces import flatdim, flatten

from modules.controller.controller import Controller
from contextlib import nullcontext

from modules.io.replayBuffer import ReplayBuffer
from modules.networks.Qtransformer import QTransformer
from modules.networks.activationVisualiser import ActivationVisualizer
from modules.networks.cvae import CVAE
from modules.networks.hrlblock import HRLBlock
from modules.networks.predictiveCodingNetwork import PredictiveCodingNetwork
from modules.networks.transformer import TransformerConfig, Transformer

seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
profile = False  # use pytorch profiler, or just simple benchmarking?

# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# -----------------------------------------------------------------------------

class AgentController(Controller):
    def __init__(self, env, config=None):
        super().__init__()
        # Network size
        self.output_size = flatdim(env.action_space)
        self.observation_size = flatdim(env.observation_space)
        self.env = env
        self.batch_size = 1
        self.context_size = 1024

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
        self.network_vis = None
        self.q_values = None

        self.prediction_loss = 0
        self.reconstruction_loss = 0
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
            n_layer=4, n_head=4, n_embd=400,  # model params
            dropout=0.0, bias=True,
            weight_decay=0.0001, learning_rate=0.001,  # Optimizer settings
            betas=(0.9, 0.95), device_type=device_type
        )

        self.model = DotMap(
            visual_cortex=CVAE(self.latent_vis_size, device=device),
            neocortex=QTransformer(config if config is not None else self.transformerConfig),
            rl_block_1=HRLBlock(self.latent_size, self.output_size, actor_only=True),
        )

        self.visualizer = ActivationVisualizer(self.model.neocortex)
        self.visualizer.register_hooks()

        torch.cuda.synchronize()

        self.load_model()
        self.reset()

    def step(self, observation: dict, reward: float):
        time = self.env.time

        # Internal curiosity reward for prediction errors (previous frame prediction error)
        # This guides the agent to explore more of the environment and improve its world model
        reward += self.prediction_loss

        # Roll over memories
        self.observations = np.roll(self.observations, shift=-1, axis=0)
        self.latents = np.roll(self.latents, shift=-1, axis=0)
        if self.predicting or self.dreaming or \
            (time % self.train_interval == 0 and self.train_interval != -1):
            self.predictions = np.roll(self.predictions, shift=-1, axis=0)
            self.q_values = np.roll(self.q_values, shift=-1, axis=0)
            self.reconstructions = np.roll(self.reconstructions, shift=-1, axis=0)
        self.actions = np.roll(self.actions, shift=-1, axis=0)
        self.rewards = np.roll(self.rewards, shift=-1, axis=0)
        self.observations[-1] = flatten(self.env.observation_space, observation)
        self.rewards[-1] = reward
        # self.memories.add(self.latents[-2], self.q_values, self.actions, reward, self.latents[-1])

        # Generate random action
        actions = self.env.random_policy()
        self.actions[-1] = actions

        # Quit early if we don't have enough data (Just for training)
        if time % self.env.reset_interval < self.context_size \
                and self.train_interval != -1 and False:
            return actions

        # Make predictions
        if self.predicting and not self.dreaming:
            self.predict()
            self.train_policy()

            latent = torch.tensor(self.latents[-1]).to(torch.float32).to(device, non_blocking=True)
            actions = self.model.rl_block_1.forward(latent)
            self.actions[-1] = actions

        if self.dreaming:
            self.dream()

        # Train the agent
        if time % self.train_interval == 0 and self.train_interval != -1:
            self.train_visual_cortex(self.observations)
            self.latents[:, self.latent_vis_size:] = self.observations[:, self.vision_size:]

            # Only start training the world model when our
            # latent space is expressive enough to be useful
            if self.reconstruction_loss < 0.001:
                x, y = self.get_batch(self.latents, self.actions, self.rewards)
                latent_predictions, q_values = self.train_neocortex(x, y)
                latent_prediction = torch.from_numpy(latent_predictions[-1, :self.latent_vis_size]) \
                    .unsqueeze(0).to(torch.float32).to(device, non_blocking=True)
                self.predictions[-1, :self.vision_size] = self.model.visual_cortex.decode(latent_prediction)

            if self.prediction_loss < 0.05:
                self.train_policy()
        # Save the model every save_interval iterations
        if self.save_interval != -1 and time % self.save_interval == 0:
            self.save_model()
            print(f"loss: {self.prediction_loss:.4f}")

        return actions

    def predict(self):
        # Produce latent representation (and reconstruction)
        x = self.observations[-1, :self.vision_size]\
            .reshape((-1, 3, 1, self.env.obs_pixels))
        x = torch.from_numpy(x).to(torch.float32)\
            .pin_memory().to(device, non_blocking=True)
        latents, reconstructions = self.model.visual_cortex.forward(x)
        self.latents[:, :self.latent_vis_size] = latents.cpu().detach().numpy()
        self.latents[:, self.latent_vis_size:] = self.observations[:, self.vision_size:]
        self.reconstructions[-1] = reconstructions.cpu().detach().numpy()[-1].reshape(-1)

        # Neocortex predicts next latent state using action taken
        neocortex_input = np.concatenate([self.latents, self.actions], axis=-1)
        neocortex_input = torch.from_numpy(neocortex_input[1:]).unsqueeze(0).to(device, non_blocking=True)
        latent_prediction, q_value, _ = self.model.neocortex(neocortex_input, None)
        self.q_values[-1] = q_value

        self.network_vis = self.visualizer.visualize_activations(self.network_vis.shape)

        # Decode latent prediction back into visual representation
        latent_vis_prediction = torch.from_numpy(latent_prediction[-1, :self.latent_vis_size]) \
            .unsqueeze(0).to(torch.float32).to(device, non_blocking=True)
        self.predictions[-1, :self.vision_size] = self.model.visual_cortex.decode(latent_vis_prediction)
        self.predictions[-1, self.vision_size:] = latent_prediction[-1, self.latent_vis_size:]
        self.prediction_loss = np.mean(np.abs(self.predictions[-2] - self.observations[-1]))

    def dream(self):
        # Neocortex predicts next latent state using action taken
        neocortex_input = np.concatenate([self.latents[:-1], self.actions[1:]], axis=-1)
        neocortex_input = torch.from_numpy(neocortex_input).unsqueeze(0).to(device, non_blocking=True)
        latent_prediction, q_value, _ = self.model.neocortex(neocortex_input, None)
        self.q_values[-1] = q_value
        # Overwrite latent with prediction
        self.latents[-1] = latent_prediction

        self.network_vis = self.visualizer.visualize_activations(self.network_vis.shape)

        # Decode latent prediction back into visual representation
        latent_prediction = torch.from_numpy(latent_prediction[-1, :self.latent_vis_size]) \
            .unsqueeze(0).to(torch.float32).to(device, non_blocking=True)
        reconstruction = self.model.visual_cortex.decode(latent_prediction)
        self.predictions[-1, :self.vision_size] = reconstruction
        self.reconstructions[-1] = reconstruction

    def train_visual_cortex(self, x):
        x = x[:, :self.vision_size].reshape((-1, 3, 1, self.env.obs_pixels))
        x = torch.from_numpy(x).to(torch.float32).pin_memory().to(device, non_blocking=True)
        latents, reconstructions, loss = self.model.visual_cortex.backward(x)
        self.latents[:, :self.latent_vis_size] = latents
        self.reconstructions[-1] = reconstructions[-1]
        self.reconstruction_loss = loss
        return latents

    def train_neocortex(self, x, y):
        with ctx:
            logits, q_values, loss = self.model.neocortex(x, y)
        self.q_values[1:] = q_values
        # Calculate loss of the final prediction
        self.prediction_loss = np.mean(np.abs(logits[-1] - y.cpu().detach().numpy()[0][-1, :-1]))
        return logits, q_values

    def train_policy(self):
        # data = self.memories.sample(batch_size)
        # L1, Q1, R2, L2, Q2
        data = (self.latents[:self.context_size],
                self.actions[:self.context_size],
                self.q_values[:self.context_size],
                self.rewards,
                self.latents[1:],
                self.q_values[1:])
        data = tuple(torch.tensor(arr).to(torch.float32).to(device, non_blocking=True) for arr in data)
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
        self.network_vis = np.full((100, 100), 0.5)

    def get_batch(self, observations, actions, rewards):
        inputs = np.concatenate([observations, actions], axis=-1)
        outputs = np.concatenate([observations[1: self.context_size + 1],
                                  rewards[:, np.newaxis]], axis=-1)
        x = torch.from_numpy(inputs[:self.context_size]).unsqueeze(0)
        y = torch.from_numpy(outputs).unsqueeze(0)
        return x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)

    def model_name(self):
        return f"cvae-wm-ac-v0"

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
