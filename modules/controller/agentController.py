import os

import numpy as np
import torch

from dotmap import DotMap
from gym.spaces import flatdim, flatten

from modules.controller.controller import Controller
from contextlib import nullcontext

from modules.io.replayBuffer import ReplayBuffer
from modules.networks.Qtransformer import QTransformer
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
        self.input_size = flatdim(env.observation_space)
        self.env = env
        self.batch_size = 1
        self.block_size = 1024
        self.latent_size = 400

        # Memories and stored info for training
        self.observations = None
        self.latents = None
        self.predictions = None
        self.actions = None
        self.rewards = None
        # self.memories = ReplayBuffer(1024)
        self.loss = 0

        # Control variables for enabling training/predicting/saving weights
        self.train_interval = 1
        self.predicting = False
        self.save_interval = -1

        # Define model architecture
        self.transformerConfig = TransformerConfig(
            input_size=self.input_size + self.output_size,
            output_size=self.input_size + self.output_size + 1,
            block_size=self.block_size,  # context size
            n_layer=4, n_head=4, n_embd=self.latent_size,  # model params
            dropout=0.2, bias=True,
            weight_decay=0.0001, learning_rate=0.001,  # Optimizer settings
            betas=(0.9, 0.95), device_type=device_type
        )

        self.models = DotMap(
            # latent_representation=PredictiveCodingNetwork(
            # input_size=(3, 1, 80), n_layers=2, n_causes=[15, 25], kernel_size=[[8, 8], [3, 3]],
            # stride=[4, 2], padding=0, lam=0.1, alpha=0.1, k1=.005, k2=0.005, sigma2=10.),
            world_model=QTransformer(config if config is not None else self.transformerConfig),
            rl_block_1=HRLBlock(self.latent_size, self.output_size),
        )

        torch.cuda.synchronize()

        """self.load_model()
        self.load_submodel("world_model",
                           "../../checkpoints/checkpoint-1024x248x400x4.pth",
                           "transformer_state_dict",
                           "optimizer_state_dict")"""
        self.reset()

    def step(self, observation: dict, reward: float, time: int):
        # Internal curiosity reward for prediction errors (previous frame prediction error)
        # This guides the agent to explore more of the environment and improve its world model
        #reward += self.loss

        # Roll over memories
        self.observations = np.roll(self.observations, shift=-1, axis=0)
        self.latents = np.roll(self.latents, shift=-1, axis=0)
        self.predictions = np.roll(self.predictions, shift=-1, axis=0)
        self.actions = np.roll(self.actions, shift=-1, axis=0)
        self.rewards = np.roll(self.rewards, shift=-1, axis=0)
        self.observations[-1] = flatten(self.env.observation_space, observation)
        self.rewards[-1] = reward

        # Produce an action
        # current_state = torch.from_numpy(input)
        # current_state = current_state.to(device, non_blocking=True)
        # actions = self.models.rl_block_1(latent_states[-1])
        # self.actions = actions.cpu().detach().numpy()
        actions = self.env.random_policy()
        self.actions[-1] = actions

        # Make predictions
        if self.predicting:
            input = np.concatenate([self.observations, self.actions], axis=-1)
            latent_states = self.predict(input[1: self.block_size + 1])
        # self.latents[-1] = latent_states.cpu().detach().numpy()[-1]

        # if self.actions is not None:
        #    self.memories.add(self.latents[-2], self.actions, reward, self.latents[-1])

        # Train the agent
        if time % self.train_interval == 0 and self.train_interval != -1:
            combined = np.concatenate([self.observations, self.actions], axis=-1)
            x, y = self.get_batch(combined, self.rewards)
            self.train_wm(x, y)
            # self.train_policy(16)

        # Save the model every save_interval iterations
        if self.save_interval != -1 and time % self.save_interval == 0:
            self.save_model()
            print(f"loss: {self.loss:.4f}")

        return actions

    def predict(self, input):
        """with ctx:
            input_image = x[:, :, :240].reshape(-1, 3, 1, 80)
            #input_image = torch.tensor(input_image).pin_memory().to(device, non_blocking=True)
            latent_representation, total_loss = self.models.latent_representation(input_image)
            self.loss = total_loss.item()

            prediction = self.models.latent_representation.prediction(input_image.shape)[-1]
            self.predictions = np.roll(self.predictions, shift=-1, axis=0)
            self.predictions[-2] = np.append(prediction.reshape(240), np.zeros(8))"""
        # new_input = np.copy(self.observations[0:self.block_size])
        # predicted_obs = new_input[-1, :240]
        # new_input[-1] = np.concatenate([predicted_obs[:240], np.copy(x)[240:]])

        new_input = torch.stack([torch.from_numpy(input)])
        new_input = new_input.to(device, non_blocking=True)
        with ctx:
            logits, loss, latents = self.models.world_model(new_input, None)
        self.predictions[-1] = logits[-1]
        self.loss = np.mean(np.abs(self.predictions[-1, :240] - self.observations[-1, :240]))
        return latents

    def train_policy(self, batch_size=16):
        data = self.memories.sample(batch_size)
        if data is not None:
            self.models.rl_block_1.train(data)

    def train_wm(self, x, y):
        with ctx:
            logits, loss, latent = self.models.world_model(x, y)
        self.predictions[-1] = logits[-1]
        # Calculate loss of the final prediction
        self.loss = np.mean(np.abs(logits[-1] - y.cpu().detach().numpy()[0][-1]))

    def reset(self):
        self.observations = np.zeros((self.block_size + 1, self.input_size), dtype=np.float16)
        self.latents = np.zeros((self.block_size, self.latent_size), dtype=np.float16)
        self.predictions = np.zeros((self.block_size, self.input_size + self.output_size + 1), dtype=np.float16)
        self.actions = np.zeros((self.block_size + 1, self.output_size), dtype=np.float16)
        self.rewards = np.zeros(self.block_size, dtype=np.float16)

    def get_batch(self, data, rewards):
        x = torch.stack([torch.from_numpy(data[:self.block_size])])
        outputs = np.column_stack((data[-self.block_size:], rewards))
        y = torch.stack([torch.from_numpy(outputs)])
        return x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)

    def model_name(self):
        return f"wm-ac-hrl-v0"

    def save_model(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path_name = f"../../checkpoints/{self.model_name()}.pth"
        filename = os.path.join(dirname, path_name)

        save_dict = {}
        for name, model_obj in self.models.items():
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
        for name, model_obj in self.models.items():
            if name in load_dict:
                try:
                    model_obj.model.load_state_dict(load_dict[f"{name}"])
                    model_obj.optimizer.load_state_dict(load_dict[f"{name}_optimizer"])
                except:
                    print("Failed to load:", name)

    def load_submodel(self, name, path, model_name, optimiser_name):
        dirname = os.path.dirname(os.path.realpath(__file__))
        load_dict = torch.load(os.path.join(dirname, path))
        self.models[name].model.load_state_dict(load_dict[model_name])
        self.models[name].optimizer.load_state_dict(load_dict[optimiser_name])
