import math
import random
from abc import ABC

import numpy as np
import pyglet
from gym.envs.classic_control import rendering
from gym.spaces import flatten, flatdim
from gym.vector.utils import spaces

from environments.environment import Environment
from environments.fishTank.fish import Fish
from environments.fishTank.health import Health
from environments.fishTank.mapGenerator import generate_map
from environments.fishTank.wall import Wall
from modules.collisions.collisions import solve_collisions
from modules.controller import HTMController
from modules.raymarch.ray import ray_march
from modules.renderingTools import Texture
from modules.utils.mathHelper import angle_to_vector, get_velocity
from matplotlib.colors import LinearSegmentedColormap

WINDOW_W = 500
WINDOW_H = 500


class FishTank(Environment, ABC):
    """
    A 2d procedurally generated fish tank containing food pellets and a physics simulation driven using Verlet
    integration. Each character in the environment receives a sequence of observations generated by casting rays out
    from the character's head. This 1D RGB image is flattened into a single array, and appended with the character's
    current actions, their body position as a percentage of the tank size in the x and y dimension, their direction in
    terms of normalised x and y coordinates, and their velocity is normalised by their maximum acceleration.

    The controller then processes the input and returns an array of outputs Actions correspond to
    [forward_thrust (0 - 1),
    rotational acceleration (0 full counter-clockwise, 0.5 - still, 1 - full clockwise]
    """
    metadata = {'render.modes': ['humaresetn']}
    reset_interval = 5000
    obs_pixels = 120
    stereoscopic = False
    num_NPCS = 3

    def __init__(self):
        super().__init__("FishTank-v0")
        self.grid = None

        self.x_size = 400
        self.y_size = 300

        self.view = 0
        self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
        bg = rendering.FilledPolygon([(0, 0), (0, WINDOW_H), (WINDOW_W, WINDOW_H), (WINDOW_W, 0)])
        bg.set_color(0, 0, 0)
        border = rendering.PolyLine([(0, 0), (0, self.y_size), (self.x_size, self.y_size), (self.x_size, 0)],
                                    close=True)
        border.set_linewidth(5)
        border.set_color(*Wall.colour)

        self.viewer.add_geom(bg)
        self.viewer.add_geom(border)
        self.viewer.transform = rendering.Transform()
        self.viewer.window.push_handlers(
            on_mouse_motion=self.on_mouse_motion
        )

        self.character = Fish(self.x_size / 2, self.y_size / 2, obs_pixels=FishTank.obs_pixels)
        self.NPCs = []

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.OrderedDict({
            "vision": spaces.Box(low=0, high=1, shape=(self.obs_pixels * 3,)),
            "dynamics": spaces.OrderedDict({
                "collision_force": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.character.body),)),
                "direction": spaces.Box(low=0, high=1, shape=(2,)),
                "velocity": spaces.Box(low=0, high=1, shape=(1,)),
                "position": spaces.Box(low=0, high=1, shape=(2,)),
            })
        })
        self.time = 1
        self.maxHealth = 8
        self.healthItems = []

        self.reset()

        for i in range(FishTank.num_NPCS):
            x, y = self.get_random_pos()
            self.NPCs.append(Fish(x, y, obs_pixels=0))

    def step(self, actions: list):
        if self.time % FishTank.reset_interval == 0 and FishTank.reset_interval != -1:
            self.reset()

        # perform character simulation
        self.character.step(self, actions)
        solve_collisions(self.grid, self.character)

        for npc in self.NPCs:
            npc.step(self, npc.random_policy())
            solve_collisions(self.grid, npc)

        # Compute reward
        reward = 0

        # Add reward for colliding with health
        for health in self.healthItems:
            dist = math.sqrt((health.x - self.character.body[0].x) ** 2 + (health.y - self.character.body[0].y) ** 2)
            if dist < self.character.size + health.size:
                self.healthItems.remove(health)
                reward += health.size * 100
                self.spawn_health()

        # Penalise collisions
        if abs(self.character.collision_force[0]) > 0.05:
            reward -= 0.5 + self.character.collision_force[0] * 3

        # Add reward for going faster (assuming forward thrust > 0 so reversing isn't penalised)
        velocity = get_velocity(self.character.body[0]) / self.character.maxAccel
        if actions is not None and actions[0] > 0:
            reward += velocity * 0.2

        self.time += 1

        observation = {
            "vision": self.get_observation(self.character),
            "dynamics": {
                "collision_force": self.character.collision_force,
                "direction": angle_to_vector(self.character.dir),
                "velocity": [velocity],
                "position": [self.character.body[0].x, self.character.body[0].y],
            }
        }

        return observation, reward

    def random_policy(self):
        return self.character.random_policy()

    def reset(self):
        # Reset the state of the environment to an initial state
        self.grid = generate_map([self.x_size, self.y_size], round(self.x_size / 20), 0.6, 0.2)
        #self.grid = generate_maze([self.x_size, self.y_size], round(self.x_size / 20))
        self.character.reset(self)
        for npc in self.NPCs:
            x, y = self.get_random_pos()
            npc.set_position(x, y)
        self.healthItems = []
        self.time = round(self.time/self.reset_interval) * self.reset_interval
        while len(self.healthItems) < self.maxHealth:
            self.spawn_health()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, agent, mode='human', close=False):
        from modules.controller.HRLController import HRLController
        from modules.controller.HTMController import HTMController

        scale = max(WINDOW_W / self.x_size, WINDOW_H / self.x_size)
        self.viewer.transform.set_scale(scale, scale)

        # Get output
        obs_shape = (self.obs_pixels, 3)
        vision_size = flatdim(self.observation_space["vision"])

        if isinstance(agent, (HRLController, HTMController)):
            o = np.reshape(agent.observations[-2, :vision_size], obs_shape)
            r = np.reshape(agent.reconstructions[-2], obs_shape)
            p = np.reshape(agent.predictions[-2, :vision_size], obs_shape)

            if self.view != 0:
                self.viewer.transform.set_scale(1, 1)
                if self.view == 1:
                    network_vis = agent.visualizer.visualize_activations((500, 500))
                else:
                    network_vis = agent.visualizer.visualize_attention(self.view - 1, (500, 500))
                self.draw_neural_map(network_vis, WINDOW_W*0.025, WINDOW_H*0.025, WINDOW_W*0.95, WINDOW_H*0.95, colour=(50, 50, 50))
                self.draw_observation(o, WINDOW_W - 180, 40, 180, 20, colour=(50, 50, 50))
                self.draw_observation(r, WINDOW_W - 180, 20, 180, 20, colour=(50, 50, 50))
                return self.viewer.render()
        else:
            o = np.reshape(agent.observation[:vision_size], obs_shape)

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if self.grid[x][y]:
                    Wall.render(self.viewer, x, y, 20)

        # Draw health
        for health in self.healthItems:
            health.render(self.viewer)

        # Draw bot
        self.character.render(self.viewer, self.stereoscopic)

        for npc in self.NPCs:
            npc.render(self.viewer, self.stereoscopic, NPC=True)

        self.draw_observation(o, WINDOW_W/scale - 180, WINDOW_H/scale - 20, 180, 20, colour=(50, 50, 50))

        info_str = f"Time: {self.time}"
        if isinstance(agent, (HRLController, HTMController)):
            self.draw_observation(r, WINDOW_W/scale - 180, WINDOW_H/scale - 40, 180, 20, colour=(50, 50, 50))
            self.draw_observation(p, WINDOW_W/scale - 180, WINDOW_H/scale - 60, 180, 20, colour=(50, 50, 50))
            if agent.prediction is not None:
                self.draw_image(agent.prediction, WINDOW_W/scale - 180, WINDOW_H/scale - 60, 180, 20, colour=(50, 50, 50))

            info_str += f"\nReconstr_loss: {agent.reconstruction_loss:.3f}"
            info_str += f"\nPred_loss: {agent.prediction_loss:.3f}"
            info_str += f"\nReward: {agent.rewards[-1]:.3f}"
            #info_str += f"\nQ_value: {agent.q_values[-1]:.3f}"
            info_str += f"\nActions: [{agent.actions[-1][0]:.3f}, {agent.actions[-1][1]:.3f}]"

        label = pyglet.text.Label(info_str, font_size=10,
                                  multiline=True,
                                  font_name="Russo One",
                                  width=200,
                                  x=0, y=WINDOW_H, anchor_x='left', anchor_y='top',
                                  color=(255, 255, 255, 255))
        self.viewer.add_label(label)

        return self.viewer.render()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y

    def spawn_health(self):
        if len(self.healthItems) >= self.maxHealth:
            return
        x, y = self.get_random_pos()
        health = Health(x, y)
        self.healthItems.append(health)

    def get_random_pos(self):
        valid_pos = False
        x, y = 0, 0
        while not valid_pos:
            x = random.randrange(0, self.x_size)
            y = random.randrange(0, self.y_size)
            valid_pos = not self.grid[math.floor(x / 20)][math.floor(y / 20)]
        return x, y

    def draw_image(self, observation, x, y, w, h, colour=(100, 100, 255)):
        # Expects W,H,C format
        observation = np.flip(np.swapaxes(observation, 0, 1), 1)
        observation = np.ascontiguousarray(observation, dtype=np.float32)

        tex = Texture(observation, x, y, w, h)
        self.viewer.add_onetime(tex)

    def draw_observation(self, observation, x, y, w, h, colour=(100, 100, 255)):
        observation = np.repeat(observation[:, np.newaxis, :], 5, axis=1)
        observation = np.flip(np.swapaxes(observation, 0, 1), 1)
        observation = np.ascontiguousarray(observation, dtype=np.float32)

        tex = Texture(observation, x, y, w, h)
        self.viewer.add_onetime(tex)

        """quad = rendering.PolyLine([(x, y), (x, y + h), (x + w, y + h), (x + w, y)], True)
        quad.set_color(*colour)
        quad.set_linewidth(1)
        self.viewer.add_onetime(quad)"""

    def draw_neural_map(self, map, x, y, w, h, colour=(100, 100, 255)):
        colors = [(1, 0.3333, 0), (0, 0, 0), (0.4157, 1, 0.2196)]  # Red, Black, Green
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

        map = np.swapaxes(map, 0, 1)
        map = cmap(map)[:, :, :3]
        map = np.ascontiguousarray(map, dtype=np.float32)

        tex = Texture(map, x, y, w, h)
        self.viewer.add_onetime(tex)

        quad = rendering.PolyLine([(x, y), (x, y + h), (x + w, y + h), (x + w, y)], True)
        quad.set_color(*colour)
        quad.set_linewidth(1)
        self.viewer.add_onetime(quad)

    def get_observation(self, character):
        observation = np.zeros((character.fidelity, 3), dtype=np.float32)
        ray_data = ray_march(character, self.NPCs, self.grid, self.healthItems, self.stereoscopic)

        # Create masks for each object type
        wall_mask = ray_data[:, 0] == 1
        health_mask = ray_data[:, 0] == 2
        npc_mask = (ray_data[:, 0] >= 3) & (ray_data[:, 0] < 4)

        # Update observation array based on object types
        observation[wall_mask, :3] = np.array(Wall.colour[:3]) / 255 * (
                1 - ray_data[wall_mask, 1][:, np.newaxis] / 300)
        observation[health_mask, :3] = np.array(Health.colour[:3]) / 255 * (
                1 - ray_data[health_mask, 1][:, np.newaxis] / 300)

        c = np.array([Fish.colour[:3]]) * (4 - ray_data[npc_mask, 0][:, np.newaxis]) \
            + np.array([[254, 200, 200]]) * (ray_data[npc_mask, 0][:, np.newaxis] - 3)
        observation[npc_mask, :3] = c / 255 * (
                1 - ray_data[npc_mask, 1][:, np.newaxis] / 300)

        noise = np.random.normal(0, 1, observation.shape)
        observation += noise * 0.005
        return observation
