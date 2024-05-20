from abc import ABC

import pyglet
from gym.envs.classic_control import rendering

from environments.fishTank.health import Health
from environments.fishTank.mapGenerator import generate_map
from environments.fishTank.wall import Wall
from environments.environment import Environment
import numpy as np
import random, math

from modules.collisions.collisions import solve_collisions
from modules.inputType.input import Input
from modules.raymarch.ray import ray_march
from modules.renderingTools import Texture

WINDOW_W = 500
WINDOW_H = 500


class FishTank(Environment, ABC):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    x_size, y_size = 0, 0
    time = 0
    periodic_reset = True
    output_size = 3
    obs_pixels = 80
    input = Input()
    input.add_input("observation", obs_pixels * 3) \
        .add_input("actions", output_size) \
        .add_input("position", 2) \
        .add_input("direction", 2) \
        .add_input("velocity", 2)

    def __init__(self, x_size, y_size):
        super().__init__("Fish Tank")
        self.grid = None
        self.viewer = None

        FishTank.x_size, FishTank.y_size = x_size, y_size

        self.maxHealth = int(8)
        self.healthItems = []
        self.characters = []
        self.reset()
        self.points = np.zeros((1000, 5), dtype=np.float32)

    def add_character(self, character):
        self.characters.append(character)

    def simulate(self):
        from modules.controller.agentController import AgentController

        # Execute one time step within the environment
        FishTank.time += 1

        # create health
        self.spawn_health()

        # Simulate characters
        for character in self.characters:
            # perform character simulation
            character.step(self)

            solve_collisions(self.grid, character)

            character.observation = self.get_observation(character)

            # Add intrinsic curiosity reward for prediction errors
            if isinstance(character.controller, AgentController):
                new_observation = np.reshape(FishTank.input.get_input(character.observation, "observation"),
                                             (FishTank.obs_pixels * 3))
                prediction = FishTank.input.get_input(character.controller.predictions[-1], "observation")
                MAE = np.mean(np.abs(new_observation - prediction))
                character.add_reward(MAE)  # 0 - 1

            # Add reward for colliding with health
            for health in self.healthItems:
                dist = math.sqrt((health.x - character.body[0].x) ** 2 + (health.y - character.body[0].y) ** 2)
                if dist < character.size + health.size:
                    self.healthItems.remove(health)
                    character.add_reward(health.size / 10)  # 0.2 - 0.5

            # Penalise collisions
            if character.colliding:
                character.add_reward(-0.05)

        if FishTank.time % 5000 == 0 and FishTank.periodic_reset:
            self.reset()

    def reset(self):
        # Reset the state of the environment to an initial state
        self.healthItems = []
        self.grid = generate_map([self.x_size, self.y_size], round(self.x_size / 20), 0.6, 0.2)
        for character in self.characters:
            character.reset(self)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human', close=False):
        from modules.controller.agentController import AgentController

        if self.viewer is None:
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

        scale = max(WINDOW_W / self.x_size, WINDOW_H / self.x_size)
        self.viewer.transform.set_scale(scale, scale)

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if self.grid[x][y]:
                    Wall.render(self.viewer, x, y, 20)

        # Draw health
        for health in self.healthItems:
            health.render(self.viewer)

        # Draw bot
        for character in self.characters:
            character.render(self.viewer)

        # Draw output view
        if len(self.characters) != 0:
            char = self.characters[0]
            new_shape = (round(FishTank.input.size_of("observation") / 3), 3)
            last_input = char.controller.inputs[-1]
            o = np.reshape(FishTank.input.get_input(last_input, "observation"), new_shape)
            self.draw_observation(0, self.y_size + 40, 180, 20, o, colour=(50, 50, 50))

            startX = 250
            startY = self.y_size + 40
            speed = math.sqrt((char.body[0].x - char.body[0].prev_x) ** 2 +
                              (char.body[0].y - char.body[0].prev_y) ** 2)
            endX = startX + math.cos(char.dir) * speed * 5
            endY = startY + math.sin(char.dir) * speed * 5
            line = self.viewer.draw_line((startX, startY), (endX, endY), linewidth=3)
            line.set_color(255, 0.0, 0.0, 1)

            if isinstance(char.controller, AgentController):
                last_prediction = char.controller.predictions[-2]
                p = np.reshape(FishTank.input.get_input(last_prediction, "observation"), new_shape)
                self.draw_observation(0, self.y_size + 20, 180, 20, p, colour=(50, 50, 50))
                label = pyglet.text.Label(f"Loss: {self.characters[0].controller.loss:.4f}", font_size=13,
                                          font_name="Russo One",
                                          x=200, y=WINDOW_H, anchor_x='left', anchor_y='top',
                                          color=(255, 255, 255, 255))
                self.viewer.add_label(label)

                err = char.controller.inputs[-1:] - char.controller.predictions[-2:]
                mean_abs_err = np.mean(np.abs(err), 0)
                e = np.reshape(FishTank.input.get_input(mean_abs_err, "observation"), new_shape)
                self.draw_observation(0, self.y_size, 180, 20, e, colour=(50, 50, 50))

                # Draw reward
                label = pyglet.text.Label(f"Reward: {self.characters[0].reward:.4f}", font_size=13,
                                          font_name="Russo One",
                                          x=200, y=WINDOW_H - 30, anchor_x='left', anchor_y='top',
                                          color=(255, 255, 255, 255))
                self.viewer.add_label(label)

                # Draw observation zone
                bounds = [self.x_size - 100, self.y_size, 100, 75]
                quad = rendering.PolyLine([(bounds[0], bounds[1]),
                                           (bounds[0], bounds[1] + bounds[3]),
                                           (bounds[0] + bounds[2], bounds[1] + bounds[3]),
                                           (bounds[0] + bounds[2], bounds[1])], True)
                quad.set_color(*(50, 50, 50))
                quad.set_linewidth(2)
                self.viewer.add_onetime(quad)
                """pred_pos = FishTank.input.get_input(char.controller.predictions[-2], "position")
                pred_dir = FishTank.input.get_input(char.controller.predictions[-2], "direction")
                pred_dir = math.atan2(pred_dir[1], pred_dir[0])

                self.points = np.roll(self.points, shift=-FishTank.obs_pixels, axis=0)

                halfFidelity = char.fidelity * 0.5
                for x in range(FishTank.obs_pixels):
                    original_colour = Wall.colour
                    if abs(p[x][1] - p[x][0])*255 >= 20:
                        original_colour = Health.colour
                    # Inverse calculation to get distance from brightness (super fucked)
                    dist = (1 - (p[x][1] * 255)/original_colour[1])
                    ang = math.radians((x - halfFidelity) / char.fidelity * char.fov)
                    posX = pred_pos[0] + dist * math.cos(pred_dir + ang)
                    posY = pred_pos[1] + dist * math.sin(pred_dir + ang)
                    self.points[- FishTank.obs_pixels + x] = [*p[x], posX, posY]

                for x in range(self.points.shape[0]):
                    point = self.points[x]
                    t = rendering.Transform(translation=(bounds[0] + bounds[2] * point[3],
                                                         bounds[1] + bounds[3] * point[4]))
                    pred_body = self.viewer.draw_circle(radius=0.5, filled=True,
                                                        color=(point[0]*255, point[1]*255, point[2]*255))
                    pred_body.add_attr(t)

                t = rendering.Transform(translation=(bounds[0] + bounds[2] * pred_pos[0], bounds[1] + bounds[3] * pred_pos[1]))
                pred_body = self.viewer.draw_circle(radius=2, filled=True, color=char.get_colour())
                pred_body.add_attr(t)"""

        label = pyglet.text.Label(f"Time: {FishTank.time}", font_size=13,
                                  font_name="Russo One",
                                  x=0, y=WINDOW_H, anchor_x='left', anchor_y='top',
                                  color=(255, 255, 255, 255))
        self.viewer.add_label(label)

        return self.viewer.render()

    def spawn_health(self):
        if len(self.healthItems) >= self.maxHealth:
            return
        x, y = self.get_random_pos()
        health = Health(x, y)
        self.healthItems.append(health)

    def get_random_pos(self):
        validPos = False
        x, y = 0, 0
        while not validPos:
            x = random.randrange(0, self.x_size)
            y = random.randrange(0, self.y_size)
            validPos = not self.grid[math.floor(x / 20)][math.floor(y / 20)]
        return x, y

    def draw_observation(self, x, y, w, h, observation, colour=(100, 100, 255)):
        observation = np.repeat(observation[:, np.newaxis, :], 5, axis=1)
        observation = np.flip(np.swapaxes(observation, 0, 1), 1)
        observation = np.ascontiguousarray(observation, dtype=np.float32)

        tex = Texture(observation, x, y, w, h)
        self.viewer.add_onetime(tex)

        quad = rendering.PolyLine([(x, y), (x, y + h), (x + w, y + h), (x + w, y)], True)
        quad.set_color(*colour)
        quad.set_linewidth(2)
        self.viewer.add_onetime(quad)

    def get_observation(self, character):
        observation = np.zeros((character.fidelity, 3), dtype=np.float32)
        ray_data = ray_march(character, self.grid, self.healthItems)

        # Create masks for each object type
        wall_mask = ray_data[:, 0] == 1
        health_mask = ray_data[:, 0] == 2

        # Update observation array based on object types
        observation[wall_mask, :3] = np.array(Wall.colour[:3]) / 255 * (
                    1 - ray_data[wall_mask, 1][:, np.newaxis] / FishTank.x_size)
        observation[health_mask, :3] = np.array(Health.colour[:3]) / 255 * (
                    1 - ray_data[health_mask, 1][:, np.newaxis] / FishTank.x_size)

        noise = np.random.normal(0, 1, observation.shape)
        observation += noise * 0.005
        return observation

    def get_reward(self):
        return 0

    def get_info(self):
        return FishTank.time
