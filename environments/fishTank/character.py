import numpy as np
from gym.envs.classic_control import rendering
import math
import random

from modules.utils.mathHelper import interpolate, clamp, angle_diff
from modules.verlet.constraints import constrain_distance, constrain_position
from modules.verlet.point import Point
from modules.verlet.utils import find_perpendicular_points


class Character:
    maxAng = math.radians(5)
    maxAccel = 5

    def __init__(self, x, y, obs_pixels):
        self.dir_change = 0
        self.dir_change_avg = 0
        self.dir = 0
        self.target_dir = 0
        self.target_speed = 0
        self.attacking = False
        self.collision_force = 0.0
        self.accel = float(0)
        self.hp = float(100)
        self.size = 5
        self.fov = 120
        self.fidelity = round(obs_pixels)
        self.view_dist = 200
        self.rayDist = [self.view_dist] * self.fidelity
        self.observation = None
        self.body = [Point(x, y, 3), Point(x, y, 4), Point(x, y, 2), Point(x, y, 1)]

    def reset(self, env):
        x, y = env.x_size / 2, env.y_size / 2
        for i in range(len(self.body)):
            self.body[i].x = x - 10 * i
            self.body[i].y = y
            self.body[i].prev_x = x - 10 * i
            self.body[i].prev_y = y
        self.dir = math.radians(random.randrange(0, 360))
        self.attacking = False
        self.accel = 0
        self.hp = 1000

    def random_policy(self):
        reset = self.target_speed == 0 and self.target_dir == 0 and self.dir_change_avg == 0

        dir_change = angle_diff(self.dir, self.target_dir)
        dir_change = min(max(-Character.maxAng, dir_change * 0.05), Character.maxAng)
        dir_change_input = dir_change / Character.maxAng
        actions = [self.target_speed, dir_change_input]
        speed, dir = self.body[0].get_velocity()
        if random.random() < 0.2 - clamp(0, speed / 5, 1) * 0.2 or reset:
            self.target_dir = random.random() * math.pi * 2
        if random.random() < 0.05 or reset:
            self.target_speed = random.random() * (-1 if random.random() < 0.2 else 1)
        if self.collision_force > 0.1:
            self.target_dir = self.dir + math.pi
        return actions

    def step(self, env, action):
        if action is not None:
            self.accel = action[0] * (1 if action[0] >= 0 else 0.2) * self.maxAccel

            self.dir_change = action[1] * self.maxAng
            self.dir_change_avg = self.dir_change_avg * 0.9 + self.dir_change * 0.1
            self.dir += self.dir_change

            force_x = self.accel * math.cos(self.dir)
            force_y = self.accel * math.sin(self.dir)
            self.body[0].apply_force(force_x, force_y)

        # Apply physics constraints to body
        for i in range(len(self.body)):
            if i + 1 < len(self.body):
                constrain_distance(self.body[i], self.body[i + 1], 10)
            collision_force = constrain_position(self.body[i], 0, 0, env.x_size, env.y_size)
            if i == 0:
                self.collision_force = collision_force
            self.body[i].update()

    def render(self, viewer):
        # draw view cone
        half_fidelity = self.fidelity * 0.5
        for i in range(self.fidelity):
            ang = math.radians((i - half_fidelity) / self.fidelity * self.fov)
            dist = self.rayDist[i]
            end_x = self.body[0].x + math.cos(self.dir + ang) * dist
            end_y = self.body[0].y + math.sin(self.dir + ang) * dist
            ray = viewer.draw_line((self.body[0].x, self.body[0].y), (end_x, end_y))
            ray.set_color(255, 0.0, 0.0, 0.2)

        # draw spike
        front_x = math.cos(self.dir) * self.size
        front_y = math.sin(self.dir) * self.size

        # draw eyes
        eye_x = math.cos(self.dir + math.pi * 0.5) * self.size * 0.7
        eye_y = math.sin(self.dir + math.pi * 0.5) * self.size * 0.7

        iris_x = math.cos(self.dir + self.dir_change_avg * 10) * self.size * 0.25
        iris_y = math.sin(self.dir + self.dir_change_avg * 10) * self.size * 0.25

        front_x *= 0.75
        front_y *= 0.75
        t = rendering.Transform(translation=(self.body[0].x + front_x + eye_x, self.body[0].y + front_y + eye_y))
        eye_l = viewer.draw_circle(radius=self.size * 0.5, filled=True, color=(255, 255, 255))
        eye_l.add_attr(t)

        t = rendering.Transform(translation=(self.body[0].x + front_x - eye_x, self.body[0].y + front_y - eye_y))
        eye_r = viewer.draw_circle(radius=self.size * 0.5, filled=True, color=(255, 255, 255))
        eye_r.add_attr(t)

        # iris
        t = rendering.Transform(
            translation=(self.body[0].x + front_x + eye_x + iris_x, self.body[0].y + front_y + eye_y + iris_y))
        iris1 = viewer.draw_circle(radius=self.size * 0.3, filled=True, color=(0.0, 0.0, 0.0))
        iris1.add_attr(t)

        t = rendering.Transform(
            translation=(self.body[0].x + front_x - eye_x + iris_x, self.body[0].y + front_y - eye_y + iris_y))
        iris2 = viewer.draw_circle(radius=self.size * 0.3, filled=True, color=(0.0, 0.0, 0.0))
        iris2.add_attr(t)

        # draw tail
        for i in range(len(self.body)):
            tail = self.body[i]
            tail.render(viewer, self.get_colour())

            if i + 1 >= len(self.body):
                continue
            next_tail = self.body[i + 1]

            if i % 2 == 0:
                t = rendering.Transform(translation=(tail.x, tail.y))
                angle = math.atan2(tail.y - next_tail.y, tail.x - next_tail.x)
                deg90 = math.pi / 2
                fin_colour = interpolate(self.get_colour(), (255, 255, 255), 0.3)
                fin1 = [[math.cos(angle + deg90) * tail.mass,
                         math.sin(angle + deg90) * tail.mass],
                        [math.cos(angle + deg90 * 2) * tail.mass * 4,
                         math.sin(angle + deg90 * 2) * tail.mass * 4],
                        [math.cos(angle + deg90 * 1.7) * tail.mass * 8,
                         math.sin(angle + deg90 * 1.7) * tail.mass * 8]]
                geom = viewer.draw_polygon(fin1, filled=True, color=fin_colour)
                geom.add_attr(t)
                fin1 = [[math.cos(angle - deg90) * tail.mass,
                         math.sin(angle - deg90) * tail.mass],
                        [math.cos(angle - deg90 * 2) * tail.mass * 4,
                         math.sin(angle - deg90 * 2) * tail.mass * 4],
                        [math.cos(angle - deg90 * 1.7) * tail.mass * 8,
                         math.sin(angle - deg90 * 1.7) * tail.mass * 8]]
                geom = viewer.draw_polygon(fin1, filled=True, color=fin_colour)
                geom.add_attr(t)

            poly1 = find_perpendicular_points(tail, next_tail,
                                              tail.mass, next_tail.mass)
            viewer.draw_polygon(poly1, filled=True, color=self.get_colour())

    def heal(self, health):
        self.hp += health.size

    def get_colour(self):
        life = max(min(1.0, self.hp / 100), 0.4)
        return (200 * life + 55, 0, 0)

