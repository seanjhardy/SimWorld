import numpy as np
from gym.envs.classic_control import rendering
import math
import random

from environments.fishTank.fishTank import FishTank
from modules.utils.mathHelper import angle_to_vector, get_velocity, interpolate, clamp, angle_diff
from modules.verlet.constraints import constrain_distance, constrain_position
from modules.verlet.point import Point
from modules.verlet.utils import find_perpendicular_points


class Character:
    maxAng = math.radians(5)
    maxAccel = 5

    def __init__(self, x, y):
        self.dirChange = 0
        self.dirChangeAvg = 0
        self.dir = 0
        self.target_dir = 0
        self.target_speed = 0
        self.attacking = False
        self.colliding = False
        self.accel = float(0)
        self.hp = float(100)
        self.size = 5
        self.fov = 120
        self.fidelity = round(FishTank.input.size_of("observation") / 3)
        self.view_dist = 200
        self.rayDist = [self.view_dist] * self.fidelity
        self.observation = None
        self.controller = None
        self.actions = [0, 0.5, 0]
        self.body = [Point(x, y, 3), Point(x, y, 4), Point(x, y, 2), Point(x, y, 1)]
        self.reward = 0

    def add_controller(self, controller):
        self.controller = controller

    def reset(self, env):
        x, y = env.get_random_pos()
        for i in range(len(self.body)):
            self.body[i].x = x
            self.body[i].y = y
            self.body[i].prev_x = x
            self.body[i].prev_y = y
        self.dir = math.radians(random.randrange(0, 360))
        self.attacking = False
        self.accel = float(0)
        self.hp = float(1000)
        self.controller.reset()

    def step(self, env):
        # Calculate movement from previous actions
        self.accel = self.actions[0] * self.maxAccel

        self.dirChange = (self.actions[1] * 2 - 1) * self.maxAng
        self.dirChangeAvg = self.dirChangeAvg * 0.9 + self.dirChange * 0.1
        self.dir += self.dirChange

        forceX = self.accel * math.cos(self.dir)
        forceY = self.accel * math.sin(self.dir)
        self.body[0].apply_force(forceX, forceY)

        # Apply physics constraints to body
        for i in range(len(self.body)):
            if i + 1 < len(self.body):
                constrain_distance(self.body[i], self.body[i + 1], 10)
            constrain_position(self.body[i], 0, 0, env.x_size, env.y_size)
            self.body[i].update()

        # Concatenate all the inputs
        input = np.concatenate([
            self.observation.flatten(),
            self.actions,
            [self.body[0].x / FishTank.x_size, self.body[0].y / FishTank.y_size],
            angle_to_vector(self.dir),
            [get_velocity(self.body[0]) / self.maxAccel]
        ])

        # Make prediction and get next actions
        self.actions = self.controller.run(input)

        # Reset reward
        self.reward = 0

        if self.actions is None:
            dirChange = angle_diff(self.dir, self.target_dir)
            dirChange = min(max(-Character.maxAng, dirChange * 0.05), Character.maxAng)
            dirChangeInput = (dirChange / Character.maxAng + 1) / 2
            self.actions = [self.target_speed, dirChangeInput, 0]
            speed, dir = self.body[0].get_velocity()
            if random.random() < 0.2 - clamp(0, speed / 5, 1) * 0.2:
                self.target_dir = random.random() * math.pi * 2
            if random.random() < 0.05:
                self.target_speed = random.random()
            if self.colliding:
                self.target_dir = self.dir + math.pi

        self.colliding = False
        return

    def render(self, viewer):
        # draw view cone
        halfFidelity = self.fidelity * 0.5
        for i in range(self.fidelity):
            ang = math.radians((i - halfFidelity) / self.fidelity * self.fov)
            dist = self.rayDist[i]
            endX = self.body[0].x + math.cos(self.dir + ang) * dist
            endY = self.body[0].y + math.sin(self.dir + ang) * dist
            ray = viewer.draw_line((self.body[0].x, self.body[0].y), (endX, endY))
            ray.set_color(255, 0.0, 0.0, 0.2)

        # draw spike
        frontX = math.cos(self.dir) * self.size
        frontY = math.sin(self.dir) * self.size

        # draw eyes
        eyeX = math.cos(self.dir + math.pi * 0.5) * self.size * 0.7
        eyeY = math.sin(self.dir + math.pi * 0.5) * self.size * 0.7

        irisX = math.cos(self.dir + self.dirChangeAvg * 10) * self.size * 0.25
        irisY = math.sin(self.dir + self.dirChangeAvg * 10) * self.size * 0.25

        frontX *= 0.75
        frontY *= 0.75
        t = rendering.Transform(translation=(self.body[0].x + frontX + eyeX, self.body[0].y + frontY + eyeY))
        eyeL = viewer.draw_circle(radius=self.size * 0.5, filled=True, color=(255, 255, 255))
        eyeL.add_attr(t)

        t = rendering.Transform(translation=(self.body[0].x + frontX - eyeX, self.body[0].y + frontY - eyeY))
        eyeR = viewer.draw_circle(radius=self.size * 0.5, filled=True, color=(255, 255, 255))
        eyeR.add_attr(t)

        # iris
        t = rendering.Transform(
            translation=(self.body[0].x + frontX + eyeX + irisX, self.body[0].y + frontY + eyeY + irisY))
        iris1 = viewer.draw_circle(radius=self.size * 0.3, filled=True, color=(0.0, 0.0, 0.0))
        iris1.add_attr(t)

        t = rendering.Transform(
            translation=(self.body[0].x + frontX - eyeX + irisX, self.body[0].y + frontY - eyeY + irisY))
        iris2 = viewer.draw_circle(radius=self.size * 0.3, filled=True, color=(0.0, 0.0, 0.0))
        iris2.add_attr(t)

        # draw cell
        # t = rendering.Transform(translation=(self.x, self.y))
        # geom = viewer.draw_circle(radius=self.size, filled=True, color=self.get_colour())
        # geom.add_attr(t)

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

    def add_reward(self, reward):
        self.reward += reward
