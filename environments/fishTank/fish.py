import numpy as np
from gym.envs.classic_control import rendering
import math
import random

from modules.utils.mathHelper import interpolate, clamp, angle_diff, bezier, norm_angle, clockwise_angle_diff
from modules.verlet.constraints import constrain_distance, constrain_position, constrain_angle
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
        self.accel = float(0)
        self.hp = float(100)
        self.size = 5
        self.fov = 120
        self.fidelity = round(obs_pixels)
        self.view_dist = 200
        self.rayDist = [self.view_dist] * self.fidelity
        self.observation = None
        self.body = [Point(x, y, 4), Point(x, y, 5), Point(x, y, 2), Point(x, y, 0.6)]
        self.collision_force = np.zeros(len(self.body))

    def reset(self, env):
        x, y = env.x_size / 2, env.y_size / 2
        for i in range(len(self.body)):
            self.body[i].x = x - 15 * i
            self.body[i].y = y
            self.body[i].prev_x = x - 15 * i
            self.body[i].prev_y = y

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
        if abs(self.collision_force[0]) > 0.1:
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

        delta = angle_diff(math.atan2(self.body[1].y - self.body[0].y,
                                      self.body[1].x - self.body[0].x), self.dir + math.pi)

        if delta <= -math.pi:
            delta += 2 * math.pi
        elif delta >= math.pi:
            delta -= 2 * math.pi

        point1 = Point(self.body[0].x * 2 - self.body[1].x, self.body[0].y * 2 - self.body[1].y)
        self.body[0].rotate(point1, -delta * 0.01)
        self.body[1].rotate(self.body[0], delta * 0.01)

        for i in range(len(self.body)):
            if i + 1 < len(self.body):
                constrain_distance(self.body[i], self.body[i + 1], 15)
            if i + 2 < len(self.body):
                constrain_distance(self.body[i], self.body[i + 2], 30, factor=0.05)
            self.collision_force[i] = constrain_position(self.body[i], 0, 0, env.x_size, env.y_size)
            self.body[i].update()

    def render(self, viewer, stereoscopic):
        # draw view cone
        half_fidelity = self.fidelity * 0.5
        for i in range(self.fidelity):
            start_x = self.body[0].x
            start_y = self.body[0].y
            if stereoscopic:
                if i % 2 == 0:  # Odd index
                    index = (self.fidelity + 1) // 2 + (i - 1) // 2
                else:  # Even index
                    index = i // 2
                sign = 1 if i % 2 == 0 else -1
                start_x += math.cos(self.dir + sign * (math.pi / 2)) * 5
                start_y += math.sin(self.dir + sign * (math.pi / 2)) * 5
            else:
                index = i
            ang = math.radians((i - half_fidelity) / self.fidelity * self.fov)
            dist = self.rayDist[index]
            end_x = start_x + math.cos(self.dir + ang) * dist
            end_y = start_y + math.sin(self.dir + ang) * dist
            ray = viewer.draw_line((start_x, start_y), (end_x, end_y))
            ray.set_color(255, 0.0, 0.0, 0.2)

        # draw tail
        fin_colour = interpolate(self.get_colour(), (255, 255, 255), 0.3)
        body_fin_colour = interpolate(self.get_colour(), (0, 0, 0), 0.4)

        a2 = self.body[1].angle_to(self.body[0])
        a3 = self.body[2].angle_to(self.body[1])
        a4 = self.body[3].angle_to(self.body[2])

        dir_point = Point(self.body[0].x + math.cos(self.dir),
                          self.body[0].y + math.sin(self.dir))
        p1 = dir_point.sub(self.body[0])
        p2 = self.body[2].sub(self.body[3])
        skew = clockwise_angle_diff(p1, p2)

        head_to_mid = angle_diff(self.dir, a3)

        total_curvature = angle_diff(self.dir, a4)

        self.draw_ventral_fins(viewer,
                               self.body[0].x, self.body[0].y,
                               a2, self.body[0].mass, fin_colour, skew)

        angle = math.atan2(self.body[1].y - self.body[2].y,
                           self.body[1].x - self.body[2].x)
        self.draw_ventral_fins(viewer,
                               self.body[2].x * 0.5 + self.body[1].x * 0.5,
                               self.body[2].y * 0.5 + self.body[1].y * 0.5,
                               angle, self.body[2].mass * 1.2, fin_colour, skew)

        for i in range(len(self.body)):
            tail = self.body[i]
            tail.render(viewer, self.get_colour())

            if i + 1 >= len(self.body):
                continue
            next_tail = self.body[i + 1]
            poly1 = find_perpendicular_points(tail, next_tail,
                                              tail.mass, next_tail.mass)
            viewer.draw_polygon(poly1, filled=True, color=self.get_colour())


        dorsal_fin = [[self.body[0].x * 0.5 + self.body[1].x * 0.5,
                       self.body[0].y * 0.5 + self.body[1].y * 0.5],
                      [self.body[1].x,
                       self.body[1].y],
                      [self.body[2].x,
                       self.body[2].y]]
        dorsal_fin += bezier(self.body[2].x,
                             self.body[2].y,
                             self.body[1].x + math.cos(a3 + math.pi / 2) * head_to_mid * 3,
                             self.body[1].y + math.sin(a3 + math.pi / 2) * head_to_mid * 3,
                             self.body[0].x * 0.5 + self.body[1].x * 0.5,
                             self.body[0].y * 0.5 + self.body[1].y * 0.5)
        viewer.draw_polygon(dorsal_fin, filled=True, color=body_fin_colour)

        angle = self.body[3].angle_to(self.body[2]) + math.pi
        tail_width = 0.5 - clamp(0, abs(total_curvature * 0.3), 0.3)
        tail_fin = [[self.body[2].x * 0.6 + self.body[3].x * 0.4,
                     self.body[2].y * 0.6 + self.body[3].y * 0.4]]
        tail_fin += [[tail_fin[0][0] + math.cos(angle + tail_width) * 10,
                      tail_fin[0][1] + math.sin(angle + tail_width) * 10]]
        tail_fin += [[tail_fin[0][0] + math.cos(angle) * 20,
                      tail_fin[0][1] + math.sin(angle) * 20]]
        tail_fin += [[tail_fin[0][0] + math.cos(angle - tail_width) * 10,
                      tail_fin[0][1] + math.sin(angle - tail_width) * 10]]
        viewer.draw_polygon(tail_fin, filled=True, color=fin_colour)

        # draw eyes
        eye1_x = math.cos(self.dir + math.pi * 0.5 + 0.3) * self.size * 0.7
        eye1_y = math.sin(self.dir + math.pi * 0.5 + 0.3) * self.size * 0.7

        eye2_x = math.cos(self.dir - math.pi * 0.5 - 0.3) * self.size * 0.7
        eye2_y = math.sin(self.dir - math.pi * 0.5 - 0.3) * self.size * 0.7

        t = rendering.Transform(translation=(self.body[0].x + eye1_x,
                                             self.body[0].y + eye1_y))
        eye_l = viewer.draw_circle(radius=self.size * 0.5, filled=True, color=(255, 190, 190))
        eye_l.add_attr(t)

        t = rendering.Transform(translation=(self.body[0].x + eye2_x,
                                             self.body[0].y + eye2_y))
        eye_r = viewer.draw_circle(radius=self.size * 0.5, filled=True, color=(255, 190, 190))
        eye_r.add_attr(t)

        # iris
        iris_x = math.cos(self.dir + self.dir_change_avg * 10) * self.size * 0.25
        iris_y = math.sin(self.dir + self.dir_change_avg * 10) * self.size * 0.25

        t = rendering.Transform(
            translation=(self.body[0].x + eye1_x + iris_x,
                         self.body[0].y + eye1_y + iris_y))
        iris1 = viewer.draw_circle(radius=self.size * 0.3, filled=True, color=(0.0, 0.0, 0.0))
        iris1.add_attr(t)

        t = rendering.Transform(
            translation=(self.body[0].x + eye2_x + iris_x,
                         self.body[0].y + eye2_y + iris_y))
        iris2 = viewer.draw_circle(radius=self.size * 0.3, filled=True, color=(0.0, 0.0, 0.0))
        iris2.add_attr(t)

    def draw_ventral_fins(self, viewer, x, y, angle, size, colour, skew):
        t = rendering.Transform(translation=(x, y))
        deg90 = math.pi / 2
        skew = math.sin(skew)
        skew_left = 0 if skew > 0 else abs(skew + 0.2)
        skew_right = 0 if skew < 0 else abs(skew - 0.2)

        fin1 = [[math.cos(angle + deg90) * size,
                 math.sin(angle + deg90) * size],
                [math.cos(angle + deg90 * (2 + 0.5 * skew_left)) * size * 3,
                 math.sin(angle + deg90 * (2 + 0.5 * skew_left)) * size * 3],
                [math.cos(angle + deg90 * (1.5 + 0.3 * skew_left)) * size * 5,
                 math.sin(angle + deg90 * (1.5 + 0.3 * skew_left)) * size * 5]]
        geom = viewer.draw_polygon(fin1, filled=True, color=colour)
        geom.add_attr(t)
        fin1 = [[math.cos(angle - deg90) * size,
                 math.sin(angle - deg90) * size],
                [math.cos(angle - deg90 * (2 + 0.5 * skew_right)) * size * 3,
                 math.sin(angle - deg90 * (2 + 0.5 * skew_right)) * size * 3],
                [math.cos(angle - deg90 * (1.5 + 0.3 * skew_right)) * size * 5,
                 math.sin(angle - deg90 * (1.5 + 0.3 * skew_right)) * size * 5]]
        geom = viewer.draw_polygon(fin1, filled=True, color=colour)
        geom.add_attr(t)

    def heal(self, health):
        self.hp += health.size

    def get_colour(self):
        life = max(min(1.0, self.hp / 100), 0.4)
        return (200 * life + 55, 0, 0)
