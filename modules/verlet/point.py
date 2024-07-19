import math

from gym.envs.classic_control import rendering


class Point:
    def __init__(self, x, y, mass=1.0):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.mass = mass
        self.force_x = 0.0
        self.force_y = 0.0

    def apply_force(self, force_x, force_y):
        self.force_x += force_x
        self.force_y += force_y

    def get_velocity(self):
        x_vel = self.x - self.prev_x
        y_vel = self.y - self.prev_y
        speed = math.sqrt(x_vel ** 2 + y_vel ** 2)
        dir = math.atan2(y_vel, x_vel)
        return speed, dir

    def update(self, dt=1):
        acceleration_x = self.force_x / self.mass
        acceleration_y = self.force_y / self.mass

        speed, dir = self.get_velocity()
        friction_x = math.cos(dir + math.pi) * speed * 0.1
        friction_y = math.sin(dir + math.pi) * speed * 0.1

        # Verlet integration
        new_x = 2 * self.x - self.prev_x \
                + (acceleration_x + friction_x) * dt ** 2
        new_y = 2 * self.y - self.prev_y \
                + (acceleration_y + friction_y) * dt ** 2

        self.prev_x = self.x
        self.prev_y = self.y

        self.x = new_x
        self.y = new_y

        self.force_x = 0.0
        self.force_y = 0.0

    def distance_to(self, other_point):
        dx = self.x - other_point.x
        dy = self.y - other_point.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def angle_to(self, other_point):
        return math.atan2(other_point.y - self.y, other_point.x - self.x)

    def sub(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def rotate(self, origin, diff):
        xdiff = self.x - origin.x
        ydiff = self.y - origin.y
        self.x = origin.x + math.cos(diff) * xdiff - math.sin(diff) * ydiff
        self.y = origin.y + math.sin(diff) * xdiff + math.cos(diff) * ydiff

    def render(self, viewer, colour):
        t = rendering.Transform(translation=(self.x, self.y))
        geom = viewer.draw_circle(radius=self.mass, filled=True, color=colour)
        geom.add_attr(t)
