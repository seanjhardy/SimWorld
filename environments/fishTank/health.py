import random
from gym.envs.classic_control import rendering
from modules.quadtree.quadpoint import QPoint
from modules.verlet.point import Point


class Health(Point):
    colour = (0, 255, 0, 1)

    def __init__(self, x, y):
        super(Health, self).__init__(x, y)
        self.size = random.randrange(2, 5)
        self.x = x
        self.y = y

    def render(self, viewer):
        t = rendering.Transform(translation=(self.x, self.y))
        geom = viewer.draw_circle(radius=self.size, color=Health.colour)
        geom.add_attr(t)