import numpy as np


class QPoint:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with a payload object. """

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None

    def remove(self):
        self.parent.points.remove(self)
        self.parent.remove()
        self.parent.simplify()

    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def sub(self, other):
        return QPoint(self.x - other.x, self.y - other.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)