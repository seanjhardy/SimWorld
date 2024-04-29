import numpy as np


class QRect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w / 2, cx + w / 2
        self.north_edge, self.south_edge = cy - h / 2, cy + h / 2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                    self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                                                         self.north_edge, self.east_edge, self.south_edge)

    def getIntersection(self, start, end):
        cx, cy, w, h = self.cx, self.cy, self.w, self.h
        lines = []
        lines.append([cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2])
        lines.append([cx - w / 2, cy + h / 2, cx + w / 2, cy + h / 2])
        lines.append([cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        lines.append([cx - w / 2, cy + h / 2, cx - w / 2, cy - h / 2])
        for line in lines:
            d = (line[3] - line[1]) * (end.x - start.x) - (line[2] - line[0]) * (end.y - start.y)
            if d:
                uA = ((line[2] - line[0]) * (start.y - line[1]) - (line[3] - line[1]) * (start.x - line[0])) / d
                uB = ((end.x - start.x) * (start.y - line[1]) - (end.y - start.y) * (start.x - line[0])) / d
            else:
                continue
            if not (0 <= uA <= 1 and 0 <= uB <= 1):
                continue
            length = np.hypot(start.x - end.x, start.y - end.y)
            d = uA * length
            return d
        return None

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (self.west_edge <= point_x < self.east_edge and
                self.north_edge <= point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def render(self, viewer):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        viewer.draw_polyline([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], color=(0.5, 0.8, 0.2, 0.3))
