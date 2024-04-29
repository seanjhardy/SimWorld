import math

from modules.quadtree.quadrect import QRect


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, parent=None, max_points=4, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """
        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        self.parent = parent
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False
        self.totalPoints = 0
        self.se = None
        self.sw = None
        self.nw = None
        self.ne = None

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
            sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
            sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        self.nw = QuadTree(QRect(cx - w / 2, cy - h / 2, w, h),
                           self, self.max_points, self.depth + 1)
        self.ne = QuadTree(QRect(cx + w / 2, cy - h / 2, w, h),
                           self, self.max_points, self.depth + 1)
        self.se = QuadTree(QRect(cx + w / 2, cy + h / 2, w, h),
                           self, self.max_points, self.depth + 1)
        self.sw = QuadTree(QRect(cx - w / 2, cy + h / 2, w, h),
                           self, self.max_points, self.depth + 1)
        self.divided = True

    def remove(self):
        """Try to remove Point point from this QuadTree."""

        self.totalPoints -= 1
        if self.parent is not None:
            self.parent.remove()

    def simplify(self):
        # parent points fall below threshold
        if self.divided and self.totalPoints < self.max_points:
            self.divided = False
            self.points = self.nw.points + self.ne.points + self.sw.points + self.se.points
            for point in self.points:
                point.parent = self
            if self.parent is not None:
                self.parent.simplify()

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False

        if self.totalPoints < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.totalPoints += 1
            self.points.append(point)
            point.parent = self
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()
            # reinsert points into leaf nodes
            tempPoints = self.points
            self.points = []
            for p in tempPoints:
                (self.ne.insert(p) or
                 self.nw.insert(p) or
                 self.se.insert(p) or
                 self.sw.insert(p))

        self.totalPoints += 1
        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def query(self, boundary, found_points):
        """Find the points in the quadtree that lie within boundary."""

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points

    def query_circle(self, boundary, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre.

        boundary is a Rect object (a square) that bounds the search circle.
        There is no need to call this method directly: use query_radius.

        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return []

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and
                    point.distance_to(centre) <= radius):
                found_points.append(point)

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre."""

        # First find the square that bounds the search circle as a Rect object.
        boundary = QRect(centre.x, centre.y, 2 * radius, 2 * radius)
        return self.query_circle(boundary, centre, radius, found_points)

    def render(self, viewer):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""
        if self.divided:
            self.nw.render(viewer)
            self.ne.render(viewer)
            self.se.render(viewer)
            self.sw.render(viewer)
        else:
            self.boundary.render(viewer)

    def reset(self):
        self.divided = False
        self.totalPoints = 0
        self.se = None
        self.sw = None
        self.nw = None
        self.ne = None


def fastIntersectLineCircle(p1, p2, c, r):
    v = p2.sub(p1)  # Vector along line segment
    a = v.dot(v)
    b = 2 * v.dot(p1.sub(c))
    c = p1.dot(p1) + c.dot(c) - 2 * p1.dot(c) - r ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return -1
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    t1Exists = 0 <= t1 <= 1
    t2Exists = 0 <= t1 <= 1
    if not (t1Exists or t2Exists):
        return -1
    else:
        if not t1Exists:
            return t2
        elif not t2Exists:
            return t1
        else:
            return min(t1, t2) * math.sqrt(a)
