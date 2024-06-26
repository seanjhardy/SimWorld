import math


def angle_to_vector(angle):
    # Calculate x and y components
    x = math.cos(angle)
    y = math.sin(angle)

    # Normalize the vectors
    magnitude = math.sqrt(x ** 2 + y ** 2)
    if magnitude != 0:
        x /= magnitude
        y /= magnitude

    return [x, y]


def get_velocity(point):
    return math.sqrt((point.x - point.prev_x) ** 2 + (point.y - point.prev_y) ** 2)


def interpolate(c1, c2, x):
    r1, g1, b1 = c1
    r2, g2, b2 = c2

    # Interpolate each component separately
    r = int(r1 + (r2 - r1) * x)
    g = int(g1 + (g2 - g1) * x)
    b = int(b1 + (b2 - b1) * x)

    return (r, g, b)


def clamp(min_val, x, ma_val):
    return max(min_val, min(x, ma_val))


def angle_diff(angle1, angle2, norm=True):
    # Compute the difference between the angles
    diff = angle2 - angle1

    # Normalize the difference to be within -π to π radians
    if norm:
        diff = norm_angle(diff)

    return diff

def clockwise_angle_diff(p1, p2):
    return math.atan2(p1.x * p2.y - p1.y * p2.x,
                      p1.x * p2.x + p1.y * p2.y)

def bezier(x0, y0, x1, y1, x2, y2, num_points=10):
    """
    Calculate a list of points in a quadratic Bézier curve defined by the points (x0, y0), (x1, y1), and (x2, y2).

    Parameters:
    x0, y0 -- coordinates of the starting point
    x1, y1 -- coordinates of the control point
    x2, y2 -- coordinates of the end point
    num_points -- the number of points to generate along the curve

    Returns:
    A list of tuples representing the points on the Bézier curve.
    """

    def bezier_interpolation(t, p0, p1, p2):
        """
        Calculate a point in a quadratic Bézier curve.

        Parameters:
        t -- parameter, which goes from 0 to 1
        p0, p1, p2 -- control points

        Returns:
        A tuple representing the point (x, y) on the Bézier curve.
        """
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        return [int(x), int(y)]

    p0 = [x0, y0]
    p1 = [x1, y1]
    p2 = [x2, y2]

    points = []

    for i in range(num_points + 1):
        t = i / num_points
        points.append(bezier_interpolation(t, p0, p1, p2))

    return points

def norm_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def smooth_angle(angle1, angle2, tolerance=90):
    """
    Return angle1 if the angle between them is exactly 180 degrees,
    but slowly decrease to 0 if within 90 degrees of angle2.
    """
    diff = angle_diff(angle1, angle2)
    tolerance = math.radians(tolerance)

    if abs(diff) < tolerance:
        if diff > 0:
            return norm_angle(angle2 - tolerance)
        else:
            return norm_angle(angle2 + tolerance)

    return angle1
