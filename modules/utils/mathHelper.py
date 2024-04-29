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


def angle_diff(angle1, angle2):
    # Compute the difference between the angles
    diff = angle2 - angle1

    # Normalize the difference to be within -π to π radians
    diff = (diff + math.pi) % (2*math.pi) - math.pi

    return diff
