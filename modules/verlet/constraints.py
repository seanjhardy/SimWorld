import math

from modules.utils.mathHelper import angle_diff


def constrain_distance(point1, point2, distance, factor = 1):
    current_distance = point1.distance_to(point2)
    delta_distance = factor * (distance - current_distance)
    if current_distance == 0:
        current_distance += 0.001

    if abs(delta_distance) < 0.01:
        return
    delta_x = (point2.x - point1.x) * delta_distance / current_distance
    delta_y = (point2.y - point1.y) * delta_distance / current_distance

    mass_ratio = point1.mass / (point1.mass + point2.mass)

    point1.x -= delta_x * (1 - mass_ratio)
    point1.y -= delta_y * (1 - mass_ratio)
    point2.x += delta_x * mass_ratio
    point2.y += delta_y * mass_ratio


def constrain_angle(point1, point2, point3, desired_angle, factor=0.001):
    angle1 = point2.angle_to(point1)
    angle2 = point2.angle_to(point3)
    current_angle = angle_diff(angle2, angle1)

    # Calculate the difference and move it by the given factor towards the desired angle
    delta_angle = desired_angle - current_angle

    if delta_angle <= -math.pi:
        delta_angle += 2*math.pi
    elif delta_angle >= math.pi:
        delta_angle -= 2*math.pi

    if abs(delta_angle) * factor < 1e-3:  # If the difference is very small, do nothing
        return

    point1.rotate(point2, factor * delta_angle)
    point3.rotate(point2, factor * delta_angle)


def constrain_position(point, xmin, ymin, xmax, ymax):
    update_dist = 0
    bounds = [xmin + point.mass,
              xmax - point.mass,
              ymin + point.mass,
              ymax - point.mass]

    if point.x < bounds[0]:
        update_dist += abs(point.x - bounds[0])
        point.x = bounds[0]
    if point.x > bounds[1]:
        update_dist += abs(point.x - bounds[1])
        point.x = bounds[1]

    if point.y < bounds[2]:
        update_dist += abs(point.y - bounds[2])
        point.y = bounds[2]
    if point.y > bounds[3]:
        update_dist += abs(point.y - bounds[3])
        point.y = bounds[3]
    return update_dist