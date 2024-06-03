def constrain_distance(point1, point2, distance):
    current_distance = point1.distance_to(point2)
    delta_distance = distance - current_distance
    if current_distance == 0:
        current_distance += 0.001
    delta_x = (point2.x - point1.x) * delta_distance / current_distance
    delta_y = (point2.y - point1.y) * delta_distance / current_distance

    point1.x -= delta_x / 2
    point1.y -= delta_y / 2
    point2.x += delta_x / 2
    point2.y += delta_y / 2


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