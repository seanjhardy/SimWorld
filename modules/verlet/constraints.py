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
    updated = False
    if point.x < xmin + point.mass:
        point.x = xmin + point.mass
        updated = True
    if point.x > xmax - point.mass:
        point.x = xmax - point.mass
        updated = True
    if point.y < ymin + point.mass:
        point.y = ymin + point.mass
        updated = True
    if point.y > ymax - point.mass:
        point.y = ymax - point.mass
        updated = True
    return updated