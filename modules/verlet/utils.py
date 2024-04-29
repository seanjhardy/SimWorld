import math


def find_perpendicular_points(point1, point2, r1, r2):
    # Circle1 properties
    x1, y1 = point1.x, point1.y
    # Circle2 properties
    x2, y2 = point2.x, point2.y

    if x1 == x2 and y1 == y2:
        x1 += 0.0001

    angle = math.atan2(y2 - y1, x2 - x1)
    # Step 4: Add and subtract 90 degrees (in radians) from the angle
    angle_plus_90 = angle + math.pi / 2
    angle_minus_90 = angle - math.pi / 2

    # Step 5: Calculate the coordinates of the points on each circle's circumference in these perpendicular directions
    point1_circle1_x = x1 + r1 * math.cos(angle_plus_90)
    point1_circle1_y = y1 + r1 * math.sin(angle_plus_90)

    point2_circle1_x = x1 + r1 * math.cos(angle_minus_90)
    point2_circle1_y = y1 + r1 * math.sin(angle_minus_90)

    point1_circle2_x = x2 + r2 * math.cos(angle_minus_90)
    point1_circle2_y = y2 + r2 * math.sin(angle_minus_90)

    point2_circle2_x = x2 + r2 * math.cos(angle_plus_90)
    point2_circle2_y = y2 + r2 * math.sin(angle_plus_90)

    return [[point1_circle1_x, point1_circle1_y],
            [point2_circle1_x, point2_circle1_y],
            [point1_circle2_x, point1_circle2_y],
            [point2_circle2_x, point2_circle2_y]]