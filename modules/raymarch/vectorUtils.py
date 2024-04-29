import math

import numpy as np
from numba import cuda


@cuda.jit(device=True, fastmath=True)
def checkCollisionCircleRec(dir, x, y, size, minX, minY, maxX, maxY, out):
    xOverlap = 0
    yOverlap = 0
    pX = x
    pY = y
    if x < minX - size or x > maxX + size or y < minY - size or y > maxY + size:
        return

    if pX < minX:
        pX = minX
    elif pX >= maxX:
        pX = maxX
    else:
        recCenterX = (minX + maxX) / 2
        xOverlap = minX - pX if pX < recCenterX else maxX - pX

    if pY < minY:
        pY = minY
    elif pY >= maxY:
        pY = maxY
    else:
        recCenterY = (minX + maxY) / 2
        yOverlap = minY - pY if pY < recCenterY else maxY - pY

    if (x - pX) ** 2 + (y - pY) ** 2 < size ** 2:
        contactX = pX
        contactY = pY
        if abs(xOverlap) < abs(yOverlap):
            contactX += xOverlap
        elif abs(yOverlap) < abs(xOverlap):
            contactY += yOverlap

        penX = x - contactX
        penY = y - contactY
        pLength = math.sqrt(penX ** 2 + penY ** 2)
        depth = size - pLength
        normalX = penX / pLength
        normalY = penY / pLength

        if xOverlap and yOverlap:
            depth -= size * 2

        if abs(normalX * depth) > abs(out[0]):
            out[0] += normalX * depth * 1.1
        if abs(normalY * depth) > abs(out[1]):
            out[1] += normalY * depth * 1.1


@cuda.jit(device=True, fastmath=True)
def fastIntersectLineCircle(x1, y1, x2, y2, cx, cy, r):
    vx = x2 - x1
    vy = y2 - y1
    a = vx*vx + vy*vy
    b = 2.0 * (vx * (x1 - cx) + vy * (y1 - cy))
    c = (x1*x1 + y1*y1) + (cx*cx + cy*cy) - 2 * (x1*cx + y1*cy) - r ** 2
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


@cuda.jit(device=True, fastmath=True)
def fastIntersectionLineAABB(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    dx = x2 - x1
    dy = y2 - y1
    t_enter = 0.0
    t_exit = 1.0

    for edge in range(4):
        p, q = 0, 0
        if edge == 0:
            p, q = -dx, x1 - xmin
        elif edge == 1:
            p, q = dx, xmax - x1
        elif edge == 2:
            p, q = -dy, y1 - ymin
        elif edge == 3:
            p, q = dy, ymax - y1

        if p == 0 and q < 0:
            return -1  # Line is outside and parallel, so completely discarded
        else:
            t = q / p
            if p < 0:
                if t > t_enter:
                    t_enter = t
            else:
                if t < t_exit:
                    t_exit = t

    if t_enter > t_exit:
        return -1  # Line is completely outside

    x1_clip = x1 + t_enter * dx
    y1_clip = y1 + t_enter * dy
    x2_clip = x1 + t_exit * dx
    y2_clip = y1 + t_exit * dy

    d1 = math.sqrt((x1_clip - x1) ** 2 + (y1_clip - y1) ** 2)
    d2 = math.sqrt((x2_clip - x1) ** 2 + (y2_clip - y1) ** 2)
    if d1 <= 0:
        return d2
    if d2 <= 0:
        return d1
    return min(d1, d2)
