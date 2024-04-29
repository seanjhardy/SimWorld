import math

import numpy as np
from numba import cuda

from modules.raymarch.vectorUtils import fastIntersectLineCircle, fastIntersectionLineAABB


def sign(x):
    return math.copysign(1, x)


@cuda.jit(fastmath=True)
def ray_march_kernel(fov, fidelity, x, y, dir, objects, health, output):
    idx = cuda.blockIdx.x
    halfFidelity = fidelity * 0.5
    ang = math.radians((idx - halfFidelity) / fidelity * fov)
    endX = x + math.cos(dir + ang) * 400
    endY = y + math.sin(dir + ang) * 400

    minWallDist = 200
    minHealthDist = 200
    # check if ray intersects wall
    dist = fastIntersectionLineAABB(x, y, endX, endY, 0, 0, 400, 400)
    hitWall = False
    hitHealth = False
    if dist != -1:
        hitWall = True
        minWallDist = dist

    for i in range(len(objects)):
        for j in range(len(objects[0])):
            if objects[i][j]:
                dist = fastIntersectionLineAABB(x, y, endX, endY,
                                                i * 20, j * 20, (i + 1) * 20, (j + 1) * 20)
                if dist != -1:
                    minWallDist = min(dist, minWallDist)

    # calculate if health intersects circle
    for i in range(health.shape[0]):
        d = fastIntersectLineCircle(x, y, endX, endY, health[i][0], health[i][1], health[i][2])
        if d != -1:
            hitHealth = True
            minHealthDist = min(minHealthDist, d)
    if hitWall:
        output[idx][0] = 1
        output[idx][1] = minWallDist

    if hitHealth and minHealthDist < minWallDist:
        output[idx][0] = 2
        output[idx][1] = minHealthDist


# Host code
def ray_march(character, objects, health):
    image = np.zeros((character.fidelity, 2), dtype=np.float32)
    objects_arr = np.array(objects, dtype=np.float32)
    d_image = cuda.to_device(image)
    d_objects = cuda.to_device(objects_arr)
    health_arr = np.zeros((len(health), 3), dtype=np.float32)
    for i in range(len(health)):
        health_arr[i] = [health[i].x, health[i].y, health[i].size]
    d_health = cuda.to_device(health_arr)

    ray_march_kernel[character.fidelity, 1](character.fov, character.fidelity, character.body[0].x, character.body[0].y, character.dir,
                                            d_objects, d_health, d_image)
    character.rayDist = d_image.copy_to_host()[:, 1]
    return d_image.copy_to_host()
