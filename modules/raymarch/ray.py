import math

import numpy as np
from numba import cuda

from modules.raymarch.vectorUtils import fastIntersectLineCircle, fastIntersectionLineAABB
from modules.utils.mathHelper import clamp


def sign(x):
    return math.copysign(1, x)


@cuda.jit(fastmath=True)
def ray_march_kernel(fov, fidelity, x, y, dir, objects, health, npcs, output, stereoscopic):
    idx = cuda.blockIdx.x
    half_fidelity = fidelity * 0.5
    ang = math.radians((idx - half_fidelity) / fidelity * fov)

    start_x = x
    start_y = y
    if stereoscopic:
        sign = 1 if idx % 2 == 0 else -1
        start_x += math.cos(dir + sign * (math.pi / 2)) * 5
        start_y += math.sin(dir + sign * (math.pi / 2)) * 5
    start_x = max(0, min(start_x, 400))
    start_y = max(0, min(start_y, 300))

    end_x = start_x + math.cos(dir + ang) * 400
    end_y = start_y + math.sin(dir + ang) * 400

    minWallDist = 200
    minHealthDist = 200
    minNPCDist = 200
    # check if ray intersects wall
    dist = fastIntersectionLineAABB(start_x, start_y, end_x, end_y, 0, 0, 400, 300)
    hitWall = False
    hitNPC = False
    hitHealth = False
    npc = None
    if dist != -1:
        hitWall = True
        minWallDist = dist

    for i in range(len(objects)):
        for j in range(len(objects[0])):
            if objects[i][j]:
                dist = fastIntersectionLineAABB(start_x, start_y, end_x, end_y,
                                                i * 20, j * 20, (i + 1) * 20, (j + 1) * 20)
                if dist != -1:
                    minWallDist = min(dist, minWallDist)

    # calculate if health intersects circle
    for i in range(health.shape[0]):
        d = fastIntersectLineCircle(start_x, start_y, end_x, end_y, health[i][0], health[i][1], health[i][2])
        if d != -1:
            hitHealth = True
            minHealthDist = min(minHealthDist, d)

    for i in range(npcs.shape[0]):
        d = fastIntersectLineCircle(start_x, start_y, end_x, end_y, npcs[i][0], npcs[i][1], npcs[i][2])
        if d != -1:
            hitNPC = True
            if d < minNPCDist:
                minNPCDist = d
                npc = npcs[i]

    if stereoscopic:
        if idx % 2 == 0:  # Odd index
            index = (output.shape[0] + 1) // 2 + (idx - 1) // 2
        else:  # Even index
            index = idx // 2
    else:
        index = idx

    if hitWall:
        output[index][0] = 1
        output[index][1] = minWallDist

    if hitHealth and minHealthDist < minWallDist:
        output[index][0] = 2
        output[index][1] = minHealthDist

    if hitNPC and minNPCDist < minHealthDist and minNPCDist < minWallDist:
        output[index][0] = 3 + npc[3] * 0.9
        output[index][1] = minNPCDist

# Host code
def ray_march(character, NPCs, objects, health, stereoscopic):
    image = np.zeros((character.fidelity, 2), dtype=np.float32)
    objects_arr = np.array(objects, dtype=np.float32)
    d_image = cuda.to_device(image)
    d_objects = cuda.to_device(objects_arr)

    health_arr = np.zeros((len(health), 3), dtype=np.float32)
    for i in range(len(health)):
        health_arr[i] = [health[i].x, health[i].y, health[i].size]
    d_health = cuda.to_device(health_arr)

    if len(NPCs) != 0:
        npc = []
        body = len(NPCs[0].body)
        for i in range(len(NPCs)):
            for j in range(body):
                if j < body - 1:
                    npc.append([NPCs[i].body[j].x*0.5 + NPCs[i].body[j + 1].x*0.5,
                                NPCs[i].body[j].y*0.5 + NPCs[i].body[j + 1].y*0.5,
                                5.5, j/body])
                npc.append([NPCs[i].body[j].x, NPCs[i].body[j].y, NPCs[i].body[j].mass, j/body])
        d_npcs = cuda.to_device(np.array(npc))
    else:
        d_npcs = cuda.to_device(np.array([]))
    ray_march_kernel[character.fidelity, 1](character.fov, character.fidelity, character.body[0].x, character.body[0].y,
                                            character.dir, d_objects, d_health, d_npcs, d_image, stereoscopic)
    character.rayDist = d_image.copy_to_host()[:, 1]
    return d_image.copy_to_host()
