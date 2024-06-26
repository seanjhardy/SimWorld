import numpy as np
from numba import cuda
import math

from modules.raymarch.vectorUtils import checkCollisionCircleRec


@cuda.jit(fastmath=True)
def check_collision(grid, coords, out):
    grid_size = 20
    i, j, z = cuda.grid(3)

    if i < grid.shape[0] and j < grid.shape[1]:
        if grid[i, j]:  # Check only filled grid squares
            checkCollisionCircleRec(z, coords[z, 0], coords[z, 1], coords[z, 2],
                                    i * grid_size, j * grid_size,
                                    (i + 1) * grid_size, (j + 1) * grid_size,
                                    out)


def solve_collisions(grid, character):
    threads_per_block = (1, 1)
    blocks_per_grid_x = math.ceil(len(grid) / threads_per_block[0])
    blocks_per_grid_y = math.ceil(len(grid[0]) / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, len(character.body))

    out = np.array([[0, 0] for x in range(len(character.body))], dtype=np.float32)
    d_out = cuda.to_device(out)
    grid_np = np.array(grid, dtype=np.float32)
    d_grid = cuda.to_device(grid_np)

    coords = np.array([[p.x, p.y, p.mass + 1] for p in character.body], dtype=np.float32)
    d_coords = cuda.to_device(coords)

    check_collision[blocks_per_grid, threads_per_block](d_grid, d_coords, d_out)
    d_out_copy = d_out.copy_to_host()

    for x in range(len(character.body)):
        character.body[x].x += d_out_copy[x, 0] * 1.1
        character.body[x].y += d_out_copy[x, 1] * 1.1

    character.collision_force = abs(np.sum(d_out_copy, axis=1))

