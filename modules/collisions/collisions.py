import numpy as np
from numba import cuda
import math

from modules.raymarch.vectorUtils import checkCollisionCircleRec


@cuda.jit(fastmath=True)
def check_collision(grid, dir, x, y, size, out):
    grid_size = 20
    i, j = cuda.grid(2)

    if i < grid.shape[0] and j < grid.shape[1]:
        if grid[i, j]:  # Check only filled grid squares
            checkCollisionCircleRec(dir, x, y, size,
                                    i * grid_size, j * grid_size,
                                    (i + 1) * grid_size, (j + 1) * grid_size,
                                    out)


def solve_collisions(grid, character):
    threads_per_block = (1, 1)
    blocks_per_grid_x = math.ceil(len(grid) / threads_per_block[0])
    blocks_per_grid_y = math.ceil(len(grid[0]) / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    out = np.array([0, 0], dtype=np.float32)
    d_out = cuda.to_device(out)
    grid_np = np.array(grid, dtype=np.float32)
    d_grid = cuda.to_device(grid_np)
    check_collision[blocks_per_grid, threads_per_block](d_grid, character.dir, character.body[0].x, character.body[0].y, character.size, d_out)
    d_out_copy = d_out.copy_to_host()

    if abs(d_out_copy[0]) > 0.01 or abs(d_out_copy[1]) > 0.01:
        character.colliding = True
    character.body[0].x += d_out_copy[0]
    character.body[0].y += d_out_copy[1]