import cupy as cp
import numpy as np
import random

def inv2x2(m):
    det = m[0] * m[3] - m[1] * m[2]
    f = 1 / det
    return [f * m[3], -f * m[1], -f * m[2], f * m[0]]

def mvmul2(m, v, f):
    return [(v[0] * m[0] + v[1] * m[1]) * f, (v[0] * m[2] + v[1] * m[3]) * f]

def gcm2d(p, scale, theta, active_cells, axis_length):
    res = cp.zeros((axis_length[0], axis_length[1]), dtype=np.uint8)
    mat = inv2x2([cp.cos(theta), -cp.sin(theta), cp.sin(theta), cp.cos(theta)])

    np_ = mvmul2(mat, p, scale)
    cell_distances = []

    px = cp.mod(np_[0], axis_length[0]) + (np_[0] < 0) * axis_length[0]
    py = cp.mod(np_[1], axis_length[1]) + (np_[1] < 0) * axis_length[1]

    for i in range(res.size):
        x = i % axis_length[0] + 0.5
        y = axis_length[1] - ((i / axis_length[0]) + 0.5) # Flip axis for math convention
        dx = px - x
        dy = py - y

        dist = cp.sqrt(dx * dx + dy * dy)

        cell_distances.append([i, dist])

    cell_distances = sorted(cell_distances, key=lambda x: x[1])[:active_cells]
    for i in range(active_cells):
        res.flat[cell_distances[i][0]] = 1

    return res.flatten()

def gridCell2d(p, num_gcm=16, active_cells_per_gcm=1, gcm_axis_length=(4, 4), scale_range=(0.3, 1.0), seed=42):
    cp.random.seed(seed)
    pi = cp.pi
    gcm_size = gcm_axis_length[0] * gcm_axis_length[1]
    encoding = cp.zeros((num_gcm, gcm_size), dtype=cp.uint8)

    for i in range(num_gcm):
        gcm_res = gcm2d(p, cp.random.uniform(scale_range[0], scale_range[1]), cp.random.uniform(0, 2 * pi), active_cells_per_gcm, gcm_axis_length)
        encoding[i,:] = gcm_res

    return encoding.flatten()
