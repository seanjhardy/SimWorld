import math
import numpy as np
import random
import noise
from environments.maze.maze import Maze


def generate_map(map_size, square_size, edge_probability, center_probability):
    x_size = round(map_size[0] / square_size)
    y_size = round(map_size[1] / square_size)
    map_grid = np.zeros((x_size, y_size)).astype(bool)

    # Calculate the center of the map
    center_x = x_size / 2
    center_y = y_size / 2

    # Calculate the maximum distance from the center of the map
    max_distance = math.sqrt(center_x ** 2 + center_y ** 2)

    seed = round(random.random() * 1000)

    # Iterate over each grid cell
    for x in range(x_size):
        for y in range(y_size):
            # Calculate the distance from the center of the map to the current grid cell
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Calculate the probability of the grid cell being filled based on the exponential function
            edge_dropoff = (edge_probability - center_probability) * (distance / max_distance) + center_probability

            # Calculate Perlin noise value at current position
            noise_value = noise.pnoise2((x / x_size + seed) * 5, (y / y_size) * 5, octaves=1)

            # Map Perlin noise value to probability between edge_probability and center_probability
            probability = (noise_value + 1) / 2

            # Add the grid cell to the map
            map_grid[x][y] = probability < edge_dropoff

    return map_grid


def generate_maze(map_size, square_size):
    x_size = round(map_size[0] / square_size)
    y_size = round(map_size[1] / square_size)
    map_grid = np.zeros((x_size, y_size))
    start_point = np.array([0, 0])
    maze = Maze(map_grid, start_point)
    return maze.maze.astype(bool)
