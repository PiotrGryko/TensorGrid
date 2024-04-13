import math

import numpy as np


class NumpyCalculator:
    def calculate_positions(self, size):
        size_x = math.ceil(math.sqrt(size))
        size_y = size_x
        indices = np.arange(size)
        grid_x = np.mod(indices, size_x)
        grid_y = np.floor_divide(indices, size_y)
        return grid_x, grid_y, [size_x, size_y]
