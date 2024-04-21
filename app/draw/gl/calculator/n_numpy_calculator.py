import math

import numpy as np

from app.draw.gl.calculator.base_calculator import BaseCalculator


class NumpyCalculator(BaseCalculator):

    def init(self):
        pass

    def cleanup(self):
        pass

    def measure(self, elements_count):
        size_x = math.ceil(math.sqrt(elements_count))
        size_y = size_x
        return [size_x, size_y]

    def calculate_positions(self, elements_count, start_index, end_index):
        size_x = math.ceil(math.sqrt(elements_count))
        size_y = size_x
        indices = np.arange(start_index, end_index)
        grid_x = np.mod(indices, size_x)
        grid_y = np.floor_divide(indices, size_y)
        return grid_x, grid_y
