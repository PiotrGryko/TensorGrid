import random
import time
from abc import abstractmethod

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class NTexFactory:

    @abstractmethod
    def get_texture(self, data_grid, level=1):
        """Implement this method"""
        pass


class NodesGridTexFactory(NTexFactory):

    def __init__(self, color_theme):
        self.color_theme = color_theme

    def get_texture(self, data_grid, level=1):
        start_time = time.time()

        new_grid = data_grid
        grid_factor = level
        new_grid = new_grid[::grid_factor, ::grid_factor]
        # new_grid = self.scale_down_and_average(new_grid)

        image_height, image_width = self.scale_down_dimensions(new_grid.shape)
        # Normalize the grid data to [0, 255] range
        cmap_rbga = self.color_theme.cmap(new_grid)
        normalized_data = (cmap_rbga * 255).astype(np.uint8)
        image = Image.fromarray(normalized_data)

        image_rgba = image.convert('RGBA').resize((image_width, image_height))
        image_rgba_flipped = image_rgba.transpose(Image.FLIP_TOP_BOTTOM)

        # Get the raw pixel data as a numpy array
        image_data = np.array(image_rgba_flipped)

        print("Texture generated", time.time() - start_time, "width", image_width, "height", image_height, "factor",
              grid_factor)
        return image_data, image_width, image_height

    def scale_down_dimensions(self, shape, max_size=(16096, 16096)):
        original_width, original_height = shape

        # # Scale down the dimensions while preserving the aspect ratio
        new_width = original_width
        new_height = original_height

        # Check if the new dimensions exceed the maximum size
        if new_width > max_size[0]:
            new_width = max_size[0]
            new_height = int(original_height * (max_size[0] / original_width))
        if new_height > max_size[1]:
            new_height = max_size[1]
            new_width = int(original_width * (max_size[1] / original_height))

        return new_width, new_height

