import os.path
import random
import time
from abc import abstractmethod

import numpy as np
import scipy
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

from app.draw.gl.n_net import unpack_shape


class NTexFactory:

    @abstractmethod
    def get_texture_data(self, data_grid, factor, details_factor):
        """Implement this method"""
        pass


class ImageTextureFactory(NTexFactory):

    def __init__(self):

        self.images = {}
        for i in range(13):
            image_name = f"tiles/test{i}.png"
            if not os.path.exists(image_name):
                image_name = f"tiles/test{i}.jpg"
            image = Image.open(image_name)
            image_data = np.array(image)
            image_width, image_height = image.width, image.height
            self.images[i] = [image_data, image_width, image_height]

    def get_texture_data(self, data_grid, factor, details_factor):
        image_data = self.images[factor]
        return image_data[0], image_data[1], image_data[2]


class RGBGridTextureFactory(NTexFactory):

    def __init__(self, color_theme):
        self.color_theme = color_theme

    def average_downsample(self, grid, initial_factor):
        # Ensure the grid size is divisible by factor for both dimensions
        factor_y = initial_factor
        factor_x = initial_factor

        # Find the largest factor that divides the height and width of the grid evenly
        while grid.shape[0] % factor_y != 0:
            factor_y -= 1
        while len(grid.shape) > 1 and grid.shape[1] % factor_x != 0:
            factor_x -= 1

        if factor_y != initial_factor or factor_x != initial_factor:
            print(f"Adjusted factors to: factor_y={factor_y}, factor_x={factor_x}, initial factor {initial_factor}")

        # Reshape the grid to a new shape that groups elements into the averaging blocks
        if len(grid.shape) > 1:
            new_shape = (grid.shape[0] // factor_y, factor_y, grid.shape[1] // factor_x, factor_x)
            reshaped_grid = grid.reshape(new_shape)
            downsampled_grid = reshaped_grid.mean(axis=(1, 3))
        else:
            new_shape = (grid.shape[0] // factor_y, factor_y)
            reshaped_grid = grid.reshape(new_shape)
            downsampled_grid = reshaped_grid.mean(axis=(1))

        # Reshape the grid to a new shape that groups elements into the averaging blocks
        # new_shape = (grid.shape[0] // factor, factor, grid.shape[1] // factor, factor)
        # reshaped_grid = grid.reshape(new_shape)

        # Calculate the mean along the axes that correspond to the blocks

        return downsampled_grid

    def get_texture_data(self, data_grid, factor, details_factor):
        start_time = time.time()
        new_grid = data_grid
        if len(new_grid.shape) == 1:
            new_grid = new_grid[::details_factor]
        else:
            new_grid = new_grid[::details_factor, ::details_factor]

        org_image_height, org_image_width = unpack_shape(new_grid)  # self.scale_down_dimensions(unpack_shape(new_grid))

        image_height = org_image_height
        image_width = org_image_width

        # Normalize the grid data to [0, 255] range
        cmap_rbga = self.color_theme.cmap(new_grid)
        normalized_data = (cmap_rbga * 255).astype(np.uint8)

        return normalized_data, image_width, image_height

    def scale_down_dimensions(self, shape, max_size=(1600, 1600)):
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
