import os.path
import random
import time
from abc import abstractmethod

import numpy as np
from PIL import Image

from app.draw.gl.n_net import unpack_shape


class NTexFactory:

    @abstractmethod
    def get_texture_data(self, data_grid, factor):
        """
        :param data_grid:  numpy array
        :param factor: down sampling factor, 1 == full quality
        :return: image_data, image_width, image_height
        """
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

    def get_texture_data(self, data_grid, factor):
        if factor == 1:
            index = random.randint(0, 12)
        else:
            index = factor % 12
        image_data = self.images[index]
        return image_data[0][::factor, ::factor], image_data[1] / factor, image_data[2] / factor


class RGBGridTextureFactory(NTexFactory):

    def __init__(self, color_theme):
        self.color_theme = color_theme

    def get_texture_data(self, data_grid, factor):
        start_time = time.time()
        new_grid = data_grid
        if len(new_grid.shape) == 1:
            new_grid = new_grid[::factor]
        else:
            new_grid = new_grid[::factor, ::factor]

        org_image_height, org_image_width = unpack_shape(new_grid)

        image_height = org_image_height
        image_width = org_image_width

        # Normalize the grid data to [0, 255] range
        cmap_rbga = self.color_theme.cmap(new_grid)
        normalized_data = (cmap_rbga * 255).astype(np.uint8)

        return normalized_data, image_width, image_height
