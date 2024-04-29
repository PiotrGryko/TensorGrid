import random
import time

import numpy as np

from app.draw.gl.draw.n_texture import NTexture
from app.draw.gl.draw.n_vertex import NVertex


class NEntity:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.background_attached = False
        self.nodes_attached = False
        self.colors_attached = False

        self.n_vertex = NVertex()
        self.material_to_texture_map = {}

        self.positions_and_values = []
        self.grid = None

        self.data = None

    def has_texture_attached(self, material_id, factor):
        return (material_id in self.material_to_texture_map
                and self.material_to_texture_map[material_id][1] == factor)

    def create_texture(self, img_data, img_width, img_height, material_id, factor):
        tex = NTexture()
        tex.create_from_image_data(self.x1, self.y1, self.x2, self.y2,
                                   img_data,
                                   img_width,
                                   img_height,
                                   material_id=material_id)
        self.material_to_texture_map[material_id] = (tex, factor)

    def create_nodes_texture(self, n_window, material_id, factor):
        if material_id in self.material_to_texture_map:
            return
        self.background_attached = True
        self.create_background_view()
        tex = NTexture()
        tex.create_from_nodes_vertices(self.n_vertex, n_window, self.x1, self.y1, self.x2, self.y2,
                                       material_id=material_id)
        self.material_to_texture_map[material_id] = (tex, factor)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def create_colors_grid(self, positions_and_values):
        start_time = time.time()
        self.positions_and_values = positions_and_values
        self.n_vertex.create_color_grid(self.positions_and_values)
        self.colors_attached = True
        # print("color vertex created", time.time() - start_time)

    def create_colors_grid_texture(self, grid, material_id, factor):
        start_time = time.time()
        self.grid = grid
        tex = NTexture()
        tex.create_from_floats_grid(self.x1, self.y1, self.x2, self.y2, grid,
                                    material_id=material_id)
        self.material_to_texture_map[material_id] = (tex, factor)
        # print("color texture created", time.time() - start_time)

    def create_nodes_view(self, positions_and_values):
        self.positions_and_values = positions_and_values
        self.n_vertex.create_nodes(self.positions_and_values)
        self.nodes_attached = True


    def draw_texture(self, material_id):
        if material_id in self.material_to_texture_map:
            texture, tex_factor = self.material_to_texture_map[material_id]
            texture.draw()

    def draw_leaf_background(self):
        if not self.background_attached:
            self.background_attached = True
            self.create_background_view()
        self.n_vertex.draw_plane()

    def draw_nodes(self):
        if not self.nodes_attached:
            return
        self.n_vertex.draw_nodes()

    def draw_colors_grid(self):
        if not self.colors_attached:
            return
        self.n_vertex.draw_colors()
