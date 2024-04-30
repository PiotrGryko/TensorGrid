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

        self.positions_and_values = []
        self.grid = None
        self.data = None

        self.attached_textures = []
        self.attached_material_id = None
        self.attached_texture_factor = None

    def has_texture_attached(self, material_id, factor):
        return material_id == self.attached_material_id and factor == self.attached_texture_factor

    def create_texture(self, img_data, img_width, img_height, material_id, factor):
        start_time = time.time()
        tex = NTexture()
        tex.create_from_image_data(self.x1, self.y1, self.x2, self.y2,
                                   img_data,
                                   img_width,
                                   img_height,
                                   material_id=material_id)
        self.attached_texture_factor = factor
        self.attached_material_id = material_id
        self.attached_textures = [tex]

    def create_texture_from_nodes_view(self, n_window, material_id, factor):
        start_time = time.time()
        tex = NTexture()
        tex.create_from_frame_buffer(n_window, self.x1, self.y1, self.x2, self.y2,
                                     material_id=material_id,
                                     draw_func=self.n_vertex.draw_nodes)
        self.attached_texture_factor = factor
        self.attached_material_id = material_id
        self.attached_textures = [tex]

    def create_texture_from_color_grid_view(self, n_window, material_id, factor):
        start_time = time.time()
        tex = NTexture()
        tex.create_from_frame_buffer(n_window, self.x1, self.y1, self.x2, self.y2,
                                     material_id=material_id,
                                     draw_func=self.n_vertex.draw_colors)
        self.attached_texture_factor = factor
        self.attached_material_id = material_id
        self.attached_textures = [tex]

    def create_colors_grid_texture(self, grid, material_id, factor):
        start_time = time.time()
        self.grid = grid
        tex = NTexture()
        tex.create_from_floats_grid(self.x1, self.y1, self.x2, self.y2, grid,
                                    material_id=material_id)
        self.attached_texture_factor = factor
        self.attached_material_id = material_id
        self.attached_textures = [tex]
        # print("color texture created", time.time() - start_time)

    def create_colors_grid_texture_from_chunks(self, chunks, dimensions, material_id, factor):
        start_time = time.time()
        self.attached_textures = []
        for c, d in zip(chunks, dimensions):
            cx1, cy1, cx2, cy2 = d
            tex = NTexture()
            tex.create_from_floats_grid(cx1, cy1, cx2, cy2,
                                        c,
                                        material_id=material_id)
            self.attached_textures.append(tex)

        self.attached_texture_factor = factor
        self.attached_material_id = material_id

        # print("color texture created", time.time() - start_time)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def create_colors_grid(self, positions_and_values):
        start_time = time.time()
        self.positions_and_values = positions_and_values
        self.n_vertex.create_color_grid(self.positions_and_values)
        self.colors_attached = True
        # print("color vertex created", time.time() - start_time)

    def create_nodes_view(self, positions_and_values):
        self.positions_and_values = positions_and_values
        self.n_vertex.create_nodes(self.positions_and_values)
        self.nodes_attached = True

    def draw_texture(self, material_id):
        for texture in self.attached_textures:
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
