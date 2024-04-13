import random

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
        self.n_vertex = NVertex()
        self.nodes = []
        self.material_to_texture_map = {}

    def create_nodes_view(self, n_net, clip=False):
        positions, colors = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2)
        flat_array = np.reshape(positions, (-1, 2))
        if clip:
            values = np.array([self.x1, self.y1])
            flat_array = flat_array - values
        self.nodes = flat_array
        if len(self.nodes) == 0:
            return
        # print("Nodes created ", self.level)
        self.n_vertex.create_nodes(self.nodes, colors)

    def create_fbo_texture(self, n_net, n_window, material_id):
        if material_id in self.material_to_texture_map:
            return
        self.nodes_attached = True
        self.create_nodes_view(n_net)
        self.background_attached = True
        self.create_background_view()
        tex = NTexture()
        tex.create_from_fbo(self.n_vertex, n_window, self.x1, self.y1, self.x2, self.y2, material_id=material_id)
        self.material_to_texture_map[material_id] = (tex, 1)

    def create_texture(self, n_net, textures_factory, material_id=1, factor=1):
        subgrid = n_net.get_subgrid(self.x1, self.y1, self.x2, self.y2)
        img_data, img_width, img_height = textures_factory.get_texture(subgrid, factor)
        tex = NTexture()
        tex.create_from_data(self.x1, self.y1, self.x2, self.y2,
                             img_data,
                             img_width,
                             img_height,
                             material_id=material_id)
        self.material_to_texture_map[material_id] = (tex, factor)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def draw_texture(self, n_net, textures_factory, material_id, factor):
        if material_id not in self.material_to_texture_map:
            self.create_texture(n_net, textures_factory, material_id, factor)
        if material_id in self.material_to_texture_map:
            texture, tex_factor = self.material_to_texture_map[material_id]
            if tex_factor != factor:
                self.create_texture(n_net, textures_factory, material_id, factor)
            texture.draw()

    def draw_fbo_texture(self, material_id):
        if material_id in self.material_to_texture_map:
            texture, tex_factor = self.material_to_texture_map[material_id]
            texture.draw()

    def draw_vertices(self, n_net):
        if not self.nodes_attached:
            self.nodes_attached = True
            self.create_nodes_view(n_net)
        self.n_vertex.draw_nodes()

    def draw_leaf_background(self):
        if not self.background_attached:
            self.background_attached = True
            self.create_background_view()
        self.n_vertex.draw_plane()
