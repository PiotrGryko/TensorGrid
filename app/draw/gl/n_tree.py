import random

import numpy as np

from app.draw.bsp_tree.tree_bsp import BSPLeaf, BSPTree
from app.draw.gl.n_texture import NTexture
from app.draw.gl.n_vertex import NVertex


class NTreeLeaf(BSPLeaf):
    def __init__(self, x, y, w, h, level):
        super().__init__(x, y, w, h, level)
        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.w = w
        self.h = h
        self.x1, self.y1 = x, y
        self.x2, self.y2 = x + w, y + h
        self.background_attached = False
        self.nodes_attached = False
        self.n_vertex = NVertex()
        self.nodes = []
        self.material_to_texture_map = {}

    def create_nodes_view(self, n_net):
        positions, colors = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2)
        flat_array = np.reshape(positions, (-1, 2))
        self.nodes = flat_array
        if len(self.nodes) == 0:
            return
        # print("Nodes created ", self.level)
        self.n_vertex.create_nodes(self.nodes, colors)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def create_texture(self, n_net, textures_factory, material_id=1, factor=1):
        print("Creating default texture ")
        subgrid = n_net.get_subgrid(self.x1, self.y1, self.x2, self.y2)
        img_data, img_width, img_height = textures_factory.get_texture(subgrid, factor)
        tex = NTexture()
        tex.create_from_data(self.x1, self.y1, self.x2, self.y2, img_data, img_width,
                             img_height,
                             material_id=material_id)
        self.material_to_texture_map[material_id] = (tex, factor)

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

    def create_fbo_texture(self, n_net, material_id):
        if material_id not in self.material_to_texture_map:
            self.nodes_attached = True
            self.create_nodes_view(n_net)
            tex = NTexture()
            tex.create_from_fbo(self.x1, self.y1, self.x2, self.y2, self.n_vertex,
                                self.w,
                                self.h,
                                material_id=material_id)
            self.material_to_texture_map[material_id] = (tex, 1)

    def draw_leaf_background(self):
        if not self.background_attached:
            self.background_attached = True
            self.create_background_view()
        self.n_vertex.draw_plane()

    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)

    def contains(self, x1, y1, x2, y2, level):
        return x1 >= self.x1 and y1 >= self.y1 and x2 <= self.x2 and y2 <= self.y2 and level == self.level


class NTree(BSPTree):
    def __init__(self, depth, n_net, textures_factory):
        super().__init__(0, 0, depth)
        self.viewport = None
        self.n_net = n_net
        self.n_vertex = NVertex()
        self.textures = []
        self.texture_max_lod = 10  # generate up to N different level of details

        self.visible_leafs = []
        self.mega_leaf = None
        self.textures_factory = textures_factory

    def set_size(self, w, h):
        super().set_size(w, h)
        self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

    def update_viewport(self, viewport):
        visible = []
        not_visible = []
        self.viewport = viewport
        self.traverse(viewport, visible, not_visible)
        if visible != self.visible_leafs:
            # print("Visible count: ", len(visible))
            self.visible_leafs = visible
            self.build_mega_leaf()

    def build_mega_leaf(self):
        if len(self.visible_leafs) == 0:
            self.mega_leaf = None
            return

        x1 = min([v.x1 for v in self.visible_leafs])
        x2 = max([v.x2 for v in self.visible_leafs])
        y1 = min([v.y1 for v in self.visible_leafs])
        y2 = max([v.y2 for v in self.visible_leafs])
        level = max([v.level for v in self.visible_leafs])

        if self.mega_leaf is None or not self.mega_leaf.contains(x1, y1, x2, y2, level):
            mega_x1 = x1 - x1 * 0
            mega_x2 = x2 + x2 * 0
            mega_y1 = y1 - y1 * 0
            mega_y2 = y2 + y2 * 0
            self.mega_leaf = NTreeLeaf(mega_x1, mega_y1, mega_x2 - mega_x1, mega_y2 - mega_y1, level)

    def draw_debug_tree(self):
        for l in self.visible_leafs:
            l.draw_leaf_background()

    def draw_leafs_vertices(self):
        for l in self.visible_leafs:
            l.draw_vertices(self.n_net)

    def draw_mega_leaf_vertices(self):
        if self.mega_leaf is not None:
            self.mega_leaf.draw_vertices(self.n_net)

    def draw_leafs_textures(self, material_id, factor=1):
        for l in self.visible_leafs:
            l.draw_texture(self.n_net, self.textures_factory, material_id, factor)

    def draw_mega_leaf_texture(self, material_id, factor=1):
        if self.mega_leaf is not None:
            self.mega_leaf.draw_texture(self.n_net, self.textures_factory, material_id, factor)

    def draw_leafs_to_texture(self, material_id):
        for l in self.visible_leafs:
            l.create_fbo_texture(self.n_net, material_id)

    def draw_mega_leaf_to_texture(self, material_id):
        if self.mega_leaf is not None:
            self.mega_leaf.create_fbo_texture(self.n_net, material_id)

    def draw_leafs_fbo_texture(self, material_id):
        for l in self.visible_leafs:
            l.draw_fbo_texture(material_id)

    def draw_mega_leaf_fbo_texture(self, material_id):
        if self.mega_leaf is not None:
            self.mega_leaf.draw_fbo_texture(material_id)
