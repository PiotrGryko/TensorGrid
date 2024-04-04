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
        subgrid = n_net.get_subgrid(self.x1, self.y1, self.x2, self.y2)
        img_data, img_width, img_height = textures_factory.get_texture(subgrid, factor)
        tex = NTexture()
        tex.create_from_data(self.x1, self.y1, self.x2, self.y2, img_data, img_width,
                             img_height,
                             material_id=material_id)
        self.material_to_texture_map[material_id] = tex

    def draw_texture(self, n_net, textures_factory, material_id, factor):
        if material_id not in self.material_to_texture_map:
            self.create_texture(n_net, textures_factory, material_id, factor)
        if material_id in self.material_to_texture_map:
            self.material_to_texture_map[material_id].draw()

    def draw_vertices(self):
        self.n_vertex.draw_nodes()

    def draw_leaf_background(self):
        self.n_vertex.draw_plane()

    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)

    def equals(self, x1, y1, x2, y2, level):
        return x1 == self.x1 and y1 == self.y1 and x2 == self.x2 and y2 == self.y2 and level == self.level


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

    def load_net_size(self):
        self.set_size(self.n_net.total_width, self.n_net.total_height)
        cc = self.n_net.grid_columns_count
        rc = self.n_net.grid_rows_count
        side = cc if cc > rc else rc
        self.depth = int(side / 500)
        self.depth = self.depth if self.depth < self.texture_max_lod else self.texture_max_lod
        print("Tree depth assigned ", self.depth)
        # self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

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

        if self.mega_leaf is None:
            self.mega_leaf = NTreeLeaf(x1, y1, x2 - x1, y2 - y1, level)
        elif not self.mega_leaf.equals(x1, y1, x2, y2, level):
            self.mega_leaf = NTreeLeaf(x1, y1, x2 - x1, y2 - y1, level)

    def draw_debug_tree(self):
        for l in self.visible_leafs:
            if not l.background_attached:
                l.background_attached = True
                l.create_background_view()
            l.draw_leaf_background()

    def draw_leafs_vertices(self):
        for l in self.visible_leafs:
            if not l.nodes_attached:
                l.nodes_attached = True
                l.create_nodes_view(self.n_net)
            l.draw_vertices()

    def draw_mega_leaf_vertices(self):
        if self.mega_leaf is not None:
            if not self.mega_leaf.nodes_attached:
                self.mega_leaf.nodes_attached = True
                self.mega_leaf.create_nodes_view(self.n_net)
            self.mega_leaf.draw_vertices()

    def draw_leafs_textures(self, material_id):
        for l in self.visible_leafs:
            l.draw_texture(self.n_net, self.textures_factory, material_id, 1)

    def draw_mega_leaf_texture(self, material_id):
        if self.mega_leaf is not None:
            self.mega_leaf.draw_texture(self.n_net, self.textures_factory, material_id, 1)
