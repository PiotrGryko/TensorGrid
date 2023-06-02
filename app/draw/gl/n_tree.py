import random
import time

import numpy as np

from app.draw.gl.n_vertex import NVertex
from app.draw.tree_bsp import BSPLeaf, BSPTree


class NTreeLeaf(BSPLeaf):
    def __init__(self, x, y, w, h, level):
        super().__init__(x, y, w, h, level)
        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.x1, self.y1 = x, y
        self.x2, self.y2 = x + w, y + h
        self.background_attached = False
        self.nodes_attached = False
        self.grid = None
        self.n_vertex = NVertex()
        self.nodes = []

    def create_nodes_view(self, n_net):
        self.grid = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2)
        flat_grid = self.grid[np.where(self.grid != -1)]
        flat_array = np.reshape(flat_grid, (-1, 2))
        self.nodes = flat_array
        if len(self.nodes) == 0:
            return
        self.n_vertex.initialize(self.nodes)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def draw(self):
        self.n_vertex.draw_nodes()
        self.n_vertex.draw_plane()

    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)


class NTree(BSPTree):
    def __init__(self, depth, n_net):
        super().__init__(0, 0, depth)
        self.visible_leafs = []
        self.viewport = None
        self.n_net = n_net

    def update_size(self):
        self.set_size(self.n_net.total_width, self.n_net.total_height)
        cc = self.n_net.grid_columns_count
        rc = self.n_net.grid_rows_count
        side = cc if cc > rc else rc
        self.depth = int(side / 500)
        self.depth =  self.depth if self.depth<6 else 6
        print("depth assigned ",self.depth)
        self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

    def set_size(self, w, h):
        super().set_size(w, h)
        self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

    def create_view(self):
        start_time = time.time()
        leafs = self.get_leafs()
        for l in leafs:
            pass
            # l.create_background_view()
            # l.create_nodes_view()
        print("View created", time.time() - start_time)

    def draw(self):
        for l in self.visible_leafs:
            #
            if not l.nodes_attached:
                x, y, w, h, zoom = self.viewport
                if zoom > 0.03:
                    l.nodes_attached = True
                    l.create_nodes_view(self.n_net)
            if not l.background_attached:
                l.background_attached = True
                l.create_background_view()
            l.draw()

    def update_viewport(self, viewport):
        visible = []
        not_visible = []
        self.viewport = viewport
        self.traverse(viewport, visible, not_visible)
        if visible != self.visible_leafs:
            self.visible_leafs = visible
