import random
import time

import numpy as np
from numpy import float32

from app.draw.gl.n_vertex import NVertex
from app.draw.tree_bsp import BSPLeaf, BSPTree


class NTreeLeaf(BSPLeaf):
    def __init__(self, x, y, w, h, level):
        super().__init__(x, y, w, h, level)
        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.x1, self.y1 = x, y
        self.x2, self.y2 = x + w, y + h
        self.n_vertex = NVertex()
        self.nodes = []
        self.nodes_attached = False
        self.background_attached = False
        self.keys = None

    def create_nodes_view(self):
        if len(self.nodes) == 0:
            return
        shape = (len(self.nodes), 2)
        nodes_positions = np.empty(shape, dtype=np.float32)
        for index, n in enumerate(self.nodes):
            nodes_positions[index][0] = n[0]
            nodes_positions[index][1] = n[1]
        self.n_vertex.initialize(nodes_positions)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def draw(self):
        self.n_vertex.draw_nodes()
        self.n_vertex.draw_plane()


    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)


class NTree(BSPTree):
    def __init__(self, depth):
        super().__init__(0, 0, depth)
        self.visible_leafs = []
        self.nodes_positions = None
        self.viewport = None

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

    def feed_one(self, x, y):
        edge_leaf = self.get_edge_leaf(x, y)
        edge_leaf.nodes.append([x, y])

    def feed(self, nodes):
        start_time = time.time()

        mapped_nodes = np.empty((len(nodes), 5), dtype=float32)
        mapped_nodes[:, :2] = nodes

        x = nodes[:, 0]
        y = nodes[:, 1]

        x_grid_pos = x / self.edge_leaf_width
        x_grid_pos = x_grid_pos.astype(int)

        y_grid_pos = y / self.edge_leaf_height
        y_grid_pos = y_grid_pos.astype(int)

        self.keys = (x_grid_pos + (self.grid_size + 12.12)) / (y_grid_pos - 0.1)
        print(len(np.unique(self.keys)))

        mapped_nodes[:, 2] = x_grid_pos
        mapped_nodes[:, 3] = y_grid_pos

        # sorted_indices = np.argsort(self.keys)
        # self.keys = self.keys[sorted_indices]
        # mapped_nodes = mapped_nodes[sorted_indices]

        mapped_nodes[:, 4] = self.keys
        self.nodes_positions = mapped_nodes

        smaller_arrays = np.array_split(mapped_nodes, 100)

        # leafs = self.get_edge_leafs()
        # for l in leafs:
        #     key = (l.key_x +(self.grid_size+12.12)) / (l.key_y-0.1)
        #     l.nodes = mapped_nodes[np.where(( self.keys== key))]

        print("Tree fed", time.time() - start_time, "edge leafs count")

    # for testing
    def retrieve_nodes(self, l):
        key = (l.key_x + (self.grid_size + 12.12)) / (l.key_y - 0.1)
        l.nodes = self.nodes_positions[np.where((self.keys == key))]

    def draw(self):
        for l in self.visible_leafs:
            #
            if not l.nodes_attached:
                x, y, w, h, zoom = self.viewport
                if zoom > 0.085:
                    l.nodes_attached = True
                    self.retrieve_nodes(l)
                    l.create_nodes_view()
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
