import random

import pyglet
from pyglet.gl import GL_TRIANGLES

from app.draw.helper import node_color, label_color, red_color
from app.draw.tree_bsp import BSPTree, BSPLeaf


class PlaneLeaf(BSPLeaf):
    def __init__(self, x, y, w, h, level):
        super().__init__(x, y, w, h, level)
        self.nodes_meta = []
        self.nodes = []
        self.batch = pyglet.graphics.Batch()
        self.color = (random.randint(0, 255),
                      random.randint(0, 255),
                      random.randint(0, 255),
                      80)
        self.label = None
        self.view = None
        self.refresh_background()
        self.merged = False
        # print(level, x, y, w, h)

    def count_nodes(self):
        count = len(self.nodes_meta)
        for c in self.children:
            count += c.count_nodes()
        return count

    def create_child(self, x, y, w, h, level):
        return PlaneLeaf(x, y, w, h, level)

    def refresh_background(self):
        if self.view:
            self.view.delete()
        self.view = pyglet.shapes.Rectangle(self.x,
                                            self.y,
                                            self.w,
                                            self.h,
                                            color=self.color,
                                            batch=self.batch
                                            )

    def add_node(self, x, y, radius):
        self.nodes_meta.append((x, y, radius))
        size = 20
        self.nodes.append(pyglet.shapes.Rectangle(x=x,
                                                  y=y,
                                                  width=size,
                                                  height=size,
                                                  color=node_color,
                                                  batch=self.batch))

    def merge_batches(self):
        # if self.merged: return
        # self.merged=True
        # for c in self.children:
        #     c.merge_batches()
        #     self.nodes+=c.nodes
        # for n in self.nodes:
        #     n._batch = self.batch
        #     n._create_vertex_list()
        #     n._update_vertices()
        #############################################

        # nodes_meta = []
        # for c in self.children:
        #     c.merge_batches()
        #     nodes_meta += c.nodes_meta
        # for n in nodes_meta:
        #     x, y, radius = n
        #     self.add_node(x, y, radius)

        #############################################

        for c in self.children:
            c.merge_batches()
            # self.batch.add_batch(c.batch)
            for v in c.get_vertex_list():
                c.batch.migrate(v, GL_TRIANGLES, None, self.batch, keep=True)
        #############################################
        # for c in self.children:
        #     c.merge_batches()
        #
        # if self.level == 1:
        #     for m in self.nodes_meta:
        #         x, y, radius = m
        #         self.nodes.append(pyglet.shapes.Circle(x=x,
        #                                                y=y,
        #                                                radius=radius,
        #                                                color=node_color,
        #                                                batch=self.batch))
        # elif self.level == 2:
        #     for c in self.children:
        #         for v in c.get_vertex_list():
        #             c.batch.migrate(v, GL_TRIANGLES, None, self.batch, keep=True)
        # else:
        #     self.label = self.nodes.append(pyglet.shapes.Circle(x=self.x + self.w/2,
        #                                                     y=self.y + self.h/2,
        #                                                     radius=self.h/10,
        #                                                     color=red_color,
        #                                                     batch=self.batch))
        # self.label = pyglet.text.Label(f"Count:{self.count_nodes()}",
        #                                x=self.x + self.w / 3, y=self.y + self.h / 2,
        #                                color=label_color,
        #                                font_size=self.h / 13,
        #                                batch=self.batch)

    def merge_dirty(self, zoom):
        if self.merged: return

        # for c in self.children:
        #     c.merge_batches()
        if len(self.nodes_meta) > 0 and zoom > 0.3:
            print(zoom)
            # if self.level == 1:
            for m in self.nodes_meta:
                x, y, radius = m
                self.nodes.append(pyglet.shapes.Circle(x=x,
                                                       y=y,
                                                       radius=radius,
                                                       color=node_color,
                                                       batch=self.batch))
            if self.label:
                self.label.delete()
                self.label = None
            # self.merged = True
        elif self.label is None:
            self.label = self.nodes.append(pyglet.shapes.Circle(x=self.x + self.w / 2,
                                                                y=self.y + self.h / 2,
                                                                radius=self.h / 10,
                                                                color=red_color,
                                                                batch=self.batch))
            # self.merged = True
            # = pyglet.text.Label(f"Count:{self.count_nodes()}",
            #                                x=self.x + self.w / 3, y=self.y + self.h / 2,
            #                                color=label_color,
            #                                font_size=self.h / 13,
            #                                batch=self.batch)
        # elif self.level == 2:
        #     for c in self.children:
        #         for v in c.get_vertex_list():
        #             c.batch.migrate(v, GL_TRIANGLES, None, self.batch, keep=True)
        # else:
        #     self.label = pyglet.text.Label(f"Count:{self.count_nodes()}",
        #                                    x=self.x+self.w/3, y=self.y+self.h/2,
        #                                    color=label_color,
        #                                    font_size=self.h/13,
        #                                    batch=self.batch)

        # for c in self.children:
        #     c.merge_batches()
        #     # self.batch.add_batch(c.batch)
        #     for v in c.get_vertex_list():
        #         c.batch.migrate(v, GL_TRIANGLES, None, self.batch, keep=True)
        # nodes = self.get_nodes()
        # for n in nodes:
        #     n._batch = self.batch
        #     n._create_vertex_list()
        #     n._update_vertices()
        # print("merged", len(nodes))

    def get_vertex_list(self):
        vertex_lists = []
        for c in self.children:
            vertex_lists += c.get_vertex_list()
        for n in self.nodes:
            vertex_lists.append(n._vertex_list)
        return vertex_lists

    def get_nodes(self):
        nodes = self.nodes
        for c in self.children:
            nodes += c.get_nodes()
        return nodes


class PlaneTree(BSPTree):
    def __init__(self, width, height, depth):
        super().__init__(width, height, depth)
        self.leaf = PlaneLeaf(0, 0, width, height, depth)
        self.batches = set()
        self.visible_leafs = set()

    def set_size(self, w, h):
        super().set_size(w, h)
        self.leaf.refresh_background()

    def add_node(self, x, y):
        leaf = self.get_edge_leaf(x, y)
        leaf.add_node(x, y, 20)

    def on_draw(self):
        for b in self.batches:
            b.draw()

    #
    def merge_batches(self):
        # pass
        self.leaf.merge_batches()

    def on_window_update(self, world_camera):

        visible = []
        not_visible = []
        viewport = world_camera.get_window_viewport()
        self.traverse(viewport, visible, not_visible)
        x, y, w, h, zoom = viewport
        # self.batches=map(lambda v : v.batch,visible)
        if self.visible_leafs == visible:
            return
        self.visible_leafs = visible
        print(len(visible), "|", len(not_visible))
        self.batches.clear()
        for v in self.visible_leafs:
            # print("visible node  children count",v.count())
            #v.merge_dirty(zoom)
            # self.batches.add(v.background_batch)
            self.batches.add(v.batch)
        # for v in not_visible:
        #     if v.batch in self.batches:
        #         self.batches.remove(v.batch)
