import random
import time

import numpy as np

from app.draw.gl.n_texture import NTexture
from app.draw.gl.n_vertex import NVertex
from app.draw.bsp_tree.tree_bsp import BSPLeaf, BSPTree


class NTreeLeaf(BSPLeaf):
    def __init__(self, x, y, w, h, level):
        super().__init__(x, y, w, h, level)
        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.x1, self.y1 = x, y
        self.x2, self.y2 = x + w, y + h
        self.background_attached = False
        self.nodes_attached = False
        self.texture_attached = False
        self.n_vertex = NVertex()
        self.nodes = []
        self.n_texture = None

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

    def create_texture(self, n_net):
        self.n_texture = NTexture()
        factor = 1 #max([1, 20 - self.level])
        child_factor = max([1, factor - 1])
        img_data, img_width, img_height = n_net.get_texture2(self.x1, self.y1, self.x2, self.y2, factor)
        self.n_texture.create_from_data(self.x1, self.y1, self.x2, self.y2, img_data, img_width, img_height)
        # img_data2, img_width2, img_height2 = n_net.get_texture2(self.x1, self.y1, self.x2, self.y2, 20)
        # self.n_texture.add_second_texture_from_data(img_data, img_width, img_height)

    def draw_vertices(self):
        self.n_vertex.draw_nodes()

    def draw_leaf_background(self):
        self.n_vertex.draw_plane()

    def draw_texture(self):
        if self.n_texture:
            self.n_texture.draw()

    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)


class NTree(BSPTree):
    def __init__(self, depth, n_net):
        super().__init__(0, 0, depth)
        self.visible_leafs = []
        self.viewport = None
        self.n_net = n_net
        self.n_vertex = NVertex()
        self.textures = []
        self.min_possible_zoom = 0  # minimal possible zoom calculated by NWindow
        self.max_possible_zoom = 0  # max possible zoom calculated by NWindow
        self.texture_zoom_threshold_percent = 0.1  # % of max_possible_zoom at which textures swap to vertices
        self.texture_zoom_threshold = 0  # value calculated from texture_zoom_threshold_percent and  max_possible_zoom
        self.texture_zoom_step = 0  # zoom delta at which textures should change LOD
        self.texture_max_lod = 10  # generate up to N different level of details

        self.texture = None
        self.mega_leaf = None
        self.mega_visible_leafs = None

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

    def load_window_zoom_values(self, min_zoom, max_zoom):
        self.min_possible_zoom = min_zoom
        self.max_possible_zoom = max_zoom
        self.texture_zoom_threshold = self.max_possible_zoom * self.texture_zoom_threshold_percent
        self.texture_zoom_step = 1 if self.depth == 0 else (self.texture_zoom_threshold - self.min_possible_zoom) / (
            self.depth)

    def update_viewport(self, viewport):
        visible = []
        not_visible = []
        self.viewport = viewport
        self.traverse(viewport, visible, not_visible)
        if visible != self.visible_leafs:
            print("Visible count: ", len(visible))
            self.visible_leafs = visible

    def create_textures(self, n_net):
        print("Greate zoom textures")
        start_time = time.time()

        materials = []
        if self.depth > 0:
            for i in range(self.depth, 0, -1):
                if i == 1:
                    factor = 1
                else:
                    factor = i*3
                factor = i * 10
                img_data, img_width, img_height = n_net.get_texture2(0, 0, self.width, self.height, factor)
                materials.append((img_data, img_width, img_height))

        for index, m in enumerate(materials):
            texture = NTexture()
            img_data, img_width, img_height = m
            texture.create_from_data(0, 0, self.width, self.height, img_data, img_width, img_height)

            if index < len(materials) - 1:
                next_material = materials[index + 1]
                img_data, img_width, img_height = next_material
                texture.add_second_texture_from_data(img_data, img_width, img_height)
            self.textures.append(texture)
        print("Textures generated", time.time() - start_time, "zoom levels", len(self.textures))

    def draw_textures(self, n_shader):
        x, y, w, h, zoom = self.viewport
        zoom_threshold = self.texture_zoom_threshold + 0.1
        texture_index = 0
        if zoom > zoom_threshold:
            # print("Zoom is larger than max zoom,_treshold returning textures draw")
            return

        if zoom > self.texture_zoom_threshold and zoom < zoom_threshold:
            max_delta = zoom_threshold - self.texture_zoom_threshold
            delta = zoom - self.texture_zoom_threshold
            factor = delta / max_delta
            factor = factor
            n_shader.update_fading_factor(factor)
            texture_index = len(self.textures) - 1

        elif zoom < self.texture_zoom_threshold:
            zoom_norm = zoom - self.min_possible_zoom
            texture_index = int((zoom_norm) / self.texture_zoom_step)

            cur_zoom_step = texture_index * self.texture_zoom_step
            next_zoom_step = (texture_index + 1) * self.texture_zoom_step
            step_delta = (zoom_norm - cur_zoom_step) / (next_zoom_step - cur_zoom_step)
            n_shader.update_fading_factor(step_delta)

        # n_shader.set_tex2_enabled(True)
        n_shader.set_tex2_enabled(zoom < self.texture_zoom_threshold)
        # print("Drawing textures", texture_index, zoom, self.texture_zoom_threshold)
        if len(self.textures) > 0:
            self.textures[texture_index].draw()

    def draw_leafs_textures(self):
        # self.n_vertex.draw_texture()
        x, y, w, h, zoom = self.viewport

        if zoom < self.texture_zoom_threshold:
            # print("Zoom is larger than max zoom, returning tree draw")
            return
        for l in self.visible_leafs:
            if not l.texture_attached:
                l.texture_attached = True
                l.create_texture(self.n_net)
            l.draw_texture()

    def draw_vertices(self):
        # self.n_vertex.draw_texture()
        x, y, w, h, zoom = self.viewport
        if zoom < self.texture_zoom_threshold:
            # print("Zoom is larger than max zoom, returning tree draw")
            return
        for l in self.visible_leafs:
            #
            if not l.nodes_attached:
                l.nodes_attached = True
                l.create_nodes_view(self.n_net)
            l.draw_vertices()

    def draw_leafs_backgrounds(self):
        for l in self.visible_leafs:
            if not l.background_attached:
                l.background_attached = True
                l.create_background_view()
            l.draw_leaf_background()

    def draw_mega_texture(self):

        if len(self.visible_leafs) > 0 and self.visible_leafs != self.mega_visible_leafs:
            self.mega_visible_leafs = self.visible_leafs
            x1 = min([v.x1 for v in self.visible_leafs])
            x2 = max([v.x2 for v in self.visible_leafs])
            y1 = min([v.y1 for v in self.visible_leafs])
            y2 = max([v.y2 for v in self.visible_leafs])
            level = max([v.level for v in self.visible_leafs])
            self.mega_leaf = NTreeLeaf(x1, y1, x2 - x1, y2 - y1, level)
            # ml.color = (0.5, 0.5, 0.5)
            self.mega_leaf.texture_attached = True
            self.mega_leaf.create_texture(self.n_net)
        if self.mega_leaf is not None:
            self.mega_leaf.draw_texture()
