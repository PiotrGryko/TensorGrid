import random
import time

import numpy as np

from app.draw.gl.n_texture import NTexture
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
        self.texture_attached = False
        self.grid = None
        self.n_vertex = NVertex()
        self.nodes = []
        self.n_texture = None

    def create_nodes_view(self, n_net):
        self.grid, self.colors  = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2)
        #flat_grid = self.grid[np.where(self.grid != n_net.default_value)]
        flat_array = np.reshape(self.grid, (-1, 2))
        self.nodes = flat_array
        if len(self.nodes) == 0:
            return

        self.n_vertex.create_nodes(self.nodes, self.colors)

    def create_background_view(self):
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)

    def create_texture(self, n_net):
        self.n_texture = NTexture()
        # grid = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2, i)
        img_data, img_width, img_height = n_net.get_texture(self.x1, self.y1, self.x2, self.y2, 1)
        self.n_texture.create_from_data(self.x1, self.y1, self.x2, self.y2, img_data, img_width, img_height)

    def draw_vertices(self):
        self.n_vertex.draw_nodes()
        #self.n_vertex.draw_plane()

    def draw_texture(self):
        if self.n_texture:
            self.n_texture.draw()

    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)


class NTree(BSPTree):
    def __init__(self, depth, n_net):
        super().__init__(0, 0, depth)
        self.visible_leafs = []
        self.visible_bounds = [0, 0, 0, 0]
        self.viewport = None
        self.n_net = n_net
        self.n_vertex = NVertex()
        self.textures = []
        self.zoom_step = 0
        self.min_zoom = 0
        self.max_zoom = 0.03

        self.texture = None
        self.draw_dirty = False

    def update_size(self):
        self.set_size(self.n_net.total_width, self.n_net.total_height)
        cc = self.n_net.grid_columns_count
        rc = self.n_net.grid_rows_count
        side = cc if cc > rc else rc
        self.depth = int(side / 500)
        self.depth = self.depth if self.depth < 7 else 7
        print("Tree depth assigned ", self.depth)
        # self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

    def set_size(self, w, h):
        super().set_size(w, h)
        self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

    def update_viewport(self, viewport):
        visible = []
        not_visible = []
        self.viewport = viewport
        self.traverse(viewport, visible, not_visible)
        if visible != self.visible_leafs:
            self.visible_leafs = visible
            x1 = min([l.x1 for l in self.visible_leafs])
            x2 = max([l.x2 for l in self.visible_leafs])
            y1 = min([l.y1 for l in self.visible_leafs])
            y2 = max([l.y2 for l in self.visible_leafs])
            self.visible_bounds = (x1, y1, x2, y2)
            self.draw_dirty = True

    def create_textures(self, n_net, min_zoom):
        print("Greate zoom textures")
        start_time = time.time()

        self.min_zoom = min_zoom

        self.zoom_step =1 if self.depth ==0 else (self.max_zoom - self.min_zoom) / (self.depth)

        materials = []
        for i in range(self.depth, 0, -1):
            img_data, img_width, img_height = n_net.get_texture(0, 0, self.width, self.height, i)
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


        # for i in range(self.depth, 0, -1):
        #     texture = NTexture()
        #     # grid = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2, i)
        #     img_data, img_width, img_height = n_net.get_texture(0, 0, self.width, self.height,i)
        #     texture.create_from_data(0, 0, self.width, self.height, img_data, img_width, img_height)
        #     self.textures.append(texture)

        #
        # self.grid = n_net.get_positions_grid(self.x1, self.y1, self.x2, self.y2, factor)
        # img_data, img_width, img_height = n_net.get_texture(self.x1, self.y1, self.x2, self.y2, factor)
        # self.n_texture.create_from_data(self.x1, self.y1, self.x2, self.y2, img_data, img_width, img_height)
        # self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)
        # self.n_texture.create(self.x1, self.y1, self.x2, self.y2)

    def draw_textures(self, n_shader):
        x, y, w, h, zoom = self.viewport
        if zoom > self.max_zoom:
            return

        zoom_norm = zoom - self.min_zoom

        texture_index = int((zoom_norm) / self.zoom_step)


        cur_zoom_step = texture_index * self.zoom_step
        next_zoom_step = (texture_index+1) * self.zoom_step
        step_delta = (zoom_norm-cur_zoom_step)/(next_zoom_step-cur_zoom_step)

        #print("index", texture_index,len(self.textures), "cur", cur_zoom_step, "next",next_zoom_step, "zoom",zoom, "step",self.zoom_step, "delta",step_delta)

        n_shader.update_fading_factor(step_delta)


        self.textures[texture_index].draw()

        # if self.draw_dirty:
        #     self.draw_dirty=False
        #     x1,y1,x2,y2 = self.visible_bounds
        #     img_data, img_width, img_height = n_net.get_texture(x1,y1,x2,y2,self.depth-texture_index)
        #     self.texture = NTexture()
        #     self.texture.create_from_data(x1,y1,x2,y2, img_data, img_width, img_height)
        #     self.textures.append(self)
        #     #print("Texture generated", self.visible_bounds)
        # #
        # if self.texture:
        #     self.texture.draw()

        # for l in self.visible_leafs:
        #     if not l.texture_attached:
        #         l.texture_attached = True
        #         l.create_texture(n_net)
        #     l.draw_texture()

    def draw(self):
        # self.n_vertex.draw_texture()
        x, y, w, h, zoom = self.viewport
        if zoom < self.max_zoom:
            return

        for l in self.visible_leafs:
            #
            if not l.nodes_attached:
                l.nodes_attached = True
                # print(zoom)
                # factor = int(0.05 / zoom)
                # print(factor)
                # level = l.level*4 if l.level>1 else 1
                l.create_nodes_view(self.n_net)
            if not l.background_attached:
                l.background_attached = True
                # l.create_texture(self.n_net,1)
                l.create_background_view()
            l.draw_vertices()
