import gc
import random
import sys
import time

import numpy as np
from memory_profiler import profile

from app.draw.gl.draw.n_texture import NTexture
from app.draw.gl.draw.n_vertex import NVertex


class NEntity:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

        self.data_width = 0
        self.data_height = 0

        self.attached_lod_type = None
        self.attached_details_factor = None

        self.n_vertex = None
        self.data_source_texture = None
        self.textures = []

        self.debug = True

    def is_attached(self, lod_type, factor, compare_factor):
        if not compare_factor:
            return self.attached_lod_type == lod_type
        else:
            return self.attached_lod_type == lod_type and factor == self.attached_details_factor

    def create_texture_from_file(self, filename, factor, lod_type):
        self.destroy()
        start_time = time.time()
        tex = NTexture()
        tex.create_from_file(self.x1, self.y1, self.x2, self.y2,
                             filename=filename)
        self.textures = [tex]
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_texture_from_file", (time.time() - start_time) * 1000, "ms", "factor:", factor, lod_type)

    def create_texture_from_image(self, img_data, img_width, img_height, factor, lod_type):
        self.destroy()
        start_time = time.time()
        tex = NTexture()
        tex.create_from_image_data(self.x1, self.y1, self.x2, self.y2,
                                   img_data,
                                   img_width,
                                   img_height)
        self.textures = [tex]
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_texture_from_image", (time.time() - start_time) * 1000, "ms", "factor:", factor, lod_type)

    def create_textures_from_images(self, chunks, dimensions, factor, lod_type):
        self.destroy()
        start_time = time.time()
        for c, d in zip(chunks, dimensions):
            cx1, cy1, cx2, cy2 = d
            img_data, img_width, img_height = c
            tex = NTexture()
            tex.create_from_image_data(cx1, cy1, cx2, cy2, img_data, img_width, img_height)
            self.textures.append(tex)
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_textures_from_images", (time.time() - start_time) * 1000, "ms", "factor:", factor, lod_type)

    def create_color_map_texture(self, chunks, dimensions, factor, lod_type):
        self.destroy()
        start_time = time.time()
        self.textures = []
        for c, d in zip(chunks, dimensions):
            cx1, cy1, cx2, cy2 = d
            tex = NTexture()
            tex.create_from_floats_grid(cx1, cy1, cx2, cy2, c)
            self.textures.append(tex)
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_color_map_texture", (time.time() - start_time) * 1000, "ms", "factor:", factor,
                  lod_type)

    def create_colormap_texture_from_textures(self, width, height, chunks, dimensions, factor, lod_type):
        self.destroy()
        start_time = time.time()
        self.textures = []
        tex = NTexture()
        tex.create_from_floats_grid_chunks(self.x1, self.y1, self.x2, self.y2, width, height, chunks, dimensions)
        self.textures.append(tex)
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_colormap_texture_from_textures", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def create_nodes_view_from_texture_data(self, width, height, chunks, dimensions, factor, lod_type):
        start_time = time.time()
        self.data_source_texture = NTexture()
        self.data_source_texture.create_from_floats_grid_chunks(self.x1, self.y1, self.x2, self.y2, width, height,
                                                                chunks, dimensions)

        self.n_vertex = NVertex()
        self.data_width = width
        self.data_height = height
        self.n_vertex.create_nodes_from_texture(width * height)
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_nodes_view_from_texture_data", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def create_texture_from_nodes_view(self, n_window, factor, lod_type):
        self.destroy()
        start_time = time.time()
        self.n_vertex = NVertex()
        tex = NTexture()
        tex.create_from_frame_buffer(n_window, self.x1, self.y1, self.x2, self.y2,
                                     draw_func=self.n_vertex.draw_node_instances)
        self.attached_details_factor = factor
        self.textures = [tex]
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_texture_from_nodes_view", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def create_texture_from_points_view(self, n_window, factor, lod_type):
        self.destroy()
        start_time = time.time()
        tex = NTexture()
        tex.create_from_frame_buffer(n_window, self.x1, self.y1, self.x2, self.y2,
                                     draw_func=self.n_vertex.draw_points)
        self.attached_details_factor = factor
        self.textures = [tex]
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_texture_from_points_view", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def create_background_view(self):
        self.destroy()
        self.n_vertex = NVertex()
        self.n_vertex.create_plane(self.x1, self.y1, self.x2, self.y2, self.color)
        self.attached_lod_type = -1
        self.attached_details_factor = 1

    def create_points_view(self, positions_and_values, factor, lod_type):
        self.destroy()
        start_time = time.time()
        self.n_vertex = NVertex()
        self.n_vertex.create_points_from_buffer(positions_and_values)
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_points_view", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def create_points_view_from_texture_data(self, width, height, chunks, dimensions, factor, lod_type):
        start_time = time.time()
        self.data_source_texture = NTexture()
        self.data_source_texture.create_from_floats_grid_chunks(self.x1, self.y1, self.x2, self.y2, width, height,
                                                                chunks, dimensions)

        self.n_vertex = NVertex()
        self.n_vertex.create_points_from_texture(width * height)
        # self.n_vertex.num_instances = width * height
        self.data_width = width
        self.data_height = height
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_points_view_from_texture_data", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def create_nodes_view(self, positions_and_values, factor, lod_type):
        self.destroy()
        start_time = time.time()
        self.n_vertex = NVertex()
        self.n_vertex.create_nodes_from_buffer(positions_and_values)
        self.attached_lod_type = lod_type
        self.attached_details_factor = factor
        if self.debug:
            print("create_nodes_view", (time.time() - start_time) * 1000, "ms",
                  "factor:", factor,
                  lod_type)

    def draw_texture(self):
        for texture in self.textures:
            texture.draw()

    def draw_leaf_background(self):
        self.n_vertex.draw_plane()

    def draw_nodes(self):
        self.n_vertex.draw_node_instances()

    def draw_nodes_from_texture(self):
        self.data_source_texture.use_texture()
        self.n_vertex.draw_node_instances()

    def draw_points(self):
        self.n_vertex.draw_points()

    def draw_points_from_texture(self):
        self.data_source_texture.use_texture()
        self.n_vertex.draw_points_instances()

    # @profile
    def destroy(self):
        for tex in self.textures:
            tex.destroy()
        self.textures = []
        if self.n_vertex is not None:
            self.n_vertex.destroy()
        self.n_vertex = None
