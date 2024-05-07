import gc
import math
import time

import numpy as np
from memory_profiler import profile

from app.draw.gl.draw.n_entity import NEntity
from app.draw.gl.n_lod import LodType
import OpenGL.GL as gl


class Quad:
    def __init__(self):
        self.tex_vbo = None
        self.vao = None
        self.ebo = None
        self.vbo = None
        self.indices = None
        self.tex_coords = None
        self.vertices = None
        self.created = False

    def create_quad(self):
        self.vertices = np.array([
            0, 0,  # Bottom-left
            1, 0,  # Bottom-right
            1, 1,  # Top-right
            0, 1  # Top-left
        ], dtype=np.float32)

        self.tex_coords = np.array([
            0.0, 0.0,  # Bottom-left becomes Top-left
            1.0, 0.0,  # Bottom-right becomes Top-right
            1.0, 1.0,  # Top-right becomes Bottom-right
            0.0, 1.0  # Top-left becomes Bottom-left
        ], dtype=np.float32)
        # Define the indices to form two triangles
        self.indices = np.array([
            0, 1, 2,  # Triangle 1
            0, 2, 3  # Triangle 2
        ], dtype=np.uint32)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_DYNAMIC_DRAW)

        #
        self.tex_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tex_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.tex_coords.nbytes, self.tex_coords, gl.GL_STATIC_DRAW)

        # Create and bind the vertex array object (VAO)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # Bind the vertex buffer object (VBO) for vertices
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # # Bind the texture
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tex_vbo)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Create and bind the element buffer object (EBO)
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)
        self.created = True

    def update_quad_position(self, x1, y1, x2, y2):
        if not self.created:
            return
        self.vertices = np.array([
            x1, y1,  # Bottom-left
            x2, y1,  # Bottom-right
            x2, y2,  # Top-right
            x1, y2  # Top-left
        ], dtype=np.float32)
        # Bind the VBO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # Update the buffer data
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.vertices.nbytes, self.vertices)

    def draw(self):
        if not self.created:
            return
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)


class EntityV2:
    def __init__(self):
        self.node_circle_vbo = None
        self.vao = None
        self.pbo = None
        self.texture = None

        self.empty_img = None
        self.created = False

        self.width = 0
        self.height = 0
        self.num_instances = 0
        self.radius = 0.06
        self.num_segments = 20

    def create_data_container_texture(self, width, height):
        self.texture = gl.glGenTextures(1)
        self.width = width
        self.height = height
        self.empty_img = np.empty((self.width, self.height), dtype=np.float32)
        self.num_instances = self.width * self.height

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, self.width, self.height, 0, gl.GL_RED, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error loading texture: {error}")

        radius = self.radius
        vertices = []
        for i in range(self.num_segments):
            theta = 2.0 * math.pi * float(i) / float(self.num_segments)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            vertices.extend([x, y])
        vertices = np.array(vertices, dtype=np.float32)

        # Instance data for multiple triangles
        self.node_circle_vbo = gl.glGenBuffers(1)
        # Bind the vertex buffer and upload vertex data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.node_circle_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # Create vertex array object (VAO)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # Bind the vertex buffer to the VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.node_circle_vbo)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error loading texture: {error}")

        self.created = True

    def update_data(self, chunks, dimensions):
        if not self.created:
            return
        start_time = time.time()
        # print("update texture data", self.width, self.height, self.num_instances, width, height)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, gl.GL_RED, gl.GL_FLOAT, self.empty_img)
        for c, d in zip(chunks, dimensions):
            cx1, cy1, cx2, cy2 = d
            # print(d)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, cx1, cy1, cx2 - cx1, cy2 - cy1, gl.GL_RED, gl.GL_FLOAT, c)

        # print("Texture updated", time.time() - start_time)

    def draw_points(self, count):
        if not self.created:
            return

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArraysInstanced(gl.GL_POINTS, 0, 1, count)

    def draw_nodes(self, count, detail):
        if not self.created:
            return
        count = min([count, self.num_instances])
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        segments_count = min([self.num_segments, math.ceil(detail * self.num_segments)])
        gl.glBindVertexArray(self.vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_FAN, 0,
                                 segments_count, count)


class NSceneV2:
    def __init__(self, n_lod, n_tree, n_net, n_window, textures_factory):
        self.n_lod = n_lod
        self.n_tree = n_tree
        self.n_net = n_net
        self.n_window = n_window
        self.textures_factory = textures_factory

        self.mega_leaf = None
        self.current_details_level = None

        self.current_width = 0
        self.current_height = 0

        self.quad = Quad()
        self.texture = EntityV2()
        self.current_size = 0

    def get_details_factor(self):
        """
        Calculate the down sample factor
        Factor is used to reduce the vertices count or texture quality
        If the zoom level is very small large amount of data could be visible inside small screen bounds
        Rendering everything without down sampling will cause issues and very low fps

        Compare screen space bounds with world grid bounds
        """
        x, y, w, h, zoom = self.n_window.viewport_to_world_cords()
        col_min, row_min, col_max, row_max = self.n_net.world_to_grid_position(x, y, x + w, y + h)
        subgrid_width = col_max - col_min
        subgrid_height = row_max - row_min

        target_width = self.n_window.width
        target_height = self.n_window.height

        width_factor = max(subgrid_width / target_width, 0.1)
        height_factor = max(subgrid_height / target_height, 0.1)
        # print("dw", subgrid_width, "dh", subgrid_height)
        # print("tw", target_width, "th", target_height)
        # print("factor", width_factor, height_factor)
        # # print("total count", int(target_height*target_width))
        factor = min(width_factor, height_factor)
        return math.ceil(factor)

    # @profile
    def update_scene_entities(self):

        if self.n_tree.mega_leaf is None:
            return
        should_update = False
        details_level = self.get_details_factor()

        if self.mega_leaf != self.n_tree.mega_leaf:
            self.mega_leaf = self.n_tree.mega_leaf
            should_update = True

        if details_level != self.current_details_level:
            self.current_details_level = details_level
            should_update = True

        if should_update:
            self.current_size = int(self.mega_leaf.w * self.mega_leaf.h)
            chunks, dimensions, width, height = self.n_net.get_subgrid_chunks_grid_dimensions(
                self.mega_leaf.x1,
                self.mega_leaf.y1,
                self.mega_leaf.x2,
                self.mega_leaf.y2,
                self.current_details_level)
            self.quad.update_quad_position(
                0,
                0,
                self.current_width,
                self.current_height,
            )
            self.texture.update_data(
                chunks,
                dimensions)

    def draw_scene(self,
                   n_points_shader,
                   n_static_texture_shader,
                   n_colors_texture_shader,
                   n_color_map_v2_texture_shader,
                   n_instances_from_buffer_shader,
                   n_instances_from_texture_shader
                   ):

        if not self.texture.created:
            self.current_width = self.n_window.width * 2
            self.current_height = self.n_window.height * 2
            self.texture.create_data_container_texture(
                self.current_width,
                self.current_height
            )
        if not self.quad.created:
            self.quad.create_quad()

        # Update
        self.update_scene_entities()

        leaf_w = int(self.mega_leaf.w / 0.2)
        leaf_h = int(self.mega_leaf.h / 0.2)
        size = leaf_w * leaf_h

        max_points_count = 1000000
        max_nodes_count = 500000

        if size >= max_points_count:
            n_color_map_v2_texture_shader.use()
            n_color_map_v2_texture_shader.update_texture_width(self.current_width)
            n_color_map_v2_texture_shader.update_texture_height(self.current_height)
            n_color_map_v2_texture_shader.update_position_offset(self.mega_leaf.x1, self.mega_leaf.y1)
            n_color_map_v2_texture_shader.update_details_factor(self.current_details_level)
            self.quad.draw()
        else:
            n_instances_from_texture_shader.use()
            n_instances_from_texture_shader.update_texture_width(self.current_width)
            n_instances_from_texture_shader.update_texture_height(self.current_height)
            n_instances_from_texture_shader.update_target_width(leaf_w)
            n_instances_from_texture_shader.update_target_height(leaf_h)
            n_instances_from_texture_shader.update_position_offset(self.mega_leaf.x1, self.mega_leaf.y1)
            n_instances_from_texture_shader.update_details_factor(self.current_details_level)
            if size < max_nodes_count:
                self.texture.draw_nodes(size, 1)
            elif size < max_points_count:
                self.texture.draw_points(size)
