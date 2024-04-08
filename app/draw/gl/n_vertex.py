import ctypes
import math
import OpenGL.GL as gl

import numpy as np
from PIL import Image


class NVertex():
    def __init__(self):
        self.num_instances = 0
        self.radius = 0.06
        self.num_segments = 20
        self.nodes_vao = None
        self.nodes_vbo = None

        self.plane_vao = None
        self.plane_vbo = None
        self.plane_color_vbo = None
        self.plane_indices = []

    def clean(self):
        gl.glDeleteBuffers(2, self.nodes_vbo)
        gl.glDeleteVertexArrays(1, self.nodes_vao)

    def draw_nodes(self):
        if self.nodes_vao is None:
            return
        gl.glBindVertexArray(self.nodes_vao)
        # Draw the triangle
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_FAN, 0, self.num_segments, self.num_instances)

    def draw_plane(self):
        if self.plane_vao is None:
            return
        # Bind the vertex array object (VAO)
        gl.glBindVertexArray(self.plane_vao)
        # Draw the square
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.plane_indices), gl.GL_UNSIGNED_INT, None)

    def create_plane(self, x1, y1, x2, y2, color=(1.0, 0.0, 0.0)):
        # Define the vertices of the square
        vertices = np.array([
            x1, y1,  # Bottom-left
            x2, y1,  # Bottom-right
            x2, y2,  # Top-right
            x1, y2  # Top-left
        ], dtype=np.float32)
        r, g, b = color

        # Define the colors for each vertex
        colors = np.array([
            r, g, b,  # Red
            r, g, b,  # Green
            r, g, b,  # Blue
            r, g, b  # Yellow
        ], dtype=np.float32)

        # Define the indices to form two triangles
        self.plane_indices = np.array([
            0, 1, 2,  # Triangle 1
            0, 2, 3  # Triangle 2
        ], dtype=np.uint32)

        # Create and bind the vertex buffer object (VBO)
        self.plane_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.plane_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        self.plane_color_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.plane_color_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)

        # Create and bind the vertex array object (VAO)
        self.plane_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.plane_vao)

        # Bind the vertex buffer object (VBO) for vertices
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.plane_vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Bind the vertex buffer object (VBO) for colors
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.plane_color_vbo)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Create and bind the element buffer object (EBO)
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.plane_indices.nbytes, self.plane_indices, gl.GL_STATIC_DRAW)

    def create_nodes(self,positions, colors=None):
        # Vertex data for a single circle
        dx, dy = positions.shape
        radius = self.radius
        self.num_instances = dx

        if colors is not None:
            instance_colors = colors
        else:
            instance_colors = np.random.uniform(0.0, 1.0, (self.num_instances, 3)).astype(np.float32)
        instance_positions = positions  # np.random.uniform(-10.0, 10.0, (self.num_instances, 2)).astype(np.float32)

        vertices = []
        for i in range(self.num_segments):
            theta = 2.0 * math.pi * float(i) / float(self.num_segments)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            vertices.extend([x, y])
        vertices = np.array(vertices, dtype=np.float32)

        # Instance data for multiple triangles

        self.nodes_vbo = gl.glGenBuffers(3)
        # Bind the vertex buffer and upload vertex data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.nodes_vbo[0])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        # Bind the instance buffer and upload instance data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.nodes_vbo[1])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_colors.nbytes, instance_colors, gl.GL_STATIC_DRAW)

        # Bind the instance position buffer and upload instance position data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.nodes_vbo[2])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_positions.nbytes, instance_positions, gl.GL_STATIC_DRAW)

        # Create vertex array object (VAO)
        self.nodes_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.nodes_vao)

        # Bind the vertex buffer to the VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.nodes_vbo[0])
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # Bind the instance buffer to the VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.nodes_vbo[1])
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribDivisor(1, 1)  # Set the instance data divisor

        # Bind the instance position buffer to the VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.nodes_vbo[2])
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribDivisor(2, 1)  # Set the instance data divisor
