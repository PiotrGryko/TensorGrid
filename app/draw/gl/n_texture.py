import ctypes

import numpy as np
from PIL import Image
import OpenGL.GL as gl


class NTexture:
    def __init__(self):
        self.material_one = None
        self.material_two = None
        self.triangle = None

    def create_from_file(self, x1, y1, x2, y2, filename):
        self.material_one = Material().from_file(filename)
        self.triangle = Triangle(x1, y1, x2, y2, color=(1.0, 1.0, 1.0))

    def create_from_data(self, x1, y1, x2, y2, img_data, img_width, img_height):
        self.material_one = Material().from_image_data(img_data, img_width, img_height)
        self.triangle = Triangle(x1, y1, x2, y2, color=(1.0, 1.0, 1.0))

    def add_second_texture_from_data(self, img_data, img_width, img_height):
        self.material_two = Material().from_image_data(img_data, img_width, img_height)

    def draw(self):
        if not self.material_one:
            return
        self.material_one.use(gl.GL_TEXTURE0)
        if self.material_two:
            self.material_two.use(gl.GL_TEXTURE1)

        gl.glBindVertexArray(self.triangle.vao)
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.triangle.indices))
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.triangle.indices), gl.GL_UNSIGNED_INT, None)


class Triangle:
    def __init__(self, x1, y1, x2, y2, color=(1.0, 0.0, 0.0)):
        # x1, y1 = -0.5, -0.5
        # x2, y2 = 0.5, 0.5

        self.vertices = np.array([
            x1, y1,  # Bottom-left
            x2, y1,  # Bottom-right
            x2, y2,  # Top-right
            x1, y2  # Top-left
        ], dtype=np.float32)
        r, g, b = color

        # Define the colors for each vertex
        self.colors = np.array([
            r, g, b,  # Red
            r, g, b,  # Green
            r, g, b,  # Blue
            r, g, b  # Yellow
        ], dtype=np.float32)

        self.tex_coords = np.array([
            0.0, 1.0,  # Bottom-left
            1.0, 1.0,  # Bottom-right
            1.0, 0.0,  # Top-right
            0.0, 0.0  # Top-left
        ], dtype=np.float32)
        # Define the indices to form two triangles
        self.indices = np.array([
            0, 1, 2,  # Triangle 1
            0, 2, 3  # Triangle 2
        ], dtype=np.uint32)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)

        self.color_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, gl.GL_STATIC_DRAW)
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

        # Bind the vertex buffer object (VBO) for colors
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # # Bind the texture
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tex_vbo)
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Create and bind the element buffer object (EBO)
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)

    def destroy(self):
        gl.glDeleteVertexArrays(1, (self.vao,))
        gl.glDeleteBuffers(1, (self.vbo,))


class Material:
    def __init__(self):
        self.texture = None

    def from_file(self, filepath):

        image = Image.open(filepath)
        # image.show()

        img_data = np.array(image)

        image_width, image_height = image.width, image.height
        print(image, image_width, image_height)
        return self.from_image_data(img_data, image_width, image_height)

        # self.texture = gl.glGenTextures(1)
        # gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        #
        # image = Image.open(filepath)
        # #image.show()
        #
        # img_data = np.array(image)
        #
        # image_width, image_height = image.width, image.height
        # print(image, image_width, image_height)
        #
        # gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image_width, image_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
        #                 img_data)
        # error = gl.glGetError()
        # if error != gl.GL_NO_ERROR:
        #     print(f"Error loading texture: {error}")
        # gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        # error = gl.glGetError()
        # if error != gl.GL_NO_ERROR:
        #     print(f"Error generating mipmap texture: {error}")

    def from_image_data(self, img_data, image_width, image_height):
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image_width, image_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        img_data)
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error loading texture: {error}")
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error generating mipmap texture: {error}")
        return self

    def use(self, tex_index=gl.GL_TEXTURE0):
        gl.glActiveTexture(tex_index)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

    def destroy(self):
        gl.glDeleteTextures(1, (self.texture,))
