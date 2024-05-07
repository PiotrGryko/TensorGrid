import math

import OpenGL.GL as gl
import numpy as np
from PIL import Image

from app.draw.gl.n_net import unpack_shape


class NTexture:
    def __init__(self):
        self.material = None
        self.triangle = None

    def create_from_file(self, x1, y1, x2, y2, filename):
        self.material = Material().from_file(filename)
        self.triangle = Triangle(x1, y1, x2, y2)

    def create_from_image_data(self, x1, y1, x2, y2, img_data, img_width, img_height):
        self.material = Material().from_image_data(img_data, img_width, img_height)
        self.triangle = Triangle(x1, y1, x2, y2)

    def create_from_frame_buffer(self, n_window, x1, y1, x2, y2, draw_func):
        self.material = Material().from_frame_buffer(n_window, x1, y1, x2, y2, draw_func)
        self.triangle = Triangle(x1, y1, x2, y2)

    def create_from_floats_grid(self, x1, y1, x2, y2, grid):
        self.material = Material().from_floats_grid(grid)
        self.triangle = Triangle(x1, y1, x2, y2)

    def create_from_floats_grid_chunks(self, x1, y1, x2, y2, width, height, chunks, dimensions):
        self.material = Material().from_floats_grid_chunks(width, height, chunks, dimensions)
        self.triangle = Triangle(x1, y1, x2, y2)

    def use_texture(self):
        if self.material is None:
            return
        self.material.use_texture1()

    def draw(self):
        if self.material is None:
            return
        self.material.use_texture1()

        gl.glBindVertexArray(self.triangle.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.triangle.indices), gl.GL_UNSIGNED_INT, None)

    def destroy(self):
        if self.material is not None:
            self.material.destroy()
        if self.triangle is not None:
            self.triangle.destroy()


class Triangle:
    def __init__(self, x1, y1, x2, y2, material_id=1):
        # x1, y1 = -0.5, -0.5
        # x2, y2 = 0.5, 0.5
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.vertices = np.array([
            x1, y1,  # Bottom-left
            x2, y1,  # Bottom-right
            x2, y2,  # Top-right
            x1, y2  # Top-left
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
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)

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
        if material_id == 1:
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        elif material_id == 2:
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Create and bind the element buffer object (EBO)
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)

    def destroy(self):
        gl.glBindVertexArray(0)  # Unbind any VAOs
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)  # Unbind any EBOs bound to GL_ELEMENT_ARRAY_BUFFER
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind any VBOs bound to GL_ARRAY_BUFFER

        if self.vao is not None:
            gl.glDeleteVertexArrays(1, (self.vao,))
        if self.vbo is not None:
            gl.glDeleteBuffers(1, (self.vbo,))
        if self.tex_vbo is not None:
            gl.glDeleteBuffers(1, (self.tex_vbo,))
        if self.ebo is not None:
            gl.glDeleteBuffers(1, (self.ebo,))
        gl.glFlush()


class Material:
    def __init__(self):
        self.nodes_vao = None
        self.circle_vbo = None
        self.texture = None
        self.img_data = None
        self.image_width = None
        self.image_height = None
        self.fbo = None
        self.pbo = None


    def create_pbo(self, img_data):
        # Generate a buffer ID for the PBO
        pbo = gl.glGenBuffers(1)
        # Bind the PBO to the PIXEL_UNPACK_BUFFER target
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        # Allocate memory for the PBO and initialize it with image data
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, img_data.nbytes, img_data, gl.GL_STREAM_DRAW)
        # Unbind the PBO from the PIXEL_UNPACK_BUFFER target
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        return pbo

    def from_file(self, filepath):
        image = Image.open(filepath)
        img_data = np.array(image)

        image_width, image_height = image.width, image.height
        print(image, image_width, image_height)
        return self.from_image_data(img_data, image_width, image_height)

    def from_image_data(self, img_data, image_width, image_height):
        self.img_data = img_data
        self.image_width = image_width
        self.image_height = image_height
        self.texture = gl.glGenTextures(1)

        self.pbo = self.create_pbo(img_data)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # Bind the PBO to load texture data
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        # Load texture data from PBO
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image_width, image_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        None)
        # Unbind the PBO
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error loading texture: {error}")

        return self

    def from_floats_grid(self, grid):
        height, width = unpack_shape(grid)
        self.texture = gl.glGenTextures(1)
        self.pbo = self.create_pbo(grid)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # Bind the PBO to load texture data
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        # Load texture data from PBO
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, width, height, 0, gl.GL_RED, gl.GL_FLOAT, None)
        # Unbind the PBO
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error loading texture: {error}")
        return self

    def from_floats_grid_chunks(self, width, height, chunks, dimensions):
        self.texture = gl.glGenTextures(1)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, width, height, 0, gl.GL_RED, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        for c, d in zip(chunks, dimensions):
            cx1, cy1, cx2, cy2 = d
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, cx1, cy1, cx2 - cx1, cy2 - cy1, gl.GL_RED, gl.GL_FLOAT, c)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"Error loading texture: {error}")
        return self

    def from_frame_buffer(self, n_window, x1, y1, x2, y2, draw_func):
        # Generate texture
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        (sx1, sy1, sx2, sy2, zoom_factor) = n_window.world_coords_to_screen_coords(x1, y1, x2, y2)
        (wx1, wy1, wx2, wy2) = n_window.screen_coords_to_window_coords(sx1, sy1, sx2, sy2)

        image_width = round(wx2 - wx1)
        image_height = round(wy2 - wy1)

        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

        # print("Write texture to FBO width and height  ", image_width, image_height)
        # print("Quad world coord ", x1, y1, x2, y2)
        # print("Quad scene coord (Normalized Device Coordinates (NDC))", sx1, sy1, sx2, sy2)
        # print("Quad window coord", wx1, wy1, wx2, wy2)
        # print(viewport)

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image_width, image_height, 0, gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE, None)

        # Generate framebuffer
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture, 0)

        # Check framebuffer status
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer is not complete.")

        gl.glViewport(0, 0, image_width, image_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        tile_projection_matrix = n_window.projection.get_projection_for_tile(sx1, sy1, sx2, sy2)
        n_window.n_instances_from_buffer_shader.update_projection(tile_projection_matrix)

        # Activate the first texture unit and bind your texture
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        # Your drawing code here
        # n_vertex.draw_plane()
        draw_func()

        # # #
        # # # # # #
        # gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        # data = gl.glReadPixels(0, 0, image_width, image_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        # image = Image.frombytes("RGBA", (int(image_width), int(image_height)), data)
        # image.save(f"tiles/output{int(image_width)}_{int(image_height)}.png")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        n_window.n_instances_from_buffer_shader.update_projection(n_window.projection.matrix)

        gl.glViewport(0, 0, viewport[2], viewport[3])

        return self

    def use_texture1(self):
        # print("use texture 0")
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

    def use_texture2(self):
        # print("use texture 1")
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

    def destroy(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)  # Unbind the FBO
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)  # Unbind the PBO
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)  # Unbind the texture

        if self.fbo is not None:
            gl.glDeleteBuffers(1, [self.fbo])
            self.fbo = None
        if self.pbo is not None:
            gl.glDeleteBuffers(1, [self.pbo])
            self.pbo = None
        if self.texture is not None:
            gl.glDeleteTextures(1, [self.texture])
            self.texture = None
        gl.glFlush()
