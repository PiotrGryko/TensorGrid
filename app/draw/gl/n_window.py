import OpenGL.GL as gl
import glfw

from app.draw.gl.n_projection import Projection
from app.draw.gl.draw.n_shader import NShader


class NWindow:
    def __init__(self):
        self.window = None
        self.dragging = False
        self.width = 1280
        self.height = 1280
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0
        self.zoom_factor = 1.0
        self.min_zoom = 0.01
        self.max_zoom = 1
        self.zoom_step = 0.01
        self.aspect_ratio = self.width / self.height
        self.projection = Projection()
        self.render_func = None
        self.viewport_updated_func = None
        self.zoom_percent = 0

        self.n_nodes_shader = NShader()
        self.n_colors_shader = NShader()
        self.n_material_one_shader = NShader()
        self.n_material_two_shader = NShader()
        self.n_colors_texture_material_one_shader = NShader()
        self.n_colors_texture_material_two_shader = NShader()

    def calculate_min_zoom(self, n_net):
        content_width, content_height = n_net.total_width, n_net.total_height
        w, h = self.window_to_normalized_cords(self.width, self.height)
        min_zoom_x = w / content_width / 1.2
        min_zoom_y = h / content_height / 1.2
        self.min_zoom = min_zoom_x if min_zoom_x < min_zoom_y else min_zoom_y
        self.zoom_factor = self.min_zoom  # current zoom value
        self.zoom_step = (self.max_zoom - self.min_zoom) / 10000  # zoom step change on every mouse wheel scroll
        self.mouse_scroll_callback(self.window, 1, 1)
        print(w, h, content_width, content_height)
        print("Min zoom calculated:", self.min_zoom, self.max_zoom, self.zoom_step)

    def create_window(self):
        # Initialize OpenGL and create a window
        glfw.init()

        # Set GLFW context hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.SAMPLES,4)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_TRUE)

        # Create a GLFW window
        self.window = glfw.create_window(self.width, self.height, "Hello World", None, None)
        # Make the created window's context current

        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.frame_buffer_size_callback)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_position_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_window_refresh_callback(self.window, self.window_refresh_callback)

        # Check if window creation succeeded
        if not self.window:
            glfw.terminate()
            raise ValueError("Failed to create GLFW window")

    def start_main_loop(self):
        while not glfw.window_should_close(self.window):
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)
            if self.render_func:
                self.render_func()
            glfw.poll_events()

    def destroy_window(self):
        glfw.terminate()

    def set_render_func(self, render_func):
        self.render_func = render_func

    def set_viewport_updated_func(self, viewport_updated_func):
        self.viewport_updated_func = viewport_updated_func

    def get_projection_matrix(self):
        return self.projection.matrix

    def window_to_normalized_cords(self, x, y):
        sx = x / self.width * 2.0
        sy = y / self.height * 2.0
        return sx, sy


    def viewport_to_world_cords(self):
        '''
        :return: current viewport in world coordinates
        '''
        # Window bottom left
        x1, y1 = self.projection.window_to_world_point(-1, -1)
        # Window top right
        x2, y2 = self.projection.window_to_world_point(1, 1)
        w, h = x2 - x1, y2 - y1
        return (x1, y1, w, h, self.zoom_factor)

    def world_coords_to_screen_coords(self, x1, y1, x2, y2):
        '''
        :return: world coordinates (for example x5400:y1300) to screen coords (-1, -1 top left corner, 1,1 top right corner)
        '''
        # Window bottom left
        sx1, sy1 = self.projection.world_to_window_point(x1, y1)
        # Window top right
        sx2, sy2 = self.projection.world_to_window_point(x2, y2)

        return (sx1, sy1, sx2, sy2, self.zoom_factor)

    def screen_coords_to_window_coords(self, x1, y1, x2, y2):
        '''
        :return:  screen coordinates (-1, -1 top left corner, 1,1 top right corner) to window coordinates (0,0 top left, 1280,1280 top right)
        '''
        sx1 = (x1 + 1) * self.width / 2.0
        sy1 = (y1 + 1) * self.height / 2.0
        sx2 = (x2 + 1) * self.width / 2.0
        sy2 = (y2 + 1) * self.height / 2.0
        return (sx1, sy1, sx2, sy2)

    def on_viewport_updated(self):
        if self.viewport_updated_func:
            self.viewport_updated_func()

    def window_refresh_callback(self, window):
        if self.render_func:
            self.render_func()

    def frame_buffer_size_callback(self, window, w, h):
        self.width, self.height = w, h
        self.aspect_ratio = w / h
        self.projection.set_aspect_ratio(self.aspect_ratio)
        print('resize', self.width, self.height)
        self.on_viewport_updated()

    def mouse_scroll_callback(self, window, x_offset, y_offset):

        delta = - y_offset * self.zoom_step
        self.zoom_factor += delta
        # print("Zoom offset ",y_offset, self.zoom_factor)
        if self.zoom_factor <= self.min_zoom:
            self.zoom_factor = self.min_zoom

        mx = self.last_mouse_x - self.width / 2
        my = self.last_mouse_y - self.height / 2
        zoom_x, zoom_y = self.window_to_normalized_cords(mx, my)
        self.projection.zoom(zoom_x, zoom_y, self.zoom_factor)
        self.on_viewport_updated()

        zoom_percent = self.zoom_factor / self.max_zoom
        formatted = "{:.1%}".format(zoom_percent)
        if self.zoom_percent != formatted:
            self.zoom_percent = formatted
            print("zoom: ", formatted)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                pass
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.dragging = True
            elif action == glfw.RELEASE:
                self.dragging = False

    def mouse_position_callback(self, window, xpos, ypos):
        xpos, ypos = xpos, self.height - ypos
        if not self.dragging:
            self.last_mouse_x = xpos
            self.last_mouse_y = ypos
            return

        # Calculate the translation offset based on mouse movement
        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y

        dx = dx / self.width * 2.0
        dy = dy / self.height * 2.0

        self.projection.translate_by(dx, dy)
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos
        self.on_viewport_updated()

    def reset_to_center(self, n_net):
        world_w, world_h = self.projection.world_to_window_point(n_net.total_width, n_net.total_height)
        # by default world is positioned (0,0) in the bottom left corner
        dx = self.width / 2
        dy = self.height / 2
        dx = dx / self.width * 2.0
        dx = dx - world_w - (dx - world_w) / 2
        dy = dy / self.height * 2.0
        dy = dy - world_h - (dy - world_h) / 2

        self.projection.translate_by(dx, dy)
        self.on_viewport_updated()
