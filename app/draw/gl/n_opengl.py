import time

import OpenGL.GL as gl
import glfw
from OpenGL.GL import *

from app.draw.gl.n_net import NNet
from app.draw.gl.n_shader import NShader
from app.draw.gl.n_tree import NTree
from app.draw.gl.n_window import NWindow

start_time = 0
frame_count = 0

n_window = NWindow()
n_shader = NShader()
n_net = NNet(n_window)

n_tree = NTree(4, n_net)


def render():
    global frame_count, start_time
    # Clear the screen
    # gl.glClearColor(0.0, 0.0, 1.0, 1.0)
    gl.glClearColor(253 / 255, 226 / 255, 243 / 255, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    # Update frame count
    frame_count += 1
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        fps_text = f"FPS: {fps:.2f}"
        frame_count = 0
        start_time = time.time()
        print(fps_text)

    # Use the shader program
    gl.glUseProgram(n_shader.shader_program)
    n_shader.update_projection(n_window.get_projection_matrix())

    n_tree.draw()
    glfw.swap_buffers(n_window.window)


def on_viewport_updated():
    n_tree.update_viewport(n_window.viewport_to_world_cords())


def main():
    n_window.create_window()
    n_window.set_render_func(render)
    n_window.set_viewport_updated_func(on_viewport_updated)

    glEnable(GL_DEPTH_TEST)
    version = glGetString(GL_VERSION)
    print(f"OpenGL version: {version.decode('utf-8')}")
    n_shader.compile()

    n_net.init(1000000000, [500000000, 1900, 190, 1])
    n_net.generate_net()
    n_tree.update_size()
    n_window.calculate_min_zoom(n_net)
    #n_tree.set_size(n_net.total_width, n_net.total_height)

    n_tree.generate()
    n_tree.create_view()

    n_window.start_main_loop()
    glfw.terminate()
    gl.glDeleteProgram(n_shader.shader_program)


if __name__ == "__main__":
    main()
