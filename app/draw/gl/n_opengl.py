import time

import OpenGL.GL as gl
import glfw
from OpenGL.GL import *

from app.draw.gl.n_net import NNet
from app.draw.gl.n_shader import NShader
from app.draw.gl.n_texture import NTexture
from app.draw.gl.n_tree import NTree
from app.draw.gl.n_window import NWindow

start_time = 0
frame_count = 0

n_window = NWindow()
n_shader = NShader()
n_texture_shader = NShader()
n_net = NNet(n_window)

n_tree = NTree(0, n_net)
#n_texture = NTexture()

def render():
    global frame_count, start_time
    # Clear the screen
    # gl.glClearColor(0.0, 0.0, 1.0, 1.0)

    r,g,b,a = n_net.color_low
    gl.glClearColor(r,g,b,a)
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
    n_shader.use()
    n_shader.update_projection(n_window.get_projection_matrix())
    n_tree.draw()

    n_texture_shader.use()
    n_texture_shader.update_projection(n_window.get_projection_matrix())
    n_tree.draw_textures(n_texture_shader)
    #n_tree.draw_mega_texture(n_texture_shader)



    glfw.swap_buffers(n_window.window)


def on_viewport_updated():
    n_tree.update_viewport(n_window.viewport_to_world_cords())


def main():
    n_window.create_window()
    n_window.set_render_func(render)
    n_window.set_viewport_updated_func(on_viewport_updated)

    glEnable(GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    version = glGetString(GL_VERSION)
    print(f"OpenGL version: {version.decode('utf-8')}")
    n_shader.compile_vertices_program()
    n_texture_shader.compile_textures_program()

    # Init net
    n_net.init(100000000, [10000,100000,1000000])
    # generate nodes grid
    n_net.generate_net()
    # update tree size and depth using grid size
    n_tree.update_size()
    # calculate min zoom using grid size
    n_window.calculate_min_zoom(n_net)
    # create textures
    n_tree.create_textures(n_net,n_window.min_zoom)
    #n_tree.create_mega_texture(n_net)
    # generate tree
    n_tree.generate()
    # start render loop
    n_window.start_main_loop()
    glfw.terminate()
    gl.glDeleteProgram(n_shader.shader_program)


if __name__ == "__main__":
    main()
