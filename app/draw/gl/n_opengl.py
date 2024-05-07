import os
import time

import OpenGL.GL as gl
import glfw
import psutil
from OpenGL.GL import *
from transformers import AutoModelForCausalLM

from app.draw.gl.n_lod import NLvlOfDetails, LodType
from app.draw.gl.n_net import NNet
from app.draw.gl.n_scene import NScene
from app.draw.gl.n_scene_v2 import NSceneV2
from app.draw.gl.n_tree import NTree
from app.draw.gl.n_window import NWindow
from app.draw.gl.utils.c_color_theme import NColorTheme
from app.draw.gl.utils.n_tex_factory import RGBGridTextureFactory, ImageTextureFactory

start_time = 0
frame_count = 0

n_window = NWindow()
color_theme = NColorTheme()
textures_factory = ImageTextureFactory()
rgb_textures_factory = RGBGridTextureFactory(color_theme)

n_net = NNet(n_window, color_theme)
n_lod = NLvlOfDetails(n_net, n_window)
n_tree = NTree(0)
n_scene = NSceneV2(n_lod, n_tree, n_net, n_window, rgb_textures_factory)
DEBUG = False

process = psutil.Process(os.getpid())
memory_usage = 0
virtual_memory_usage = 0


def print_memory_usage():
    global memory_usage
    global virtual_memory_usage
    mem_info = process.memory_info()
    usage = f"{mem_info.rss / (1024 * 1024 * 1024):.2f}"
    if memory_usage != usage:
        memory_usage = usage
        print(f"Memory usage: {memory_usage} GB (Resident Set Size)")


def render():
    global frame_count, start_time

    r, g, b, a = color_theme.color_low
    gl.glClearColor(r, g, b, a)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    # Update frame count
    frame_count += 1
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        if fps < 80:
            fps_text = f"FPS: {fps:.2f}"
            frame_count = 0
            start_time = time.time()
            print(fps_text)

    if DEBUG:
        # Use the shader program
        n_window.n_debug_shader.use()
        n_window.n_debug_shader.update_projection(n_window.get_projection_matrix())
        n_lod.load_current_level()
        n_scene.draw_debug_tree()

        n_window.n_debug_shader.use()
        n_window.n_debug_shader.update_projection(n_window.get_projection_matrix())

    else:

        n_window.n_instances_from_buffer_shader.use()
        n_window.n_instances_from_buffer_shader.update_projection(n_window.get_projection_matrix())
        n_window.n_instances_from_buffer_shader.update_color_map(color_theme.name, color_theme.color_array)

        n_window.n_points_shader.use()
        n_window.n_points_shader.update_projection(n_window.get_projection_matrix())
        n_window.n_points_shader.update_color_map(color_theme.name, color_theme.color_array)

        n_window.n_static_texture_shader.use()
        n_window.n_static_texture_shader.update_projection(n_window.get_projection_matrix())
        n_window.n_static_texture_shader.update_color_map(color_theme.name, color_theme.color_array)

        n_window.n_color_map_texture_shader.use()
        n_window.n_color_map_texture_shader.update_projection(n_window.get_projection_matrix())
        n_window.n_color_map_texture_shader.update_color_map(color_theme.name, color_theme.color_array)

        n_window.n_color_map_v2_texture_shader.use()
        n_window.n_color_map_v2_texture_shader.update_projection(n_window.get_projection_matrix())
        n_window.n_color_map_v2_texture_shader.update_color_map(color_theme.name, color_theme.color_array)

        n_window.n_instances_from_texture_shader.use()
        n_window.n_instances_from_texture_shader.update_projection(n_window.get_projection_matrix())
        n_window.n_instances_from_texture_shader.update_color_map(color_theme.name, color_theme.color_array)

        n_lod.load_current_level()
        n_scene.draw_scene(
            n_window.n_points_shader,
            n_window.n_static_texture_shader,
            n_window.n_color_map_texture_shader,
            n_window.n_color_map_v2_texture_shader,
            n_window.n_instances_from_buffer_shader,
            n_window.n_instances_from_texture_shader
        )

    glfw.swap_buffers(n_window.window)
    print_memory_usage()


def on_viewport_updated():
    viewport = n_window.viewport_to_world_cords()
    n_tree.update_viewport(viewport)
    n_net.update_viewport(viewport)
    n_lod.update_viewport(viewport)


def create_level_of_details():
    print("Creating level of details")

    n_lod.add_level(LodType.LEAVES_COLOR_MAP_TEXTURE, 0.0)
    n_lod.add_level(LodType.LEAVES_COLOR_MAP_TEXTURE, 0.0001)
    n_lod.add_level(LodType.LEAVES_POINTS_FROM_TEXTURE, 0.003)
    n_lod.add_level(LodType.LEAVES_NODES_FROM_TEXTURE, 0.025)
    n_lod.add_level(LodType.LEAVES_NODES, 0.035)

    n_lod.dump()


def main():
    n_window.create_window()
    n_window.set_render_func(render)
    n_window.set_viewport_updated_func(on_viewport_updated)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    gl.glEnable(gl.GL_MULTISAMPLE)

    version = glGetString(GL_VERSION)
    print(f"OpenGL version: {version.decode('utf-8')}")
    n_window.n_debug_shader.compile_debug_program()
    n_window.n_points_shader.compile_points_program()
    n_window.n_static_texture_shader.compile_static_texture_program()
    n_window.n_color_map_texture_shader.compile_color_map_texture_program()
    n_window.n_color_map_v2_texture_shader.compile_color_map_v2_texture_program()
    n_window.n_instances_from_buffer_shader.compile_instances_from_buffer_program()
    n_window.n_instances_from_texture_shader.compile_instances_v2_program()

    print_memory_usage()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model =  AutoModelForCausalLM.from_pretrained(model_name)
    all_layers_and_parameters = list(model.named_parameters())
    tensors = [tensor for name, tensor in all_layers_and_parameters]
    print_memory_usage()
    n_net.init_from_tensors(tensors)
    #n_net.init_from_size([1000000])
    print_memory_usage()
    # update tree size and depth using grid size
    n_tree.set_size(n_net.total_width, n_net.total_height)
    # n_tree.load_net_size()
    # calculate min zoom using grid size
    n_window.calculate_min_zoom(n_net)
    # create level of details
    create_level_of_details()

    # generate tree
    print("Generating tree")
    n_tree.generate()
    # start render loop
    n_window.reset_to_center(n_net)
    print("Main loop")
    n_window.start_main_loop()
    glfw.terminate()
    gl.glDeleteProgram(n_window.n_debug_shader.shader_program)
    gl.glDeleteProgram(n_window.n_points_shader.shader_program)
    gl.glDeleteProgram(n_window.n_static_texture_shader.shader_program)
    gl.glDeleteProgram(n_window.n_color_map_texture_shader.shader_program)
    gl.glDeleteProgram(n_window.n_instances_from_buffer_shader.shader_program)
    gl.glDeleteProgram(n_window.n_instances_from_texture_shader.shader_program)


if __name__ == "__main__":
    main()
