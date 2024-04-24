import time

import OpenGL.GL as gl
import glfw
from OpenGL.GL import *
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.draw.gl.calculator.n_numpy_calculator import NumpyCalculator
from app.draw.gl.utils.c_color_theme import NColorTheme
from app.draw.gl.n_lod import NLvlOfDetails, LodType
from app.draw.gl.n_net import NNet
from app.draw.gl.calculator.n_open_cl_calculator import OpenCLPositionsCalculator
from app.draw.gl.n_scene import NScene
from app.draw.gl.utils.n_tex_factory import RGBGridTextureFactory, ImageTextureFactory
from app.draw.gl.n_tree import NTree
from app.draw.gl.n_window import NWindow

start_time = 0
frame_count = 0

n_window = NWindow()

color_theme = NColorTheme()

textures_factory = ImageTextureFactory()
rgb_textures_factory = RGBGridTextureFactory(color_theme)

calculator = NumpyCalculator()
n_net = NNet(n_window, color_theme, calculator)
n_lod = NLvlOfDetails(n_net, n_window)
n_tree = NTree(0)
n_scene = NScene(n_lod, n_tree, n_net, n_window, rgb_textures_factory)
DEBUG = False


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
        if fps < 60:
            fps_text = f"FPS: {fps:.2f}"
            frame_count = 0
            start_time = time.time()
            print(fps_text)

    if DEBUG:
        # Use the shader program
        n_window.n_vertices_shader.use()
        n_window.n_vertices_shader.update_projection(n_window.get_projection_matrix())
        n_lod.load_current_level()
        n_scene.draw_debug_tree()

        n_window.n_vertices_shader.use()
        n_window.n_vertices_shader.update_projection(n_window.get_projection_matrix())

        n_window.n_material_one_shader.use()
        n_window.n_material_one_shader.update_projection(n_window.get_projection_matrix())

        n_window.n_material_two_shader.use()
        n_window.n_material_two_shader.update_projection(n_window.get_projection_matrix())
    else:

        n_lod.load_current_level()
        n_scene.draw_scene(
            n_window.n_vertices_shader,
            n_window.n_material_one_shader,
            n_window.n_material_two_shader
        )

        n_window.n_vertices_shader.use()
        n_window.n_vertices_shader.update_projection(n_window.get_projection_matrix())

        n_window.n_material_one_shader.use()
        n_window.n_material_one_shader.update_projection(n_window.get_projection_matrix())

        n_window.n_material_two_shader.use()
        n_window.n_material_two_shader.update_projection(n_window.get_projection_matrix())

    glfw.swap_buffers(n_window.window)


def on_viewport_updated():
    viewport = n_window.viewport_to_world_cords()
    n_tree.update_viewport(viewport)
    n_net.update_viewport(viewport)
    n_lod.update_viewport(viewport)


def create_level_of_details():
    # n_lod.add_level(LodType.STATIC_TEXTURE, 0.0, file_path="tiles/test2.png")
    # n_lod.add_level(LodType.STATIC_TEXTURE, 0.02, file_path="tiles/test3.png")
    print("Creating level of details")
    #n_lod.add_level(LodType.STATIC_TEXTURE, 0.0, file_path="tiles/test2.png")

    # grid = n_net.grid.get_visible_area(0,0,n_net.grid_columns_count, n_net.grid_rows_count)
    # img_data, img_width, img_height = rgb_textures_factory.get_texture(grid, 25)
    #n_lod.add_level(LodType.STATIC_TEXTURE, 0.0, img_data=img_data, img_width=img_width, img_height=img_height)

    n_lod.add_level(LodType.ALL_LAYERS_TEXTURES, 0.0, texture_factor=10)
    n_lod.add_level(LodType.VISIBLE_LAYERS_TEXTURES, 0.001, texture_factor=5)
    n_lod.add_level(LodType.LEAFS_TEXTURES, 0.01, texture_factor=3)
    n_lod.add_level(LodType.MEGA_LEAF_TEXTURE, 0.02, texture_factor=1)
    n_lod.add_level(LodType.MEGA_LEAF_VERTICES_TO_TEXTURE, 0.03, texture_factor=1)
    n_lod.add_level(LodType.LEAFS_VERTICES_TO_TEXTURE, 0.07, texture_factor=1)
    n_lod.add_level(LodType.LEAFS_VERTICES, 0.1, texture_factor=1)
    n_lod.dump()


def main():
    n_window.create_window()
    n_window.set_render_func(render)
    n_window.set_viewport_updated_func(on_viewport_updated)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    version = glGetString(GL_VERSION)
    print(f"OpenGL version: {version.decode('utf-8')}")
    n_window.n_vertices_shader.compile_vertices_program()
    n_window.n_material_one_shader.compile_textures_material_one_program()
    n_window.n_material_two_shader.compile_textures_material_two_program()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #model_name = "bczhou/TinyLLaVA-3.1B"

    # model_name = "your-llm-model-name"  # Replace with the name or path of your LLM model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    all_layers_and_parameters = list(model.named_parameters())
    tensors = [tensor for name, tensor in all_layers_and_parameters]
    #n_net.init_from_size(tensors)
    n_net.init_from_tensors(tensors)

    # Init net
    #n_net.init_from_size([10000000, 100000000, 14000000, 1004000])
    # n_net.init(tmp_layers[0], tmp_layers[1:])
    # generate nodes grid
    # update tree size and depth using grid size
    n_tree.set_size(n_net.total_width, n_net.total_height)
    # n_tree.load_net_size()
    # calculate min zoom using grid size
    n_window.calculate_min_zoom(n_net)
    # n_lod.load_window_zoom_values(n_window.min_zoom, n_window.max_zoom, n_tree.depth)
    # create level of details
    # n_lod.generate_levels(n_tree.depth)
    create_level_of_details()

    # generate tree
    print("Generating tree")
    n_tree.generate()
    # start render loop
    n_window.reset_to_center(n_net)
    print("Main loop")
    n_window.start_main_loop()
    glfw.terminate()
    gl.glDeleteProgram(n_window.n_vertices_shader.shader_program)
    gl.glDeleteProgram(n_window.n_material_one_shader.shader_program)
    gl.glDeleteProgram(n_window.n_material_two_shader.shader_program)


if __name__ == "__main__":
    main()
