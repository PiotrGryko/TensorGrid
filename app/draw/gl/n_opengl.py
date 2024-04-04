import time

import OpenGL.GL as gl
import glfw
from OpenGL.GL import *
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.draw.gl.c_color_theme import NColorTheme
from app.draw.gl.n_lod import NLvlOfDetails, LodType
from app.draw.gl.n_net import NNet
from app.draw.gl.n_shader import NShader
from app.draw.gl.n_tex_factory import NodesGridTexFactory
from app.draw.gl.n_tree import NTree
from app.draw.gl.n_window import NWindow

start_time = 0
frame_count = 0

n_window = NWindow()
n_shader = NShader()
# n_texture_shader = NShader()

n_material_one_shader = NShader()
n_material_two_shader = NShader()

color_theme = NColorTheme()
textures_factory = NodesGridTexFactory(color_theme)

n_net = NNet(n_window, color_theme)
n_lod = NLvlOfDetails(n_net)
n_tree = NTree(0, n_net, textures_factory)

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
        fps_text = f"FPS: {fps:.2f}"
        frame_count = 0
        start_time = time.time()
        # print(fps_text)

    if DEBUG:
        # Use the shader program
        n_shader.use()
        n_shader.update_projection(n_window.get_projection_matrix())
        n_tree.draw_debug_tree()
    else:

        n_lod.load_current_level()
        # Use the shader program
        n_shader.use()
        n_lod.draw_lod_vertices(n_tree)
        n_shader.update_projection(n_window.get_projection_matrix())

        n_lod.draw_lod_textures(n_tree, n_material_one_shader, n_material_two_shader)

        n_material_one_shader.use()
        n_material_one_shader.update_projection(n_window.get_projection_matrix())

        n_material_two_shader.use()
        n_material_two_shader.update_projection(n_window.get_projection_matrix())

    glfw.swap_buffers(n_window.window)


def on_viewport_updated():
    viewport = n_window.viewport_to_world_cords()
    n_tree.update_viewport(viewport)
    n_lod.update_viewport(viewport)

def create_level_of_details():

    texture_factor = int(n_net.total_size / 20000000)
    print("Texture factor: ",texture_factor)

    img_data, img_width, img_height = textures_factory.get_texture(n_net.grid, 4 * texture_factor)

    n_lod.add_level(LodType.STATIC_TEXTURE, img_data, img_width, img_height)

    img_data, img_width, img_height = textures_factory.get_texture(n_net.grid, 3 * texture_factor)
    n_lod.add_level(LodType.STATIC_TEXTURE, img_data, img_width, img_height)

    img_data, img_width, img_height = textures_factory.get_texture(n_net.grid, 2 * texture_factor)
    n_lod.add_level(LodType.STATIC_TEXTURE, img_data, img_width, img_height)

    img_data, img_width, img_height = textures_factory.get_texture(n_net.grid, 1 * texture_factor)
    n_lod.add_level(LodType.STATIC_TEXTURE, img_data, img_width, img_height)

    n_lod.add_level(LodType.MEGA_LEAF_TEXTURE)
    n_lod.add_level(LodType.MEGA_LEAF_VERTICES)
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
    n_shader.compile_vertices_program()
    # n_texture_shader.compile_textures_program()
    n_material_one_shader.compile_textures_material_one_program()
    n_material_two_shader.compile_textures_material_two_program()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # model_name = "your-llm-model-name"  # Replace with the name or path of your LLM model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Get a list of all layers and parameters
    all_layers_and_parameters = list(model.named_parameters())

    tmp_layers = []
    for name, tensor in all_layers_and_parameters:
        print("layer name",name)
        tmp_layers.append(tensor.numel())

    # Init net
    n_net.init(10000000, [10000000, 1400000, 1004000, 3000000])
    #n_net.init(tmp_layers[0], tmp_layers[1:])
    # generate nodes grid
    n_net.generate_net()
    # update tree size and depth using grid size
    n_tree.load_net_size()
    # calculate min zoom using grid size
    n_window.calculate_min_zoom(n_net)
    n_lod.load_window_zoom_values(n_window.min_zoom, n_window.max_zoom, n_tree.depth)
    # create level of details
    #n_lod.generate_levels(n_tree.depth)
    create_level_of_details()



    # generate tree
    print("Generating tree")
    n_tree.generate()
    # start render loop
    print("Main loop")
    n_window.start_main_loop()
    glfw.terminate()
    gl.glDeleteProgram(n_shader.shader_program)


if __name__ == "__main__":
    main()
