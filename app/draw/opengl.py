import time

from pyglet.gl import *
from pyglet.gui import *
from pyglet.math import Mat4

from app.draw.camera import CenteredCamera
from app.draw.net import update_net, on_window_update, on_net_clicked, on_net_hovered, on_net_long_clicked
from app.draw.process import PygletProcess, SharedContainer
from app.draw.gui import gui_batch, on_train_button_clicked, set_state_active_training, \
    on_generate_button_clicked, update_result_label, set_state_training_idle, update_loss_label, show_node_details, \
    on_node_details_clicked, \
    init_gui, gui_on_window_resize, on_gui_scroll, on_gui_drag, update_net_gui
from app.draw.tree_net import generate_net2, draw_net2, on_window_update2
from app.examples.hand_written_digits.digits_recognition import create_mlp

print("init")

training_process = None
generate_process = None
shared_container = None

window = pyglet.window.Window(1280, 1280, resizable=True)
fps_display = pyglet.window.FPSDisplay(window=window)

world_camera = CenteredCamera(scroll_speed=5, min_zoom=0.02, max_zoom=4, window=window)
gui_camera = CenteredCamera(scroll_speed=5, min_zoom=0.02, max_zoom=4, window=window)

mouse_pressed_time = time.time()
mouse_released_time = time.time()


@window.event
def on_draw():
    window.clear()
    fps_display.draw()
    world_camera.begin()
    draw_net2()
    #draw_net()
    # net_batch.draw()
    world_camera.end()
    gui_camera.begin()
    gui_batch.draw()
    gui_camera.end()


@window.event
def on_resize(width, height):
    glViewport(0, 0, *window.get_framebuffer_size())
    window.projection = Mat4.orthogonal_projection(0, width, 0, height, -255, 255)
    gui_on_window_resize(window.width, window.height)


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    if on_gui_scroll(x, y, scroll_x, scroll_y, gui_camera):
        pass
    else:
        world_camera.zoom += scroll_y * -0.01
        print(world_camera.zoom)
        #print(world_camera.get_window_viewport())
        on_window_update(world_camera)
        #on_window_update2(world_camera)


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if on_gui_drag(x, y, dx, dy, gui_camera):
        pass
    else:
        world_camera.offset_x += -dx / world_camera.zoom
        world_camera.offset_y += -dy / world_camera.zoom
        #print(world_camera.get_window_viewport())
        #on_window_update(world_camera)
        on_window_update2(world_camera)


@window.event
def on_mouse_motion(x, y, dx, dy):
    on_net_hovered(x, y, world_camera)


@window.event
def on_mouse_press(x, y, button, modifiers):
    global mouse_pressed_time
    mouse_pressed_time = time.time()


@window.event
def on_mouse_release(x, y, button, modifiers):
    global mouse_released_time
    mouse_released_time = time.time()
    time_diff_ms = (mouse_released_time - mouse_pressed_time) * 1000
    if time_diff_ms < 200:
        on_mouse_click(x, y, button, modifiers)
    elif time_diff_ms > 500:
        on_mouse_long_click(x, y, button, modifiers)


def on_mouse_long_click(x, y, button, modifiers):
    print("on long click")
    on_net_long_clicked(x, y, world_camera)


def on_mouse_click(x, y, button, modifiers):
    print('on click')
    if on_train_button_clicked(x, y, gui_camera):
        set_state_active_training()
        start_training()
    elif on_generate_button_clicked(x, y, gui_camera):
        start_generate()
    elif on_node_details_clicked(x, y, gui_camera):
        return
    elif node := on_net_clicked(x, y, world_camera):
        show_node_details(node, window.width, window.height)


def start_generate():
    print("GENERATE BUTTON PRESSED")
    if generate_process.is_active():
        generate_process.kill()
    generate_process.launch(target=run_generate,
                            args=(shared_container.shared_memory_name,),
                            on_job_finished=on_result_generated)


def run_generate(memory_name):
    shared_dict = SharedContainer(memory_name)
    mlp = shared_dict.get_mlp()
    result = mlp.generate_custom()
    shared_dict.set_result(result)


def on_result_generated():
    update_result_label(shared_container.get_result())


def start_training():
    print(f"TRAIN BUTTON PRESSED {training_process.is_active()}")
    if training_process.is_active():
        training_process.kill()
        print("training stopped")
    else:
        training_process.launch(target=run_training,
                                args=(shared_container.shared_memory_name,),
                                update_ui_func=training_update_ui,
                                on_job_finished=on_training_finished)


def run_training(memory_name):
    shared_dict = SharedContainer(memory_name)
    mlp = shared_dict.get_mlp()

    def process_step(loss, mlp):
        params = mlp.parameters()
        params_map = {p.label: p for p in params}
        shared_dict.set_loss(loss)
        shared_dict.set_mlp(mlp)
        shared_dict.set_parameters(set(params))
        shared_dict.set_parameters_map(params_map)
        print("step updated")

    mlp.train_custom(100, process_step)


def training_update_ui():
    start_time = time.time()

    if shared_container.should_update():
        parameters = shared_container.get_parameters()
        parameters_map = shared_container.get_parameters_map()
        shared_container.mark_as_updated()
        print("Dict time:", time.time() - start_time)
        update_loss_label(shared_container.get_loss())
        update_net(parameters_map)
        update_net_gui(parameters)
        on_window_update(world_camera)
    end_time = time.time()
    print("Time taken:", end_time - start_time)


def on_training_finished():
    set_state_training_idle()


if __name__ == "__main__":
    init_gui()
    print("creating mlp")
    mlp = create_mlp()

    shared_container = SharedContainer()
    shared_container.set_mlp(mlp)
    training_process = PygletProcess()
    generate_process = PygletProcess()
    generate_net2(10000,[500, 50,1],world_camera)
    #generate_net(mlp, world_camera)
    glClearColor(253 / 255, 226 / 255, 243 / 255, 1)
    # glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    pyglet.app.run()
