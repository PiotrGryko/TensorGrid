from datetime import datetime

import pyglet
from pyglet import image
from pyglet.graphics import Group

from app.draw.helper import generate_line_color

width = 1280
height = 1280
gui_batch = pyglet.graphics.Batch()

train_button = None
generate_button = None
loss_label = None
result_label = None

play_image = image.load('resources/play.png')
pause_image = image.load('resources/pause.png')
forward_image = image.load('resources/forward.png')

gui_group = Group(3)
details_group = Group(2)

node_details = []
current_node = None


class TextLabelMeta:
    def __init__(self, id, index, x, y, width, height, text, color):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.index = index

    def create_text_label(self):
        l = pyglet.text.Label(
            text=self.text,
            width=self.width,
            height=20,
            font_size=12,
            multiline=True,
            x=self.x,
            y=self.y,
            color=self.color,
            group=details_group,
            batch=gui_batch)
        l.background_group = Group(2)
        return l


class ScrollableLayout(pyglet.shapes.Rectangle):
    def __init__(self,
                 x,
                 y,
                 width,
                 height,
                 color,
                 batch=gui_batch,
                 group=details_group):
        super(ScrollableLayout, self).__init__(x, y, width, height, color=color, batch=batch, group=group)
        self.children_meta = {}
        self.children = {}
        self.total_scroll_x = 0
        self.total_scroll_y = 0
        self.total_height = 0
        print(x,y, height)

    def contains(self, item):
        return item.x > self.x and item.x < self.x + self.width and item.y > self.y and item.y < self.y + self.height

    def update_child(self, id):
        meta = self.children_meta[id]
        if self.contains(meta):
            if id not in self.children or self.children[id] is None:
                self.children[id] = meta.create_text_label()
            else:
                self.children[id].y = meta.y
                self.children[id].text = meta.text
                self.children[id].color = meta.color

        else:
            if id in self.children and self.children[id]:
                self.children[id].delete()
                self.children[id] = None

    def add_text_meta(self,
                      id,
                      x,
                      y,
                      width,
                      height,
                      text,
                      color,
                      index=0):
        self.children_meta[id] = TextLabelMeta(id,
                                               index,
                                               self.x + x,
                                               self.y + y + self.height,
                                               width,
                                               height,
                                               text,
                                               color)
        self.total_height += height
        self.update_child(id)

    def on_scroll(self, scroll_x, scroll_y):
        if self.total_height < self.height:
            return

        d_min = 0
        d_max = self.total_height - self.height

        if scroll_y != 0:

            if self.total_scroll_y + scroll_y < d_min:
                scroll_y = -self.total_scroll_y
            elif self.total_scroll_y + scroll_y > d_max:
                scroll_y = d_max - self.total_scroll_y
            if 0 <= self.total_scroll_y + scroll_y < d_max:
                for key, meta in self.children_meta.items():
                    meta.y += scroll_y
                    self.update_child(key)
                self.total_scroll_y += scroll_y


def on_gui_scroll(x, y, scroll_x, scroll_y, gui_camera):
    if len(node_details) == 0:
        return

    scroll_layout = node_details[0]
    if gui_camera.contains_rectangle(x, y, scroll_layout):
        scroll_layout.on_scroll(scroll_x, scroll_y)
        return True
    return False


def on_gui_drag(x, y, drag_x, drag_y, gui_camera):
    if len(node_details) == 0:
        return

    scroll_layout = node_details[0]
    if gui_camera.contains_rectangle(x, y, scroll_layout):
        scroll_layout.on_scroll(drag_x, drag_y)
        return True
    return False


def update_net_gui(parameters):
    if len(node_details) == 0:
        return
    scroll_layout = node_details[0]
    now = datetime.now()

    for p in parameters:
        if f"w{p.label}" in scroll_layout.children_meta:
            value_label_meta = scroll_layout.children_meta[f"w{p.label}"]
            grad_label_meta = scroll_layout.children_meta[f"g{p.label}"]
            w = p.data
            g = p.grad
            value_label_meta.text = f"{value_label_meta.index}: Weight: {'%.7f' % w}"
            value_label_meta.color = generate_line_color(w)
            grad_label_meta.text = f"{value_label_meta.index}: Grad: {'%.7f' % g}"
            grad_label_meta.color = generate_line_color(g)
            scroll_layout.update_child(value_label_meta.id)
            scroll_layout.update_child(grad_label_meta.id)

    updated_time = now.strftime('%H:%M:%S')
    update_label_meta = scroll_layout.children_meta["update"]
    update_label_meta.text = f"Updated: {updated_time}"
    scroll_layout.update_child(update_label_meta.id)


def init_gui():
    global train_button
    global generate_button
    global loss_label
    global result_label
    train_button = pyglet.sprite.Sprite(play_image,

                                        batch=gui_batch,
                                        group=gui_group)
    generate_button = pyglet.sprite.Sprite(forward_image,

                                           batch=gui_batch,
                                           group=gui_group)
    loss_label = pyglet.text.Label(
        f"Loss: 0",
        width=250,
        multiline=True,

        color=(0, 0, 0, 255),
        group=gui_group,
        batch=gui_batch)
    result_label = pyglet.text.Label(
        "Result:None",
        width=1000,
        font_size=12,
        multiline=True,

        color=(0, 0, 0, 255),
        group=gui_group,
        batch=gui_batch)


def show_node_details(node, window_width, window_height):
    global current_node
    if current_node == node:
        dismiss_node_details()
        return
    else:
        dismiss_node_details()
    current_node = node
    total_width = 300
    total_height = window_height
    x = window_width / 2 - total_width
    y = -window_height / 2
    layout = ScrollableLayout(
        x=x,
        y=y,
        width=total_width,
        height=total_height,
        batch=gui_batch,
        color=(255, 255, 255),
        group=details_group
    )
    node_details.append(layout)
    layout.add_text_meta(
        id=f"Key:{node.key}",
        x=10,
        y=-20,
        width=total_width - 20,
        height=20,
        text=f"Key:{node.key}",
        color=(0, 0, 0, 255)
    )

    layout.add_text_meta(
        id=f"Bias:{node.bias}",
        x=10,
        y=- 40,
        width=total_width - 20,
        height=20,
        text=f"Bias:{node.bias}",
        color=(0, 0, 0, 255)
    )
    layout.add_text_meta(
        id="update",
        x=10,
        y=- 60,
        width=total_width - 20,
        height=20,
        text=f"Updated: {node.updated_time}",
        color=(0, 0, 0, 255)
    )
    if len(node.weights) > 0:
        layout.add_text_meta(
            id=f"Weights:",
            x=10,
            y=-80,
            width=total_width - 20,
            height=20,
            text=f"Weights:",
            color=(0, 0, 0, 255)
        )

    for index, key in enumerate(reversed(node.weights.keys())):
        w = node.weights[key]
        g = node.grads[key]
        offset = index * 20
        if len(node.weights) > 0:
            layout.add_text_meta(
                id=f"w{key}",
                x=10,
                y=-20 * (index + 5) - offset,
                width=total_width - 20,
                height=20,
                text=f"{index + 1}: Weight: {'%.7f' % w}",
                color=generate_line_color(w),
                index=index + 1
            )
            layout.add_text_meta(
                id=f"g{key}",
                x=10,
                y=-20 * (index + 5) - offset - 20,
                width=total_width - 20,
                height=20,
                text=f"{index + 1}: Grad: {'%.7f' % g}",
                color=generate_line_color(g),
                index=index + 1
            )


def gui_on_window_resize(width, height):
    global train_button
    global generate_button
    global loss_label
    global result_label

    train_button.x = 20 + -width / 2
    train_button.y = height / 2 - 75

    generate_button.x = 20 + -width / 2
    generate_button.y = height / 2 - 125

    loss_label.x = 100 + - width / 2
    loss_label.y = height / 2 - 55

    result_label.x = 100 + - width / 2
    result_label.y = height / 2 - 105

    details_width = 300
    x = width / 2 - details_width
    y = -height / 2

    for index, n in enumerate(node_details):
        if index == 0:
            n.x = x
            n.y = y
            n.height = height
        else:
            n.x = x + 10
            n.y = y + height - 20 * index


def dismiss_node_details():
    global current_node
    for c in node_details:
        c.delete()
    node_details.clear()
    current_node = None


def on_node_details_clicked(x, y, gui_camera):
    for c in node_details:
        if c and gui_camera.contains_rectangle(x, y, c):
            return True
    return False


def on_train_button_clicked(x, y, gui_camera):
    return gui_camera.contains_rectangle(x, y, train_button)


def on_generate_button_clicked(x, y, gui_camera):
    return gui_camera.contains_rectangle(x, y, generate_button)


def update_loss_label(loss):
    formatted_loss = '%.4f' % loss
    loss_label.text = f'Loss: {formatted_loss}'


def update_result_label(result):
    result_label.text = result


def set_state_active_training():
    train_button.image = pause_image


def set_state_training_idle():
    train_button.image = play_image
