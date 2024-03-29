import math

from app.draw.pyglet.helper import *
from app.draw.pyglet.meta import MetaContainer
from app.micrograd.nn import Neuron

layers = []
meta_container = MetaContainer()


class NodesLayer:
    def __init__(self, layer_index, layer_sizes):
        self.size_threshold = 20
        self.layer_index = layer_index
        self.size = layer_sizes[self.layer_index]
        self.formatted = self.size > self.size_threshold
        self.formatted_size_y = self.size if not self.formatted else int(math.sqrt(self.size))
        self.formatted_size_x = 1 if not self.formatted else self.formatted_size_y
        self.node_gap_x = 100#node_size * 4 if self.formatted else node_size
        self.node_gap_y = 100#node_size * 4 if self.formatted else node_size
        print(layer_index,self.node_gap_x, self.node_gap_y)
        self.width = self.formatted_size_x * self.node_gap_x
        self.height = self.formatted_size_y * self.node_gap_y

    def width_offset(self):
        return sum([l.width for l in layers[:self.layer_index]])

    def max_height(self):
        return max([l.height for l in layers])

    def get_neuron_position(self, neuron_index):
        width_offset = self.width_offset()
        height_offset = (self.max_height() - self.height) / 2
        if self.formatted:
            size = self.formatted_size_y
            x = width_offset + int(neuron_index % size) * self.node_gap_x
            y = height_offset + int(neuron_index / size) * self.node_gap_y
            return (x, y)
        else:
            x = width_offset + self.layer_index * self.node_gap_x
            y = height_offset + self.node_gap_y * neuron_index
            return (x, y)


def generate_net(mlp, world_camera):
    print(f"generate net {len(mlp.parameters())}")
    params_map = {p.label: p for p in mlp.parameters()}
    meta_container.update_parameters(params_map)
    input_size = mlp.inputs_per_neuron
    layer_sizes = [input_size] + [len(l.neurons) for l in mlp.layers]
    for index, l in enumerate(layer_sizes):
        layers.append(NodesLayer(index, layer_sizes))

    print(f"generate input neurons {len(mlp.layers)}")
    input_neurons = [Neuron(0, i, -1) for i in range(0, input_size)]
    print("generate  neurons")
    lm = {l.layer_index + 1: l.neurons for l in mlp.layers}
    lm[0] = input_neurons

    print("generating net...")
    for layer_index, layer_neurons in lm.items():
        prev_layer_neurons = [] if layer_index == 0 else lm[layer_index - 1]
        for n in layer_neurons:
            layer = layers[n.layer_index + 1]
            x, y = layer.get_neuron_position(n.neuron_index)
            meta_container.add_node_meta(x, y, n, node_size)
            for p, weight in zip(prev_layer_neurons, n.weights):
                layer = layers[p.layer_index + 1]
                px, py = layer.get_neuron_position(p.neuron_index)
                meta_container.add_line_meta(x, y, px, py, weight, p, n)

    print("drawing...")
    meta_container.on_window_update(world_camera)


def draw_net():
    #tree.on_draw()
    meta_container.on_draw()


def on_net_hovered(x, y, world_camera):
    meta_container.on_hover(x, y, world_camera)


def on_net_long_clicked(x, y, world_camera):
    return meta_container.on_long_click(x, y, world_camera)


def on_net_clicked(x, y, world_camera):
    return meta_container.on_click(x, y, world_camera)


def update_net(parameters_map):
    meta_container.update_parameters(parameters_map)


def on_window_update(world_camera):
    meta_container.on_window_update(world_camera)

