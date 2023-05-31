import math
import time

from app.draw.tree_plane import PlaneTree

layers = []
tree = PlaneTree(0, 0,4)


class SquareLayer:
    def __init__(self, layer_index, layer_sizes):
        self.layer_index = layer_index
        self.size = layer_sizes[self.layer_index]
        self.size_x = math.isqrt(self.size)
        self.size_y = self.size_x
        self.node_gap_x = 100  # node_size * 4 if self.formatted else node_size
        self.node_gap_y = 100  # node_size * 4 if self.formatted else node_size
        self.layer_gap = 200

        self.offset_x = self.node_gap_x / 2
        self.offset_y = self.node_gap_y / 2

        self.width = self.size_x * self.node_gap_x + self.layer_gap + self.offset_x
        self.height = self.size_y * self.node_gap_y +      self.layer_gap

    def width_offset(self):
        return sum([l.width for l in layers[:self.layer_index]])

    def max_height(self):
        return max([l.height for l in layers])

    def get_neuron_position2(self, neuron_index, nodes_layers):
        width_offset = sum([l.width for l in nodes_layers[:self.layer_index]])
        max_height = max([l.height for l in nodes_layers])
        height_offset = (max_height - self.height) / 2
        x = self.offset_x + width_offset + int(neuron_index % self.size_x) * self.node_gap_x
        y = self.offset_y + height_offset + int(neuron_index / self.size_y) * self.node_gap_y
        return (x, y)


class LineLayer:
    def __init__(self, layer_index, layer_sizes):
        self.layer_index = layer_index
        self.size = layer_sizes[self.layer_index]
        self.formatted_size_y = self.size
        self.formatted_size_x = 1
        self.node_gap_x = 100  # node_size * 4 if self.formatted else node_size
        self.node_gap_y = 100  # node_size * 4 if self.formatted else node_size
        self.layer_gap = 200
        self.width = self.layer_gap
        self.height = self.formatted_size_y * self.node_gap_y
        self.offset_x = self.layer_gap / 2
        self.offset_y = self.node_gap_y / 2

    def width_offset(self):
        return sum([l.width for l in layers[:self.layer_index]])

    def max_height(self):
        return max([l.height for l in layers])

    def get_neuron_position2(self, neuron_index, nodes_layers):
        width_offset = sum([l.width for l in nodes_layers[:self.layer_index]])
        max_height = max([l.height for l in nodes_layers])
        height_offset = (max_height - self.height) / 2
        x = self.offset_x + width_offset
        y = self.offset_y + height_offset + self.node_gap_y * neuron_index
        return (x, y)


def get_layer(index, all_layers):
    size = all_layers[index]
    if size > 20:
        return SquareLayer(index, all_layers)
    else:
        return LineLayer(index, all_layers)


def generate_net2(input_size, layers_sizes, world_camera):
    global tree
    print(f"generate net {input_size} {layers_sizes}")
    start_time = time.time()
    layers = []
    all_layers = [input_size] + layers_sizes
    for index, l in enumerate(all_layers):
        layers.append(get_layer(index, all_layers))

    total_w = sum([l.width for l in layers])
    total_h = max([l.height for l in layers])
    print(total_w, total_h)
    tree.set_size(total_w,total_h)
    # tree.width = total_w
    # tree.height = total_h
    # tree.leaf.w = total_w
    # tree.leaf.h = total_h
    print("generating tree")
    tree.generate()

    #tree.add_view()

    print("drwaing tree")
    print(tree.dump())
    for c in tree.leaf.children:
        print(c.dump(""))

    print("leafs count:", tree.count(),"edge leafs count:", tree.edge_leafs_count())
    tree.generate_edge_leafs_grid()
    for layer_index, l in enumerate(layers):
        for neuron_index in range(l.size):
            x, y = l.get_neuron_position2(neuron_index, layers)

    print("iteration time ",time.time()-start_time)
    start_time = time.time()
    print("generate layers")
    for layer_index, l in enumerate(layers):
        # prev_l = layers[index-1] if layer_index >0 else None
        for neuron_index in range(l.size):
            x, y = l.get_neuron_position2(neuron_index, layers)
            # to do
            tree.add_node(
                x=x,
                y=y
            )
    print("nodes generated",time.time()-start_time)
    start_time = time.time()
    #
    #
    print("merge batches")
    tree.merge_batches()
    print("batches merged",time.time()-start_time)
    print("drawing...")
    tree.on_window_update(world_camera)


def draw_net2():
    global tree
    # tree.on_draw()
    tree.on_draw()


def on_net_hovered(x, y, world_camera):
    pass


def on_net_long_clicked(x, y, world_camera):
    pass


def on_net_clicked(x, y, world_camera):
    pass


def update_net(parameters_map):
    pass


def on_window_update2(world_camera):
    tree.on_window_update(world_camera)
    # global tree
    # tree.on_window_update(world_camera)
