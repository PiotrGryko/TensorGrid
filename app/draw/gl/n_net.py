import math
import time

import numpy as np


class SquareLayer:
    def __init__(self, layer_index, layer_sizes):
        self.layer_index = layer_index
        self.size = layer_sizes[self.layer_index]
        self.size_x = math.isqrt(self.size)
        self.size_y = self.size_x
        self.node_gap_x = 100  # node_size * 4 if self.formatted else node_size
        self.node_gap_y = 100  # node_size * 4 if self.formatted else node_size
        self.layer_gap = 200

        self.offset_x = 0  # self.node_gap_x / 2
        self.offset_y = 0  # self.node_gap_y / 2

        self.width = self.size_x * self.node_gap_x + self.layer_gap + self.offset_x
        self.height = self.size_y * self.node_gap_y + self.layer_gap

        self.width_offset = 0
        self.max_height = 0

    def associate_layers(self, nodes_layers):
        self.width_offset = sum([l.width for l in nodes_layers[:self.layer_index]])
        self.max_height = max([l.height for l in nodes_layers])

    def get_neuron_positions(self,viewport_width, viewport_height):


        indices = np.arange(self.size)
        height_offset = (self.max_height - self.height) / 2
        grid_x = np.mod(indices, self.size_x)
        grid_y = np.floor_divide(indices, self.size_y)
        x = self.offset_x + self.width_offset + grid_x  * self.node_gap_x
        y = self.offset_y + height_offset + grid_y * self.node_gap_y
        sx = x / viewport_width * 2.0
        sy = y / viewport_height * 2.0
        positions = np.column_stack((sx, sy)).astype(np.float32)

       #  grid_size = self.size_x+2
       #  # Create an empty grid using None values
       #  grid = np.empty((grid_size,grid_size, 3)).astype(np.float32)
       #  #indices = np.zeros_like(positions, dtype=int)
       #
       #  # Assign positions to the grid using vectorized operations
       #  mapped_positions = np.empty((len(positions), 3)).astype(np.float32)
       #  mapped_positions[:, :2] = positions
       #  mapped_positions[:, 2] = indices+1
       #
       #  # grid_x = grid_x.reshape(-1, 1)
       #  # grid_y = grid_y.reshape(-1, 1)
       #
       #  # Create an index array for indexing into the grid
       #  #indices = np.concatenate((grid_x, grid_y), axis=1)
       #
       #  # Assign positions to the grid using indices
       #  #grid[indices[:, 0], indices[:, 1]] = positions
       #  # print(grid_y, grid_x)
       #  grid[grid_y,grid_x] = mapped_positions
       #  #np.set_printoptions(threshold=np.inf)
       #  #print(mapped_positions)
       #  #print(indices)
       # # print(grid_x, grid_y)
       #  first_row = grid[0, :, :][:, 0]
       #  first_column = grid[:, 0, :][:, 1]
       #
       #  min_x = 0.4
       #  min_y = 0.2
       #  # print("row",first_row)
       #  # print("column",first_column)
       #  y_index = len(first_row)- np.searchsorted(first_row, min_x, side='right')
       #  x_index = len(first_column)- np.searchsorted(first_column, min_y, side='right')

        #
        # print("index",x_index,y_index)
        # print("result")
        # print(grid[-x_index:, -y_index:])
        #print(grid[:, 0, :])
        # for row in grid:
        #     print("row",row)
        #print(self.layer_index, positions)

        #
        # positions = np.empty((len(neuron_indices), 2), dtype=np.float32)
        # positions[:, 0] = x
        # positions[:, 1] = y
        return positions
        #return np.column_stack((x, y))

    def get_neuron_position(self, neuron_index):
        height_offset = (self.max_height - self.height) / 2
        x = self.offset_x + self.width_offset + int(neuron_index % self.size_x) * self.node_gap_x
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

        self.width_offset = 0
        self.max_height = 0

    def associate_layers(self, nodes_layers):
        self.width_offset = sum([l.width for l in nodes_layers[:self.layer_index]])
        self.max_height = max([l.height for l in nodes_layers])

    def get_neuron_position(self, neuron_index):
        height_offset = (self.max_height - self.height) / 2
        x = self.offset_x + self.width_offset
        y = self.offset_y + height_offset + self.node_gap_y * neuron_index
        return (x, y)

    def get_neuron_positions(self,viewport_width, viewport_height):
        neuron_indices = np.arange(self.size)
        height_offset = (self.max_height - self.height) / 2
        x = self.offset_x + self.width_offset * neuron_indices
        y = self.offset_y + height_offset + self.node_gap_y * neuron_indices
        sx = x / viewport_width * 2.0
        sy = y / viewport_height * 2.0
        print(sx)
        print(sy)
        positions = np.column_stack((sx, sy)).astype(np.float32)
        return positions



class NNet:
    def __init__(self, n_window):
        self.n_window = n_window
        self.total_width = 0
        self.total_height = 0
        self.layers = []

    def get_layer(self, index, all_layers):
        size = all_layers[index]
        if size > 20:
            return SquareLayer(index, all_layers)
        else:
            return LineLayer(index, all_layers)

    def init(self, input_size, layers_sizes):
        print(f"generate net {input_size} {layers_sizes}")
        all_layers = [input_size] + layers_sizes
        for index, l in enumerate(all_layers):
            self.layers.append(self.get_layer(index, all_layers))

        self.total_width = sum([l.width for l in self.layers])
        self.total_height = max([l.height for l in self.layers])

        for l in self.layers:
            l.associate_layers(self.layers)

    def generate_net(self):
        print(f"generate net")
        print(self.total_width, self.total_height)
        start_time = time.time()
        size = sum([l.size for l in self.layers])

        # N elements,  (x,y) each
        shape = (size, 2)
        arrays = []
        for layer_index, l in enumerate(self.layers):
            layer_nodes = l.get_neuron_positions(self.n_window.width, self.n_window.height)
            arrays.append(layer_nodes)
            # for neuron_index in range(l.size):
            #     x, y = l.get_neuron_position(neuron_index)
            #     sx, sy = self.n_window.window_to_normalized_cords(x, y)
            #     nodes[index, 0] = 0
            #     nodes[index, 1] = 0
            #     index += 1
            #     tree.feed_one(sx, sy)
        print("Net generated", time.time() - start_time)
        return np.concatenate(arrays)
