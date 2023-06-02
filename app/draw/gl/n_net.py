import math
import time

import numpy as np


class Layer:
    def __init__(self, size, column_offset, row_offset):
        self.column_offset = column_offset
        self.row_offset = row_offset
        self.size = size

        self.is_square = self.size > 20
        if self.is_square:
            self.size_x = math.ceil(math.sqrt(self.size))
            self.size_y = self.size_x
        else:
            self.size_y = self.size
            self.size_x = 1

        self.max_batch_size = 50000000  # Numpy performance drops with large numbers

        self.sublayers = []
        self.split()

    def split(self):
        if self.size > self.max_batch_size:
            sub_colum_offset = int(self.size_x / 2)
            sub_row_offset = int(self.size_y / 2)
            sub_size = int(self.size / 4)
            print("splitted", sub_size)
            self.sublayers.append(Layer(sub_size, self.column_offset, self.row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset, self.row_offset+sub_row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset+sub_colum_offset, self.row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset+sub_colum_offset, self.row_offset+sub_row_offset))


    def collect(self):
        result = []
        for sl in self.sublayers:
            result += sl.collect()
        if len(result)==0:
            result.append(self)
        return result

    def get_neuron_positions(self):
        indices = np.arange(self.size)
        positions = np.full(self.size, 1)

        grid_x = np.mod(indices, self.size_x)
        if self.is_square:
            grid_y = np.floor_divide(indices, self.size_y)
        else:
            grid_y = indices

        columns_indices = grid_x + self.column_offset if self.column_offset > 0 else grid_x
        rows_indices = grid_y + self.row_offset if self.row_offset > 0 else grid_y
        return positions, columns_indices, rows_indices


class NNet:
    def __init__(self, n_window):
        self.n_window = n_window
        self.total_width = 0
        self.total_height = 0
        self.layers = []
        self.grid_columns_count = 0
        self.grid_rows_count = 0
        self.grid = None
        self.first_row = None
        self.first_column = None
        self.node_gap_x = 100 / self.n_window.width * 2.0
        self.node_gap_y = 100 / self.n_window.width * 2.0

    def init(self, input_size, layers_sizes):
        print(f"generate net {input_size} {layers_sizes}")
        all_layers = [input_size] + layers_sizes
        for index, l in enumerate(all_layers):
            column_offset = sum([l.size_x for l in self.layers]) + len(self.layers)
            row_offset = 0
            self.layers.append(Layer(l, column_offset, row_offset))

        self.grid_columns_count = sum([l.size_x for l in self.layers]) + len(self.layers)
        self.grid_rows_count = max([l.size_y for l in self.layers])
        self.total_width = self.grid_columns_count * self.node_gap_x
        self.total_height = self.grid_rows_count * self.node_gap_y
        self.grid = np.full((self.grid_rows_count, self.grid_columns_count), -1).astype(np.float32)
        print("grid", self.grid.shape)

    def generate_net(self):
        print(f"generate net")
        print(self.total_width, self.total_height)
        start_time = time.time()

        for layer_index, l in enumerate(self.layers):
            sublayers = l.collect()
            for sl in sublayers:
                positions, columns_indices, rows_indices = sl.get_neuron_positions()
                self.grid[rows_indices, columns_indices] = positions

        print("Net generated", time.time() - start_time)

    def get_positions_grid(self, x1, y1, x2, y2):
        start_time = time.time()
        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)
        subgrid = self.grid[row_min:row_max, col_min:col_max]
        indices = np.where(subgrid != -1)
        rows, columns = indices
        columns = columns + col_min
        rows = rows + row_min
        rows_pos = rows * node_gap_y
        columns_pos = columns * node_gap_x
        positions = np.column_stack((columns_pos, rows_pos)).astype(np.float32)
        # print("Positions generated", time.time() - start_time)
        return positions
