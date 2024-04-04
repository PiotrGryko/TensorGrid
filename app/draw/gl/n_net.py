import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class Layer:
    def __init__(self, size, column_offset, row_offset):
        self.column_offset = column_offset
        self.row_offset = row_offset
        self.size = size
        self.size_x = 0
        self.size_y = 0
        self.is_square = self.size > 20

        self.max_batch_size = 50000000  # Numpy performance drops with large numbers
        self.sublayers = []
        # self.split()

    def define_layer_size(self):
        if self.is_square:
            self.size_x = math.ceil(math.sqrt(self.size))
            self.size_y = self.size_x
        else:
            self.size_y = self.size
            self.size_x = 1

    def define_layer_offset(self, column_offset, row_offset):
        self.column_offset = column_offset
        self.row_offset = row_offset

    def split(self):
        if self.size > self.max_batch_size:
            sub_colum_offset = int(self.size_x / 2)
            sub_row_offset = int(self.size_y / 2)
            sub_size = int(self.size / 4)
            # print("splitted", sub_size)
            self.sublayers.append(Layer(sub_size, self.column_offset, self.row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset, self.row_offset + sub_row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset + sub_colum_offset, self.row_offset))
            self.sublayers.append(
                Layer(sub_size, self.column_offset + sub_colum_offset, self.row_offset + sub_row_offset))
            for s in self.sublayers:
                s.define_layer_size()

    def collect(self):
        result = []
        for sl in self.sublayers:
            result += sl.collect()
        if len(result) == 0:
            result.append(self)
        return result

    def get_neuron_positions(self):
        indices = np.arange(self.size)
        positions = np.random.uniform(0, 1, self.size)

        grid_x = np.mod(indices, self.size_x)
        if self.is_square:
            grid_y = np.floor_divide(indices, self.size_y)
        else:
            grid_y = indices

        columns_indices = grid_x + self.column_offset if self.column_offset > 0 else grid_x
        rows_indices = grid_y + self.row_offset if self.row_offset > 0 else grid_y
        return positions, columns_indices, rows_indices


class NNet:
    def __init__(self, n_window, color_theme):
        self.n_window = n_window
        self.color_theme = color_theme
        self.layers = []
        self.grid_columns_count = 0
        self.grid_rows_count = 0
        self.total_width = 0
        self.total_height = 0
        self.total_size = 0
        self.grid = None
        self.first_row = None
        self.first_column = None
        self.node_gap_x = 0.2  # 100 / self.n_window.width * 2.0
        self.node_gap_y = 0.2  # 100 / self.n_window.width * 2.0
        self.default_value = -2

    def init(self, input_size, layers_sizes):
        print(f"Init net {input_size} {layers_sizes}")
        start_time = time.time()
        all_layers = [input_size] + layers_sizes
        for index, l in enumerate(all_layers):
            grid_layer = Layer(l, 0, 0)
            grid_layer.define_layer_size()
            self.layers.append(grid_layer)

        max_row_count = max(l.size_y for l in self.layers)
        gap_between_layers = 20  # 10 columns
        for index, grid_layer in enumerate(self.layers):
            column_offset = sum([l.size_x for l in self.layers[:index]])
            layer_offset = index * gap_between_layers
            current_row_count = grid_layer.size_y
            row_offset = 0
            if current_row_count < max_row_count:
                row_offset = int((max_row_count - current_row_count) / 2)
            grid_layer.define_layer_offset(column_offset + layer_offset, row_offset)
            grid_layer.split()

        self.grid_columns_count = sum([l.size_x for l in self.layers]) + gap_between_layers * len(self.layers)
        self.grid_rows_count = max([l.size_y for l in self.layers])
        self.total_width = self.grid_columns_count * self.node_gap_x
        self.total_height = self.grid_rows_count * self.node_gap_y
        self.grid = np.full((self.grid_rows_count, self.grid_columns_count), self.default_value).astype(np.float32)
        self.total_size = sum(all_layers)
        print("Net initialized", time.time() - start_time, "total size: ", self.total_size, "node gaps: ",
              self.node_gap_x, self.node_gap_y)

    def process_batch(self, sublayer, index, grid):
        # Perform some computation on the chunk
        positions, columns_indices, rows_indices = sublayer.get_neuron_positions()
        grid[rows_indices, columns_indices] = positions

    def generate_net(self):
        print(f"generate net")
        print(f"Grid size, columns: {self.grid_columns_count} rows: {self.grid_rows_count}")
        print("Size in Pixels", self.total_width, self.total_height)
        start_time = time.time()

        executor = ThreadPoolExecutor()
        print("Splitting data into batches for parallel load")
        sublayers = []
        for layer in self.layers:
            sublayers.extend(layer.collect())

        print("Batches count:", len(sublayers))
        futures = []
        for index, batch in enumerate(sublayers):
            futures.append(executor.submit(self.process_batch, batch, index, self.grid))

        # Load the net and print progress bar
        for index, future in enumerate(as_completed(futures)):
            print(f"\rLoading net: {int(100 * index / len(futures))}%", end="")
        print(f"\rLoading net: 100%", end="\n")

        print("Net generated", time.time() - start_time)

    def get_subgrid(self, x1, y1, x2, y2):
        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)
        subgrid = self.grid[row_min:row_max, col_min:col_max]

        return subgrid

    def get_positions_grid(self, x1, y1, x2, y2):
        start_time = time.time()
        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)
        subgrid = self.grid[row_min:row_max, col_min:col_max]

        indices = np.where(subgrid != self.default_value)
        rows, columns = indices
        columns = columns + col_min
        rows = rows + row_min
        rows_pos = rows * node_gap_y
        columns_pos = columns * node_gap_x

        positions = np.column_stack((columns_pos, rows_pos)).astype(np.float32)
        color_values = self.color_theme.cmap(subgrid[indices]).astype(np.float32)[:, :3]

        return positions, color_values
