import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def unpack_shape(array):
    shape = array.shape
    if len(shape) == 1:
        return (shape[0], 1)  # or (shape[0], 1) if you prefer to treat it as a single column with many rows
    return shape


class Grid:
    def __init__(self):
        # Store subgrids with their offsets as keys (row_offset, column_offset)
        self.layers = []
        self.default_value = -2

    def add_layers(self, layers):
        # Add or update a subgrid at the specified offsets
        self.layers = layers

    def rectangles_intersect(self, x1, y1, x2, y2, grid_x1, grid_y1, grid_x2, grid_y2):
        # Check if one rectangle is on left side of other
        if x1 > grid_x2 or grid_x1 > x2:
            return False
        # Check if one rectangle is above the other
        if y1 > grid_y2 or grid_y1 > y2:
            return False
        return True

    def get_visible_layers(self, x1, y1, x2, y2):
        visible_layers = []
        for sublayer in self.layers:
            grid_x1 = sublayer.column_offset
            grid_y1 = sublayer.row_offset
            grid_x2 = grid_x1 + sublayer.columns_count
            grid_y2 = grid_y1 + sublayer.rows_count
            if self.rectangles_intersect(x1, y1, x2, y2, grid_x1, grid_y1, grid_x2, grid_y2):
                visible_layers.append(sublayer)
        return visible_layers

    def get_visible_area(self, x1, y1, x2, y2):

        # Initialize the result array to the dimensions of the view area
        result_grid = np.full((y2 - y1, x2 - x1), self.default_value, dtype=np.float32)
        # Iterate over each subgrid to check for intersections
        for sublayer in self.layers:
            grid_x1 = sublayer.column_offset
            grid_y1 = sublayer.row_offset
            grid_x2 = grid_x1 + sublayer.columns_count
            grid_y2 = grid_y1 + sublayer.rows_count

            if self.rectangles_intersect(x1, y1, x2, y2, grid_x1, grid_y1, grid_x2, grid_y2):
                # Calculate the overlap area
                overlap_x1 = max(x1, grid_x1)
                overlap_y1 = max(y1, grid_y1)
                overlap_x2 = min(x2, grid_x2)
                overlap_y2 = min(y2, grid_y2)
                # Copy the data from subgrid to the result grid
                # Adjust indices for result grid indexing
                result_slice = result_grid[overlap_y1 - y1:overlap_y2 - y1, overlap_x1 - x1:overlap_x2 - x1]
                # print(result_slice.shape)
                # print(sublayer.layer_grid.shape)
                if sublayer.layer_grid.ndim == 1:
                    subgrid_slice = sublayer.layer_grid[
                                    overlap_y1 - grid_y1:overlap_y2 - grid_y1]
                    result_slice[:, overlap_x1 - grid_x1] = subgrid_slice
                else:
                    subgrid_slice = sublayer.layer_grid[
                                    overlap_y1 - grid_y1:overlap_y2 - grid_y1,
                                    overlap_x1 - grid_x1:overlap_x2 - grid_x1]
                    result_slice[:] = subgrid_slice

        return result_grid

    def get_visible_data_chunks(self, x1, y1, x2, y2, width_factor, height_factor):

        rows_list = []
        columns_list = []
        values_list = []

        # Iterate over each subgrid to check for intersections
        for sublayer in self.layers:
            grid_x1 = sublayer.column_offset
            grid_y1 = sublayer.row_offset
            grid_x2 = grid_x1 + sublayer.columns_count
            grid_y2 = grid_y1 + sublayer.rows_count

            if self.rectangles_intersect(x1, y1, x2, y2, grid_x1, grid_y1, grid_x2, grid_y2):
                # Calculate the overlap area
                overlap_x1 = max(x1, grid_x1)
                overlap_y1 = max(y1, grid_y1)
                overlap_x2 = min(x2, grid_x2)
                overlap_y2 = min(y2, grid_y2)

                if sublayer.layer_grid.ndim == 1:
                    overlap_data = sublayer.layer_grid[
                                   overlap_y1 - grid_y1:overlap_y2 - grid_y1]
                    chunk = overlap_data[::height_factor]
                    chunk_indices = np.where(chunk != self.default_value)
                    chunk_rows = chunk_indices[0]
                    chunk_columns = np.full(chunk_rows.size, 1)
                else:
                    overlap_data = sublayer.layer_grid[
                                   overlap_y1 - grid_y1:overlap_y2 - grid_y1,
                                   overlap_x1 - grid_x1:overlap_x2 - grid_x1]
                    chunk = overlap_data[::height_factor, ::width_factor]
                    chunk_indices = np.where(chunk != self.default_value)
                    chunk_rows, chunk_columns = chunk_indices

                chunk_col_min, chunk_row_min = (overlap_x1, overlap_y1)
                chunk_values = chunk[chunk_indices]
                chunk_columns = (chunk_columns * width_factor) + chunk_col_min
                chunk_rows = (chunk_rows * height_factor) + chunk_row_min
                rows_list.append(chunk_rows)
                columns_list.append(chunk_columns)
                values_list.append(chunk_values)

        if len(values_list) == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        rows = np.concatenate(rows_list)
        columns = np.concatenate(columns_list)
        values = np.concatenate(values_list)

        return rows, columns, values


class Layer:
    def __init__(self, layer_grid):
        # layer_grid =  np.random.uniform(0, 1, (150, 3900)).astype(np.float32)

        self.column_offset = 0
        self.row_offset = 0
        self.layer_grid = layer_grid
        self.rows_count, self.columns_count = unpack_shape(self.layer_grid)
        self.size = self.layer_grid.size

        # print(self.layer_grid.shape, self.unpack_shape(self.layer_grid))
        self.layer_grid = np.clip(self.layer_grid * 50, 0, 1)
        # self.layer_grid[:] = 1
        self.id = None

    def define_layer_offset(self, column_offset, row_offset):
        self.column_offset = column_offset
        self.row_offset = row_offset
        self.id = f"{self.column_offset}-{self.row_offset}-{self.columns_count}-{self.rows_count}-{self.size}"


class NNet:
    def __init__(self, n_window, color_theme, calculator):
        self.n_window = n_window
        self.color_theme = color_theme
        self.calculator = calculator
        self.layers = []
        self.grid_columns_count = 0
        self.grid_rows_count = 0
        self.total_width = 0
        self.total_height = 0
        self.total_size = 0
        # self.grid = None
        self.node_gap_x = 0.2  # 100 / self.n_window.width * 2.0
        self.node_gap_y = 0.2  # 100 / self.n_window.width * 2.0
        self.default_value = -2
        self.grid = Grid()

        self.visible_layers = []

    def init_from_size(self, all_layers_sizes):
        print("Init net from sizes")
        layers = []
        print("Generating layers data")
        for size in all_layers_sizes:
            calculated_size = self.calculator.measure(size)
            rows_count, columns_count = calculated_size[0], calculated_size[1]
            layer_grid = np.random.uniform(0, 1, (rows_count, columns_count)).astype(np.float32)
            layers.append(layer_grid)
        print("Creating layers")
        self.create_layers(layers)
        self.init_grid()

    def init_from_tensors(self, tensors):
        print("Init net from tensors")
        size = len(tensors)
        layers = []
        print("")
        for index, tensor in enumerate(tensors):
            print(f"\rDetaching tensors: {int(100 * index / size)}%", end="")
            layers.append(tensor.detach().numpy())
        print(f"\rDetaching tensors: 100%", end="")
        print("")
        self.create_layers(layers)
        self.init_grid()

    def create_layers(self, all_layers):
        print("Creating layers", len(all_layers))
        for index, layer_data in enumerate(all_layers):
            # size = layer_data.size
            # calculated_size = self.calculator.measure(size)
            # print("Create layer", layer_data.shape)
            grid_layer = Layer(layer_data)
            self.layers.append(grid_layer)

    def init_grid(self):
        start_time = time.time()
        print(f"Init net, layers count:", len(self.layers))
        max_row_count = max(l.rows_count for l in self.layers)
        gap_between_layers = 200
        print("Loading offsets")
        for index, grid_layer in enumerate(self.layers):
            column_offset = sum([l.columns_count for l in self.layers[:index]])
            layer_offset = index * gap_between_layers
            current_row_count = grid_layer.rows_count
            row_offset = 0
            if current_row_count < max_row_count:
                row_offset = int((max_row_count - current_row_count) / 2)
            grid_layer.define_layer_offset(column_offset + layer_offset, row_offset)
            print(f"\rLoading offsets: {int(100 * index / len(self.layers))}%", end="")
        print(f"\rLoading offsets: 100%", end="")
        print("")
        self.grid_columns_count = sum([l.columns_count for l in self.layers]) + gap_between_layers * len(self.layers)
        self.grid_rows_count = max([l.rows_count for l in self.layers])
        self.total_width = self.grid_columns_count * self.node_gap_x
        self.total_height = self.grid_rows_count * self.node_gap_y
        self.total_size = sum([l.size for l in self.layers])
        print(f"Grid dimmensions: {self.grid_rows_count}x{self.grid_columns_count}")
        self.grid.add_layers(self.layers)
        # self.grid = np.full((self.grid_rows_count, self.grid_columns_count), self.default_value).astype(np.float16)
        print("Net initialized", time.time() - start_time, "s",
              # "grid",self.grid,
              "total size: ", self.total_size,
              "node gaps: ", self.node_gap_x, self.node_gap_y)

    def update_viewport(self, viewport):
        x, y, w, h, zoom = viewport
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h

        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)
        visible = self.grid.get_visible_layers(col_min, row_min, col_max, row_max)

        if visible != self.visible_layers:
            self.visible_layers = visible

    def get_subgrid(self, x1, y1, x2, y2):
        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)
        # subgrid = self.grid[row_min:row_max, col_min:col_max]

        subgrid = self.grid.get_visible_area(col_min, row_min, col_max, row_max)

        return subgrid

    def get_positions_grid(self, x1, y1, x2, y2, factor=1):
        if factor < 1:
            raise "Factor <1 !"
        start_time = time.time()
        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)

        rows, columns, values = self.grid.get_visible_data_chunks(col_min, row_min, col_max, row_max, factor, factor)

        rows_pos = rows * node_gap_y
        columns_pos = columns * node_gap_x
        positions = np.column_stack((columns_pos, rows_pos)).astype(np.float32)
        color_values = self.color_theme.cmap(values).astype(np.float32)[:, :3]

        return positions, color_values

    def get_positions_and_colors_array(self, x1, y1, x2, y2, target_width, target_height):

        start_time = time.time()
        node_gap_x = self.node_gap_x
        node_gap_y = self.node_gap_y

        col_min = int(x1 / node_gap_x)
        col_max = math.ceil(x2 / node_gap_x)

        row_min = int(y1 / node_gap_y)
        row_max = math.ceil(y2 / node_gap_y)

        # subgrid_height, subgrid_width = unpack_shape(subgrid)
        subgrid_width = col_max - col_min
        subgrid_height = row_max - row_min
        # Determine the sampling factor
        width_factor = max(int(subgrid_width / target_width), 1)
        height_factor = max(int(subgrid_height / target_height), 1)

        rows, columns, values = self.grid.get_visible_data_chunks(col_min,
                                                                  row_min,
                                                                  col_max,
                                                                  row_max,
                                                                  width_factor,
                                                                  height_factor)

        rows_pos = rows * node_gap_y
        columns_pos = columns * node_gap_x

        # Stack the indices with the subgrid values
        # This results in a 3D array where the last dimension has three elements: row, col, value
        combined_array = np.stack((columns_pos, rows_pos, values), axis=-1)

        # Reshape to a 2D array where each row is [row_index, col_index, value]
        result_array = combined_array.reshape(-1, 3).astype(np.float32)
        # print(result_array)
        print("colors generated", time.time() - start_time)
        return result_array
