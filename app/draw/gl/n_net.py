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
    def __init__(self, n_window):
        self.n_window = n_window
        self.layers = []
        self.grid_columns_count = 0
        self.grid_rows_count = 0
        self.total_width = 0
        self.total_height = 0
        self.grid = None
        self.first_row = None
        self.first_column = None
        self.node_gap_x = 0.2 #100 / self.n_window.width * 2.0
        self.node_gap_y = 0.2 #100 / self.n_window.width * 2.0
        self.default_value = -2

        self.cmap_options = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
                             'BuPu_r',
                             'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys',
                             'Greys_r',
                             'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
                             'Pastel1',
                             'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                             'PuBu_r', 'PuOr',
                             'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
                             'RdPu',
                             'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r',
                             'Set2',
                             'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'summer_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                             'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                             'autumn',
                             'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
                             'cividis',
                             'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
                             'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
                             'gist_heat',
                             'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
                             'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
                             'gnuplot_r', 'gray',
                             'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
                             'magma_r',
                             'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma',
                             'plasma_r',
                             'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
                             'summer',
                             'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                             'tab20c_r',
                             'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
                             'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

        self.Interpolation_methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
                                      'hamming',
                                      'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
                                      'sinc', 'lanczos']

        self.interpolation = random.choice(self.Interpolation_methods)
        print("Interpolation option:", self.interpolation)

        random_cmap = random.choice(self.cmap_options)
        print("Color option:", random_cmap)
        self.cmap = plt.cm.get_cmap(random_cmap)
        self.color_low = self.cmap(-1)

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
        total_size = sum(all_layers)
        print("Net initialized", time.time() - start_time, "total size: ", total_size, "node gaps: ", self.node_gap_x, self.node_gap_y)
        print("grid", self.grid.shape)

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

    def plot_grid(self):
        print(f"Plot net")
        start_time = time.time()
        fig, ax = plt.subplots()
        # Draw a grid
        ax.grid(False)

        # Remove x and y axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Customize the grid
        # ax.grid(color='gray')

        new_grid = np.empty_like(self.grid)
        new_grid.fill(self.default_value)
        print(len(new_grid))
        factor = int(len(new_grid) / 10)
        new_grid = self.grid[::factor, ::factor]

        new_grid = np.pad(new_grid, pad_width=1, mode='constant', constant_values=self.default_value)

        ax.imshow(new_grid, cmap='jet', interpolation='nearest', alpha=1, origin='lower')

        # # Save the plot
        image_path = 'gl/tiles/grid_image.png'
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

        # Close the plot
        plt.close()
        print("Plot generated", time.time() - start_time)

    def get_texture2(self, x1, y1, x2, y2, factor=1):
        # print(f"NNet get texture ", factor)
        start_time = time.time()

        new_grid = self.get_subgrid(x1, y1, x2, y2)
        grid_factor = factor
        new_grid = new_grid[::grid_factor, ::grid_factor]
        # new_grid = self.scale_down_and_average(new_grid)

        image_height, image_width =  self.scale_down_dimensions(new_grid.shape)
        # Normalize the grid data to [0, 255] range
        cmap_rbga = self.cmap(new_grid)
        normalized_data = (cmap_rbga * 255).astype(np.uint8)
        image = Image.fromarray(normalized_data)

        image_rgba = image.convert('RGBA').resize((image_width, image_height))
        image_rgba_flipped = image_rgba.transpose(Image.FLIP_TOP_BOTTOM)

        # Get the raw pixel data as a numpy array
        image_data = np.array(image_rgba_flipped)

        print("Texture generated", time.time() - start_time, "width", image_width, "height", image_height, "factor",
              grid_factor)
        return image_data, image_width, image_height

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
        color_values = self.cmap(subgrid[indices]).astype(np.float32)[:, :3]

        return positions, color_values

    def scale_down_dimensions(self, shape, max_size=(16096, 16096)):
        original_width, original_height = shape

        # # Scale down the dimensions while preserving the aspect ratio
        new_width = original_width
        new_height = original_height

        # Check if the new dimensions exceed the maximum size
        if new_width > max_size[0]:
            new_width = max_size[0]
            new_height = int(original_height * (max_size[0] / original_width))
        if new_height > max_size[1]:
            new_height = max_size[1]
            new_width = int(original_width * (max_size[1] / original_height))

        return new_width, new_height

    def scale_down_and_average(self, grid, max_size=(8096, 8096)):
        # Get the dimensions of the original grid
        height, width = grid.shape

        # Calculate the aspect ratio of the original grid
        aspect_ratio = width / height

        # # Define the original 10x10 grid
        original_grid = grid
        # Define the pooling factor (2x smaller in each dimension)
        # Get the dimensions of the grid
        original_rows, original_cols = grid.shape

        # Calculate the pooling factor separately for rows and columns
        pooling_factor_rows = original_rows / max_size[0]
        pooling_factor_cols = original_cols / max_size[1]

        # Choose the smaller pooling factor to ensure the resulting grid fits within max_size
        pooling_factor = max(pooling_factor_rows, pooling_factor_cols)
        pooling_factor = int(pooling_factor)
        if pooling_factor > 0:
            # Calculate the number of rows and columns needed to pad the original grid
            rows_to_pad = int(pooling_factor - (original_grid.shape[0] % pooling_factor))
            cols_to_pad = int(pooling_factor - (original_grid.shape[1] % pooling_factor))

            # Pad the original grid to make it evenly divisible by the pooling factor
            padded_grid = np.pad(original_grid, ((0, rows_to_pad), (0, cols_to_pad)), mode='constant')

            # Reshape the padded grid into non-overlapping blocks of the desired size
            reshaped_grid = padded_grid.reshape(padded_grid.shape[0] // pooling_factor, pooling_factor,
                                                padded_grid.shape[1] // pooling_factor, pooling_factor)

            # Perform average pooling or max pooling to combine values in each block
            # For example, to use average pooling:
            pooled_grid = np.mean(reshaped_grid, axis=(1, 3))
            # print(original_grid, original_grid.shape)
            # print(pooled_grid, pooled_grid.shape)
            original_grid = pooled_grid
        return original_grid
        #
        #
        #
        # # Check if any of the dimensions exceed the maximum size
        # if width > max_size[0] or height > max_size[1]:
        #     # If either dimension exceeds the maximum size, adjust the dimensions
        #     if aspect_ratio > 1:
        #         # Width is larger, so resize width to max_size[0] and adjust height accordingly
        #         new_width = max_size[0]
        #         new_height = int(new_width / aspect_ratio)
        #     else:
        #         # Height is larger, so resize height to max_size[1] and adjust width accordingly
        #         new_height = max_size[1]
        #         new_width = int(new_height * aspect_ratio)
        #     print(grid.shape)
        #     print(new_height, new_width)
        #     scaled_grid = np.mean(np.reshape(grid, (new_height, height // new_height, new_width, width // new_width)),
        #                           axis=(1, 3))
        # else:
        #     # If neither dimension exceeds the maximum size, keep the original dimensions
        #     new_height, new_width = height, width
        #     scaled_grid = grid
        #
        #
        # # Reshape the grid to have the new dimensions
        #
        #
        # return scaled_grid
