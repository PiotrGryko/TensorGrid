import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib._tight_bbox import adjust_bbox


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
            # print("splitted", sub_size)
            self.sublayers.append(Layer(sub_size, self.column_offset, self.row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset, self.row_offset + sub_row_offset))
            self.sublayers.append(Layer(sub_size, self.column_offset + sub_colum_offset, self.row_offset))
            self.sublayers.append(
                Layer(sub_size, self.column_offset + sub_colum_offset, self.row_offset + sub_row_offset))

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

        self.cmap = plt.cm.get_cmap('jet')
        self.color_low = self.cmap(-1)

    def init(self, input_size, layers_sizes):
        print(f"Init net net {input_size} {layers_sizes}")
        start_time = time.time()
        all_layers = [input_size] + layers_sizes
        for index, l in enumerate(all_layers):
            column_offset = sum([l.size_x for l in self.layers]) + len(self.layers)
            row_offset = 0
            self.layers.append(Layer(l, column_offset, row_offset))

        self.grid_columns_count = sum([l.size_x for l in self.layers]) + len(self.layers)
        self.grid_rows_count = max([l.size_y for l in self.layers])
        self.total_width = self.grid_columns_count * self.node_gap_x
        self.total_height = self.grid_rows_count * self.node_gap_y
        self.grid = np.full((self.grid_rows_count, self.grid_columns_count), self.default_value).astype(np.float32)
        print("Net initialized", time.time() - start_time)
        print("grid", self.grid.shape)

    def process_batch(self, sublayer, index, grid):
        # Perform some computation on the chunk
        positions, columns_indices, rows_indices = sublayer.get_neuron_positions()
        grid[rows_indices, columns_indices] = positions

    def generate_net(self):
        print(f"generate net")
        print(self.total_width, self.total_height)
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

        methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
                   'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
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

    def get_texture(self, x1, y1, x2, y2, factor=1):
        # print(f"NNet get texture ", factor)
        start_time = time.time()
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(dpi = (1/factor)*800)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        new_grid = self.get_subgrid(x1, y1, x2, y2)
        grid_factor = factor * 10
        new_grid = new_grid[::grid_factor, ::grid_factor]

        if factor < 3:
            interpolation = "nearest"
        elif factor < 4:
            interpolation = "nearest"
        elif factor < 6:
            interpolation = "mitchell"
        else:
            interpolation = "bicubic"
        #new_grid = np.pad(new_grid, pad_width=1, constant_values=self.default_value)

        ax.imshow(new_grid, cmap=self.cmap, alpha=1, origin='lower', interpolation=interpolation)
        # image_width, image_height = fig.canvas.get_width_height()
        bbox = fig.get_tightbbox()
        # bbox=bbox.padded(1)
        adjust_bbox(fig, bbox)

        # # Get the image data from the canvas
        canvas = fig.canvas
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        image_data = buffer.getvalue()
        #
        image_width, image_height = fig.canvas.get_width_height()
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_data))
        # Convert the image to RGBA format
        image_rgba = image.convert('RGBA')
        # Get the raw pixel data as a numpy array
        image_data = np.array(image_rgba)
        # plt.savefig("gl/tiles/test2.png")
        plt.close(fig)
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
