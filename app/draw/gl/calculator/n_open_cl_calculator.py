import math

import numpy as np
import pyopencl as cl

from app.draw.gl.calculator.base_calculator import BaseCalculator

# OpenCL compute shader source
opencl_compute_shader = """
__kernel void calculate_grid_positions(
    __global int *grid_x,
    __global int *grid_y,
    const int numElements,
    const int start_index,
    const int end_index)
{
    int id = get_global_id(0) + start_index;
    if (id >= end_index) return;

    int size = (int)(sqrt((float)numElements));
    if (size * size < numElements) size++;

    grid_x[id] = id % size;
    grid_y[id] = id / size;

}
"""


class OpenCLPositionsCalculator(BaseCalculator):
    def __init__(self):
        self.context = None
        self.queue = None
        self.program = None
        self.init()

    def init(self):
        # Set up OpenCL context and queue
        platform = cl.get_platforms()[0]  # Select the first platform
        device = platform.get_devices()[0]  # Select the first GPU or CPU
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)

        # Compile the OpenCL program
        self.program = cl.Program(self.context, opencl_compute_shader).build()

    def cleanup(self):
        pass

    def measure(self, elements_count):
        size_x = math.ceil(math.sqrt(elements_count))
        size_y = size_x
        return [size_x, size_y]

    def calculate_positions(self, elements_count, start_index, end_index):
        # Setup position data

        # Create buffers
        mf = cl.mem_flags
        grid_x_buf = cl.Buffer(self.context, mf.WRITE_ONLY, elements_count * 4)
        grid_y_buf = cl.Buffer(self.context, mf.WRITE_ONLY, elements_count * 4)

        # Set kernel arguments and dispatch the kernel
        self.program.calculate_grid_positions(
            self.queue, (elements_count,), None,
            grid_x_buf,
            grid_y_buf,
            np.int32(elements_count),
            np.int32(start_index),
            np.int32(end_index)
        )

        # Read back results
        grid_x = np.empty(elements_count, dtype=np.int32)
        grid_y = np.empty(elements_count, dtype=np.int32)
        cl.enqueue_copy(self.queue, grid_x, grid_x_buf)
        cl.enqueue_copy(self.queue, grid_y, grid_y_buf)
        self.queue.finish()

        return grid_x, grid_y

# # Example usage
# calculator = OpenCLPositionsCalculator()
# size = 1024
# grid_x, grid_y, grid_size = calculator.calculate_positions(size)
# print("Grid X:", grid_x)
# print("Grid Y:", grid_y)
# print("Grid Size:", grid_size)
