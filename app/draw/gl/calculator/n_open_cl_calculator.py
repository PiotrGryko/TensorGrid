import numpy as np
import pyopencl as cl

# OpenCL compute shader source
opencl_compute_shader = """
__kernel void calculate_grid_positions(
    __global int *grid_x,
    __global int *grid_y,
    __global int *grid_size,
    const int numElements)
{
    int id = get_global_id(0);
    if (id >= numElements) return;

    int size = (int)(sqrt((float)numElements));
    if (size * size < numElements) size++;

    grid_x[id] = id % size;
    grid_y[id] = id / size;

    if (id == 0) {
        grid_size[0] = size;
        grid_size[1] = size;
    }
}
"""


class OpenCLPositionsCalculator:
    def __init__(self):
        self.context = None
        self.queue = None
        self.program = None
        self.init_opencl()

    def init_opencl(self):
        # Set up OpenCL context and queue
        platform = cl.get_platforms()[0]  # Select the first platform
        device = platform.get_devices()[0]  # Select the first GPU or CPU
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)

        # Compile the OpenCL program
        self.program = cl.Program(self.context, opencl_compute_shader).build()

    def calculate_positions(self, size):
        # Setup position data

        # Create buffers
        mf = cl.mem_flags
        grid_x_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size * 4)
        grid_y_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size * 4)
        grid_size_buf = cl.Buffer(self.context, mf.WRITE_ONLY, 8)  # 2 integers

        # Set kernel arguments and dispatch the kernel
        self.program.calculate_grid_positions(
            self.queue, (size,), None,
            grid_x_buf,
            grid_y_buf,
            grid_size_buf,
            np.int32(size)
        )

        # Read back results
        grid_x = np.empty(size, dtype=np.int32)
        grid_y = np.empty(size, dtype=np.int32)
        grid_size = np.empty(2, dtype=np.int32)
        cl.enqueue_copy(self.queue, grid_x, grid_x_buf)
        cl.enqueue_copy(self.queue, grid_y, grid_y_buf)
        cl.enqueue_copy(self.queue, grid_size, grid_size_buf)
        self.queue.finish()

        return grid_x, grid_y, grid_size


# # Example usage
# calculator = OpenCLPositionsCalculator()
# size = 1024
# grid_x, grid_y, grid_size = calculator.calculate_positions(size)
# print("Grid X:", grid_x)
# print("Grid Y:", grid_y)
# print("Grid Size:", grid_size)
