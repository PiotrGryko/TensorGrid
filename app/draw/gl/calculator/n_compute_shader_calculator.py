import OpenGL.GL as gl
import numpy as np

from app.draw.gl.calculator.base_calculator import BaseCalculator

compute_shader_source_square = """
#version 450 core

layout(local_size_x = 256) in;


// Output buffers for grid coordinates
layout(std430, binding = 0) buffer GridX {
    int grid_x[];
};

layout(std430, binding = 1) buffer GridY {
    int grid_y[];
};

// Output grid size (could be adjusted by shader logic for different shapes)
layout(std430, binding = 2) buffer GridSize {
    int grid_size[2]; // Holds width and height of the grid
};

uniform int numElements; // Total number of elements

void main() {
    int id = int(gl_GlobalInvocationID.x);
    if (id >= numElements) return;

    int size = int(sqrt(float(numElements))); // Calculate size of the square grid
    if (size * size < numElements) size++;    // Adjust grid size if not perfect square

    grid_x[id] = id % size; // Calculate the x-coordinate on the grid
    grid_y[id] = id / size; // Calculate the y-coordinate on the grid

    // Update the grid size
    if (id == 0) { // Only let the first thread update the size
        grid_size[0] = size;
        grid_size[1] = size;
    }
}
"""


class NPositionsCalculator(BaseCalculator):
    """
    TO DO !
    """
    def __init__(self):
        self.compute_program = None

    def cleanup(self):
        pass

    def measure(self, elements_count):
        pass

    def init(self):
        self.compile_compute_shader(compute_shader_source_square)

    def compile_compute_shader(self, compute_shader_source):
        # Create and compile the compute shader
        compute_shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        gl.glShaderSource(compute_shader, compute_shader_source)
        gl.glCompileShader(compute_shader)

        # Check the compilation status
        status = gl.glGetShaderiv(compute_shader, gl.GL_COMPILE_STATUS)
        if status != gl.GL_TRUE:
            # Compilation failed, retrieve the error message
            error_message = gl.glGetShaderInfoLog(compute_shader)
            print("Compute Shader compilation failed:\n", error_message)
            return None

        # Create the shader program and attach the compute shader
        self.compute_program = gl.glCreateProgram()
        gl.glAttachShader(self.compute_program, compute_shader)
        gl.glLinkProgram(self.compute_program)

        # Check the linking status
        status = gl.glGetProgramiv(self.compute_program, gl.GL_LINK_STATUS)
        if gl.GL_TRUE != status:
            # Linking failed, retrieve the error message
            error_message = gl.glGetProgramInfoLog(self.compute_program)
            print("Compute shader program linking failed:\n", error_message)
            return None

    def use(self):
        gl.glUseProgram(self.compute_program)

    def read_buffer(self, buffer_id, dtype, count):
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        mapped_buffer = gl.glMapBufferRange(gl.GL_SHADER_STORAGE_BUFFER, 0, count * np.dtype(dtype).itemsize,
                                            gl.GL_MAP_READ_BIT)
        data = np.frombuffer(mapped_buffer, dtype=dtype, count=count)
        gl.glUnmapBuffer(gl.GL_SHADER_STORAGE_BUFFER)
        return data

    def calculate_positions(self, size, start_index, end_index):
        # Setup position data (even if it's not fully used for now)

        # Set the uniform for number of elements
        gl.glUniform1i(gl.glGetUniformLocation(self.compute_program, "numElements"), size)

        # Dispatch compute shader
        gl.glDispatchCompute((size + 255) // 256, 1, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

        grid_x_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, grid_x_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, size * 4, None, gl.GL_DYNAMIC_DRAW)  # int size
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, grid_x_buffer)

        grid_y_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, grid_y_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, size * 4, None, gl.GL_DYNAMIC_DRAW)  # int size
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, grid_y_buffer)

        grid_size_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, grid_size_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, 8, None, gl.GL_DYNAMIC_DRAW)  # Assuming int[2]
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, grid_size_buffer)

        grid_x = self.read_buffer(grid_x_buffer, np.int32, size)
        grid_y = self.read_buffer(grid_y_buffer, np.int32, size)
        grid_size = self.read_buffer(grid_size_buffer, np.int32, 2)  # Read 2 integers

        gl.glDeleteBuffers(1, [grid_x_buffer])
        gl.glDeleteBuffers(1, [grid_y_buffer])
        gl.glDeleteBuffers(1, [grid_size_buffer])

        return grid_x, grid_y, grid_size
