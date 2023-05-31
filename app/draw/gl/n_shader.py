import ctypes

import OpenGL.GL as gl
import numpy as np

# manage shaders

# Vertex shader source code
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 instance_color;
layout(location = 2) in vec2 instance_position;

out vec3 color;
uniform mat4 projection_matrix;


void main()
{
    gl_Position = projection_matrix * vec4(position + instance_position, 0.0, 1.0);
    color = instance_color;
}
"""

# Fragment shader source code
fragment_shader_source = """
#version 330 core

in vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(color, 1.0);
}
"""


class NShader:
    def __init__(self):
        self.shader_program = None
        self.shader_version = None

    def update_projection(self, projection_matrix):
        projection_matrix_uniform = gl.glGetUniformLocation(self.shader_program, "projection_matrix")
        gl.glUniformMatrix4fv(projection_matrix_uniform, 1, gl.GL_FALSE, projection_matrix)

    def compile(self):
        self.shader_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        print("Supported shader version:", self.shader_version.decode())

        # Create and compile the vertex shader
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader, vertex_shader_source)
        gl.glCompileShader(vertex_shader)

        # Check the compilation status
        status = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
        if status != gl.GL_TRUE:
            # Compilation failed, retrieve the error message
            error_message = gl.glGetShaderInfoLog(vertex_shader)
            print("Vertex Shader compilation failed:\n", error_message)

        # Create and compile the fragment shader
        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader, fragment_shader_source)
        gl.glCompileShader(fragment_shader)

        # Check the compilation status
        status = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
        if status != gl.GL_TRUE:
            # Compilation failed, retrieve the error message
            error_message = gl.glGetShaderInfoLog(fragment_shader)
            print("Fragment Shader compilation failed:\n", error_message)

        # Create the shader program and attach the shaders
        self.shader_program = gl.glCreateProgram()
        gl.glAttachShader(self.shader_program, vertex_shader)
        gl.glAttachShader(self.shader_program, fragment_shader)
        gl.glLinkProgram(self.shader_program)

        # Check the linking status
        status = gl.glGetProgramiv(self.shader_program, gl.GL_LINK_STATUS)
        if gl.GL_TRUE != status:
            # Linking failed, retrieve the error message
            error_message = gl.glGetProgramInfoLog(self.shader_program)
            print("Shader program linking failed:\n", error_message)

