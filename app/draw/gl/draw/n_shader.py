import OpenGL.GL as gl

# manage shaders

# Vertex shader source code
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 instance_color;
layout(location = 2) in vec2 instance_position;
layout(location = 3) in vec2 tex_coord;
layout(location = 4) in vec2 tex_coord2;

out vec3 color;
out vec2 frag_tex_coord;
out vec2 frag_tex_coord2;
uniform mat4 projection_matrix;


void main()
{
    gl_Position = projection_matrix * vec4(position + instance_position, 0.0, 1.0);
    color = instance_color;
    frag_tex_coord = tex_coord;
    frag_tex_coord2 = tex_coord2;
}
"""

# Fragment shader source code for drawing vertices
fragment_shader_source = """
#version 330 core

in vec3 color;
out vec4 frag_color;

uniform float fading_factor = 1.0f;


void main()
{
    frag_color = vec4(color, fading_factor);
}
"""


# Fragment shader source code for drawing textures
texture_fragment_shader_material_one = """
#version 330 core

in vec3 color;
in vec2 frag_tex_coord;
out vec4 frag_color;

uniform sampler2D tex;
uniform sampler2D tex2;


uniform float fading_factor = 1.0f;
uniform float alpha_factor = 1.0f;
uniform bool tex2_enabled = true;


void main()
{
    vec4 tex_color = texture(tex, frag_tex_coord);
    tex_color.a = fading_factor;
    frag_color = tex_color;
}
"""


# Fragment shader source code for drawing textures
texture_fragment_shader_material_two = """
#version 330 core

in vec3 color;
in vec2 frag_tex_coord2;
out vec4 frag_color;

uniform sampler2D tex2;
uniform sampler2D tex;


uniform float fading_factor = 1.0f;
uniform float alpha_factor = 1.0f;
uniform bool tex2_enabled = true;


void main()
{
    vec4 tex_color2 = texture(tex2, frag_tex_coord2);
    tex_color2.a = fading_factor;
    frag_color = tex_color2;
}
"""


class NShader:
    def __init__(self):
        self.shader_program = None
        self.shader_version = None

    def use(self):
        gl.glUseProgram(self.shader_program)

    def update_projection(self, projection_matrix):
        projection_matrix_uniform = gl.glGetUniformLocation(self.shader_program, "projection_matrix")
        gl.glUniformMatrix4fv(projection_matrix_uniform, 1, gl.GL_FALSE, projection_matrix)

        text_uniform = gl.glGetUniformLocation(self.shader_program, "tex1")
        gl.glUniform1i(text_uniform, 0)
        text_uniform1 = gl.glGetUniformLocation(self.shader_program, "tex2")
        gl.glUniform1i(text_uniform1, 1)

    def update_fading_factor(self, factor):
        fading_factor = gl.glGetUniformLocation(self.shader_program, "fading_factor")
        gl.glUniform1f(fading_factor, factor)

    def update_alpha_factor(self, factor):
        fading_factor = gl.glGetUniformLocation(self.shader_program, "alpha_factor")
        gl.glUniform1f(fading_factor, factor)

    def set_tex2_enabled(self, enabled):
        tex2_enabled = gl.glGetUniformLocation(self.shader_program, "tex2_enabled")
        gl.glUniform1f(tex2_enabled, enabled)

    def compile_vertices_program(self):
        self.compile(vertex_shader_source, fragment_shader_source)

    # def compile_textures_program(self):
    #     self.compile(vertex_shader_source, texture_fragment_shader_source_test)

    def compile_textures_material_one_program(self):
        self.compile(vertex_shader_source, texture_fragment_shader_material_one)

    def compile_textures_material_two_program(self):
        self.compile(vertex_shader_source, texture_fragment_shader_material_two)

    def compile(self, vertex_shader_source, fragment_shader_source):
        self.shader_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        #print("Supported shader version:", self.shader_version.decode())

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

        # # Create and compile the geometry shader
        # geometry_shader = gl.glCreateShader(gl.GL_GEOMETRY_SHADER)
        # gl.glShaderSource(geometry_shader, geometry_shader_source)
        # gl.glCompileShader(geometry_shader)
        #
        # # Check the compilation status
        # status = gl.glGetShaderiv(geometry_shader, gl.GL_COMPILE_STATUS)
        # if status != gl.GL_TRUE:
        #     # Compilation failed, retrieve the error message
        #     error_message = gl.glGetShaderInfoLog(geometry_shader)
        #     print("Geometry Shader compilation failed:\n", error_message)

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
        # gl.glAttachShader(self.shader_program, geometry_shader)
        gl.glAttachShader(self.shader_program, fragment_shader)
        gl.glLinkProgram(self.shader_program)

        # Check the linking status
        status = gl.glGetProgramiv(self.shader_program, gl.GL_LINK_STATUS)
        if gl.GL_TRUE != status:
            # Linking failed, retrieve the error message
            error_message = gl.glGetProgramInfoLog(self.shader_program)
            print("Shader program linking failed:\n", error_message)
