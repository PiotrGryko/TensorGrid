import OpenGL.GL as gl

# manage shaders

# NODES

# Vertex shader source code for drawing nodes using instances drawing
# Each node is drawn as a large circle
# Circles vertices are stored in position layout
# Instances positions and their values (float 0 to 1) are stored in positions_and_values
vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 positions_and_value;

out float color_value;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * vec4(position + positions_and_value.xy, 0.0, 1.0);
    color_value = positions_and_value.z; 
}
"""

# Fragment shader source code for drawing nodes
# Use instance value (float 0 to 1) extract color from color_map
fragment_shader_source = """
#version 330 core

uniform float fading_factor = 1.0f;
uniform sampler1D color_map;
in float color_value;
out vec4 frag_color;

void main()
{
    frag_color = texture(color_map, color_value);
}
"""

# STATIC TEXTURES

# Vertex shader source code for drawing textures
texture_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coord;
layout(location = 2) in vec2 tex_coord2;

out vec2 frag_tex_coord;
out vec2 frag_tex_coord2;
uniform mat4 projection_matrix;


void main()
{
    gl_Position = projection_matrix * vec4(position, 0.0, 1.0);
    frag_tex_coord = tex_coord;
    frag_tex_coord2 = tex_coord2;
}
"""

# Fragment shader source code for drawing textures
texture_fragment_shader_material_one = """
#version 330 core

in vec2 frag_tex_coord;
out vec4 frag_color;

uniform sampler1D color_map; 
uniform sampler2D tex1;
uniform sampler2D tex2;


uniform float fading_factor = 1.0f;

void main()
{
    vec4 tex_color = texture(tex1, frag_tex_coord);
    tex_color.a = fading_factor;
    frag_color = tex_color;
}
"""


# Fragment shader source code for drawing textures
texture_fragment_shader_material_two = """
#version 330 core

in vec2 frag_tex_coord2;
out vec4 frag_color;

uniform sampler1D color_map;
uniform sampler2D tex1;
uniform sampler2D tex2;


uniform float fading_factor = 1.0f;

void main()
{
    vec4 tex_color2 = texture(tex2, frag_tex_coord2);
    tex_color2.a = fading_factor;
    frag_color = tex_color2;
}
"""

# DRAWING POINTS

# Vertex shader source code for drawing points
color_grid_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position_and_value; // x, y for position, z for color scaling
uniform mat4 projection_matrix;
out float color_value;

void main() {
    gl_Position = projection_matrix * vec4(position_and_value.xy, 0.0, 1.0);
    color_value = position_and_value.z; // Pass the color scaling value to the fragment shader

}
"""

# Fragment shader source code for drawing points
color_grid_fragment_shader_source = """
#version 330 core

uniform sampler1D color_map;
in float color_value;
out vec4 fragColor;

void main() {
    fragColor = texture(color_map, color_value);
}
"""



# Vertex shader source code for drawing texture color map
color_grid_texture_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coord;
layout(location = 2) in vec2 tex_coord2;

uniform mat4 projection_matrix;
out float color_value;
out vec2 frag_tex_coord;
out vec2 frag_tex_coord2;

void main() {
    gl_Position = projection_matrix * vec4(position, 0.0, 1.0);
    frag_tex_coord = tex_coord;
    frag_tex_coord2 = tex_coord2;
}
"""

# Fragment shader source code for drawing texture color map
color_grid_texture_material_one_fragment_shader_source = """
#version 330 core

uniform sampler1D color_map;
uniform sampler2D tex1;

in vec2 frag_tex_coord;
in vec2 frag_tex_coord2;

out vec4 fragColor;

void main() {
    float value = texture(tex1, frag_tex_coord).r; 
    vec3 color = texture(color_map, value).rgb;
    fragColor = vec4(color, 1.0);
}
"""

color_grid_texture_material_two_fragment_shader_source = """
#version 330 core

uniform sampler1D color_map;
uniform sampler2D tex2;

in vec2 frag_tex_coord;
in vec2 frag_tex_coord2;

out vec4 fragColor;

void main() {
    float value = texture(tex2, frag_tex_coord2).r; 
    vec3 color = texture(color_map, value).rgb;
    fragColor = vec4(color, 1.0);
}
"""





class NShader:
    def __init__(self):
        self.shader_program = None
        self.shader_version = None
        self.current_cmap_name = None

    def use(self):
        gl.glUseProgram(self.shader_program)

        # assign shader tex1 and tex2 to openGl texture. gl.GL_TEXTURE1 and gl.GL_TEXTURE2
        text_uniform = gl.glGetUniformLocation(self.shader_program, "color_map")
        gl.glUniform1i(text_uniform, 0)
        text_uniform1 = gl.glGetUniformLocation(self.shader_program, "tex1")
        gl.glUniform1i(text_uniform1, 1)
        text_uniform2 = gl.glGetUniformLocation(self.shader_program, "tex2")
        gl.glUniform1i(text_uniform2, 2)

    def update_color_map(self,cmap_name, color_values):
        if self.current_cmap_name == cmap_name:
            return
        # Generate and bind the texture
        colorMapTextureID = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_1D, colorMapTextureID)

        # Upload the 1D color map data
        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGB, 255, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, color_values)

        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        self.current_cmap_name = cmap_name


    def update_projection(self, projection_matrix):
        projection_matrix_uniform = gl.glGetUniformLocation(self.shader_program, "projection_matrix")
        gl.glUniformMatrix4fv(projection_matrix_uniform, 1, gl.GL_FALSE, projection_matrix)


    def update_fading_factor(self, factor):
        fading_factor = gl.glGetUniformLocation(self.shader_program, "fading_factor")
        gl.glUniform1f(fading_factor, factor)


    def compile_nodes_program(self):
        self.compile(vertex_shader_source, fragment_shader_source)

    def compile_color_grid_program(self):
        self.compile(color_grid_vertex_shader_source, color_grid_fragment_shader_source)

    def compile_textures_material_one_program(self):
        self.compile(texture_vertex_shader_source, texture_fragment_shader_material_one)

    def compile_textures_material_two_program(self):
        self.compile(texture_vertex_shader_source, texture_fragment_shader_material_two)

    def compile_colors_textures_material_one_program(self):
        self.compile(color_grid_texture_vertex_shader_source, color_grid_texture_material_one_fragment_shader_source)

    def compile_colors_textures_material_two_program(self):
        self.compile(color_grid_texture_vertex_shader_source, color_grid_texture_material_two_fragment_shader_source)

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
