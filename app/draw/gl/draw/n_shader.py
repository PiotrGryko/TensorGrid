import OpenGL.GL as gl

# manage shaders

# DEBUG BSP TREE TILES

# Simple shader for drawing vertices
# Positions are vec2 buffer "position"
# Colors  are vec3 buffer "color"
debug_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

out vec3 color_value;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * vec4(position, 0.0, 1.0);
    color_value = color; 
}
"""
debug_fragment_shader_source = """
#version 330 core

in vec3 color_value;
out vec4 frag_color;

void main()
{
    frag_color = vec4(color_value.xyz, 1.0);
}
"""

# DRAWING POINTS

# Vertex shader source code for drawing points
# Positions and float values are passed as a vec3 buffer "position_and_value"
# Color is sampled from the color map texture
points_from_buffer_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec3 position_and_value; // x, y for position, z for color scaling
uniform mat4 projection_matrix;
float color_multiplier = 50;
out float color_value;

void main() {
    gl_Position = projection_matrix * vec4(position_and_value.xy, 0.0, 1.0);
    float intensified_color_value = clamp(position_and_value.z  * color_multiplier, 0, 1);
    color_value = intensified_color_value; // Pass the color scaling value to the fragment shader

}
"""
# Fragment shader source code for drawing points
points_from_buffer_fragment_shader_source = """
#version 330 core

uniform sampler1D color_map;
in float color_value;
out vec4 fragColor;

void main() {
    fragColor = texture(color_map, color_value);
}
"""

# DRAWING INSTANCES

# Vertex shader source code for instanced drawing
# Instance vertices are stored in vec2 "position"
# Instances positions and their values (float 0 to 1) are stored in vec3 "positions_and_values"
instances_from_buffer_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 positions_and_value;

out float color_value;
uniform mat4 projection_matrix;
float color_multiplier = 50;

void main()
{
    gl_Position = projection_matrix * vec4(position + positions_and_value.xy, 0.0, 1.0);
    float intensified_color_value = clamp(positions_and_value.z  * color_multiplier, 0, 1);
    color_value = intensified_color_value; 
}
"""

# Fragment shader source code for drawing instances
# Use instance value (float 0 to 1) extract color from color_map
instances_from_buffer_fragment_shader_source = """
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

# Vertex shader source code for instanced drawing
# Instance vertices are stored in vec2 position
# Instances positions and their values (float 0 to 1) are stored in texture buffer "tex1"
# Final position on the screen is calculated from texture (x,y)
# Colors is sampled from the color map
instances_from_texture_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
uniform sampler1D color_map;
uniform sampler2D tex1;
uniform int texture_width;
uniform int texture_height;
uniform int factor =1;
uniform vec2 position_offset = vec2(0.0, 0.0); 
uniform mat4 projection_matrix;
float node_gap = 0.2;
float color_multiplier = 50;

out vec4 color_value;

void main()
{
    float x = gl_InstanceID % texture_width;
    float y = gl_InstanceID / texture_width;
    float value = texelFetch(tex1, ivec2(x, y), 0).r;
    float entity_factor =  factor;
    float scaled_x = x * entity_factor;
    float scaled_y = y * entity_factor;
    
    vec2 instance_position = vec2(scaled_x * node_gap, scaled_y * node_gap);
    float intensified_color_value = clamp(value  * color_multiplier, 0, 1);
    
    gl_Position = projection_matrix * vec4(position.xy + instance_position + position_offset, 0.0, 1.0);
    color_value = texture(color_map, intensified_color_value);
    
}
"""

# Fragment shader source code for drawing instances
instances_from_texture_fragment_shader_source = """
#version 330 core

uniform float fading_factor = 1.0f;
uniform sampler1D color_map;
in vec4 color_value;
out vec4 frag_color;

void main()
{
    frag_color = color_value;
}
"""

# COLOR MAP TEXTURES

# Vertex shader source code for drawing texture color map
# Data is passed as a texture buffer of floats "tex1"
# Colors is sampled from the color map
cmap_texture_vertex_shader_source = """
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
cmap_texture_fragment_shader_source = """
#version 330 core

uniform sampler1D color_map;
uniform sampler2D tex1;
uniform float fading_factor = 1.0;

in vec2 frag_tex_coord;
in vec2 frag_tex_coord2;

out vec4 fragColor;
float color_multiplier = 50;

void main() {
    float value = texture(tex1, frag_tex_coord).r; 
    float intensified_color_value = clamp(value  * color_multiplier, 0, 1);
    vec3 color = texture(color_map, intensified_color_value).rgb;
    fragColor = vec4(color, 1.0 * fading_factor);
}
"""



# STATIC TEXTURES

# Vertex shader source code for drawing static images
static_texture_vertex_shader_source = """
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
static_texture_fragment_shader_source = """
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





### V2

cmap_v2_texture_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coord;

uniform vec2 position_offset = vec2(0.0, 0.0); 
uniform mat4 projection_matrix;
uniform sampler1D color_map;
uniform sampler2D tex1;
uniform int factor =1;

float node_gap = 0.2;
out vec2 frag_tex_coord;

void main()
{
    float x = position.x;
    float y = position.y;

    float entity_factor =  factor;    
    float scaled_x = x * entity_factor * node_gap;
    float scaled_y = y * entity_factor * node_gap;

    vec2 instance_position = vec2(scaled_x, scaled_y);
    gl_Position =  projection_matrix * vec4(instance_position  + position_offset, 0.0, 1.0);
    frag_tex_coord = tex_coord;

}
"""
# Fragment shader source code for drawing instances
cmap_v2_texture_fragment_shader_source = """
#version 330 core

uniform sampler1D color_map;
uniform sampler2D tex1;
uniform float fading_factor = 1.0;

in vec2 frag_tex_coord;

out vec4 fragColor;
float color_multiplier = 50;

void main() {
    float value = texture(tex1, frag_tex_coord).r; 
    float intensified_color_value = clamp(value  * color_multiplier, 0, 1);
    vec3 color = texture(color_map, intensified_color_value).rgb;
    fragColor =  vec4(color, 1.0 * fading_factor);
}
"""

instances_v2_vertex_shader_source = """
#version 330 core

layout(location = 0) in vec2 position;
uniform sampler1D color_map;
uniform sampler2D tex1;

uniform int texture_width;
uniform int texture_height;

uniform int target_width;
uniform int target_height;

uniform vec2 position_offset = vec2(0.0, 0.0); 
uniform mat4 projection_matrix;
uniform int factor =1;

float node_gap = 0.2;
float color_multiplier = 50;

out vec4 color_value;

void main()
{
    
    int selected_width = min(texture_width, target_width);
    int selected_height = min(texture_height, target_height);

    float x = gl_InstanceID % selected_width;
    float y = gl_InstanceID / selected_width;
    float value = texelFetch(tex1, ivec2(x, y), 0).r;
    float entity_factor =  factor;
    float scaled_x = x * entity_factor;
    float scaled_y = y * entity_factor;

    vec2 instance_position = vec2(scaled_x * node_gap, scaled_y * node_gap);
    float intensified_color_value = clamp(value  * color_multiplier, 0, 1);

    gl_Position = projection_matrix * vec4(position.xy + instance_position + position_offset, 0.0, 1.0);
    color_value = texture(color_map, intensified_color_value);

}
"""

# Fragment shader source code for drawing instances
instances_v2_fragment_shader_source = """
#version 330 core

uniform float fading_factor = 1.0f;
uniform sampler1D color_map;
in vec4 color_value;
out vec4 frag_color;

void main()
{
    frag_color = color_value;
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

    def update_color_map(self, cmap_name, color_values):
        if self.current_cmap_name == cmap_name:
            return
        # Generate and bind the texture
        colorMapTextureID = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_1D, colorMapTextureID)

        # color_values = np.clip(color_values * 50, 0, 255)
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

    def update_texture_width(self, width):
        texture_width = gl.glGetUniformLocation(self.shader_program, "texture_width")
        gl.glUniform1i(texture_width, width)

    def update_texture_height(self, height):
        texture_height = gl.glGetUniformLocation(self.shader_program, "texture_height")
        gl.glUniform1i(texture_height, height)

    def update_target_width(self, width):
        target_width = gl.glGetUniformLocation(self.shader_program, "target_width")
        gl.glUniform1i(target_width, width)

    def update_target_height(self, height):
        target_height = gl.glGetUniformLocation(self.shader_program, "target_height")
        gl.glUniform1i(target_height, height)

    def update_details_factor(self, details_factor):
        factor = gl.glGetUniformLocation(self.shader_program, "factor")
        gl.glUniform1i(factor, details_factor)

    def update_position_offset(self, x1, y1):
        position_offset = gl.glGetUniformLocation(self.shader_program, "position_offset")
        gl.glUniform2f(position_offset, x1, y1)

    def compile_debug_program(self):
        self.compile(debug_vertex_shader_source, debug_fragment_shader_source)

    def compile_instances_from_buffer_program(self):
        self.compile(instances_from_buffer_vertex_shader_source, instances_from_buffer_fragment_shader_source)

    def compile_instances_from_texture_program(self):
        self.compile(instances_from_texture_vertex_shader_source, instances_from_texture_fragment_shader_source)

    def compile_points_program(self):
        self.compile(points_from_buffer_vertex_shader_source, points_from_buffer_fragment_shader_source)

    def compile_static_texture_program(self):
        self.compile(static_texture_vertex_shader_source, static_texture_fragment_shader_source)

    def compile_color_map_texture_program(self):
        self.compile(cmap_texture_vertex_shader_source, cmap_texture_fragment_shader_source)

    def compile_color_map_v2_texture_program(self):
        self.compile(cmap_v2_texture_vertex_shader_source, cmap_v2_texture_fragment_shader_source)

    def compile_instances_v2_program(self):
        self.compile(instances_v2_vertex_shader_source, instances_v2_fragment_shader_source)

    def compile(self, vertex_shader_source, fragment_shader_source):
        self.shader_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        # print("Supported shader version:", self.shader_version.decode())

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
