from ctypes import c_void_p

import OpenGL.GL as gl
# import OpenGL.GLUT as glut
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo as glvbo
from OpenGL.raw.GL._types import GL_TRUE

# Vertex shader source code
vertex_shader_source = """
    #version 330 core
    in vec3 vp;

    void main() {
        gl_Position = vec4(vp, 1.0);
    }
"""

# Fragment shader source code
fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color
    }
"""

shader_program = None
vbo = None
vao= None
window = None


def initialize():
    global shader_program
    global vbo
    global vao
    print("init")

    shader_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
    print("Supported shader version:", shader_version.decode())

    # Create and compile the vertex shader
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, vertex_shader_source)
    gl.glCompileShader(vertex_shader)

    # Check the compilation status
    status = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if status != GL_TRUE:
        # Compilation failed, retrieve the error message
        error_message = glGetShaderInfoLog(vertex_shader)
        print("Vertex Shader compilation failed:\n", error_message)

    # Create and compile the fragment shader
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, fragment_shader_source)
    gl.glCompileShader(fragment_shader)

    # Check the compilation status
    status = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if status != GL_TRUE:
        # Compilation failed, retrieve the error message
        error_message = glGetShaderInfoLog(fragment_shader)
        print("Fragment Shader compilation failed:\n", error_message)

    # Create the shader program and attach the shaders
    shader_program = gl.glCreateProgram()
    gl.glAttachShader(shader_program, vertex_shader)
    gl.glAttachShader(shader_program, fragment_shader)
    gl.glLinkProgram(shader_program)

    # Check the linking status
    status = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if GL_TRUE != status:
        # Linking failed, retrieve the error message
        error_message = glGetProgramInfoLog(shader_program)
        print("Shader program linking failed:\n", error_message)

    # Set up the vertex data for the triangle
    # vertices = glvbo.VBO(
    #     np.array([[0.0, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]], 'f')
    # )

    #  (x, y, z)
    vertices = [
        -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
        0.0, 0.5, 0.0,
    ]
    vertices = (GLfloat * len(vertices))(*vertices)
    vbo = None
    vbo = gl.glGenBuffers(1, vbo)
    #vertex_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)

    vao = None
    vao = gl.glGenVertexArrays(1, vao)
    gl.glBindVertexArray(vao)
    gl.glEnableVertexAttribArray(0)
    gl.glBindBuffer(GL_ARRAY_BUFFER, vbo)
    gl.glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)


def render():
    global shader_program
    global vbo
    global vao
    # Clear the screen
    gl.glClearColor(0.0, 0.0, 1.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    # Use the shader program
    gl.glUseProgram(shader_program)
    gl.glBindVertexArray(vao)
    # # Bind the vertex buffer
    # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    #
    # # Specify the layout of the vertex data
    # gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    # gl.glEnableVertexAttribArray(0)

    # Draw the triangle
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    # Disable the vertex attribute array
    #gl.glDisableVertexAttribArray(0)

    # Swap buffers to display the rendered image


def processInput(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)
def framebuffer_size_callback():
    glViewport(0, 0, 800, 600)
def main():
    global window
    # Initialize OpenGL and create a window
    glfw.init()

    # Set GLFW context hints
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    # Create a GLFW window
    window = glfw.create_window(800, 600,"Hello World", None, None)
    # Make the created window's context current

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback())

    # Check if window creation succeeded
    if not window:
        glfw.terminate()
        raise ValueError("Failed to create GLFW window")

    initialize()
    # Main loop
    while not glfw.window_should_close(window):
        processInput(window)


        # Render the frame
        render()
        glfw.swap_buffers(window)
        # Process events
        glfw.poll_events()
    # Terminate GLFW
    glfw.terminate()


#
# def render2():
#     global shader_program
#     gl.glClear(gl.GL_COLOR_BUFFER_BIT)
#     gl.glUseProgram(shader_program)
#
#     gl.glLoadIdentity()
#     # Set the color to red
#     gl.glColor3f(1.0, 0.0, 0.0)
#
#     # Draw a rectangle
#     gl.glBegin(gl.GL_QUADS)
#     gl.glVertex2f(-0.5, -0.5)
#     gl.glVertex2f(0.5, -0.5)
#     gl.glVertex2f(0.5, 0.5)
#     gl.glVertex2f(-0.5, 0.5)
#     gl.glEnd()
#
#     gl.glFlush()
#
# def main2():
#     global shader_program
#     global vertex_buffer
#     glut.glutInit()
#     glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB)
#     glut.glutInitWindowSize(800, 600)
#     glut.glutCreateWindow(b"Simple OpenGL Window")
#
#     shader_version = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
#     print("Supported shader version:", shader_version.decode())
#
#
#     vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
#     print(vertex_shader)
#     gl.glShaderSource(vertex_shader, vertex_shader_source)
#     gl.glCompileShader(vertex_shader)
#
#     # Check the compilation status
#     status = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
#     if status != GL_TRUE:
#         # Compilation failed, retrieve the error message
#         error_message = glGetShaderInfoLog(vertex_shader)
#         print("Vertex Shader compilation failed:\n", error_message)
#
#     # Create and compile the fragment shader
#     fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
#     print(fragment_shader)
#     gl.glShaderSource(fragment_shader, fragment_shader_source)
#     gl.glCompileShader(fragment_shader)
#
#     # Check the compilation status
#     status = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
#     if status != GL_TRUE:
#         # Compilation failed, retrieve the error message
#         error_message = glGetShaderInfoLog(fragment_shader)
#         print("Fragment Shader compilation failed:\n", error_message)
#
#     # Create the shader program and attach the shaders
#     shader_program = gl.glCreateProgram()
#     gl.glAttachShader(shader_program, vertex_shader)
#     gl.glAttachShader(shader_program, fragment_shader)
#     gl.glLinkProgram(shader_program)
#
#     # Check the linking status
#     status = glGetProgramiv(shader_program, GL_LINK_STATUS)
#     if GL_TRUE != status:
#         # Linking failed, retrieve the error message
#         error_message = glGetProgramInfoLog(shader_program)
#         print("Shader program linking failed:\n", error_message)
#
#     gl.glClearColor(0.0, 0.0, 0.0, 1.0)
#     glut.glutDisplayFunc(render2)
#     glut.glutMainLoop()


if __name__ == "__main__":
    main()

#
# import pygame
# from OpenGL.GL import *
# from ctypes import sizeof, c_void_p
#
# pygame.init()
# display = pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.OPENGL)
# clock = pygame.time.Clock()
# FPS = 60
#
# VERTEX_SHADER_SOURCE = '''
#     #version 330 core
#     layout (location = 0) in vec3 aPos;
#
#     void main()
#     {
#         gl_Position = vec4(aPos, 1.0);
#     }
# '''
#
# FRAGMENT_SHADER_SOURCE = '''
#     #version 330 core
#
#     out vec4 fragColor;
#     void main()
#     {
#         fragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
#     }
# '''
#
# shader_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
# print("Supported shader version:", shader_version.decode())
# #  (x, y, z)
# vertices = [
#     -0.5, -0.5, 0.0,
#     0.5, -0.5, 0.0,
#     0.0, 0.5, 0.0,
# ]
# vertices = (GLfloat * len(vertices))(*vertices)
#
# # we created program object
# program = glCreateProgram()
#
# # we created vertex shader
# vertex_shader = glCreateShader(GL_VERTEX_SHADER)
# # we passed vertex shader's source to vertex_shader object
# glShaderSource(vertex_shader, VERTEX_SHADER_SOURCE)
# # and we compile it
# glCompileShader(vertex_shader)
#
# # we created fragment shader
# fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
# # we passed fragment shader's source to fragment_shader object
# glShaderSource(fragment_shader, FRAGMENT_SHADER_SOURCE)
# # and we compile it
# glCompileShader(fragment_shader)
#
# # attach these shaders to program
# glAttachShader(program, vertex_shader)
# glAttachShader(program, fragment_shader)
#
# # link the program
# glLinkProgram(program)
#
# # create vbo object
# vbo = glGenBuffers(1)
#
# # enable buffer(VBO)
# glBindBuffer(GL_ARRAY_BUFFER, vbo)
#
# # send the data
# glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
#
# # create vao object
# vao = glGenVertexArrays(1)
# # enable VAO and then finally binding to VBO object what we created before.
# glBindVertexArray(vao)
#
# # we activated to the slot of position in VAO (vertex array object)
# glEnableVertexAttribArray(0)
#
# # explaining to the VAO what data will be used for slot 0 (position slot)
# glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), c_void_p(0))
#
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#
#     glClearColor(1.0, 0.6, 0.0, 1.0)
#     glClear(GL_COLOR_BUFFER_BIT)
#
#     glUseProgram(program)
#     glBindVertexArray(vao)
#     glDrawArrays(GL_TRIANGLES, 0, 3)
#
#     pygame.display.flip()
#     clock.tick(FPS)