import numpy as np


# mat4
# projection = mat4(
#     vec4(zoom_factor / aspect_ratio, 0.0, 0.0, 0.0),
#     vec4(0.0, zoom_factor, 0.0, 0.0),
#     vec4(0.0, 0.0, 1.0, 0.0),
#     vec4(
#         translation.x - anchor_x * (zoom_factor / aspect_ratio - 1.0),
#         translation.y - anchor_y * (zoom_factor - 1.0),
#         0.0,
#         1.0
#     )
# );
class Projection:
    def __init__(self):
        self.matrix = np.array([
            [1.0,
             0.0,
             0.0,
             0.0],
            [0.0,
             1.0,
             0.0,
             0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0,
             0.0,
             0.0,
             1.0]
        ], dtype=np.float32)

    def get_projection_for_tile(self, x1, y1, x2, y2):
        # Calculate the midpoint of the tile
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Calculate the scale factors needed
        scale_x = 2.0 / (x2 - x1)
        scale_y = 2.0 / (y2 - y1)

        # Create translation and scaling matrices
        translation_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-mid_x, -mid_y, 0.0, 1.0]
        ], dtype=np.float32)

        scaling_matrix = np.array([
            [scale_x, 0, 0, 0],
            [0, scale_y, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Update the projection matrix by first translating, then scaling
        _matrix = np.dot(self.matrix, translation_matrix)
        _matrix = np.dot(_matrix, scaling_matrix)
        return _matrix

    def window_to_world_point(self, x, y):
        # Map window point in [-1,1] cords to world cords
        # Usage example: node_position = window_to_world_point(mouse_position)
        point = np.array([x, y, 0.0, 1.0], dtype=np.float32)
        inverse_projection_matrix = np.linalg.inv(self.matrix)
        world_point_homogeneous = point @ inverse_projection_matrix
        world_point_homogeneous /= world_point_homogeneous[3]
        return [world_point_homogeneous[0], world_point_homogeneous[1]]

    def world_to_window_point(self, x, y):
        point = np.array([x, y, 0.0, 1.0], dtype=np.float32)
        world_point_homogeneous = point @ self.matrix
        world_point_homogeneous /= world_point_homogeneous[3]
        return [world_point_homogeneous[0], world_point_homogeneous[1]]

    def set_aspect_ratio(self, aspect_ratio):
        self.matrix[0][0] = self.get_zoom() / aspect_ratio

    def get_translation(self):
        return [
            self.matrix[3][0],
            self.matrix[3][1]
        ]

    def get_zoom(self):
        return self.matrix[1][1]

    def translate_by(self, dx, dy):
        self.matrix[3][0] += dx
        self.matrix[3][1] += dy

    def translate_to(self, x, y):
        self.matrix[3][0] = x
        self.matrix[3][1] = y

    def zoom(self, x, y, new_zoom):
        current_zoom = self.get_zoom()
        zoom_factor = new_zoom / current_zoom

        translation_matrix_1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-x, -y, 0.0, 1.0]
        ])

        # Step 2: Apply zoom
        zoom_matrix = np.array([
            [zoom_factor, 0.0, 0.0, 0.0],
            [0.0, zoom_factor, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Step 3: Translate back to original translation
        translation_matrix_2 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, 0.0, 1.0]
        ])

        result = translation_matrix_1 @ zoom_matrix @ translation_matrix_2
        self.matrix = self.matrix @ result
