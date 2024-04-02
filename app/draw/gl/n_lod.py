import time
from enum import Enum

from app.draw.gl.n_texture import NTexture


class LodType(Enum):
    STATIC_TEXTURE = 1
    LEAFS_VERTICES = 2
    LEAFS_TEXTURES = 3,
    MEGA_LEAF_VERTICES = 4
    MEGA_LEAF_TEXTURE = 5


class Lod:
    def __init__(self, level, lod_type):
        self.level = level
        self.lod_type = lod_type
        self.texture = None
        self.factor = None
        self.material_id = None
        self.mega_leaf = None
        self.visible_leafs = []

    def print(self):
        print(f"level: {self.level}, lod_type: {self.lod_type}, material_id: {self.material_id}, factor: {self.factor}")

    def draw_vertices(self, n_tree):
        if self.lod_type == LodType.LEAFS_VERTICES:
            n_tree.draw_leafs_vertices()
        if self.lod_type == LodType.MEGA_LEAF_VERTICES:
            n_tree.draw_mega_leaf_vertices()

    def draw_textures(self, n_tree):
        if self.lod_type == LodType.STATIC_TEXTURE:
            self.texture.draw()
        if self.lod_type == LodType.LEAFS_TEXTURES:
            n_tree.draw_leafs_textures(self.material_id)
        if self.lod_type == LodType.MEGA_LEAF_TEXTURE:
            n_tree.draw_mega_leaf_texture(self.material_id)

    @staticmethod
    def build_static_texture_level(n_net, level, max_depth, material_id):
        total_width = n_net.total_width
        total_height = n_net.total_height
        factor = max([max_depth - level, 1])
        factor = 1 if factor == 1 else factor
        factor = factor
        lod = Lod(level, LodType.STATIC_TEXTURE)
        lod.texture = NTexture()
        lod.material_id = material_id
        img_data, img_width, img_height = n_net.get_texture2(0, 0, total_width, total_height, factor)
        lod.texture.create_from_data(0, 0, total_width, total_height, img_data, img_width, img_height,
                                     material_id=material_id)
        lod.factor = factor
        print("Creted stastic texture ", level, material_id, factor)
        return lod

    @staticmethod
    def build_leafs_texture_level(level, material_id):
        lod = Lod(level, LodType.LEAFS_TEXTURES)
        lod.material_id = material_id
        return lod

    @staticmethod
    def build_mega_leaf_texture_level(level, material_id):
        lod = Lod(level, LodType.MEGA_LEAF_TEXTURE)
        lod.material_id = material_id
        return lod

    @staticmethod
    def build_leafs_vertices_level(level):
        lod = Lod(level, LodType.LEAFS_VERTICES)
        return lod

    @staticmethod
    def build_mega_leaf_vertices_level(level):
        lod = Lod(level, LodType.MEGA_LEAF_VERTICES)
        return lod


class NLvlOfDetails:
    def __init__(self):
        self.textures = []
        self.min_possible_zoom = 0  # minimal possible zoom calculated by NWindow
        self.max_possible_zoom = 0  # max possible zoom calculated by NWindow
        self.max_lod_level = 10  # generate up to N different level of details
        self.viewport = None
        self.current_level = None
        self.prev_level = None
        self.lod_levels = []

        self.last_lod_threshold = 0.3
        self.lod_zoom_step = 0

    def update_viewport(self, viewport):
        self.viewport = viewport

    def load_window_zoom_values(self, min_zoom, max_zoom, depth):
        self.min_possible_zoom = min_zoom
        self.max_possible_zoom = max_zoom
        self.lod_zoom_step = ((self.max_possible_zoom * self.last_lod_threshold) - self.min_possible_zoom) / depth

    def generate_levels(self, n_net, depth):
        depth = 6
        print("Create levels of detail, depth:",depth)
        start_time = time.time()
        for i in range(depth):
            is_even = i % 2 == 0
            material_id = 2 if is_even else 1
            if i <2:
                lod = Lod.build_static_texture_level(n_net, i, depth, material_id=material_id)
            else:
                lod = Lod.build_mega_leaf_texture_level(i, material_id=material_id)
            self.lod_levels.append(lod)
        last_lod = Lod.build_leafs_vertices_level(depth)
        last_lod.texture = self.lod_levels[-1].texture
        self.lod_levels.append(last_lod)
        last_lod = Lod.build_leafs_vertices_level(depth)
        last_lod.texture = self.lod_levels[-1].texture
        self.lod_levels.append(last_lod)

        print("Generated levels of details", time.time() - start_time, len(self.lod_levels))
        for index, l in enumerate(self.lod_levels):
            print("Level of detail: ", index, l.level, l.lod_type, l.factor, "material: ", l.material_id, "factor",
                  l.factor)

    def load_current_level(self):
        x, y, w, h, zoom = self.viewport
        zoom_norm = zoom - self.min_possible_zoom
        lod_index = int((zoom_norm) / self.lod_zoom_step)
        lod_index = min([self.max_lod_level, lod_index])

        if lod_index >= len(self.lod_levels):
            return
        if self.current_level is not None and self.current_level.level == lod_index:
            return

        self.prev_level = self.current_level
        self.current_level = self.lod_levels[lod_index]
        self.current_level.print()

    def draw_lod_vertices(self, n_net, n_tree):
        if self.current_level.lod_type == LodType.LEAFS_VERTICES:
            n_tree.draw_leafs_vertices()
        if self.current_level.lod_type == LodType.MEGA_LEAF_VERTICES:
            n_tree.draw_mega_leaf_vertices()

    def draw_lod_textures(self, n_net, n_tree, n_material_one_shader, n_material_two_shader):
        if self.current_level.material_id == 1:
            n_material_one_shader.use()
            n_material_one_shader.update_fading_factor(1.0)
            self.current_level.draw_textures(n_tree)
        elif self.current_level.material_id == 2:
            n_material_two_shader.use()
            n_material_two_shader.update_fading_factor(1.0)
            self.current_level.draw_textures(n_tree)

        if self.prev_level is not None:
            offset = self.get_offset_from_previous_level()
            if self.prev_level.material_id == 1:
                n_material_one_shader.use()
                n_material_one_shader.update_fading_factor(1.0-offset)
                self.prev_level.draw_textures(n_tree)
            elif self.prev_level.material_id == 2:
                n_material_two_shader.use()
                n_material_two_shader.update_fading_factor(1.0 - offset)
                self.prev_level.draw_textures(n_tree)

    def get_offset_from_previous_level(self):
        x, y, w, h, zoom = self.viewport
        # fade out textures in the second half of pod
        start_level = self.lod_zoom_step * self.current_level.level
        end_level = start_level + self.lod_zoom_step
        norm_zoom = zoom - self.min_possible_zoom
        # print(start_level, end_level)
        if norm_zoom < start_level:
            offset = 0
        elif norm_zoom > end_level:
            offset = 1
        else:
            offset = (norm_zoom - start_level) / (end_level - start_level)

        if self.prev_level is not None:
            if self.prev_level.level > self.current_level.level:
                offset = 1 - offset
        return offset
