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

    def add_second_texture(self, lod):
        if self.texture is None:
            return
        if lod.texture is None:
            return
        self.texture.add_second_texture_from_object(lod.texture.material_one)

    def print(self):
        print(f"level: {self.level}, lod_type: {self.lod_type}")

    @staticmethod
    def build_static_texture_level(n_net, level, max_depth):
        total_width = n_net.total_width
        total_height = n_net.total_height
        factor = max([max_depth - level, 1])
        factor = 1 if factor == 1 else factor
        lod = Lod(level, LodType.STATIC_TEXTURE)
        img_data, img_width, img_height = n_net.get_texture2(0, 0, total_width, total_height, factor)
        lod.texture = NTexture()
        lod.texture.create_from_data(0, 0, total_width, total_height, img_data, img_width, img_height)
        lod.factor = factor
        return lod

    @staticmethod
    def build_leafs_texture_level(level):
        lod = Lod(level, LodType.LEAFS_TEXTURES)
        return lod

    @staticmethod
    def build_mega_leaf_texture_level(level):
        lod = Lod(level, LodType.MEGA_LEAF_TEXTURE)
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
        self.current_lod = 0
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
        print("Create levels of detail")
        start_time = time.time()
        for i in range(depth ):
            lod = Lod.build_static_texture_level(n_net, i, depth)
            self.lod_levels.append(lod)
        # self.lod_levels.append(Lod.build_leafs_texture_level(depth - 4))
        # self.lod_levels.append(Lod.build_leafs_texture_level(depth - 3))
        # self.lod_levels.append(Lod.build_leafs_texture_level(depth - 2))
        # self.lod_levels.append(Lod.build_leafs_texture_level(depth - 1))
        last_lod = Lod.build_mega_leaf_vertices_level(depth)
        last_lod.texture = self.lod_levels[-1].texture
        self.lod_levels.append(last_lod)
        for index, lod in enumerate(self.lod_levels):
            if index < len(self.lod_levels) - 1:
                next_lod = self.lod_levels[index + 1]
                lod.add_second_texture(next_lod)

        print("Generated levels of details", time.time() - start_time, len(self.lod_levels))
        for index, l in enumerate(self.lod_levels):
            print("Level of detail: ", index, l.level, l.lod_type, l.factor)

    def get_current_lod(self):
        x, y, w, h, zoom = self.viewport
        zoom_norm = zoom - self.min_possible_zoom
        lod = int((zoom_norm) / self.lod_zoom_step)
        lod = min([self.max_lod_level, lod])
        if self.current_lod != lod:
            self.current_lod = lod
            print("Current level of detail:")
            self.lod_levels[self.current_lod].print()
        return lod

    def draw_debug_tree(self, n_tree):
        visible_leafs = n_tree.visible_leafs
        for l in visible_leafs:
            if not l.background_attached:
                l.background_attached = True
                l.create_background_view()
            l.draw_leaf_background()

    def draw_lod_vertices(self, n_net, n_tree):
        lod = self.lod_levels[self.current_lod]
        visible_leafs = n_tree.visible_leafs
        mega_leaf = n_tree.mega_leaf
        if lod.lod_type == LodType.LEAFS_VERTICES:
            for l in visible_leafs:
                if not l.nodes_attached:
                    l.nodes_attached = True
                    l.create_nodes_view(n_net)
                l.draw_vertices()
        if lod.lod_type == LodType.MEGA_LEAF_VERTICES:
            if mega_leaf is not None:
                if not mega_leaf.nodes_attached:
                    mega_leaf.nodes_attached = True
                    mega_leaf.create_nodes_view(n_net)
                mega_leaf.draw_vertices()

    def draw_lod_textures(self, n_net, n_tree, n_texture_shader):
        self.fade_lod_texture_levels(n_texture_shader)
        lod = self.lod_levels[self.current_lod]
        visible_leafs = n_tree.visible_leafs
        mega_leaf = n_tree.mega_leaf
        if lod.texture is not None:
            lod.texture.draw()
        if lod.lod_type == LodType.LEAFS_TEXTURES:
            for l in visible_leafs:
                if not l.texture_attached:
                    l.texture_attached = True
                    l.create_texture(n_net)
                l.draw_texture()
        if lod.lod_type == LodType.MEGA_LEAF_TEXTURE:
            if mega_leaf is not None:
                if not mega_leaf.texture_attached:
                    mega_leaf.texture_attached = True
                    mega_leaf.create_texture(n_net, self.current_lod)
                    print("Created mega leaf texture", self.current_lod)
                mega_leaf.draw_texture()

    def fade_lod_texture_levels(self, n_texture_shader):
        x, y, w, h, zoom = self.viewport
        lod = self.lod_levels[self.current_lod]
        # fade out textures in the second half of pod
        fade_end = self.lod_zoom_step * lod.level + self.lod_zoom_step
        # fade start is end of the pod - step/2
        fade_start = fade_end - self.lod_zoom_step
        norm_zoom = zoom - self.min_possible_zoom
        if fade_start < norm_zoom < fade_end:
            step_delta = (norm_zoom - fade_start) / (fade_end - fade_start)
            n_texture_shader.update_fading_factor(step_delta)
        n_texture_shader.set_tex2_enabled(self.current_lod < len(self.lod_levels) - 1)
        # n_texture_shader.set_tex2_enabled(False)
