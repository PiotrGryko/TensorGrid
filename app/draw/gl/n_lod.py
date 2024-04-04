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
        self.material_id = None
        self.was_fully_scrolled = True

    def dump(self):
        return f"level: {self.level}, lod_type: {self.lod_type}, material_id: {self.material_id}"

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


class NLvlOfDetails:
    def __init__(self, n_net):
        self.n_net = n_net
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
        self.msg = None

    def dump(self):
        for lvl in self.lod_levels:
            print(lvl.dump())

    def update_viewport(self, viewport):
        self.viewport = viewport

    def load_window_zoom_values(self, min_zoom, max_zoom, depth):
        self.min_possible_zoom = min_zoom
        self.max_possible_zoom = max_zoom
        self.lod_zoom_step = ((self.max_possible_zoom * self.last_lod_threshold) - self.min_possible_zoom) / depth

    def get_material_for_level(self, level):
        is_even = level % 2 == 0
        material_id = 2 if is_even else 1
        return material_id

    def add_level(self,
                  lod_type,
                  img_data=None,
                  img_width=None,
                  img_height=None
                  ):
        level = len(self.lod_levels)
        material_id = self.get_material_for_level(level)
        total_width = self.n_net.total_width
        total_height = self.n_net.total_height

        if lod_type == LodType.STATIC_TEXTURE:
            lod = Lod(level, LodType.STATIC_TEXTURE)
            lod.texture = NTexture()
            lod.material_id = material_id
            lod.texture.create_from_data(0, 0, total_width, total_height, img_data, img_width, img_height,
                                         material_id=material_id)
        elif lod_type == LodType.LEAFS_TEXTURES:
            lod = Lod(level, LodType.LEAFS_TEXTURES)
            lod.material_id = material_id
        elif lod_type == LodType.MEGA_LEAF_TEXTURE:
            lod = Lod(level, LodType.MEGA_LEAF_TEXTURE)
            lod.material_id = material_id
        elif lod_type == LodType.LEAFS_VERTICES:
            lod = Lod(level, LodType.LEAFS_VERTICES)
        elif lod_type == LodType.MEGA_LEAF_VERTICES:
            lod = Lod(level, LodType.MEGA_LEAF_VERTICES)
        else:
            print("Unsupported lod type", lod_type)
            return
        self.lod_levels.append(lod)

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
        print(self.current_level.dump())

    def draw_lod_vertices(self, n_tree):
        if self.current_level.lod_type == LodType.LEAFS_VERTICES:
            n_tree.draw_leafs_vertices()
        if self.current_level.lod_type == LodType.MEGA_LEAF_VERTICES:
            n_tree.draw_mega_leaf_vertices()

    def draw_lod_textures(self, n_tree, n_material_one_shader, n_material_two_shader):
        #
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
            self.current_level.was_fully_scrolled = round(offset) == 1
            if not self.prev_level.was_fully_scrolled:
                return
            if self.prev_level.material_id == 1:
                n_material_one_shader.use()
                n_material_one_shader.update_fading_factor(1 - offset)
                self.prev_level.draw_textures(n_tree)
            elif self.prev_level.material_id == 2:
                n_material_two_shader.use()
                n_material_two_shader.update_fading_factor(1 - offset)
                self.prev_level.draw_textures(n_tree)

    def get_offset_from_previous_level(self):
        x, y, w, h, zoom = self.viewport
        start_level = self.lod_zoom_step * self.current_level.level
        end_level = start_level + self.lod_zoom_step
        norm_zoom = zoom - self.min_possible_zoom

        if norm_zoom < start_level:
            offset = 0
        elif norm_zoom > end_level:
            offset = 1
        else:
            offset = (norm_zoom - start_level) / (end_level - start_level)

        if self.prev_level.level > self.current_level.level:
            offset = 1 - offset
        return offset
