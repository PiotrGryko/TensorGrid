from enum import Enum

from app.draw.gl.draw.n_texture import NTexture


class LodType(Enum):
    STATIC_TEXTURE = 1
    LEAFS_VERTICES = 2
    LEAFS_TEXTURES = 3,
    MEGA_LEAF_VERTICES = 4
    MEGA_LEAF_TEXTURE = 5,
    LEAFS_VERTICES_TO_TEXTURE = 6,
    MEGA_LEAF_VERTICES_TO_TEXTURE = 7
    ALL_LAYERS_TEXTURES = 8
    VISIBLE_LAYERS_TEXTURES = 9


class Lod:
    def __init__(self, level, lod_type):
        self.level = level
        self.lod_type = lod_type
        self.texture = None
        self.material_id = None
        self.texture_factor = 1
        self.level_zoom_start = 0  # Min zoom value at which this level should be visible

    def dump(self):
        return f"level: {self.level}, lod_type: {self.lod_type}, material_id: {self.material_id}, zoom start {self.level_zoom_start}, factor {self.texture_factor}"


class NLvlOfDetails:
    def __init__(self, n_net, n_window):
        self.n_net = n_net
        self.n_window = n_window
        self.textures = []
        self.max_lod_level = 10  # generate up to N different level of details
        self.viewport = None
        self.current_level = None
        self.prev_level = None
        self.next_level = None
        self.lod_levels = []

        self.last_lod_threshold = 0.1
        self.lod_zoom_step = 0
        self.msg = None

    def dump(self):
        for lvl in self.lod_levels:
            print(lvl.dump())

    def update_viewport(self, viewport):
        self.viewport = viewport

    def get_material_for_level(self, level):
        is_even = level % 2 == 0
        material_id = 2 if is_even else 1
        return material_id

    def add_level(self,
                  lod_type,
                  zoom_percent,
                  img_data=None,
                  img_width=None,
                  img_height=None,
                  file_path=None,
                  texture_factor=None
                  ):
        level = len(self.lod_levels)
        material_id = self.get_material_for_level(level)
        total_width = self.n_net.total_width
        total_height = self.n_net.total_height
        level_zoom_start = self.n_window.min_zoom + (self.n_window.max_zoom - self.n_window.min_zoom) * zoom_percent

        if lod_type == LodType.STATIC_TEXTURE:
            lod = Lod(level, LodType.STATIC_TEXTURE)
            lod.texture = NTexture()
            lod.material_id = material_id
            if file_path is not None:
                lod.texture.create_from_file(0, 0, total_width, total_height, file_path, material_id=material_id)
            else:
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
        elif lod_type == LodType.LEAFS_VERTICES_TO_TEXTURE:
            lod = Lod(level, LodType.LEAFS_VERTICES_TO_TEXTURE)
            lod.material_id = material_id
        elif lod_type == LodType.MEGA_LEAF_VERTICES_TO_TEXTURE:
            lod = Lod(level, LodType.MEGA_LEAF_VERTICES_TO_TEXTURE)
            lod.material_id = material_id
        elif lod_type == LodType.ALL_LAYERS_TEXTURES:
            lod = Lod(level, LodType.ALL_LAYERS_TEXTURES)
            lod.material_id = material_id
        elif lod_type == LodType.VISIBLE_LAYERS_TEXTURES:
            lod = Lod(level, LodType.VISIBLE_LAYERS_TEXTURES)
            lod.material_id = material_id
        else:
            raise f"Unsupported lod type: {lod_type}"
        if texture_factor is not None:
            lod.texture_factor = texture_factor
        lod.level_zoom_start = level_zoom_start
        self.lod_levels.append(lod)
        self.lod_zoom_step = ((self.n_window.max_zoom * self.last_lod_threshold) - self.n_window.min_zoom) / len(
            self.lod_levels)

    def load_current_level(self):
        x, y, w, h, zoom = self.viewport
        zoom_norm = zoom - self.n_window.min_zoom
        lod_index = None
        for index, l in enumerate(self.lod_levels):
            if zoom_norm < l.level_zoom_start:
                lod_index = index - 1 if index > 0 else 0
                break
        if lod_index is None:
            lod_index = len(self.lod_levels) - 1

        if lod_index >= len(self.lod_levels):
            return
        if self.current_level is not None and self.current_level.level == lod_index:
            return

        self.current_level = self.lod_levels[lod_index]
        if lod_index > 0:
            self.prev_level = self.lod_levels[lod_index - 1]
        else:
            self.prev_level = None
        if lod_index < len(self.lod_levels) - 1:
            self.next_level = self.lod_levels[lod_index + 1]
        else:
            self.next_level = None

        print(self.current_level.dump())

    def get_offset_from_previous_level(self):
        x, y, w, h, zoom = self.viewport

        start_level = self.current_level.level_zoom_start
        end_level = start_level + 0.1 if self.next_level is None else self.next_level.level_zoom_start
        end_level = start_level + (end_level - start_level)

        norm_zoom = zoom - self.n_window.min_zoom

        if norm_zoom < start_level:
            offset = 0
        elif norm_zoom > end_level:
            offset = 1
        else:
            offset = (norm_zoom - start_level) / (end_level - start_level)
        return 1 - offset
