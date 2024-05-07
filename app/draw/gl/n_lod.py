from enum import Enum


class LodType(Enum):
    LAYERS_STATIC_TEXTURES = 1  # Visible layers using texture generator
    LEAVES_STATIC_TEXTURES = 2  # Visible BSP tree leafs data using texture generator
    LEAVES_COLOR_MAP_TEXTURE = 3  # Drawing FLOATS texture colored using color_map texture sampling
    LEAVES_POINTS = 4  # Drawing POINTS vertices colored using color_map texture sampling
    LEAVES_POINTS_TO_TEXTURE = 5  # Drawing LEAFS_COLORS to texture frame buffer
    LEAVES_POINTS_FROM_TEXTURE = 6  # Drawing colors vertices using instanced drawing and texture
    LEAVES_NODES = 7  # Drawing nodes vertices using instanced drawing
    LEAVES_NODES_TO_TEXTURE = 8  # Drawing LEAVES_NODES to texture frame buffer
    LEAVES_NODES_FROM_TEXTURE = 9  # Drawing nodes vertices using instanced drawing and texture

class Lod:
    def __init__(self, level, lod_type):
        self.level = level
        self.lod_type = lod_type
        self.texture = None
        self.level_zoom_start = 0  # Min zoom value at which this level should be visible

    def dump(self):
        return f"level: {self.level}, lod_type: {self.lod_type}, zoom start {self.level_zoom_start}"


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

    def add_level(self,
                  lod_type,
                  zoom_percent):
        level = len(self.lod_levels)
        level_zoom_start = self.n_window.min_zoom + (self.n_window.max_zoom - self.n_window.min_zoom) * zoom_percent
        if lod_type.value not in LodType._value2member_map_:
            raise f"Unsupported lod type: {lod_type}"

        lod = Lod(level, lod_type)
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
