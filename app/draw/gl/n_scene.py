from app.draw.gl.draw.n_entity import NEntity
from app.draw.gl.n_lod import LodType


class NScene:

    def __init__(self, n_lod, n_tree, n_net, n_window, textures_factory):
        self.n_lod = n_lod
        self.n_tree = n_tree
        self.n_net = n_net
        self.n_window = n_window
        self.textures_factory = textures_factory

        self.visible_entities = {}
        self.visible_layers_entities = {}
        self.cache = {}

    def _create_layer_entity(self, layer):
        x1 = layer.column_offset * self.n_net.node_gap_x
        y1 = layer.row_offset * self.n_net.node_gap_y
        x2 = x1 + layer.columns_count * self.n_net.node_gap_x
        y2 = y1 + layer.rows_count * self.n_net.node_gap_y
        entity = NEntity(x1, y1, x2, y2)
        entity.data = layer.layer_grid
        return entity

    def update_scene_entities(self):
        # Visible layers entities
        visible_layers = {obj.id: obj for obj in self.n_net.visible_layers}
        visible_layers_keys = set(visible_layers.keys())
        visible_current_layers_keys = set(self.visible_layers_entities.keys())
        visible_keys_to_remove = visible_current_layers_keys - visible_layers_keys
        visible_keys_to_add = visible_layers_keys - visible_current_layers_keys

        for key in visible_keys_to_remove:
            self.cache[key] = self.visible_layers_entities[key]
            del self.visible_layers_entities[key]
        for key in visible_keys_to_add:
            if key in self.cache:
                self.visible_layers_entities[key] = self.cache[key]
            else:
                layer = visible_layers[key]
                entity = self._create_layer_entity(layer)
                self.visible_layers_entities[key] = entity

        # Tree leaves entities
        leaves = {obj.id: obj for obj in self.n_tree.visible_leafs}
        current_keys = set(self.visible_entities.keys())
        new_keys = set(leaves.keys())

        keys_to_remove = current_keys - new_keys
        keys_to_add = new_keys - current_keys

        # Perform removal
        for key in keys_to_remove:
            self.cache[key] = self.visible_entities[key]
            del self.visible_entities[key]

        # Perform addition
        for key in keys_to_add:
            if key in self.cache:
                self.visible_entities[key] = self.cache[key]
            else:
                leaf = leaves[key]
                entity = NEntity(leaf.x1, leaf.y1, leaf.x2, leaf.y2)
                self.visible_entities[key] = entity

    def get_entity_details_factor(self, entity):
        (sx1, sy1, sx2, sy2, zoom_factor) = self.n_window.world_coords_to_screen_coords(entity.x1, entity.y1,
                                                                                        entity.x2, entity.y2)
        (wx1, wy1, wx2, wy2) = self.n_window.screen_coords_to_window_coords(sx1, sy1, sx2, sy2)
        target_width = wx2 - wx1
        target_height = wy2 - wy1
        col_min, row_min, col_max, row_max = self.n_net.world_to_grid_position(entity.x1,
                                                                               entity.y1,
                                                                               entity.x2,
                                                                               entity.y2)
        subgrid_width = col_max - col_min
        subgrid_height = row_max - row_min

        width_factor = max(int(subgrid_width / target_width), 1)
        height_factor = max(int(subgrid_height / target_height), 1)

        return max(width_factor, height_factor)

    def attach_leaf_texture(self, entity, lod):
        if not entity.has_texture_attached(lod.material_id, lod.texture_factor):
            details_factor = self.get_entity_details_factor(entity)
            subgrid = self.n_net.get_subgrid(entity.x1, entity.y1, entity.x2, entity.y2, details_factor)
            img_data, img_width, img_height = self.textures_factory.get_texture_data(subgrid, lod.texture_factor,
                                                                                     details_factor)
            entity.create_texture(img_data, img_width, img_height, lod.material_id, lod.texture_factor)

    def attach_layer_texture(self, entity, lod, data):
        if not entity.has_texture_attached(lod.material_id, lod.texture_factor):
            details_factor = self.get_entity_details_factor(entity)
            img_data, img_width, img_height = self.textures_factory.get_texture_data(data, lod.texture_factor,
                                                                                     details_factor)
            entity.create_texture(img_data, img_width, img_height, lod.material_id, lod.texture_factor)

    def attach_leaf_colors_grid(self, entity):
        if not entity.colors_attached:
            details_factor = self.get_entity_details_factor(entity)
            positions_and_colors = self.n_net.get_positions_and_values_array(entity.x1, entity.y1, entity.x2, entity.y2,
                                                                             details_factor)
            entity.create_colors_grid(positions_and_colors)

    def attach_leaf_colors_grid_texture(self, entity, lod):
        if not entity.has_texture_attached(lod.material_id, lod.texture_factor):
            details_factor = self.get_entity_details_factor(entity)
            subgrid = self.n_net.get_subgrid(entity.x1, entity.y1, entity.x2, entity.y2, details_factor)
            entity.create_colors_grid_texture(subgrid, lod.material_id, lod.texture_factor)

    def attach_leaf_colors_grid_texture_from_data_chunk(self, entity, lod):
        details_factor = self.get_entity_details_factor(entity)
        if not entity.has_texture_attached(lod.material_id, lod.texture_factor):

            chunks, dimensions = self.n_net.get_subgrid_chunks(entity.x1, entity.y1, entity.x2,
                                                               entity.y2, details_factor)
            entity.create_colors_grid_texture_from_chunks(chunks, dimensions, lod.material_id,
                                                          lod.texture_factor)

    def attach_leaf_nodes_view(self, entity):
        if not entity.nodes_attached:
            details_factor = self.get_entity_details_factor(entity)
            positions_and_colors = self.n_net.get_positions_and_values_array(entity.x1, entity.y1, entity.x2, entity.y2,
                                                                             details_factor)
            entity.create_nodes_view(positions_and_colors)

    def draw_vertices_level(self,
                            lod,
                            n_vertices_shader,
                            n_colors_shader):
        if lod.lod_type == LodType.LEAFS_NODES:
            n_vertices_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_nodes_view(entity)
                entity.draw_nodes()
        if lod.lod_type == LodType.LEAFS_NODES_TO_TEXTURE:
            n_vertices_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_nodes_view(entity)
                entity.create_texture_from_nodes_view(self.n_window, lod.material_id, lod.texture_factor)
        if lod.lod_type == LodType.LEAFS_COLORS:
            n_colors_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_colors_grid(entity)
                entity.draw_colors_grid()
        if lod.lod_type == LodType.LEAFS_COLORS_TO_TEXTURE:
            n_colors_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_colors_grid(entity)
                entity.create_texture_from_color_grid_view(self.n_window, lod.material_id, lod.texture_factor)

    def draw_texture_level(self,
                           lod,
                           n_material_shader,
                           n_colors_texture_material_shader
                           ):
        if lod.lod_type == LodType.STATIC_TEXTURE:
            n_material_shader.use()
            lod.texture.draw()
        if lod.lod_type == LodType.LEAFS_TEXTURES:
            n_material_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_texture(entity, lod)
                entity.draw_texture(lod.material_id)
        if lod.lod_type == LodType.LEAFS_NODES_TO_TEXTURE:
            n_material_shader.use()
            for key, entity in self.visible_entities.items():
                entity.draw_texture(lod.material_id)
        if lod.lod_type == LodType.LEAFS_COLORS_TO_TEXTURE:
            n_material_shader.use()
            for key, entity in self.visible_entities.items():
                entity.draw_texture(lod.material_id)
        if lod.lod_type == LodType.VISIBLE_LAYERS_TEXTURES:
            n_material_shader.use()
            for key, entity in self.visible_layers_entities.items():
                self.attach_layer_texture(entity, lod, entity.data)
                entity.draw_texture(lod.material_id)
        if lod.lod_type == LodType.LEAFS_COLORS_TEXTURE:
            n_colors_texture_material_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_colors_grid_texture(entity, lod)
                entity.draw_texture(lod.material_id)
        if lod.lod_type == LodType.LEAFS_COLORS_TEXTURE_FROM_CHUNKS:
            n_colors_texture_material_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_colors_grid_texture_from_data_chunk(entity, lod)
                entity.draw_texture(lod.material_id)

    def draw_scene(self,
                   n_vertices_shader,
                   n_colors_shader,
                   n_material_one_shader,
                   n_material_two_shader,
                   n_colors_texture_material_one_shader,
                   n_colors_texture_material_two_shader):

        self.update_scene_entities()
        # draw vertices

        current_level = self.n_lod.current_level
        prev_level = self.n_lod.prev_level
        self.draw_vertices_level(current_level, n_vertices_shader, n_colors_shader)

        # draw textures
        if current_level.material_id == 1:
            n_material_one_shader.use()
            n_material_one_shader.update_fading_factor(1.0)
            self.draw_texture_level(current_level, n_material_one_shader, n_colors_texture_material_one_shader)
        elif current_level.material_id == 2:
            n_material_two_shader.use()
            n_material_two_shader.update_fading_factor(1.0)
            self.draw_texture_level(current_level, n_material_two_shader, n_colors_texture_material_two_shader)

        # if prev_level is not None:
        #     offset = self.n_lod.get_offset_from_previous_level()
        #     if prev_level.material_id == 1:
        #         n_material_one_shader.use()
        #         n_material_one_shader.update_fading_factor(offset)
        #         self.draw_texture_level(prev_level)
        #     elif prev_level.material_id == 2:
        #         n_material_two_shader.use()
        #         n_material_two_shader.update_fading_factor(offset)
        #         self.draw_texture_level(prev_level)

    def draw_debug_tree(self):
        self.update_scene_entities()
        for key, entity in self.visible_entities.items():
            entity.draw_leaf_background()
