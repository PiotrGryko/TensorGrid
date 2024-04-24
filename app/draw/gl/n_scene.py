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
        self.mega_entity = {}
        self.all_layers_entities = {}
        self.visible_layers_entities = {}

    def _create_layer_entity(self, layer):
        x1 = layer.column_offset * self.n_net.node_gap_x
        y1 = layer.row_offset * self.n_net.node_gap_y
        x2 = x1 + layer.columns_count * self.n_net.node_gap_x
        y2 = y1 + layer.rows_count * self.n_net.node_gap_y
        entity = NEntity(x1, y1, x2, y2)
        entity.data = layer.layer_grid
        return entity

    def update_scene_entities(self):
        # All layers entities
        all_layers = {obj.id: obj for obj in self.n_net.layers}
        all_layers_keys = set(all_layers.keys())
        all_current_layers_keys = set(self.all_layers_entities.keys())
        all_keys_to_remove = all_current_layers_keys - all_layers_keys
        all_keys_to_add = all_layers_keys - all_current_layers_keys

        for key in all_keys_to_remove:
            del self.all_layers_entities[key]
        for key in all_keys_to_add:
            layer = all_layers[key]
            entity = self._create_layer_entity(layer)
            self.all_layers_entities[key] = entity

        # Visible layers entities
        visible_layers = {obj.id: obj for obj in self.n_net.visible_layers}
        visible_layers_keys = set(visible_layers.keys())
        visible_current_layers_keys = set(self.visible_layers_entities.keys())
        visible_keys_to_remove = visible_current_layers_keys - visible_layers_keys
        visible_keys_to_add = visible_layers_keys - visible_current_layers_keys

        for key in visible_keys_to_remove:
            del self.visible_layers_entities[key]
        for key in visible_keys_to_add:
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
            del self.visible_entities[key]

        # Perform addition
        for key in keys_to_add:
            leaf = leaves[key]
            entity = NEntity(leaf.x1, leaf.y1, leaf.x2, leaf.y2)
            self.visible_entities[key] = entity

        if self.n_tree.mega_leaf is not None:
            leaf = self.n_tree.mega_leaf
            leaf_id = leaf.id
            if leaf_id not in self.mega_entity:
                self.mega_entity = {leaf_id: NEntity(leaf.x1, leaf.y1, leaf.x2, leaf.y2)}

    def draw_vertices_level(self, lod):
        if lod.lod_type == LodType.LEAFS_VERTICES:
            for key, entity in self.visible_entities.items():
                entity.draw_vertices(self.n_net, lod.texture_factor)
        if lod.lod_type == LodType.LEAFS_VERTICES_TO_TEXTURE:
            for key, entity in self.visible_entities.items():
                entity.create_fbo_texture(self.n_net, self.n_window, lod.material_id, lod.texture_factor)
        if lod.lod_type == LodType.MEGA_LEAF_VERTICES:
            for key, entity in self.mega_entity.items():
                entity.draw_vertices(self.n_net, lod.texture_factor)
        if lod.lod_type == LodType.MEGA_LEAF_VERTICES_TO_TEXTURE:
            for key, entity in self.mega_entity.items():
                entity.create_fbo_texture(self.n_net, self.n_window, lod.material_id, lod.texture_factor)

    def draw_texture_level(self, lod):
        if lod.lod_type == LodType.STATIC_TEXTURE:
            lod.texture.draw()
        if lod.lod_type == LodType.LEAFS_TEXTURES:
            for key, entity in self.visible_entities.items():
                subgrid = self.n_net.get_subgrid(entity.x1, entity.y1, entity.x2, entity.y2)
                entity.draw_texture(subgrid, self.textures_factory, lod.material_id, lod.texture_factor)
        if lod.lod_type == LodType.LEAFS_VERTICES_TO_TEXTURE:
            for key, entity in self.visible_entities.items():
                entity.draw_fbo_texture(lod.material_id)
        if lod.lod_type == LodType.MEGA_LEAF_TEXTURE:
            for key, entity in self.mega_entity.items():
                subgrid = self.n_net.get_subgrid(entity.x1, entity.y1, entity.x2, entity.y2)
                entity.draw_texture(subgrid, self.textures_factory, lod.material_id, lod.texture_factor)
        if lod.lod_type == LodType.MEGA_LEAF_VERTICES_TO_TEXTURE:
            for key, entity in self.mega_entity.items():
                entity.draw_fbo_texture(lod.material_id)
        if lod.lod_type == LodType.ALL_LAYERS_TEXTURES:
            for key, entity in self.all_layers_entities.items():
                entity.draw_texture(entity.data, self.textures_factory, lod.material_id, lod.texture_factor)
        if lod.lod_type == LodType.VISIBLE_LAYERS_TEXTURES:
            for key, entity in self.visible_layers_entities.items():
                entity.draw_texture(entity.data, self.textures_factory, lod.material_id, lod.texture_factor)

    def draw_scene(self, n_vertices_shader, n_material_one_shader, n_material_two_shader):

        self.update_scene_entities()
        # draw vertices
        n_vertices_shader.use()
        current_level = self.n_lod.current_level
        prev_level = self.n_lod.prev_level
        self.draw_vertices_level(current_level)

        # draw textures
        if current_level.material_id == 1:
            n_material_one_shader.use()
            n_material_one_shader.update_fading_factor(1.0)
            self.draw_texture_level(current_level)
        elif current_level.material_id == 2:
            n_material_two_shader.use()
            n_material_two_shader.update_fading_factor(1.0)
            self.draw_texture_level(current_level)

        if prev_level is not None:
            offset = self.n_lod.get_offset_from_previous_level()
            if prev_level.material_id == 1:
                n_material_one_shader.use()
                n_material_one_shader.update_fading_factor(offset)
                self.draw_texture_level(prev_level)
            elif prev_level.material_id == 2:
                n_material_two_shader.use()
                n_material_two_shader.update_fading_factor(offset)
                self.draw_texture_level(prev_level)

    def draw_debug_tree(self):
        self.update_scene_entities()
        for key, entity in self.visible_entities.items():
            entity.draw_leaf_background()
