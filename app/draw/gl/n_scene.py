import gc
import math

from memory_profiler import profile

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

        self.refresh_on_factor_change = True
        self.use_mega_leaf = True

    def _create_layer_entity(self, layer):
        x1 = layer.column_offset * self.n_net.node_gap_x
        y1 = layer.row_offset * self.n_net.node_gap_y
        x2 = x1 + layer.columns_count * self.n_net.node_gap_x
        y2 = y1 + layer.rows_count * self.n_net.node_gap_y
        entity = NEntity(x1, y1, x2, y2)
        entity.data = layer.layer_grid
        return entity

    # @profile
    def update_scene_entities(self):
        # Visible layers entities
        # visible_layers = {obj.id: obj for obj in self.n_net.visible_layers}
        # visible_layers_keys = set(visible_layers.keys())
        # visible_current_layers_keys = set(self.visible_layers_entities.keys())
        # visible_keys_to_remove = visible_current_layers_keys - visible_layers_keys
        # visible_keys_to_add = visible_layers_keys - visible_current_layers_keys
        #
        # for key in visible_keys_to_remove:
        #     entity =  self.visible_layers_entities[key]
        #     self.visible_layers_entities[key] = None
        #     del self.visible_layers_entities[key]
        #     entity.destroy()
        #
        # for key in visible_keys_to_add:
        #     layer = visible_layers[key]
        #     entity = self._create_layer_entity(layer)
        #     self.visible_layers_entities[key] = entity

        # Tree leaves entities
        if self.use_mega_leaf:
            if self.n_tree.mega_leaf is not None:
                leaves = {self.n_tree.mega_leaf.id: self.n_tree.mega_leaf}
            else:
                leaves = {}
        else:
            leaves = {obj.id: obj for obj in self.n_tree.visible_leaves}
        current_keys = set(self.visible_entities.keys())
        new_keys = set(leaves.keys())

        keys_to_remove = current_keys - new_keys
        keys_to_add = new_keys - current_keys

        # Perform removal
        for key in keys_to_remove:
            # self.cache[key] = self.visible_entities[key]
            self.visible_entities[key].destroy()
            # print("removing old leaf", keys_to_remove, keys_to_add)
            self.visible_entities[key] = None
            del self.visible_entities[key]

        # Perform addition
        for key in keys_to_add:
            leaf = leaves[key]
            entity = NEntity(leaf.x1, leaf.y1, leaf.x2, leaf.y2)
            # print("adding new leaf")
            self.visible_entities[key] = entity

    def get_details_factor(self):
        """
        Calculate the down sample factor
        Factor is used to reduce the vertices count or texture quality
        If the zoom level is very small large amount of data could be visible inside small screen bounds
        Rendering everything without down sampling will cause issues and very low fps

        Compare screen space bounds with world grid bounds
        """
        x, y, w, h, zoom = self.n_window.viewport_to_world_cords()
        col_min, row_min, col_max, row_max = self.n_net.world_to_grid_position(
            x, y, x + w, y + h)
        subgrid_width = col_max - col_min
        subgrid_height = row_max - row_min

        target_width = self.n_window.width
        target_height = self.n_window.height

        width_factor = max(math.ceil(subgrid_width / target_width), 1)
        height_factor = max(math.ceil(subgrid_height / target_height), 1)
        # print("dw", subgrid_width, "dh", subgrid_height)
        # print("tw", target_width, "th", target_height)
        # print("factor", width_factor, height_factor)
        # # print("total count", int(target_height*target_width))
        return min(width_factor, height_factor)

    # def get_entity_details_factor(self, entity):
    #     """
    #     Calculate the down sample factor
    #     Factor is used to reduce the entity vertices count or texture quality
    #     If the zoom level is very small large amount of data could be packed inside small visible bounds
    #     Rendering everything without down sampling will cause issues and very low fps
    #
    #     Compare screen space bounds with world grid bounds
    #     """
    #     (sx1, sy1, sx2, sy2, zoom_factor) = self.n_window.world_coords_to_screen_coords(
    #         entity.x1,
    #         entity.y1,
    #         entity.x2,
    #         entity.y2)
    #     (wx1, wy1, wx2, wy2) = self.n_window.screen_coords_to_window_coords(sx1, sy1, sx2, sy2)
    #     target_width = wx2 - wx1
    #     target_height = wy2 - wy1
    #     col_min, row_min, col_max, row_max = self.n_net.world_to_grid_position(
    #         entity.x1,
    #         entity.y1,
    #         entity.x2,
    #         entity.y2)
    #     subgrid_width = col_max - col_min
    #     subgrid_height = row_max - row_min
    #
    #     capped_width = min(target_height, self.n_window.width)
    #     capped_height = min(target_height, self.n_window.height)
    #
    #     width_factor = max(int(subgrid_width / capped_width), 1)
    #     height_factor = max(int(subgrid_height / capped_height), 1)
    #     # print("dw",subgrid_width, "tw",target_width)
    #     # print("dh",subgrid_height, "th",target_height)
    #     # print("total count", int(target_height*target_width))
    #     return min(width_factor, height_factor)

    #### Lazy loading entities #####

    def is_entity_attached(self, lod_type, entity, factor):
        return entity.is_attached(lod_type, factor, self.refresh_on_factor_change)

    def attach_leaf_static_texture(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return

        chunks, dimensions = self.n_net.get_subgrid_chunks_screen_dimensions(entity.x1, entity.y1, entity.x2,
                                                                             entity.y2, details_factor)
        images_data = []
        for c in chunks:
            images_data.append(self.textures_factory.get_texture_data(c, details_factor))
        entity.create_textures_from_images(images_data, dimensions, details_factor, lod_type)

    def attach_layer_static_texture(self, entity, data, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        img_data, img_width, img_height = self.textures_factory.get_texture_data(data, details_factor)
        entity.create_texture_from_image(img_data, img_width, img_height, details_factor, lod_type)

    def attach_leaf_color_map_texture(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        if details_factor <= 2:
            chunks, dimensions = self.n_net.get_subgrid_chunks_screen_dimensions(
                entity.x1, entity.y1,
                entity.x2, entity.y2,
                details_factor)
            entity.create_color_map_texture(chunks, dimensions, details_factor, lod_type)
        else:
            chunks, dimensions, width, height = self.n_net.get_subgrid_chunks_grid_dimensions(
                entity.x1, entity.y1,
                entity.x2, entity.y2,
                details_factor)
            entity.create_colormap_texture_from_textures(
                width, height, chunks, dimensions,
                details_factor,
                lod_type)

    def attach_nodes_instances_from_texture(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        chunks, dimensions, width, height = self.n_net.get_subgrid_chunks_grid_dimensions(
            entity.x1, entity.y1,
            entity.x2,
            entity.y2,
            details_factor)
        entity.create_nodes_view_from_texture_data(
            width, height, chunks, dimensions,
            details_factor, lod_type)

    def attach_points_instances_from_texture(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        chunks, dimensions, width, height = self.n_net.get_subgrid_chunks_grid_dimensions(
            entity.x1, entity.y1,
            entity.x2, entity.y2,
            details_factor)
        entity.create_points_view_from_texture_data(
            width, height,
            chunks, dimensions,
            details_factor, lod_type)

    def attach_leaf_points_view(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        positions_and_colors = self.n_net.get_positions_and_values_array(
            entity.x1, entity.y1,
            entity.x2, entity.y2,
            details_factor)
        entity.create_points_view(positions_and_colors, details_factor, lod_type)

    def attach_leaf_nodes_view(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        positions_and_colors = self.n_net.get_positions_and_values_array(
            entity.x1, entity.y1,
            entity.x2, entity.y2,
            details_factor)
        entity.create_nodes_view(positions_and_colors, details_factor, lod_type)

    def attach_leaf_nodes_view_to_texture(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        entity.create_texture_from_nodes_view(self.n_window, details_factor, lod_type)

    def attach_leaf_points_view_to_texture(self, entity, lod_type):
        details_factor = self.get_details_factor()
        if self.is_entity_attached(lod_type, entity, details_factor):
            return
        entity.create_texture_from_points_view(self.n_window, details_factor, lod_type)

    ####### Drawing #######
    def draw_scene(self,
                   n_points_shader,
                   n_static_texture_shader,
                   n_colors_texture_shader,
                   n_color_map_v2_texture_shader,
                   n_instances_from_buffer_shader,
                   n_instances_from_texture_shader
                   ):

        # Update
        self.update_scene_entities()

        # Draw
        lod = self.n_lod.current_level

        if lod.lod_type == LodType.LEAVES_NODES:
            n_instances_from_buffer_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_nodes_view(entity, lod.lod_type)
                entity.draw_nodes()
        if lod.lod_type == LodType.LEAVES_POINTS:
            n_points_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_points_view(entity, lod.lod_type)
                entity.draw_points()
        if lod.lod_type == LodType.LEAVES_STATIC_TEXTURES:
            n_static_texture_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_static_texture(entity, lod.lod_type)
                entity.draw_texture()
        if lod.lod_type == LodType.LAYERS_STATIC_TEXTURES:
            n_static_texture_shader.use()
            for key, entity in self.visible_layers_entities.items():
                self.attach_layer_static_texture(entity, entity.data, lod.lod_type)
                entity.draw_texture()
        if lod.lod_type == LodType.LEAVES_COLOR_MAP_TEXTURE:
            n_colors_texture_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_color_map_texture(entity, lod.lod_type)
                entity.draw_texture()
        if lod.lod_type == LodType.LEAVES_NODES_TO_TEXTURE:
            n_instances_from_buffer_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_nodes_view(entity, lod.lod_type)
                self.attach_leaf_nodes_view_to_texture(entity, lod.lod_type)
            n_static_texture_shader.use()
            for key, entity in self.visible_entities.items():
                entity.draw_texture()
        if lod.lod_type == LodType.LEAVES_POINTS_TO_TEXTURE:
            n_points_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_leaf_points_view(entity, lod.lod_type)
                self.attach_leaf_points_view_to_texture(entity, lod.lod_type)
            n_static_texture_shader.use()
            for key, entity in self.visible_entities.items():
                entity.draw_texture()
        if lod.lod_type == LodType.LEAVES_NODES_FROM_TEXTURE:
            n_instances_from_texture_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_nodes_instances_from_texture(entity, lod.lod_type)
                n_instances_from_texture_shader.update_texture_width(entity.data_width)
                n_instances_from_texture_shader.update_position_offset(entity.x1, entity.y1)
                n_instances_from_texture_shader.update_details_factor(entity.attached_details_factor)
                entity.draw_nodes_from_texture()
        if lod.lod_type == LodType.LEAVES_POINTS_FROM_TEXTURE:
            n_instances_from_texture_shader.use()
            for key, entity in self.visible_entities.items():
                self.attach_points_instances_from_texture(entity, lod.lod_type)
                n_instances_from_texture_shader.update_texture_width(entity.data_width)
                n_instances_from_texture_shader.update_position_offset(entity.x1, entity.y1)
                n_instances_from_texture_shader.update_details_factor(entity.attached_details_factor)
                entity.draw_points_from_texture()

    # @profile
    def draw_debug_tree(self):
        self.update_scene_entities()
        for key, entity in self.visible_entities.items():
            entity.create_background_view()
            entity.draw_leaf_background()
