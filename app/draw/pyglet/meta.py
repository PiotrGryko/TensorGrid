import pyglet
from pyglet.graphics import Group

from helper import *


class MetaContainer:
    def __init__(self):
        self.pyglet_nodes = {}
        self.pyglet_lines = {}
        self.nodes_meta = {}
        self.lines_meta = {}
        self.visible_lines_meta = {}
        self.parameters_map = {}
        self.net_batch = pyglet.graphics.Batch()
        self.net_group = Group(1)
        self.lines_group = Group(0)
        # :(
        self.hovered_node = None
        self.selected_node = None

    def on_draw(self):
        self.net_batch.draw()

    def update_parameters(self, parameters_map):
        self.parameters_map = parameters_map
        for k, pn in self.pyglet_nodes.items():
            meta = self.nodes_meta[k]
            meta.update_view(pn, self.parameters_map)
        for k, pl in self.pyglet_lines.items():
            meta = self.lines_meta[k]
            meta.update_view(pl, self.parameters_map)
        print(f"updated {len(parameters_map)} {len(self.pyglet_nodes)} {len(self.pyglet_lines)}")

    def add_line_meta(self, x, y, px, py, weight, prev_node, next_node):
        self.lines_meta[weight.label] = LineMeta(
            key=weight.label,
            x=x,
            y=y,
            px=px,
            py=py,
            color=generate_line_color(weight.grad),
            prev_node=prev_node.label,
            next_node=next_node.label
        )

    def add_node_meta2(self, key, x, y, weights, grads, bias, radius):
        self.nodes_meta[key] = NodeMeta(
            key=key,
            x=x,
            y=y,
            color=generate_node_color_w(weights),
            weights={index: w for index, w in enumerate(weights)},
            grads={index: g for index, g in enumerate(grads)},
            bias=bias,
            radius=radius
        )

    def add_node_meta(self, x, y, node, radius):
        self.nodes_meta[node.label] = NodeMeta(
            key=node.label,
            x=x,
            y=y,
            color=generate_node_color_w([w.data for w in node.weights]),
            weights={w.label: w.data for w in node.weights},
            grads={w.label: w.grad for w in node.weights},
            bias=node.bias.data,
            radius=radius
        )

    def update_node(self, meta):
        pn = self.pyglet_nodes.get(meta.key)
        if pn: meta.update_view(pn, self.parameters_map)

    def attach_lines(self, meta):
        for k, w in meta.weights.items():
            self.visible_lines_meta[k] = self.lines_meta[k]
            self.pyglet_lines[k] = self.lines_meta[k].create_background_view(
                self.parameters_map,
                self.net_batch,
                self.lines_group)

    def remove_lines(self, meta):
        for k, w in meta.weights.items():
            l = self.pyglet_lines.get(k)
            if l:
                l.delete()
                self.pyglet_lines.pop(k)
                self.visible_lines_meta.pop(k)

    def get_node_by_screen_x_y(self, x, y, world_camera):
        for key, meta in self.nodes_meta.items():
            hovered = meta.contains(x, y, world_camera)
            if hovered:
                return meta
        return None

    def on_long_click(self, x, y, world_camera):
        def pin(meta):
            meta.pinned = True
            self.update_node(meta)
            self.attach_lines(meta)

        def unpin(meta):
            meta.pinned = False
            self.update_node(meta)
            self.remove_lines(meta)

        m = self.get_node_by_screen_x_y(x, y, world_camera)
        if m and not m.selected:
            if not m.pinned:
                pin(m)
                return m
            else:
                unpin(m)
                return m
        return None

    def on_click(self, x, y, world_camera):
        def select(meta):
            meta.selected = True
            meta.pinned = False
            self.selected_node = meta
            self.update_node(meta)
            self.attach_lines(meta)

        def unselect(meta):
            meta.selected = False
            self.selected_node = None
            self.update_node(meta)
            self.remove_lines(meta)

        m = self.get_node_by_screen_x_y(x, y, world_camera)
        if m:
            if not m.selected:
                if self.selected_node and self.selected_node is not m:
                    unselect(self.selected_node)
                select(m)
                return m
            else:
                unselect(m)
                return m
        return None

    def on_hover(self, x, y, world_camera):
        m = self.get_node_by_screen_x_y(x, y, world_camera)
        if m and not self.hovered_node:
            m.hovered = True
            self.update_node(m)
            self.hovered_node = m
        if self.hovered_node and not m:
            self.hovered_node.hovered = False
            self.update_node(self.hovered_node)
            self.hovered_node = None

    def on_window_update(self, world_camera):
        # TO DO, Iterating over all nodes slows performance
        for k, m in self.nodes_meta.items():
            pn = self.pyglet_nodes.get(k)
            if m.is_visible(world_camera):
                if pn is None:
                    self.pyglet_nodes[k] = m.create_background_view(self.parameters_map, self.net_batch, self.net_group)
                else:
                    m.update_view(pn, self.parameters_map)
            elif pn:
                pn.delete()
                self.pyglet_nodes.pop(k)

        for k, m in self.visible_lines_meta.items():
            pl = self.pyglet_lines.get(k)
            if m.is_visible(world_camera):
                if pl is None:
                    self.pyglet_lines[k] = m.create_background_view(self.parameters_map, self.net_batch, self.net_group)
                else:
                    m.update_view(pl, self.parameters_map)
            elif pl:
                pl.delete()
                self.pyglet_lines.pop(k)


class Meta:
    def __init__(self,
                 key=None,
                 x=None,
                 y=None,
                 px=None,
                 py=None,
                 radius=None,
                 color=None,
                 bias=None,
                 prev_node=None,
                 next_node=None,
                 weights={},
                 grads={}
                 ):
        self.key = key
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.radius = radius
        self.color = color
        self.bias = bias
        self.prev_node = prev_node
        self.next_node = next_node
        self.weights = weights
        self.grads = grads
        self.selected = False
        self.pinned = False
        self.hovered = False
        self.updated_time = None

    def update(self, parameters_map):
        pass

    def create_view(self, parameters_map, batch, group):
        pass

    def update_view(self, pyglet_view, parameters_map):
        pass

    def is_visible(self, world_camera):
        pass

    def contains(self, x, y, word_camera):
        pass


class LineMeta(Meta):

    def update(self, parameters_map):
        p = parameters_map[self.key]
        self.color = generate_line_color(p.grad)

    def create_view(self, parameters_map, batch, group):
        self.update(parameters_map)
        return pyglet.shapes.Line(x=self.x,
                                  y=self.y,
                                  x2=self.px,
                                  y2=self.py,
                                  width=2,
                                  color=self.color,
                                  batch=batch,
                                  group=group)

    def update_view(self, pyglet_view, parameters_map):
        self.update(parameters_map)
        pyglet_view.color = self.color

    def is_visible(self, world_camera):
        # return world_camera.contains(self.x, self.y) or world_camera.contains(self.px, self.py)
        return True


class NodeMeta(Meta):
    def get_action_color(self):
        if self.selected:
            return selected_color
        elif self.pinned:
            return pinned_color
        elif self.hovered:
            return hover_color
        else:
            return self.color

    def update(self, parameters_map):
        for k, w in self.weights.items():
            self.weights[k] = parameters_map[k].data
            self.grads[k] = parameters_map[k].data
        self.color = generate_node_color_w(self.weights.values())

    def create_view(self, parameters_map, batch, group):
        self.update(parameters_map)
        return pyglet.shapes.Circle(x=self.x,
                                    y=self.y,
                                    radius=self.radius,
                                    color=self.get_action_color(),
                                    batch=batch,
                                    group=group)

    def update_view(self, pyglet_view, parameters_map):
        self.update(parameters_map)
        pyglet_view.color = self.get_action_color()

    def is_visible(self, world_camera):
        # return world_camera.contains(self.x, self.y)
        return True

    def contains(self, x, y, word_camera):
        return word_camera.contains_circle(x, y, self)
