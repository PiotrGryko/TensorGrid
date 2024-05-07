import time

import numpy as np


class BSPLeaf:
    def __init__(self, x, y, w, h, level):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.level = level
        self.children = []
        self.key_x = 0 if self.x == 0 else int(self.x / self.w)
        self.key_y = 0 if self.y == 0 else int(self.y / self.h)
        self.generated = False

    def is_visible(self, viewport):
        x, y, w, h, zoom = viewport
        # plane
        px1, py1, px2, py2 = self.x, self.y, self.x + self.w, self.y + self.h
        # viewport
        wx1, wy1, wx2, wy2 = x, y, x + w, y + h

        # viewport inside leaf
        viewport_inside_x = (px1 <= wx1 and px2 >= wx2)
        viewport_inside_y = (py1 <= wy1 and py2 >= wy2)

        # leaf inside viewport
        leaf_inside_x = (wx1 <= px1 and wx2 >= px2)
        leaf_inside_y = (wy1 <= py1 and wy2 >= py2)

        # partially visible
        horizontal = (px1 >= wx1 and px1 <= wx2) or (px2 >= wx1 and px2 <= wx2)
        vertical = (py1 >= wy1 and py1 <= wy2) or (py2 >= wy1 and py2 <= wy2)
        is_visible = (horizontal or viewport_inside_x) and (vertical or viewport_inside_y)
        is_fully_visible = leaf_inside_x and leaf_inside_y
        contains_viewport = viewport_inside_x and viewport_inside_y
        return is_visible, is_fully_visible, contains_viewport

    def contains_point(self, x, y):
        px1, py1, px2, py2 = self.x, self.y, self.x + self.w, self.y + self.h
        result = (x >= px1 and x <= px2) and (y >= py1 and y <= py2)
        return result

    def find_leaf_by_x_y(self, x, y):
        child = None
        contains = self.contains_point(x, y)
        if not contains:
            return None
        for c in self.children:
            result = c.find_leaf_by_x_y(x, y)
            if result:
                child = result
        if child:
            return child
        elif contains:
            return self
        return None

    def create_child(self, x, y, w, h, level):
        return BSPLeaf(x, y, w, h, level)

    # split in 4
    def generate(self, depth, step=0):
        if step < depth:
            step += 1
            # 300.0 600.0 | 300.0 600.0
            self.generate_leafs()
            for c in self.children:
                c.generate(depth, step)

    def generate_leafs(self):
        self.generated = True
        print("generated leafs", self)
        self.children = [
            self.create_child(
                self.x,
                self.y,
                self.w / 2,
                self.h / 2,
                self.level+1
            ),
            self.create_child(
                self.x,
                self.y + self.h / 2,
                self.w / 2,
                self.h / 2,
                self.level+1
            ),
            self.create_child(
                self.x + self.w / 2,
                self.y,
                self.w / 2,
                self.h / 2,
                self.level+1
            ),
            self.create_child(
                self.x + self.w / 2,
                self.y + self.h / 2,
                self.w / 2,
                self.h / 2,
                self.level+1
            )
        ]

    def traverse(self, viewport, visible, not_visible, max_depth=None):
        is_visible, is_fully_visible, contains_viewport = self.is_visible(viewport)
        # Not visible discard
        if not is_visible:
            not_visible.append(self)
            return False
        # Is fully visible
        if is_fully_visible:
            visible.append(self)
            return True
        if contains_viewport:
            max_depth = self.level + 4
        if max_depth is None:
            max_depth = self.level + 4
        if self.level == max_depth:
            visible.append(self)
            return True
        if not self.generated:
            self.generate_leafs()
        if len(self.children) == 0:
            visible.append(self)
            return True
        for c in self.children:
            c.traverse(viewport, visible, not_visible, max_depth)
        return True

    def count(self):
        count = 1
        for c in self.children:
            count += c.count()
        return count

    def edge_leafs_count(self):
        count = 0
        if len(self.children) == 0:
            count = 1
        for c in self.children:
            count += c.edge_leaves_count()
        return count

    def edge_leafs(self):
        result = []
        if len(self.children) == 0:
            result.append(self)
        for c in self.children:
            result += c.edge_leaves()
        return result

    def leafs(self):
        result = [self]
        for c in self.children:
            result += c.leaves()
        return result

    def dump(self, indent):
        value = f"{indent}{self.x} {self.x + self.w} | {self.y}  {self.y + self.h}"
        indent = indent + " "
        for index, child in enumerate(self.children):
            value += f"\n{index} {child.dump(indent)}"
        return value

    def __repr__(self):
        return f"(x1:{self.x} y1:{self.y} x2:{self.x + self.w} y2:{self.y + self.h})"


# starts at 0,0
class BSPTree:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.leaf = BSPLeaf(0, 0, self.width, self.height, 0)
        self.edge_leafs = {}
        self.edge_leafs_grid = None
        self.edge_leaf_width = 0
        self.edge_leaf_height = 0
        self.grid_size = 0
        self.depth = depth

    def set_size(self, w, h):
        self.width = w
        self.height = h
        self.leaf.w = w
        self.leaf.h = h

    def generate(self):
        print("generating tree....")
        start_time = time.time()
        # self.leaf.generate(self.depth)
        print("Tree generated", time.time() - start_time)

    def traverse(self, viewport, visible, not_visible):
        self.leaf.traverse(viewport, visible, not_visible)

    def dump(self):
        result = f"BSPTree w:{self.width} h:{self.height}"
        for c in self.leaf.children:
            result += c.dump("")
        result += f"leafs count: {self.count()} edge leafs count: {self.edge_leafs_count()}"
        return result

    def count(self):
        return self.leaf.count()

    def get_edge_leaf(self, x, y):
        key_x = int(x / self.edge_leaf_width)
        key_y = int(y / self.edge_leaf_height)
        leaf = self.edge_leafs_grid[key_x][key_y]
        return leaf

    def edge_leafs_count(self):
        return self.leaf.edge_leafs_count()

    def get_edge_leafs(self):
        return self.leaf.edge_leafs()

    def get_leafs(self):
        return self.leaf.leafs()

    def generate_edge_leafs_grid(self):
        edge_leafs = self.leaf.edge_leafs()
        edge_leaf = edge_leafs[0]
        self.edge_leaf_width = edge_leaf.w
        self.edge_leaf_height = edge_leaf.h
        self.grid_size = int(self.width / edge_leaf.w)
        self.edge_leafs_grid = np.empty((self.grid_size, self.grid_size), dtype=object)
        for leaf in edge_leafs:
            self.edge_leafs_grid[leaf.key_x][leaf.key_y] = leaf
