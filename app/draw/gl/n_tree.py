import random

from app.draw.bsp_tree.tree_bsp import BSPLeaf, BSPTree


class NTreeLeaf(BSPLeaf):
    def __init__(self, x, y, w, h, level):
        super().__init__(x, y, w, h, level)
        self.color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.w = w
        self.h = h
        self.x1, self.y1 = x, y
        self.x2, self.y2 = x + w, y + h
        self.id = f"{x}-{y}-{w}-{h}-{level}"

    def dump(self):
        return f'x:{self.x}, y:{self.y}, x2:{self.x2}, y2:{self.y2}, w:{self.w}, h:{self.h}'

    def create_child(self, x, y, w, h, level):
        return NTreeLeaf(x, y, w, h, level)

    def contains(self, x1, y1, x2, y2, level):
        return x1 >= self.x1 and y1 >= self.y1 and x2 <= self.x2 and y2 <= self.y2 and level == self.level


class NTree(BSPTree):
    def __init__(self, depth):
        super().__init__(0, 0, depth)
        self.viewport = None
        self.visible_leaves = []
        self.mega_leaf = None

    def set_size(self, w, h):
        super().set_size(w, h)
        self.leaf = NTreeLeaf(0, 0, self.width, self.height, 0)

    def update_viewport(self, viewport):
        visible = []
        not_visible = []
        self.viewport = viewport
        self.traverse(viewport, visible, not_visible)
        if visible != self.visible_leaves:
            # print("Visible count: ", len(visible))
            self.visible_leaves = visible
            self.build_mega_leaf()

    def build_mega_leaf(self):
        if len(self.visible_leaves) == 0:
            self.mega_leaf = None
            return

        x1 = min([v.x1 for v in self.visible_leaves])
        x2 = max([v.x2 for v in self.visible_leaves])
        y1 = min([v.y1 for v in self.visible_leaves])
        y2 = max([v.y2 for v in self.visible_leaves])
        level = max([v.level for v in self.visible_leaves])

        if self.mega_leaf is None:
            self.mega_leaf = NTreeLeaf(x1, y1, x2 - x1, y2 - y1, level)
        elif not self.mega_leaf.contains(x1, y1, x2, y2, level):
            self.mega_leaf = NTreeLeaf(x1, y1, x2 - x1, y2 - y1, level)
            #print("mega leaf updated", self.mega_leaf.w, self.mega_leaf.h)
