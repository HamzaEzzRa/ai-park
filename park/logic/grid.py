from __future__ import annotations

import math
from typing import List

from park.internal.layers import CollisionLayer
from park.internal.math import Vector2D
from park.internal.physics import Physics
from park.logic.node import Node


class Grid2D:
    _grids: List[Grid2D] = []

    def __init__(
        self,
        width: int,
        height: int,
        cell_size: float,
        mask_bits: int | CollisionLayer = CollisionLayer.ALL_BITS,
    ):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.mask_bits = int(mask_bits)

        self.cols = math.ceil(width / cell_size)
        self.rows = math.ceil(height / cell_size)
        self.nodes: List[List[Node]] = self.create_grid()
        self._dirty_nodes: list[Node] = []
        Grid2D._grids.append(self)

    def create_grid(self) -> List[List[Node]]:
        grid = []
        for y in range(self.rows):
            row = []
            for x in range(self.cols):
                center = Vector2D(x + 0.5, y + 0.5) * self.cell_size
                node = Node(center, self.cell_size / 2)
                row.append(node)
            grid.append(row)
        return grid

    def world_to_node(self, position: Vector2D) -> Node | None:
        col = int(position.x // self.cell_size)
        row = int(position.y // self.cell_size)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return self.nodes[row][col]
        return None

    def update_nodes(self):
        self._dirty_nodes.clear()
        for row in self.nodes:
            for node in row:
                node.occupants.clear()

        if self.mask_bits != CollisionLayer.NONE:
            for collider in Physics.active_colliders():
                if not CollisionLayer.can_collide(
                    collider.layer_bits,
                    collider.mask_bits,
                    CollisionLayer.ALL_BITS,
                    self.mask_bits,
                ):
                    continue

                bounds = collider.bounds()
                if bounds.width <= 0 or bounds.height <= 0:
                    continue

                min_col = max(0, int(math.floor(bounds.x / self.cell_size)))
                max_col = min(
                    self.cols - 1,
                    int(math.floor((bounds.x + bounds.width - 1e-6) / self.cell_size)),
                )
                min_row = max(0, int(math.floor(bounds.y / self.cell_size)))
                max_row = min(
                    self.rows - 1,
                    int(math.floor((bounds.y + bounds.height - 1e-6) / self.cell_size)),
                )
                if min_col > max_col or min_row > max_row:
                    continue
 
                for row_idx in range(min_row, max_row + 1):
                    node_row = self.nodes[row_idx]
                    for col_idx in range(min_col, max_col + 1):
                        node_row[col_idx].occupants.append(collider)

        for row in self.nodes:
            for node in row:
                if node.occupancy_changed():
                    self._dirty_nodes.append(node)
                node.was_occupied = len(node.occupants) > 0

    def has_dirty_nodes(self) -> bool:
        return bool(self._dirty_nodes)

    def consume_dirty_nodes(self) -> list[Node]:
        dirty = list(self._dirty_nodes)
        self._dirty_nodes.clear()
        return dirty
