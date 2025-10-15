from __future__ import annotations

from typing import TYPE_CHECKING, List

from ai.pathfinding.core import MovementStrategy
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.logic.grid import Grid2D
    from park.logic.node import Node


class LinearStrategy(MovementStrategy):
    def __init__(
        self,
        grid: Grid2D,
        *,
        allow_diagonal: bool = True,
        path_smoothing: bool = True,
        samples_per_segment: int = 5,
        min_distance: float = 1e-5
    ):
        super().__init__(
            grid,
            allow_diagonal=allow_diagonal,
            path_smoothing=path_smoothing,
            samples_per_segment=samples_per_segment,
            min_distance=min_distance
        )

    def find_node_path(
        self,
        start_node: Node,
        end_node: Node,
        force_walk_nodes: List[Node]
    ) -> List[Node]:
        if self._allow_diagonal:
            return [start_node, end_node]
        else:
            middle_nodes: List[Node] = [
                self._grid.world_to_node(
                    Vector2D(
                        start_node.center.x,
                        end_node.center.y
                    )
                ),
                self._grid.world_to_node(
                    Vector2D(
                        end_node.center.x,
                        start_node.center.y
                    )
                )
            ]
            for node in middle_nodes:  # Try to find a walkable node
                if (
                    node is not None
                    and node is not start_node
                    and node is not end_node
                    and len(node.occupants) == 0
                ):
                    return [start_node, node, end_node]

            # If both middle nodes are not walkable, return first valid one
            middle_node = middle_nodes[0] or middle_nodes[1]
            if (
                middle_node is not None
                and middle_node is not start_node
                and middle_node is not end_node
            ):
                return [start_node, middle_node, end_node]
            else:
                return [start_node, end_node]
