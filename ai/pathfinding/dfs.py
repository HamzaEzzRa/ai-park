from __future__ import annotations

from typing import TYPE_CHECKING, List

from ai.pathfinding.core import MovementStrategy

if TYPE_CHECKING:
    from park.logic.grid import Grid2D
    from park.logic.node import Node


class DepthFirstStrategy(MovementStrategy):
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
        open_set: List[Node] = [start_node]
        closed_set: List[Node] = []
        came_from: dict[Node, Node] = {}

        while open_set:
            current = open_set.pop()
            if current == end_node:
                return self._reconstruct_path(came_from, current)

            closed_set.append(current)

            for neighbor in self._grid.get_node_neighbors(
                current,
                allow_diagonal=self._allow_diagonal
            ):
                if (
                    neighbor in closed_set
                    or not (
                        neighbor.walkable
                        or neighbor in force_walk_nodes
                    )
                ):
                    continue

                if neighbor not in open_set:
                    open_set.append(neighbor)
                    came_from[neighbor] = current

        return []

    def _reconstruct_path(
        self,
        came_from: dict[Node, Node],
        current: Node
    ) -> List[Node]:
        total_path: List[Node] = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        return total_path
