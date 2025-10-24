from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List

from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.logic.grid import Grid2D
    from park.logic.node import Node


@dataclass(frozen=True)
class MovementState:
    position: Vector2D
    velocity: Vector2D
    target: Vector2D | None
    force_walk_nodes: List[Node]


class MovementPlan:
    def __init__(
        self,
        *,
        nodes: List[Node],
        waypoints: List[Vector2D]
    ):
        self.nodes = nodes
        self.waypoints = waypoints
        self._current_index = 0

    @staticmethod
    def idle() -> MovementPlan:
        return MovementPlan(nodes=[], waypoints=[])

    @property
    def progress(self) -> float:
        value = self._current_index / max(1, len(self.waypoints) - 1)
        return min(max(value, 0.0), 1.0)

    @property
    def remaining_waypoints(self) -> List[Vector2D]:
        return self.waypoints[self._current_index:]

    def next_waypoint(self) -> Vector2D | None:
        if self.progress >= 1.0:
            return None
        self._current_index += 1
        waypoint = self.waypoints[self._current_index]
        return waypoint

    def previous_waypoint(self) -> Vector2D | None:
        if self.progress <= 0.0:
            return None
        self._current_index -= 1
        return self.waypoints[self._current_index]

    def peek_waypoint(self) -> Vector2D | None:
        if (
            self._current_index < 0
            or self._current_index >= len(self.waypoints)
        ):
            return None
        return self.waypoints[self._current_index]


class MovementStrategy:
    class Type(Enum):
        LINEAR = "linear"
        ASTAR = "astar"
        BFS = "bfs"
        DFS = "dfs"

    def __init__(
        self,
        grid: Grid2D,
        allow_diagonal: bool = True,
        path_smoothing: bool = True,
        samples_per_segment: int = 5,
        min_distance: float = 1e-5
    ):
        self._grid = grid
        self._allow_diagonal = allow_diagonal
        self._path_smoothing = path_smoothing
        self._samples_per_segment = samples_per_segment
        self._min_distance = min_distance

    def find_node_path(
        self,
        start_node: Node,
        end_node: Node,
        force_walk_nodes: List[Node]
    ) -> List[Node]:
        """
        Find a path from start_node to end_node as a list of nodes.
        Start and end nodes should be included in the returned list.
        """
        raise NotImplementedError

    def plan(
        self,
        state: MovementState
    ) -> MovementPlan:
        if state.target is None:
            return MovementPlan.idle()
        nodes, path = self._find_path(state)
        if not nodes or not path:
            return MovementPlan.idle()
        return MovementPlan(nodes=nodes, waypoints=path)

    def _find_path(
        self,
        state: MovementState
    ) -> tuple[List[Node], List[Vector2D]]:
        start = state.position
        end = state.target
        if end is None:
            return [], []

        start_node = self._grid.world_to_node(start)
        end_node = self._grid.world_to_node(end)
        if start_node is None or end_node is None:
            return [], []

        nodes = self.find_node_path(
            start_node,
            end_node,
            state.force_walk_nodes
        )
        if self._path_smoothing:
            return nodes, self._smoothen_path(state, nodes)

        waypoints = [node.center for node in nodes]
        waypoints.insert(0, start)
        waypoints.append(end)
        return nodes, waypoints

    def _smoothen_path(
        self,
        state: MovementState,
        nodes: List[Node]
    ) -> List[Vector2D]:
        """ Smooth a path from nodes centers using Catmull-Rom splines. """
        if (
            not nodes
            or state.position.distance_from(state.target) < self._min_distance
        ):
            return []

        points: List[Vector2D] = [state.position]
        for node in nodes:
            center = node.center
            if center.distance_from(points[-1]) > self._min_distance:
                points.append(center)
        points.append(state.target)

        cr_spline: List[Vector2D] = [points[0]]
        for i in range(len(points) - 1):
            p0 = points[i - 1] if i > 0 else None
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2] if i + 2 < len(points) else None

            for j in range(self._samples_per_segment):
                t = (j + 1) / (self._samples_per_segment + 1)
                candidate = catmull_rom(p0=p0, p1=p1, p2=p2, p3=p3, t=t)
                candidate = Vector2D(
                    max(0.0, min(self._grid.width, candidate.x)),
                    max(0.0, min(self._grid.height, candidate.y))
                )
                if candidate.distance_from(cr_spline[-1]) > self._min_distance:
                    cr_spline.append(candidate)

        cr_spline.append(state.target)
        return cr_spline


def catmull_rom(
    *,
    p0: Vector2D | None = None,
    p1: Vector2D,
    p2: Vector2D,
    p3: Vector2D | None = None,
    t: float
) -> Vector2D:
    # If a control point is not specified, duplicate using the adjacent point
    if p0 is None:
        p0 = p1.copy()
    if p3 is None:
        p3 = p2.copy()

    t2 = t * t
    t3 = t2 * t
    a_x = 2 * p1.x
    a_y = 2 * p1.y
    b_x = -p0.x + p2.x
    b_y = -p0.y + p2.y
    c_x = 2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x
    c_y = 2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y
    d_x = -p0.x + 3 * p1.x - 3 * p2.x + p3.x
    d_y = -p0.y + 3 * p1.y - 3 * p2.y + p3.y

    return Vector2D(
        0.5 * (a_x + b_x * t + c_x * t2 + d_x * t3),
        0.5 * (a_y + b_y * t + c_y * t2 + d_y * t3),
    )
