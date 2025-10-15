from __future__ import annotations

from typing import TYPE_CHECKING

from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.internal.collider import Collider


class Node:
    def __init__(self, center: Vector2D, radius: float):
        self.center = center
        self.radius = radius

        self.was_occupied: bool = False
        self.occupants: list[Collider] = []

    @property
    def diameter(self) -> float:
        return self.radius * 2

    @property
    def diagonal(self) -> float:
        return self.radius * 1.41421356237

    @property
    def walkable(self) -> bool:
        return len(self.occupants) == 0

    def occupancy_changed(self) -> bool:
        return self.was_occupied != (len(self.occupants) > 0)
