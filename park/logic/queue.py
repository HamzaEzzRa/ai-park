from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from park.entities.visitor import Visitor
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.world import World


class VisitorQueue:
    def __init__(
        self,
        world: World,
        head: Vector2D,
        tail: Vector2D,
        spacing: float,
        max_groups: Optional[int] = None,
        max_members: Optional[int] = None,
    ):
        self.world = world
        self.head = head
        self.tail = tail
        self.direction = (tail - head).normalized()
        self.base_gap = max(0.0, float(spacing))
        self.max_groups = max_groups if (max_groups is None or max_groups >= 0) else None
        self.max_members = max_members if (max_members is None or max_members >= 0) else None

        self._groups: List[Visitor] = []
        self._member_total: int = 0

    def set_head(self, head: Vector2D):
        self.head = head
        self.direction = (self.tail - self.head).normalized()

    def set_tail(self, tail: Vector2D):
        self.tail = tail
        self.direction = (self.tail - self.head).normalized()

    def set_spacing(self, spacing: float):
        self.base_gap = max(0.0, float(spacing))

    def set_capacity(self, max_groups: Optional[int]):
        if max_groups is None:
            self.max_groups = None
        else:
            self.max_groups = max(0, int(max_groups))

    def set_member_capacity(self, max_members: Optional[int]):
        if max_members is None:
            self.max_members = None
        else:
            self.max_members = max(0, int(max_members))

    @property
    def groups(self) -> List[Visitor]:
        return list(self._groups)

    @property
    def count(self) -> int:
        return len(self._groups)

    @property
    def member_count(self) -> int:
        return self._member_total

    @property
    def average_group_size(self) -> float:
        if not self._groups:
            return 0.0
        return self._member_total / len(self._groups)

    def clear(self) -> None:
        for visitor in self._groups:
            visitor.set_queue_state(queue=None, index=None, size=0)
        self._groups.clear()
        self._member_total = 0

    def can_accept(self, groups: int) -> bool:
        return (
            self.max_groups is None
            or (self.count + groups) <= self.max_groups
        )

    def can_accept_members(self, members: int) -> bool:
        return (
            self.max_members is None
            or (self._member_total + members) <= self.max_members
        )

    def can_fit(self, visitor: Visitor) -> bool:
        projected = self._groups + [visitor]
        return self._fits_within_bounds(projected)

    def add(self, visitor: Visitor) -> None:
        if visitor in self._groups:
            return
        self._groups.append(visitor)
        self._member_total += visitor.group_size

    def remove(self, visitor: Visitor) -> bool:
        if visitor in self._groups:
            idx = self._groups.index(visitor)
            self._groups.pop(idx)
            self._member_total = max(0, self._member_total - visitor.group_size)
            visitor.set_queue_state(queue=None, index=None, size=0)
            return True
        return False

    def pop(self, index: int = 0) -> Optional[Visitor]:
        if not self._groups:
            return None
        visitor = self._groups.pop(index)
        self._member_total = max(0, self._member_total - visitor.group_size)
        visitor.set_queue_state(queue=None, index=None, size=0)
        return visitor

    def slot_position(self, index: int) -> Vector2D:
        centers = self._compute_centers(self._groups)
        if index >= len(centers):
            return self.world.snap(self.head)
        dir_norm = self._direction_normalized()
        slot = self.head + dir_norm * centers[index]
        return self.world.snap(slot)

    def iter_assignments(self) -> Iterable[Tuple[Visitor, Vector2D]]:
        dir_norm = self._direction_normalized()
        centers = self._compute_centers(self._groups)
        for visitor, offset in zip(self._groups, centers):
            pos = self.head + dir_norm * offset
            yield visitor, self.world.snap(pos)

    def update_targets(self):
        dir_norm = self._direction_normalized()
        centers = self._compute_centers(self._groups)
        queue_size = self.count
        for idx, (visitor, offset) in enumerate(zip(self._groups, centers)):
            pos = self.head + dir_norm * offset
            visitor.set_queue_state(queue=self, index=idx, size=queue_size)
            if visitor.target_position != pos:
                visitor.state = Visitor.State.QUEUEING
            visitor.move_to(self.world.snap(pos))

    def _direction_normalized(self) -> Vector2D:
        mag = self.direction.magnitude()
        if mag == 0:
            return Vector2D(1, 0)
        return Vector2D(self.direction.x / mag, self.direction.y / mag)

    def _item_extent(self, visitor: Visitor) -> float:
        dir_norm = self._direction_normalized()
        size_vec = None

        collider = getattr(visitor, "collider", None)
        if collider is not None and hasattr(collider, "size"):
            size_vec = collider.size
        elif hasattr(visitor, "sprite") and getattr(visitor, "sprite") is not None:
            size_vec = visitor.sprite.size

        if size_vec is None:
            return max(self.base_gap, 1.0)

        extent = abs(dir_norm.x) * size_vec.x + abs(dir_norm.y) * size_vec.y
        return max(extent, 1.0)

    def _compute_centers(self, items: List[Visitor]) -> List[float]:
        centers: List[float] = []
        if not items:
            return centers

        offset = 0.0
        prev_half = 0.0
        for idx, item in enumerate(items):
            extent = self._item_extent(item)
            half = extent * 0.5
            if idx == 0:
                offset = half
            else:
                offset += prev_half + self.base_gap + half
            centers.append(offset)
            prev_half = half
        return centers

    def _required_distance(self, items: List[Visitor]) -> float:
        if not items:
            return 0.0
        centers = self._compute_centers(items)
        last_extent = self._item_extent(items[-1]) * 0.5
        return centers[-1] + last_extent

    def _max_distance(self) -> float:
        dir_norm = self._direction_normalized()
        origin = self.head
        tmin = 0.0
        tmax = float("inf")

        dir_norm = self._direction_normalized()
        to_tail = self.tail - origin
        projected = to_tail.x * dir_norm.x + to_tail.y * dir_norm.y
        return max(0.0, projected)

    def _fits_within_bounds(self, items: List[Visitor]) -> bool:
        if not items:
            return True
        max_distance = self._max_distance()
        if max_distance <= 0:
            return False
        required = self._required_distance(items)
        return required <= max_distance + 1e-6
