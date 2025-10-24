from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ai.pathfinding.core import MovementPlan
from park.entities.visitor import Visitor
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.entities.core import BaseEntity
    from park.entities.robot import Robot
    from park.world import World


class Queue:
    def __init__(
        self,
        name: str,
        world: World,
        head: Vector2D,
        tail: Vector2D,
        spacing: float,
        max_capacity: Optional[int] = None,
    ):
        self.name = name
        self.world = world
        self.head = head
        self.tail = tail
        self.direction = (tail - head).normalized()
        self.base_gap = max(0.0, float(spacing))
        self.max_capacity = (
            max_capacity if (max_capacity is None or max_capacity >= 0)
            else None
        )

        self._entities: List[BaseEntity] = []

    @property
    def count(self) -> int:
        return len(self._entities)

    @property
    def entities(self) -> List[BaseEntity]:
        return self._entities

    def set_head(self, head: Vector2D):
        self.head = head
        self.direction = (self.tail - self.head).normalized()

    def set_tail(self, tail: Vector2D):
        self.tail = tail
        self.direction = (self.tail - self.head).normalized()

    def set_spacing(self, spacing: float):
        self.base_gap = max(0.0, float(spacing))

    def set_capacity(self, max_capacity: Optional[int]):
        if max_capacity is None:
            self.max_capacity = None
        else:
            self.max_capacity = max(0, int(max_capacity))

    def can_accept(self, to_add: int) -> bool:
        return (
            self.max_capacity is None
            or (self.count + to_add) <= self.max_capacity
        )

    def clear(self) -> None:
        self._entities.clear()

    def add(self, entity: BaseEntity) -> None:
        if entity in self._entities:
            return
        self._entities.append(entity)

    def pop(self, index: int = 0) -> Optional[BaseEntity]:
        if not self._entities:
            return None
        return self._entities.pop(index)

    def remove(self, entity: BaseEntity) -> bool:
        if entity in self._entities:
            idx = self._entities.index(entity)
            self._entities.pop(idx)
            return True
        return False

    def can_fit(self, item: BaseEntity) -> bool:
        projected = self._entities + [item]
        return self._fits_within_bounds(projected)

    def update_targets(self):
        dir_norm = self._direction_normalized()
        centers = self._compute_centers(self._entities)
        for idx, (entity, offset) in enumerate(zip(self._entities, centers)):
            pos = self.head + dir_norm * offset
            entity.transform.set_position(pos)

    def _direction_normalized(self) -> Vector2D:
        mag = self.direction.magnitude()
        if mag == 0:
            return Vector2D(1, 0)
        return Vector2D(self.direction.x / mag, self.direction.y / mag)

    def _item_extent(self, item: BaseEntity) -> float:
        dir_norm = self._direction_normalized()
        size_vec = None

        if hasattr(item, "collider") and getattr(item, "collider") is not None:
            size_vec = item.collider.size
        elif hasattr(item, "sprite") and getattr(item, "sprite") is not None:
            size_vec = item.sprite.size

        if size_vec is None:
            return max(self.base_gap, 1.0)

        extent = abs(dir_norm.x) * size_vec.x + abs(dir_norm.y) * size_vec.y
        return max(extent, 1.0)

    def _compute_centers(self, items: List[BaseEntity]) -> List[float]:
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

    def _required_distance(self, items: List[BaseEntity]) -> float:
        if not items:
            return 0.0
        centers = self._compute_centers(items)
        last_extent = self._item_extent(items[-1]) * 0.5
        return centers[-1] + last_extent

    def _max_distance(self) -> float:
        dir_norm = self._direction_normalized()
        origin = self.head

        dir_norm = self._direction_normalized()
        to_tail = self.tail - origin
        projected = to_tail.x * dir_norm.x + to_tail.y * dir_norm.y
        return max(0.0, projected)

    def _fits_within_bounds(self, items: List[BaseEntity]) -> bool:
        if not items:
            return True
        max_distance = self._max_distance()
        if max_distance <= 0:
            return False
        required = self._required_distance(items)
        return required <= max_distance + 1e-6


class RobotQueue(Queue):
    def __init__(
        self,
        name: str,
        world: World,
        head: Vector2D,
        tail: Vector2D,
        spacing: float,
        max_capacity: Optional[int] = None,
    ):
        super().__init__(name, world, head, tail, spacing, max_capacity)
        self._entities: List[Robot] = []

    @property
    def robots(self) -> List[Robot]:
        return list(self._entities)

    def update_targets(self):
        dir_norm = self._direction_normalized()
        centers = self._compute_centers(self._entities)
        for idx, (robot, offset) in enumerate(zip(self._entities, centers)):
            pos = self.head + dir_norm * offset
            new_plan = MovementPlan(
                nodes=[],
                waypoints=[robot.transform.position.copy(), pos]
            )
            robot.set_plan(new_plan)

class VisitorQueue(Queue):
    def __init__(
        self,
        name: str,
        world: World,
        head: Vector2D,
        tail: Vector2D,
        spacing: float,
        max_capacity: Optional[int] = None,
        max_members: Optional[int] = None,
    ):
        super().__init__(name, world, head, tail, spacing, max_capacity)

        self.max_members = (
            max_members if (max_members is None or max_members >= 0)
            else None
        )
        self._entities: List[Visitor] = []
        self._member_total: int = 0

    def set_member_capacity(self, max_members: Optional[int]):
        if max_members is None:
            self.max_members = None
        else:
            self.max_members = max(0, int(max_members))

    @property
    def groups(self) -> List[Visitor]:
        return list(self._entities)

    @property
    def member_count(self) -> int:
        return self._member_total

    @property
    def average_group_size(self) -> float:
        if not self._entities:
            return 0.0
        return self._member_total / len(self._entities)

    def clear(self) -> None:
        for visitor in self._entities:
            visitor.set_queue_state(queue=None, index=None, size=0)
        self._entities.clear()
        self._member_total = 0

    def can_accept_members(self, members: int) -> bool:
        return (
            self.max_members is None
            or (self._member_total + members) <= self.max_members
        )

    def add(self, visitor: Visitor) -> None:
        if visitor in self._entities:
            return
        if visitor.queue_ref not in (None, self):
            visitor.queue_ref.remove(visitor)  # Force single queue membership
        visitor.set_queue_state(queue=self, index=self.count - 1, size=self.count)
        self._entities.append(visitor)
        self._member_total += visitor.group_size

    def pop(self, index: int = 0) -> Optional[Visitor]:
        if not self._entities:
            return None
        visitor = self._entities.pop(index)
        self._member_total = max(0, self._member_total - visitor.group_size)
        visitor.set_queue_state(queue=None, index=None, size=0)
        return visitor

    def remove(self, visitor: Visitor) -> bool:
        if visitor in self._entities:
            idx = self._entities.index(visitor)
            self._entities.pop(idx)
            self._member_total = max(0, self._member_total - visitor.group_size)
            visitor.set_queue_state(queue=None, index=None, size=0)
            return True
        return False

    # def slot_position(self, index: int) -> Vector2D:
    #     centers = self._compute_centers(self._entities)
    #     if index >= len(centers):
    #         return self.head
    #     dir_norm = self._direction_normalized()
    #     slot = self.head + dir_norm * centers[index]
    #     return slot

    # def iter_assignments(self) -> Iterable[Tuple[Visitor, Vector2D]]:
    #     dir_norm = self._direction_normalized()
    #     centers = self._compute_centers(self._entities)
    #     for visitor, offset in zip(self._entities, centers):
    #         pos = self.head + dir_norm * offset
    #         yield visitor, pos

    def update_targets(self):
        dir_norm = self._direction_normalized()
        centers = self._compute_centers(self._entities)
        queue_size = self.count
        for idx, (visitor, offset) in enumerate(zip(self._entities, centers)):
            pos = self.head + dir_norm * offset
            visitor.set_queue_state(queue=self, index=idx, size=queue_size)
            if visitor.target_position != pos:
                visitor.state = Visitor.State.QUEUEING
            visitor.move_to(pos)
