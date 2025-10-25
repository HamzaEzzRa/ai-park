from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from park.entities.core import BaseEntity
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.entities.ride import Ride
    from park.entities.robot import Robot
    from park.logic.queue import VisitorQueue
    from park.simulation import Simulation


class Visitor(BaseEntity):
    class MemberType(Enum):
        ADULT = "adult"
        CHILD = "child"

    class GroupType(Enum):
        ALL_ADULTS = 0
        WITH_CHILDREN = 1

    class State(Enum):
        QUEUEING = "queuing"
        IN_QUEUE = "in_queue"
        ON_BOARD = "on_board"
        ON_RIDE = "on_ride"
        RIDING = "riding"
        EXITING = "exiting"

    _tooltip_visitors: set["Visitor"] = set()

    def __init__(
        self,
        simulation: Simulation,
        position: Vector2D,
        group_size: int,
        move_speed: float,
        desired_rides: int,
        member_spacing: float,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(simulation, position)
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.group_size = group_size
        self.move_speed = move_speed
        self.desired_rides = desired_rides

        # At least one adult, rest can be adults or children
        self.members: List[Visitor.MemberType] = (
            [Visitor.MemberType.ADULT]  
            + self.rng.choice(
                list(Visitor.MemberType),
                size=group_size - 1,
                replace=True
            ).tolist()
        )
        self.member_spacing = member_spacing

        self.group_type = self._get_group_type()

        self.state = Visitor.State.QUEUEING
        self.completed_rides = 0
        self.time_in_queues = 0
        self.time_in_rides = 0
        self.satisfaction = 0.5

        self.base_dissatisfaction_rate = 0.000075
        self.max_tail_boost = 0.4

        self.queue_ref: Optional[VisitorQueue] = None
        self.queue_index: Optional[int] = None
        self.queue_size: int = 0

        self.assigned_robot: Optional[Robot] = None
        self.assigned_ride: Optional[Ride] = None

        self.target_position = self.transform.position.copy()

        self._tooltip_visible = False

    @staticmethod
    def get_tooltip_visitors() -> List[Visitor]:
        return list(Visitor._tooltip_visitors)

    @property
    def tooltip_visible(self) -> bool:
        return self._tooltip_visible

    def set_tooltip_visible(self, value: bool) -> None:
        self._tooltip_visible = value
        if value:
            Visitor._tooltip_visitors.add(self)
        elif not value:
            Visitor._tooltip_visitors.discard(self)

    @property
    def dissatisfaction_rate(self) -> float:
        if not self.queue_ref or self.queue_size <= 1 or self.queue_index is None:
            return self.base_dissatisfaction_rate

        rel_pos = self.queue_index / max(self.queue_size - 1, 1)
        rel_pos = rel_pos * rel_pos * (3 - 2 * rel_pos)
        position_factor = 1.0 + self.max_tail_boost * rel_pos

        queue_pressure = 2 * (1 / (1 + 2**(-self.queue_size / 5)) - 0.5)
        queue_size_factor = 1.0 + 0.5 * queue_pressure

        final_rate = self.base_dissatisfaction_rate * position_factor * queue_size_factor
        return final_rate

    def set_queue_state(self, queue, index: Optional[int], size: int) -> None:
        self.queue_ref = queue
        self.queue_index = index
        self.queue_size = size

    def move_to(self, target: Vector2D):
        self.target_position = target.copy()

    def group_bounds(self) -> Vector2D:
        n = max(1, int(self.group_size))
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        spacing = self.member_spacing
        width = max(spacing, cols * spacing)
        height = max(spacing, rows * spacing)
        padding = spacing * 0.25
        return Vector2D(width + padding, height + padding)

    def update(self):
        super().update()
        self.sprite.set_enabled(True)
        self.collider.set_enabled(True)

        if self.state == Visitor.State.ON_BOARD or self.state == Visitor.State.ON_RIDE:
            self.satisfaction = max(
                0.0, min(1.0, self.satisfaction - self.base_dissatisfaction_rate)
            )
        elif self.state == Visitor.State.IN_QUEUE: # apply dissatisfaction with queue factors
            self.satisfaction = max(
                0.0, min(1.0, self.satisfaction - self.dissatisfaction_rate)
            )

        if self.state == Visitor.State.QUEUEING:
            if not self.target_position:
                return
            if self.transform.position == self.target_position:
                self.state = Visitor.State.IN_QUEUE
            else:
                delta = self.target_position - self.transform.position
                direction = delta.normalized()
                distance = delta.magnitude()
                move_distance = min(self.move_speed, distance)
                self.transform.translate(direction * move_distance)
        elif self.state == Visitor.State.IN_QUEUE:
            self.time_in_queues += 1
            if (
                self.simulation.exit_queue is not None 
                and self.simulation.exit_queue.count > 0
                and self.queue_ref == self.simulation.exit_queue
                and self == self.simulation.exit_queue.groups[0]
            ):
                self.simulation.exit_queue.remove(self)
                self.state = Visitor.State.EXITING
        elif self.state == Visitor.State.ON_BOARD:
            self.collider.set_enabled(False)
            robot_pos = self.assigned_robot.transform.position.copy()
            self.transform.set_position(robot_pos)
            self.target_position = robot_pos
        elif self.state == Visitor.State.ON_RIDE:
            self.sprite.set_enabled(False)
            self.collider.set_enabled(False)
        elif self.state == Visitor.State.RIDING:
            self.sprite.set_enabled(False)
            self.collider.set_enabled(False)
            if self.assigned_ride is not None:
                self.time_in_rides += 1
                self.satisfaction += self.assigned_ride.enjoyment_rate
                self.satisfaction = max(0.0, min(1.0, self.satisfaction))
        elif self.state == Visitor.State.EXITING:
            self.sprite.set_enabled(False)
            self.collider.set_enabled(False)
            self.satisfaction += (
                (2 * self.time_in_rides - self.time_in_queues)
                / (self.time_in_rides + self.time_in_queues)
            )
            self.satisfaction = max(0.0, min(1.0, self.satisfaction))
            self.simulation.remove_visitor(self)
            self.delete()

    def _get_group_type(self) -> Visitor.GroupType:
        if all(member == Visitor.MemberType.ADULT for member in self.members):
            return Visitor.GroupType.ALL_ADULTS
        else:
            return Visitor.GroupType.WITH_CHILDREN
