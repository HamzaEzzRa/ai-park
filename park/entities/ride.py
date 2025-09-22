from enum import Enum

from park.entities.core import BaseEntity
from park.entities.visitor import Visitor
from park.internal.math import Vector2D
from park.logic.queue import VisitorQueue


class Ride(BaseEntity):
    class OperationalState(Enum):
        OPEN = "open"
        CLOSED = "closed"
        BROKEN = "broken"

    class RunState(Enum):
        LOADING = "loading"
        RUNNING = "running"
        UNLOADING = "unloading"

    def __init__(
        self,
        simulation,
        position: Vector2D,
        name: str,
        capacity: int,
        duration: int,
        entrance_queue: VisitorQueue,
        exit_queue: VisitorQueue
    ):
        super().__init__(simulation, position)
        self.name = name
        self.capacity = capacity
        self.duration = duration
        self.enjoyment_rate = 0.0005 # TODO: make this more interesting (varies per ride, group size, ...), and less effective when repeating the same ride
        self.entrance_queue = entrance_queue
        self.exit_queue = exit_queue

        # Treat riders as a queue for practicality
        self.riders = VisitorQueue(
            world=simulation.world,
            head=position,
            tail=position,
            spacing=0.0,
            max_members=capacity,
        )

        self.operational_state = Ride.OperationalState.OPEN
        self.run_state = Ride.RunState.LOADING
        self.timer = 0

    def update(self):
        super().update()

        if (
            self.operational_state == Ride.OperationalState.BROKEN
            or self.operational_state == Ride.OperationalState.CLOSED
        ):
            return

        self.entrance_queue.update_targets()
        self.exit_queue.update_targets()

        if self.run_state == Ride.RunState.LOADING:
            if not self.entrance_queue.count:
                return
            visitor = self.entrance_queue.groups[0]
            if visitor.state == Visitor.State.IN_QUEUE:
                if self.riders.can_accept_members(visitor.group_size):
                    self.riders.add(visitor)
                    visitor.state = Visitor.State.ON_RIDE
                    visitor.assigned_ride = self
                    self.entrance_queue.pop(0)
            if (
                self.riders.member_count == self.capacity
                or not self.riders.can_accept_members(visitor.group_size)
            ):
                self.timer = 0
                self.run_state = Ride.RunState.RUNNING
                for rider in self.riders.groups:
                    rider.state = Visitor.State.RIDING
        elif self.run_state == Ride.RunState.RUNNING:
            self.timer += 1
            if self.timer >= self.duration:
                for visitor in self.riders.groups:
                    visitor.completed_rides += 1
                    visitor.state = Visitor.State.ON_RIDE
                self.run_state = Ride.RunState.UNLOADING
        elif self.run_state == Ride.RunState.UNLOADING:
            if self.riders.count > 0:
                visitor = self.riders.groups[0]
                if not (
                    self.exit_queue.can_accept(1)
                    and self.exit_queue.can_accept_members(visitor.group_size)
                    and self.exit_queue.can_fit(visitor)
                ): return
                self.riders.remove(visitor)
                self.exit_queue.add(visitor)
                visitor.state = Visitor.State.QUEUEING
                visitor.assigned_ride = None
                visitor.target_position = None
                visitor.transform.set_position(self.exit_queue.tail)
            else:
                self.run_state = Ride.RunState.LOADING

    def can_accept(self, group: Visitor) -> bool:
        can_accept = True
        if (
            self.operational_state != Ride.OperationalState.OPEN
            or not self.entrance_queue.can_accept(1)
            or not self.entrance_queue.can_accept_members(group.group_size)
            or not self.entrance_queue.can_fit(group)
        ):
            can_accept = False
        return can_accept
