from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from park.entities.core import BaseEntity
from park.entities.visitor import Visitor
from park.entities.ride import Ride
from park.internal.math import Vector2D
from park.world import World

if TYPE_CHECKING:
    from park.internal.collider import Collider


class Robot(BaseEntity):
    class State(Enum):
        ROAMING = "roaming"
        ACQUIRED_VISITOR = "acquired_visitor"
        BOARDED = "boarded"

    def __init__(
        self,
        simulation,
        position: Vector2D,
        move_speed: float,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(simulation, position)
        self.move_speed = move_speed * 60
        self.move_direction = Vector2D(0, 0)
        self.state = Robot.State.ROAMING
        self.target_visitor: Optional[Visitor] = None
        self.target_ride: Optional[Ride] = None

        self._random_direction_chance = 0.01
        self._failure_cooldown = 60  # in simulation steps
        self._last_failure_step = 0

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def set_direction(self, x: float, y: float):
        self.move_direction = Vector2D(x, y)

    def update(self):
        super().update()

        self.rigidbody.set_velocity(self.move_direction * self.move_speed)

        if (self._current_step - self._last_failure_step) < self._failure_cooldown:
            if self.rng.random() < self._random_direction_chance:  # randomly change direction
                self.move_direction = self._random_direction()
        else:
            if self.state == Robot.State.ROAMING:
                if self.rng.random() < self._random_direction_chance:
                    self.move_direction = self._random_direction()
                self._acquire_visitor()
            elif self.state == Robot.State.ACQUIRED_VISITOR:
                if self.target_visitor is None:
                    self._acquire_visitor()
                if self.target_visitor is not None:
                    direction = self.target_visitor.transform.position - self.transform.position
                    self.move_direction = direction.normalized()
                    self._maybe_pickup_visitor()
            elif self.state == Robot.State.BOARDED:
                if self.target_visitor.completed_rides < self.target_visitor.desired_rides:
                    if self.target_ride is None:
                        self.target_ride = self._find_closest_ride()
                    if self.target_ride is not None:
                        direction = self.target_ride.entrance_queue.tail - self.transform.position
                        self.move_direction = direction.normalized()
                        self._maybe_dropoff_at_ride()
                elif self.simulation.exit_queue is not None:
                    direction = self.simulation.exit_queue.tail - self.transform.position
                    self.move_direction = direction.normalized()
                    self._maybe_dropoff_at_exit()

    def on_collision_enter(self, other: Collider):
        if (
            self.state == Robot.State.ACQUIRED_VISITOR
            or self.state == Robot.State.BOARDED
        ):
            self._release_targets()
            self._last_failure_step = self._current_step

        if other is None:
            self.move_direction = -self.move_direction
            return

        normal = self._collision_normal(other)
        if normal.magnitude() == 0:
            self.move_direction = -self.move_direction
            return

        incoming = self.move_direction
        if incoming.magnitude() == 0:
            incoming = normal * -1

        reflected = incoming - normal * (2 * incoming.dot(normal))
        if reflected.magnitude() > 1e-6:
            self.move_direction = reflected.normalized()
        else:
            self.move_direction = (-incoming).normalized()

    def _random_direction(self) -> Vector2D:
        angle = self.rng.uniform(0, 2 * np.pi)
        return Vector2D(np.cos(angle), np.sin(angle))

    def _collision_normal(self, other: Collider) -> Vector2D:
        self_collider = self.collider
        if self_collider is None or other is None:
            return Vector2D.zero()

        try:
            other_center = other.center
        except AttributeError:
            other_center = None

        if other_center is None and hasattr(other, "entity") and other.entity:
            other_center = other.entity.transform.position

        self_center = self_collider.center if hasattr(self_collider, "center") else self.transform.position
        other_center = other_center or getattr(getattr(other, "entity", None), "transform", self.transform).position

        displacement = self_center - other_center
        if displacement.magnitude() == 0:
            # fall back to separating axis based on current direction
            displacement = self.move_direction if self.move_direction.magnitude() > 0 else Vector2D(1.0, 0.0)

        return displacement.normalized()

    def _is_on_bounds(self, position: Vector2D) -> bool:
        world = self.simulation.world if self.simulation else self.world
        min_x, max_x, min_y, max_y = self._world_limits(world)
        return (
            position.x <= min_x
            or position.y <= min_y
            or position.x >= max_x
            or position.y >= max_y
        )

    def _half_extents(self) -> Vector2D:
        if self.collider and hasattr(self.collider, "half_extents"):
            return self.collider.half_extents
        size = self.collider.size if self.collider else Vector2D(0, 0)
        return Vector2D(size.x * 0.5, size.y * 0.5)

    def _world_limits(self, world: World) -> Tuple[float, float, float, float]:
        half = self._half_extents()
        min_x = min(max(half.x, 0.0), world.width)
        max_x = max(min(world.width - half.x, world.width), min_x)
        min_y = min(max(half.y, 0.0), world.height)
        max_y = max(min(world.height - half.y, world.height), min_y)
        return min_x, max_x, min_y, max_y

    def _clamp_to_world(self, position: Vector2D, world: World) -> Vector2D:
        min_x, max_x, min_y, max_y = self._world_limits(world)
        clamped_x = min(max(position.x, min_x), max_x)
        clamped_y = min(max(position.y, min_y), max_y)
        return Vector2D(clamped_x, clamped_y)

    def _interaction_radius(self, other: BaseEntity) -> float:
        robot_half = self._half_extents()
        other_half = Vector2D(0, 0)
        if hasattr(other, "collider") and other.collider:
            other_half = other.collider.half_extents
        return max(robot_half.x, robot_half.y) + max(other_half.x, other_half.y) + 4.0

    def _acquire_visitor(self) -> None:
        if self.target_visitor is not None:
            return
        best_visitor: Optional[Visitor] = None
        best_distance = float("inf")
        visitors = []
        for ride in self.simulation.rides:
            visitors.extend(ride.exit_queue.groups)
        visitors.extend(self.simulation.entrance_queue.groups)
        for visitor in visitors:
            if (
                visitor.state != Visitor.State.IN_QUEUE
                or visitor.assigned_robot not in (None, self)
            ): continue
            distance = (visitor.transform.position - self.transform.position).magnitude()
            if distance < best_distance:
                best_distance = distance
                best_visitor = visitor

        if best_visitor is not None:
            self.target_visitor = best_visitor
            self.target_visitor.assigned_robot = self
            self.state = Robot.State.ACQUIRED_VISITOR
        else:
            self._last_failure_step = self._current_step

    def _release_targets(self) -> bool:
        if self.target_visitor is None:
            return
        if self.state == Robot.State.ACQUIRED_VISITOR:
            self.target_visitor.assigned_robot = None
            self.target_visitor = None
        elif self.state == Robot.State.BOARDED:
            self.target_visitor.assigned_ride = None
            self.target_ride = None

    def _maybe_pickup_visitor(self) -> None:
        if self.target_visitor is None:
            return
        visitor = self.target_visitor
        distance = (visitor.transform.position - self.transform.position).magnitude()
        radius = self._interaction_radius(visitor)
        if distance <= radius:
            self.move_direction = Vector2D(0, 0)
            if visitor.queue_ref is not None:
                visitor.queue_ref.remove(visitor)
            visitor.assigned_robot = self
            visitor.state = Visitor.State.ON_BOARD
            self.state = Robot.State.BOARDED

    def _find_closest_ride(self, exclude_list: List[Ride] = []) -> Optional[Ride]:
        best_ride: Optional[Ride] = None
        best_distance = float("inf")
        for ride in self.simulation.rides:
            if (
                ride in exclude_list
                or (self.target_visitor and not ride.can_accept(self.target_visitor))
            ): continue
            dist = (ride.transform.position - self.transform.position).magnitude()
            if dist < best_distance:
                best_distance = dist
                best_ride = ride

        if best_ride is None:
            self._last_failure_step = self._current_step
        return best_ride

    def _maybe_dropoff_at_ride(self) -> None:
        if self.target_visitor is None or self.target_ride is None:
            return

        distance = (self.target_ride.entrance_queue.tail - self.transform.position).magnitude()
        if distance > self._interaction_radius(self.target_visitor):
            return

        if self.target_ride.can_accept(self.target_visitor):
            self.target_ride.entrance_queue.add(self.target_visitor)
            self.target_visitor.assigned_robot = None
            self.target_visitor.transform.set_position(self.target_ride.entrance_queue.tail)
            self.target_visitor.state = Visitor.State.QUEUEING
            self.target_ride = None
            self.target_visitor = None
            self.state = Robot.State.ROAMING
            self.move_direction = Vector2D(0, 0)
        else: # Try another ride
            self.target_ride = self._find_closest_ride(exclude_list=[self.target_ride])

    def _maybe_dropoff_at_exit(self) -> None:
        if self.target_visitor is None or self.simulation.exit_queue is None:
            return

        distance = (self.simulation.exit_queue.tail - self.transform.position).magnitude()
        if distance > self._interaction_radius(self.target_visitor):
            return

        if (
            self.simulation.exit_queue.can_accept(1)
            and self.simulation.exit_queue.can_accept_members(self.target_visitor.group_size)
            and self.simulation.exit_queue.can_fit(self.target_visitor)
        ):
            self.simulation.exit_queue.add(self.target_visitor)
            self.target_visitor.assigned_robot = None
            self.target_visitor.transform.set_position(self.simulation.exit_queue.tail)
            self.target_visitor.state = Visitor.State.QUEUEING
            self.target_ride = None
            self.target_visitor = None
            self.state = Robot.State.ROAMING
            self.move_direction = Vector2D(0, 0)
