from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ai.pathfinding.astar import AStarStrategy
from ai.pathfinding.core import MovementPlan, MovementState, MovementStrategy
from ai.pathfinding.linear import LinearStrategy
from park.entities.core import BaseEntity
from park.entities.ride import Ride
from park.entities.visitor import Visitor
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.internal.collider import Collider
    from park.logic.node import Node
    from park.simulation import Simulation


class Robot(BaseEntity):
    class State(Enum):
        ROAMING = "roaming"
        ACQUIRED_VISITOR = "acquired_visitor"
        BOARDED = "boarded"

    _debug_path_robots: List[Robot] = []

    def __init__(
        self,
        simulation: Simulation,
        position: Vector2D,
        move_speed: float,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(simulation, position)
        self.move_speed = move_speed * 60
        self._move_direction = Vector2D(0, 0)
        self.state = Robot.State.ROAMING
        self.target_visitor: Optional[Visitor] = None
        self.target_ride: Optional[Ride] = None

        self._failure_cooldown = 30  # in simulation steps
        self._last_failure_step = -self._failure_cooldown

        self._linear_strategy = LinearStrategy(
            simulation.world.grid,
            allow_diagonal=True,
            path_smoothing=False,
            samples_per_segment=5,
            min_distance=1e-2
        )
        self._astar_strategy = AStarStrategy(
            simulation.world.grid,
            allow_diagonal=True,
            path_smoothing=False,
            samples_per_segment=5,
            min_distance=1e-2
        )
        self._current_strategy: MovementStrategy = self._linear_strategy

        self._target_position: Vector2D | None = None
        self._current_plan: MovementPlan | None = None

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self._debug_path = False

    @staticmethod
    def get_debug_path_robots() -> List[Robot]:
        return list(Robot._debug_path_robots)

    @property
    def debug_path(self) -> bool:
        return self._debug_path

    def set_debug_path(self, value: bool) -> None:
        self._debug_path = value
        if value and self not in Robot._debug_path_robots:
            Robot._debug_path_robots.append(self)
        elif not value and self in Robot._debug_path_robots:
            Robot._debug_path_robots.remove(self)

    @property
    def current_plan(self) -> MovementPlan | None:
        return self._current_plan

    def set_direction(self, direction: Vector2D):
        if direction.squared_magnitude() != 1.0:
            direction = direction.normalized()
        self._move_direction = direction

    def set_movement_strategy(self, strategy: MovementStrategy.Type):
        if strategy == MovementStrategy.Type.LINEAR:
            self._current_strategy = self._linear_strategy
        elif strategy == MovementStrategy.Type.ASTAR:
            self._current_strategy = self._astar_strategy
        else:
            raise ValueError(f"Unknown movement strategy: {strategy}")
        print(f"Robot {self.id} set movement strategy to {strategy}")

    def set_target_position(self, position: Vector2D):
        same_target = (
            self._target_position is not None
            and self._target_position == position
        )
        plan_active = (
            self._current_plan is not None
            and self._current_plan.peek_waypoint() is not None
        )

        if same_target and plan_active:
            return

        self._target_position = position
        force_walk_nodes: List[Node] = []
        if (
            self.state == Robot.State.ACQUIRED_VISITOR
            and self.target_visitor is not None
            and self.target_visitor.bounds is not None
        ):
            force_walk_nodes = self.simulation.world.grid.get_nodes_in_rect(
                self.target_visitor.bounds
            )
        elif (
            self.state == Robot.State.BOARDED
            and self.target_ride is not None
            and self.target_ride.bounds is not None
        ):
            force_walk_nodes = self.simulation.world.grid.get_nodes_in_rect(
                self.target_ride.bounds
            )

        self._current_plan = self._current_strategy.plan(
            MovementState(
                position=self.transform.position,
                velocity=self.rigidbody.velocity,
                target=self._target_position,
                force_walk_nodes=force_walk_nodes
            )
        )

    def clear_plan(self):
        self._target_position = None
        self._current_plan = None

    def random_world_position(self) -> Vector2D:
        world = self.simulation.world
        half_extents = self._half_extents()
        min_x = half_extents.x
        min_y = half_extents.y
        max_x = world.width - half_extents.x
        max_y = world.height - half_extents.y
        position = Vector2D(
            self.rng.uniform(min_x, max_x),
            self.rng.uniform(min_y, max_y)
        )
        return position

    def update(self):
        super().update()

        if (self._current_step - self._last_failure_step) < self._failure_cooldown:
            return

        if self.state == Robot.State.ROAMING:
            self._roam()
            self._acquire_visitor()
        elif self.state == Robot.State.ACQUIRED_VISITOR:
            if self.target_visitor is None:
                self._acquire_visitor()
            if self.target_visitor is not None:
                self.set_target_position(self.target_visitor.transform.position)
                self._maybe_pickup_visitor()
        elif self.state == Robot.State.BOARDED:
            if self.target_visitor.completed_rides < self.target_visitor.desired_rides:
                if self.target_ride is None:
                    self.target_ride = self._find_closest_ride()
                if self.target_ride is not None:
                    self.set_target_position(self.target_ride.entrance_queue.tail)
                    self._maybe_dropoff_at_ride()
            elif self.simulation.exit_queue is not None:
                self.set_target_position(self.simulation.exit_queue.tail)
                self._maybe_dropoff_at_exit()

        if self._current_plan is not None:
            current_waypoint = self._current_plan.peek_waypoint()
            current_waypoint = self._clamp_to_world(current_waypoint)
            if self.transform.position == current_waypoint:
                next_waypoint = self._current_plan.next_waypoint()
                if next_waypoint is not None:
                    next_waypoint = self._clamp_to_world(next_waypoint)
                    self.rigidbody.move_position(next_waypoint, speed=self.move_speed)
                else:
                    self.clear_plan()
            else:
                self.rigidbody.move_position(current_waypoint, speed=self.move_speed)

    def on_collision_enter(self, other: Collider):
        if (
            self.state == Robot.State.ACQUIRED_VISITOR
            or self.state == Robot.State.BOARDED
        ):
            self._release_targets()

        self.clear_plan()
        self._last_failure_step = self._current_step

    def _roam(self):
        if self._current_plan is None:
            self.set_target_position(self.random_world_position())

    def _random_direction(self) -> Vector2D:
        angle = self.rng.uniform(0, 2 * np.pi)
        return Vector2D(np.cos(angle), np.sin(angle))

    def _half_extents(self) -> Vector2D:
        if self.collider and hasattr(self.collider, "half_extents"):
            return self.collider.half_extents
        size = self.collider.size if self.collider else Vector2D(0, 0)
        return Vector2D(size.x * 0.5, size.y * 0.5)

    def _clamp_to_world(self, position: Vector2D) -> Vector2D | None:
        if position is None:
            return None

        half_extents = self._half_extents()
        clamped_x = max(half_extents.x, min(self.simulation.world.width - half_extents.x, position.x))
        clamped_y = max(half_extents.y, min(self.simulation.world.height - half_extents.y, position.y))
        return Vector2D(clamped_x, clamped_y)

    def _interaction_radius(self, other: BaseEntity) -> float:
        robot_half = self._half_extents()
        other_half = Vector2D(0, 0)
        if hasattr(other, "collider") and other.collider:
            other_half = other.collider.half_extents
        return max(robot_half.x, robot_half.y) + max(other_half.x, other_half.y) + 6.0

    def _acquire_visitor(self) -> None:
        if self.target_visitor is not None:
            return
        closest_visitor: Optional[Visitor] = None
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
                closest_visitor = visitor

        if closest_visitor is not None:
            self.target_visitor = closest_visitor
            self.target_visitor.assigned_robot = self
            self.state = Robot.State.ACQUIRED_VISITOR
        # else:
            # self._last_failure_step = self._current_step

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
            # self._move_direction = Vector2D(0, 0)
            self.clear_plan()
            if visitor.queue_ref is not None:
                visitor.queue_ref.remove(visitor)
            visitor.assigned_robot = self
            visitor.state = Visitor.State.ON_BOARD
            self.state = Robot.State.BOARDED

    def _find_closest_ride(self, exclude_list: List[Ride] = []) -> Optional[Ride]:
        closest_ride: Optional[Ride] = None
        best_distance = float("inf")
        for ride in self.simulation.rides:
            if (
                ride in exclude_list
                or (self.target_visitor and not ride.can_accept(self.target_visitor))
            ): continue
            dist = (ride.entrance_queue.tail - self.transform.position).magnitude()
            if dist < best_distance:
                best_distance = dist
                closest_ride = ride

        # if closest_ride is None:
            # self._last_failure_step = self._current_step
        return closest_ride

    def _maybe_dropoff_at_ride(self) -> None:
        if self.target_visitor is None or self.target_ride is None:
            return

        distance = (self.target_ride.entrance_queue.tail - self.transform.position).magnitude()
        if distance > self._interaction_radius(self.target_visitor):
            return

        if self.target_ride.can_accept(self.target_visitor):
            # self._move_direction = Vector2D(0, 0)
            self.clear_plan()
            self.target_ride.entrance_queue.add(self.target_visitor)
            self.target_visitor.assigned_robot = None
            self.target_visitor.transform.set_position(self.target_ride.entrance_queue.tail)
            self.target_visitor.state = Visitor.State.QUEUEING
            self.target_ride = None
            self.target_visitor = None
            self.state = Robot.State.ROAMING
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
            # self._move_direction = Vector2D(0, 0)
            self.clear_plan()
            self.simulation.exit_queue.add(self.target_visitor)
            self.target_visitor.assigned_robot = None
            self.target_visitor.transform.set_position(self.simulation.exit_queue.tail)
            self.target_visitor.state = Visitor.State.QUEUEING
            self.target_ride = None
            self.target_visitor = None
            self.state = Robot.State.ROAMING
