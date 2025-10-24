from __future__ import annotations

from enum import Enum, IntFlag
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ai.fsm.core import Machine
from ai.pathfinding.astar import AStarStrategy
from ai.pathfinding.bfs import BreadthFirstStrategy
from ai.pathfinding.core import MovementPlan, MovementState, MovementStrategy
from ai.pathfinding.dfs import DepthFirstStrategy
from ai.pathfinding.linear import LinearStrategy
from park.entities.core import BaseEntity
from park.entities.ride import Ride
from park.entities.visitor import Visitor
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.entities.charger import Charger
    from park.internal.collider import Collider
    from park.logic.node import Node
    from park.simulation import Simulation


class Robot(BaseEntity):
    class State(Enum):
        ROAMING = "roaming"
        PICK_VISITOR = "pick_visitor"
        PICK_RIDE = "pick_ride"
        CHARGING = "charging"

    class Trigger(IntFlag):
        VISITOR_IN_QUEUE = 1 << 1
        VISITOR_ON_BOARD = 1 << 2
        VISITOR_DROP_OFF = 1 << 3
        LOW_BATTERY = 1 << 4
        FULL_BATTERY = 1 << 5

    _tooltip_robots: set["Robot"] = set()

    def __init__(
        self,
        simulation: Simulation,
        position: Vector2D,
        move_speed: float,
        max_health: float = 100.0,
        max_battery: float = 100.0,
        low_battery_threshold: float = 0.15,
        respawn_cost: float = 1000.0,
        repair_cost: float = 150.0,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(simulation, position)
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.move_speed = move_speed * 60

        self.state_machine = Machine(
            states=Robot.State,
            initial_state=Robot.State.ROAMING
        )
        self._setup_state_transitions()

        self.attached_charger: Optional[Charger] = None
        self.target_charger: Optional[Charger] = None
        self.target_ride: Optional[Ride] = None
        self.target_visitor: Optional[Visitor] = None

        self._max_health = max_health
        self._max_battery = max_battery

        self._health = max_health
        self._battery_level = rng.uniform(0.8 * max_battery, max_battery)
        # self._battery_level = low_battery_threshold * max_battery  # For charger testing

        self._battery_base_drain = 0.02  # per tick
        self._low_battery_threshold = low_battery_threshold

        self._respawn_cost = respawn_cost
        self._repair_cost = repair_cost

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
        self._bfs_strategy = BreadthFirstStrategy(
            simulation.world.grid,
            allow_diagonal=True,
            path_smoothing=False,
            samples_per_segment=5,
            min_distance=1e-2
        )
        self._dfs_strategy = DepthFirstStrategy(
            simulation.world.grid,
            allow_diagonal=True,
            path_smoothing=False,
            samples_per_segment=5,
            min_distance=1e-2
        )
        self._current_strategy: MovementStrategy = self._linear_strategy

        self._target_position: Vector2D | None = None
        self._current_plan: MovementPlan | None = None

        self._tooltip_visible = False

    @staticmethod
    def get_tooltip_robots() -> List[Robot]:
        return list(Robot._tooltip_robots)

    @property
    def tooltip_visible(self) -> bool:
        return self._tooltip_visible

    def set_tooltip_visible(self, value: bool) -> None:
        self._tooltip_visible = value
        if value:
            Robot._tooltip_robots.add(self)
        elif not value:
            Robot._tooltip_robots.discard(self)

    @property
    def max_health(self) -> float:
        return self._max_health

    @property
    def max_battery(self) -> float:
        return self._max_battery

    @property
    def health(self) -> float:
        return self._health

    @property
    def health_percentage(self) -> float:
        return (
            100 * (self._health / self._max_health) if self._max_health > 0
            else 0.0
        )

    @property
    def battery_level(self) -> float:
        return self._battery_level

    @property
    def battery_percentage(self) -> float:
        return (
            100 * (self._battery_level / self._max_battery) if self._max_battery > 0
            else 0.0
        )

    @property
    def battery_drain_rate(self) -> float:
        drain_rate = self._battery_base_drain
        if self.state.value == Robot.State.PICK_RIDE.value:
            drain_rate *= 1.5  # 50% more drain when carrying visitor to ride
        if self.state.value == Robot.State.CHARGING.value:
            drain_rate *= 0.75  # 25% less drain when going to charger
        return drain_rate

    @property
    def state(self) -> Robot.State:
        return self.state_machine.current_state

    @property
    def current_plan(self) -> MovementPlan | None:
        return self._current_plan

    def charge(self, charge_amount: float):
        self._battery_level = min(
            self._max_battery,
            self._battery_level + charge_amount
        )

    def stop_charging(self):
        if self.state.value == Robot.State.CHARGING.value:
            if self.rigidbody is not None:
                self.rigidbody.set_static(False)

            if self.target_visitor is not None:
                self.state_machine.trigger(
                    Robot.Trigger.FULL_BATTERY
                    | Robot.Trigger.VISITOR_ON_BOARD
                )
            else:
                self.state_machine.trigger(Robot.Trigger.FULL_BATTERY)
        if self.attached_charger is not None:
            self.attached_charger = None

    def set_movement_strategy(self, strategy: MovementStrategy.Type):
        if strategy == MovementStrategy.Type.LINEAR:
            self._current_strategy = self._linear_strategy
        elif strategy == MovementStrategy.Type.ASTAR:
            self._current_strategy = self._astar_strategy
        elif strategy == MovementStrategy.Type.BFS:
            self._current_strategy = self._bfs_strategy
        elif strategy == MovementStrategy.Type.DFS:
            self._current_strategy = self._dfs_strategy
        else:
            raise ValueError(f"Unknown movement strategy: {strategy}")

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
        force_walk_nodes: List[Node] = self.simulation.world.grid.get_nodes_in_rect(
            self.bounds
        )
        if (
            self.state.value == Robot.State.PICK_VISITOR.value
            and self.target_visitor is not None
            and self.target_visitor.bounds is not None
        ):
            force_walk_nodes += self.simulation.world.grid.get_nodes_in_rect(
                self.target_visitor.bounds
            )
        elif (
            self.state.value == Robot.State.PICK_RIDE.value
            and self.target_ride is not None
            and self.target_ride.bounds is not None
        ):
            force_walk_nodes += self.simulation.world.grid.get_nodes_in_rect(
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

    def set_plan(self, plan: MovementPlan):
        self._current_plan = plan

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

        if not (
            self.state.value == Robot.State.CHARGING.value
            and self.attached_charger is not None
        ):
            self._battery_level -= self.battery_drain_rate
            if self.battery_percentage <= self._low_battery_threshold * 100.0:
                self.state_machine.trigger(Robot.Trigger.LOW_BATTERY)
            if self._battery_level <= 0:
                self._respawn()

        if self.state.value == Robot.State.ROAMING.value:
            self._roam()
            self._find_best_visitor()
        elif self.state.value == Robot.State.PICK_VISITOR.value:
            if self.target_visitor is None:
                self._find_best_visitor()
            if self.target_visitor is not None:
                self.set_target_position(self.target_visitor.transform.position)
                self._maybe_pickup_visitor()
        elif self.state.value == Robot.State.PICK_RIDE.value:
            if self.target_visitor.completed_rides < self.target_visitor.desired_rides:
                if self.target_ride is None:
                    self.target_ride = self._find_best_ride()
                if self.target_ride is not None:
                    if not self._is_waypoint_ok(self.target_ride.entrance_queue.tail):
                        self.target_ride = self._find_best_ride(
                            exclude_list=[self.target_ride]
                        )
                    if self.target_ride is not None:
                        self.set_target_position(self.target_ride.entrance_queue.tail)
                        self._maybe_dropoff_at_ride()
            elif self.simulation.exit_queue is not None:
                self.set_target_position(self.simulation.exit_queue.tail)
                self._maybe_dropoff_at_exit()
        elif self.state.value == Robot.State.CHARGING.value:
            if self.target_charger is None:
                self.target_charger = self._find_best_charger()
            if self.target_charger is not None:
                if self.attached_charger is None:
                    self.set_target_position(self.target_charger.charge_queue.tail)
                self._maybe_attach_to_charger()
            else:
                self.clear_plan()
                self.rigidbody.set_velocity(Vector2D(0, 0))

        if self._current_plan is not None:
            current_waypoint = self._current_plan.peek_waypoint()
            current_waypoint = self._clamp_to_world(current_waypoint)
            if current_waypoint is None:
                self.clear_plan()
            else:
                if self.transform.position == current_waypoint:
                    next_waypoint = self._current_plan.next_waypoint()
                    next_waypoint = self._clamp_to_world(next_waypoint)
                    if next_waypoint is not None:
                        if not self._is_waypoint_ok(next_waypoint):
                            self._replan_path()
                        else:
                            self.rigidbody.move_position(next_waypoint, speed=self.move_speed)
                    else:
                        self.clear_plan()
                else:
                    if not self._is_waypoint_ok(current_waypoint):
                        self._replan_path()
                    else:
                        self.rigidbody.move_position(current_waypoint, speed=self.move_speed)

    def on_collision_enter(self, other: Collider):
        if (
            self.target_visitor is not None
            and other.transform == self.target_visitor.transform
        ):
            if self.state.value == Robot.State.PICK_VISITOR.value:
                self._maybe_pickup_visitor()
            return

        if (
            self.state.value == Robot.State.PICK_VISITOR.value
            or self.state.value == Robot.State.PICK_RIDE.value
            or self.state.value == Robot.State.CHARGING.value
        ):
            self._release_targets()

        if (
            self.state.value != Robot.State.CHARGING.value
            or self.attached_charger is None
        ):
            # damage robot depending on velocity
            if not self.rigidbody.is_static:
                impact_velocity = self.rigidbody.velocity.magnitude()
                damage = impact_velocity * 0.05
                self._health -= damage
            if self._health <= 0:
                self._respawn()

        self.clear_plan()
        if (self._current_step - self._last_failure_step) >= self._failure_cooldown:
            self._last_failure_step = self._current_step

    def _setup_state_transitions(self):
        self.state_machine.add_transition(
            sources=Robot.State.ROAMING,
            dest=Robot.State.PICK_VISITOR,
            trigger=Robot.Trigger.VISITOR_IN_QUEUE
        )
        self.state_machine.add_transition(
            sources=Robot.State.PICK_VISITOR,
            dest=Robot.State.PICK_RIDE,
            trigger=Robot.Trigger.VISITOR_ON_BOARD
        )
        self.state_machine.add_transition(
            sources=Robot.State.PICK_RIDE,
            dest=Robot.State.ROAMING,
            trigger=Robot.Trigger.VISITOR_DROP_OFF
        )

        self.state_machine.add_transition(
            sources=[
                Robot.State.ROAMING,
                Robot.State.PICK_VISITOR,
                Robot.State.PICK_RIDE
            ],
            dest=Robot.State.CHARGING,
            trigger=Robot.Trigger.LOW_BATTERY
        )
        self.state_machine.add_transition(
            sources=Robot.State.CHARGING,
            dest=Robot.State.ROAMING,
            trigger=Robot.Trigger.FULL_BATTERY
        )
        self.state_machine.add_transition(
            sources=Robot.State.CHARGING,
            dest=Robot.State.PICK_RIDE,
            trigger=(
                Robot.Trigger.FULL_BATTERY
                | Robot.Trigger.VISITOR_ON_BOARD
            )
        )

    def _is_waypoint_ok(self, waypoint: Vector2D) -> bool:
        node = self.simulation.world.grid.world_to_node(waypoint)
        if node is None:
            return False
        if (
            not node.walkable
            and (
                self.collider not in node.occupants
                and (self.target_visitor is not None
                and self.target_visitor.collider not in node.occupants)
            )
        ):
            return False
        return True

    def _replan_path(self):
        if self.state.value == Robot.State.ROAMING.value:
            self.clear_plan()
            self._roam()
            return
        if (
            self.state.value == Robot.State.PICK_RIDE.value
            and self.target_ride is not None
        ):
            self.target_ride = None
            self.clear_plan()
            return
        if self._target_position is not None:
            old_target = self._target_position
            self.clear_plan()
            self.set_target_position(old_target)
            return

    def _respawn(self):
        self.simulation.funds -= self._respawn_cost
        self._health = self._max_health
        self._battery_level = self._max_battery
        self.clear_plan()
        if self.rigidbody is not None:
            self.rigidbody.set_static(False)

        self.transform.set_position(self.random_world_position())
        if (
            self.state.value == Robot.State.PICK_RIDE.value
            and self.target_visitor is not None
        ):
            self.simulation.remove_visitor(self.target_visitor)
            self.target_visitor.delete()
            self.target_visitor = None
        self._release_targets()
        self.state_machine.reset()

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

    def _interaction_radius(self, other: Optional[BaseEntity]) -> float:
        robot_half = self._half_extents()
        other_half = Vector2D(0, 0)
        if other is not None and hasattr(other, "collider") and other.collider:
            other_half = other.collider.half_extents
        return max(robot_half.x, robot_half.y) + max(other_half.x, other_half.y) + 8.0

    def _get_visitor_score(self, visitor: Visitor) -> float:
        distance = (visitor.transform.position - self.transform.position).magnitude()
        score = distance
        return score

    def _get_ride_score(self, ride: Ride) -> float:
        distance = (ride.entrance_queue.tail - self.transform.position).magnitude()
        score = distance
        return score

    def _get_charger_score(self, charger: Charger) -> float:
        distance = (charger.charge_queue.tail - self.transform.position).magnitude()
        score = distance
        return score

    def _find_best_visitor(self) -> None:
        if self.target_visitor is not None:
            return
        best_visitor: Optional[Visitor] = None
        best_score = float("inf")
        visitors: List[Visitor] = []
        for ride in self.simulation.rides:
            visitors.extend(ride.exit_queue.groups)
        visitors.extend(self.simulation.entrance_queue.groups)
        for visitor in visitors:
            if (
                visitor.state != Visitor.State.IN_QUEUE
                or visitor.assigned_robot not in (None, self)
            ): continue
            score = self._get_visitor_score(visitor)
            if score < best_score:
                best_score = score
                best_visitor = visitor

        if best_visitor is not None:
            self.target_visitor = best_visitor
            self.target_visitor.assigned_robot = self
            self.state_machine.trigger(Robot.Trigger.VISITOR_IN_QUEUE)
        else:
            self.state_machine.set_state(Robot.State.ROAMING)

    def _release_targets(self) -> bool:
        if self.target_visitor is not None:
            if self.state.value == Robot.State.PICK_VISITOR.value:
                self.target_visitor.assigned_robot = None
                self.target_visitor = None
            elif self.state.value == Robot.State.PICK_RIDE.value:
                self.target_visitor.assigned_ride = None
                self.target_ride = None

        if self.state.value == Robot.State.CHARGING.value:
            if (
                self.target_charger is not None
                and self.attached_charger is None
            ):
                self.target_charger = None

    def _maybe_pickup_visitor(self) -> None:
        if self.target_visitor is None:
            return
        visitor = self.target_visitor
        distance = (visitor.transform.position - self.transform.position).magnitude()
        radius = self._interaction_radius(visitor)
        if distance <= radius:
            self.clear_plan()
            if visitor.queue_ref is not None:
                visitor.queue_ref.remove(visitor)
            visitor.assigned_robot = self
            visitor.state = Visitor.State.ON_BOARD
            self.state_machine.trigger(Robot.Trigger.VISITOR_ON_BOARD)

    def _find_best_ride(self, exclude_list: List[Ride] = []) -> Optional[Ride]:
        best_ride: Optional[Ride] = None
        best_score = float("inf")
        for ride in self.simulation.rides:
            if (
                ride in exclude_list
                or (self.target_visitor and not ride.can_accept(self.target_visitor))
            ): continue
            score = self._get_ride_score(ride)
            if score < best_score:
                best_score = score
                best_ride = ride

        if best_ride is None:
            self.clear_plan()
            self._last_failure_step = self._current_step
        return best_ride

    def _maybe_dropoff_at_ride(self) -> None:
        if self.target_visitor is None or self.target_ride is None:
            return

        distance = (self.target_ride.entrance_queue.tail - self.transform.position).magnitude()
        if distance > self._interaction_radius(self.target_visitor):
            return

        if self.target_ride.can_accept(self.target_visitor):
            self.clear_plan()
            self.target_ride.entrance_queue.add(self.target_visitor)
            self.target_visitor.assigned_robot = None
            self.target_visitor.transform.set_position(self.target_ride.entrance_queue.tail)
            self.target_visitor.state = Visitor.State.QUEUEING
            self.simulation.funds += self.target_ride.entry_price * self.target_visitor.group_size

            self.target_ride = None
            self.target_visitor = None
            self.state_machine.trigger(Robot.Trigger.VISITOR_DROP_OFF)
        else: # Try another ride
            self.target_ride = self._find_best_ride(exclude_list=[self.target_ride])

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
            self.clear_plan()
            self.simulation.exit_queue.add(self.target_visitor)
            self.target_visitor.assigned_robot = None
            self.target_visitor.transform.set_position(self.simulation.exit_queue.tail)
            self.target_visitor.state = Visitor.State.QUEUEING
            self.target_ride = None
            self.target_visitor = None
            self.state_machine.trigger(Robot.Trigger.VISITOR_DROP_OFF)

    def _find_best_charger(self, exclude_list: List[Charger] = []) -> Optional[Charger]:
        best_charger: Optional[Charger] = None
        best_score = float("inf")
        for charger in self.simulation.chargers:
            if (
                charger in exclude_list
                or not charger.charge_queue.can_accept(1)
                or not charger.charge_queue.can_fit(self)
            ):
                continue
            score = self._get_charger_score(charger)
            if score < best_score:
                best_score = score
                best_charger = charger

        if best_charger is not None:
            self.target_charger = best_charger
        return best_charger

    def _maybe_attach_to_charger(self) -> None:
        if (
            self.target_charger is None
            or self.attached_charger is not None
        ):
            return

        distance = (self.target_charger.charge_queue.tail - self.transform.position).magnitude()
        if distance > self._interaction_radius(self):
            return

        if (
            self.target_charger.charge_queue.can_accept(1)
            and self.target_charger.charge_queue.can_fit(self)
        ):
            self.clear_plan()
            self.target_charger.attach_robot(self)
            # self.transform.set_position(self.target_charger.charge_queue.tail)
            # if self.rigidbody is not None:
            #     self.rigidbody.set_static(True)

            # Repair the robot if damaged
            repair_cost = self._repair_cost * (1 - (self._health / self._max_health) ** 2)
            self.simulation.funds -= repair_cost
            self._health = self._max_health
        else:
            self.target_charger = self._find_best_charger(exclude_list=[self.target_charger])
