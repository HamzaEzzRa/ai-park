from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

from park.entities.ride import Ride
from park.entities.robot import Robot
from park.entities.visitor import Visitor
from park.internal.collider import BoxCollider
from park.internal.layers import CollisionLayer
from park.internal.math import Vector2D
from park.internal.rigidbody import RigidBody
from park.internal.sprite import Sprite, SpriteShape
from park.logic.queue import VisitorQueue
from park.render import Colors

if TYPE_CHECKING:
    from park.world import World


class Simulation:
    def __init__(
            self,
            world: World,
            rng: Optional[np.random.Generator] = None,
        ):
        self.world = world
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.rides: List[Ride] = []
        self.robots: List[Robot] = []
        self.visitors: List[Visitor] = []

        self.left_visitors: int = 0
        self.left_satisfaction: float = 0.0

        self.current_step = 0

        # Entrance queue configuration
        entrance_queue_head = Vector2D(
            self.world.width // 2,
            self.world.height - self.world.cell_size // 2
        )
        entrance_queue_tail = Vector2D(
            self.world.width,
            self.world.height - self.world.cell_size // 2
        )
        entrance_queue_spacing = max(self.world.cell_size // 16, 4.0)
        self.entrance_queue = VisitorQueue(
            world=self.world,
            head=entrance_queue_head,
            tail=entrance_queue_tail,
            spacing=entrance_queue_spacing
        )

        # Exit queue
        exit_queue_head = Vector2D(self.world.width // 2, 0.0)
        exit_queue_tail = Vector2D(self.world.width // 2, self.world.cell_size)
        exit_queue_spacing = max(self.world.cell_size // 16, 4.0)
        self.exit_queue = VisitorQueue(
            world=self.world,
            head=exit_queue_head,
            tail=exit_queue_tail,
            spacing=exit_queue_spacing
        )

        self.spawn_rate = 0.005
        self.dynamic_spawn = True
        self._spawn_accum = 0.0

    def spawn_ride(
        self,
        name: str,
        position: Vector2D,
        capacity: int,
        duration: int,
        entrance_queue: VisitorQueue,
        exit_queue: VisitorQueue
    ):
        ride = Ride(
            simulation=self,
            name=name,
            position=position,
            capacity=capacity,
            duration=duration,
            entrance_queue=entrance_queue,
            exit_queue=exit_queue
        )
        sprite_size = Vector2D(24.0, 24.0)
        ride.attach_component(
            Sprite(
                size=sprite_size,
                color=Colors.ride,
                shape=SpriteShape.RECT,
                data={"label": ride.name},
            )
        )
        ride.attach_component(BoxCollider(
            size=sprite_size,
            layer_bits=CollisionLayer.RIDE,
            mask_bits=CollisionLayer.ROBOT
        ))
        ride.attach_component(RigidBody(mass=0.0, is_static=True))
        self.rides.append(ride)

    def spawn_robot(
        self,
        position: Optional[Vector2D] = None,
        move_speed: Optional[float] = None
    ):
        sprite_size = Vector2D(18.0, 18.0)
        if position is None:
            position = Vector2D(
                self.rng.uniform(
                    sprite_size.x // 2,
                    self.world.width - sprite_size.x // 2
                ),
                self.rng.uniform(
                    sprite_size.y // 2,
                    self.world.height - sprite_size.y // 2
                )
            )
        if move_speed is None:
            move_speed = 1.0

        robot = Robot(
            simulation=self,
            position=position,
            move_speed=move_speed
        )
        robot.attach_component(
            Sprite(
                size=sprite_size,
                color=Colors.robot,
                shape=SpriteShape.RECT,
            )
        )
        robot.attach_component(BoxCollider(
            size=sprite_size,
            layer_bits=CollisionLayer.ROBOT,
            mask_bits=CollisionLayer.ALL_BITS
        ))
        robot.attach_component(RigidBody(
            mass=1.0,
            is_static=False,
            friction=0.0,
            bounciness=1.0
        ))
        self.robots.append(robot)

    def spawn_visitor(
        self,
        group_size: Optional[int] = None,
        move_speed: float = 0.5,
        desired_rides: Optional[int] = None
    ) -> bool:
        if group_size is None:
            group_size = self.rng.integers(1, 4)
        if desired_rides is None:
            desired_rides = self.rng.integers(1, 4)

        visitor = Visitor(
            simulation=self,
            position=self.entrance_queue.tail,
            group_size=group_size,
            move_speed=move_speed,
            desired_rides=desired_rides
        )
        visitor._member_spacing = 8.0
        group_bounds = visitor.group_bounds()
        visitor.attach_component(
            Sprite(
                size=group_bounds,
                color=Colors.visitor,
                shape=SpriteShape.GROUP,
                data={"spacing": visitor._member_spacing, "members": visitor.group_size},
            )
        )
        # Check if we can add to entrance queue
        if not (
            self.entrance_queue.can_accept(1)
            and self.entrance_queue.can_accept_members(visitor.group_size)
            and self.entrance_queue.can_fit(visitor)
        ):
            visitor.delete()
            return False

        visitor.attach_component(BoxCollider(
            size=group_bounds,
            layer_bits=CollisionLayer.VISITOR,
            mask_bits=CollisionLayer.ROBOT
        ))
        visitor.attach_component(RigidBody(is_static=True))
        self.entrance_queue.add(visitor)
        self.visitors.append(visitor)
        return True

    def remove_visitor(self, visitor: Visitor) -> None:
        if visitor in self.visitors:
            self.left_visitors += 1
            self.left_satisfaction += visitor.satisfaction
            if visitor.queue_ref is not None:
                visitor.queue_ref.remove(visitor)
            self.visitors.remove(visitor)

    def step(self):
        self.current_step += 1
        self._maybe_spawn_visitors()
        self.entrance_queue.update_targets()
        self.exit_queue.update_targets()
        for ride in self.rides:
            ride.update()
        for robot in self.robots:
            robot.update()
        for visitor in self.visitors:
            visitor.update()
        self.world.update()

    def set_spawn_rate(self, rate: float):
        self.spawn_rate = max(0.0, float(rate))

    def set_dynamic_spawn(self, enabled: bool):
        self.dynamic_spawn = bool(enabled)

    def get_avg_satisfaction(self) -> float:
        vals = [v.satisfaction for v in self.visitors]
        final_value = (
            (sum(vals) + self.left_satisfaction) / (len(vals) + self.left_visitors)
            if (vals or self.left_visitors) else 0.0
        )
        return final_value

    def get_members_in_queues(self) -> int:
        members = 0
        members += self.entrance_queue.member_count
        for ride in self.rides:
            members += ride.entrance_queue.member_count
            members += ride.exit_queue.member_count
        return members

    def get_members_on_rides(self) -> int:
        members = 0
        for ride in self.rides:
            members += ride.riders.member_count
        return members

    def _effective_spawn_rate(self) -> float:
        rate = self.spawn_rate
        if self.dynamic_spawn:
            avg_sat = self.get_avg_satisfaction()  # 0..1
            rate *= (0.5 + avg_sat)  # scale with avg satisfaction from 0.5x to 1.5x
        return rate

    def _maybe_spawn_visitors(self):
        # Compute expected arrivals this step
        rate = self._effective_spawn_rate()
        expected_per_step = rate / 60.0
        if expected_per_step <= 0:
            return

        # Accumulate fractional expected visitors for smooth spawning
        self._spawn_accum += expected_per_step
        spawn_n = int(self._spawn_accum)
        self._spawn_accum -= spawn_n
        # Probabilistic extra spawn for remainder
        if self.rng.random() < self._spawn_accum:
            spawn_n += 1
            self._spawn_accum = 0.0

        for _ in range(spawn_n):
            if not self.spawn_visitor(move_speed=0.5):
                # Queue is at capacity; stop spawning this step
                break
