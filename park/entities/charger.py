from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from park.entities.core import BaseEntity
from park.entities.robot import Robot
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.logic.queue import RobotQueue


class Charger(BaseEntity):
    def __init__(
        self,
        simulation,
        position: Vector2D,
        charge_queue: RobotQueue,
        charging_rate: float,
        cost_rate: float
    ):
        super().__init__(simulation, position)
        self._charge_queue = charge_queue
        self._charging_rate = charging_rate
        self._cost_rate = cost_rate

    @property
    def charge_queue(self) -> RobotQueue:
        return self._charge_queue

    def attach_robot(self, robot: Robot):
        robot.attached_charger = self
        self.charge_queue.add(robot)

    def update(self):
        super().update()
        self.charge_queue.update_targets()

        for robot in self.charge_queue.robots:
            if robot.battery_level >= robot.max_battery:
                robot.stop_charging()
                self.charge_queue.remove(robot)
            else:
                robot.charge(self._charging_rate)
                self.simulation.funds_delta -= self._cost_rate
