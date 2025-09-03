from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------- Entities ----------
@dataclass
class Attraction:
    name: str
    x: float
    y: float
    visit_time: float
    capacity: int


@dataclass
class Robot:
    id: int
    x: float
    y: float
    speed: float
    controller: "RobotController" = None
    state: "RobotState" = None


@dataclass
class Visitor:
    id: int
    x: float
    y: float
    group_size: int
    state: "VisitorState" = None


# ---------- Simulation ----------
class World:
    def __init__(
        self,
        width: int,
        height: int,
        rng: Optional[np.random.Generator] = None
    ):
        self.width = width
        self.height = height

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.attractions: List[Attraction] = []
        self.robots: List[Robot] = []
        self.visitors: List[Visitor] = []

    def spawn_demo(self, n_visitors=30, n_robots=5):
        self.attractions = [
            Attraction(
                "Carousel",
                350, 250, visit_time=12, capacity=10
            ),
            Attraction(
                "RollerCo",
                800, 200, visit_time=10, capacity=5
            ),
            Attraction(
                "Haunted",
                1250, 250, visit_time=8, capacity=8
            ),
            Attraction(
                "Bumper",
                400, 700, visit_time=5, capacity=12
            ),
            Attraction(
                "Ferris",
                900, 650, visit_time=14, capacity=6
            ),
            Attraction(
                "Arcade",
                1400, 700, visit_time=6, capacity=15
            ),
        ]

        self.visitors = []
        for i in range(n_visitors):
            x = self.rng.uniform(100, 300)
            y = self.rng.uniform(150, self.height - 150)
            size = int(self.rng.integers(1, 6))
            self.visitors.append(Visitor(i, x, y, group_size=size))

        self.robots = []
        for r in range(n_robots):
            rx = 80
            ry = 150 + r * (self.height - 300) / max(1, n_robots - 1)
            speed = self.rng.uniform(50, 80)
            self.robots.append(Robot(r, rx, ry, speed=speed))


class Simulation:
    def __init__(self, world_width, world_height):
        self.world = World(world_width, world_height)
        self.world.spawn_demo()

    def step(self, delta_time: float):
        pass
