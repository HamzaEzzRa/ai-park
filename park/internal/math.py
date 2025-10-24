from dataclasses import dataclass

import numpy as np


@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        if scalar == 0:
            return Vector2D(self.x, self.y)
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    @staticmethod
    def zero() -> "Vector2D":
        return Vector2D(0.0, 0.0)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def copy(self) -> "Vector2D":
        return Vector2D(self.x, self.y)

    def squared_magnitude(self) -> float:
        return self.x**2 + self.y**2

    def magnitude(self) -> float:
        return np.sqrt(self.squared_magnitude())

    def normalized(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def distance_from(self, other: "Vector2D") -> float:
        return (self - other).magnitude()


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    def contains(self, point: Vector2D) -> bool:
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.x + self.width < other.x
            or self.x > other.x + other.width
            or self.y + self.height < other.y
            or self.y > other.y + other.height
        )

    @staticmethod
    def from_center(
        center: Vector2D,
        size: Vector2D
    ) -> "Rect":
        return Rect(
            center.x - size.x / 2,
            center.y - size.y / 2,
            size.x,
            size.y
        )
