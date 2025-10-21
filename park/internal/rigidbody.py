from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from park.internal.collider import Collider
from park.internal.math import Vector2D

if TYPE_CHECKING:
    from park.internal.transform import Transform


class RigidBody:
    def __init__(
        self,
        mass: float = 1.0,
        is_static: bool = False,
        bounciness: float = 0.0,
        friction: float = 0.0,
        freeze_position: List[bool] = [False, False]  # x, y
    ):
        self._transform: Transform = None
        self._velocity: Vector2D = Vector2D(0.0, 0.0)
        self._mass: float = mass
        self._is_static: bool = is_static
        self._bounciness: float = bounciness
        self._friction: float = friction
        self._freeze_position: List[bool] = freeze_position
        self._enabled: bool = True

    @property
    def transform(self) -> Transform:
        return self._transform

    @transform.setter
    def transform(self, transform: Transform) -> None:
        self._transform = transform

    @property
    def colliders(self) -> List[Collider]:
        if self.transform is None:
            raise ValueError("RigidBody has no associated Transform")
        _colliders = self.transform.get_components(Collider)
        if not _colliders:
            raise ValueError("RigidBody has no Collider associated with its Transform")
        return _colliders

    @property
    def velocity(self) -> Vector2D:
        return self._velocity

    def set_velocity(self, velocity: Vector2D) -> None:
        self._move_target = None
        self._move_speed = None
        self._velocity = self._apply_linear_constraints(velocity)

    @property
    def mass(self) -> float:
        return self._mass

    def set_mass(self, mass: float) -> None:
        self._mass = mass

    @property
    def inverse_mass(self) -> float:
        if self._is_static or self._mass <= 0.0:
            return 0.0
        return 1.0 / self._mass

    @property
    def is_static(self) -> bool:
        return self._is_static

    def set_static(self, is_static: bool) -> None:
        self._is_static = is_static

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def bounciness(self) -> float:
        return self._bounciness

    def set_bounciness(self, bounciness: float) -> None:
        self._bounciness = max(0.0, min(1.0, bounciness))

    @property
    def friction(self) -> float:
        return self._friction

    def set_friction(self, friction: float) -> None:
        self._friction = max(0.0, friction)

    @property
    def position(self) -> Vector2D:
        return self.transform.position

    def apply_impulse(self, impulse: Vector2D) -> None:
        if self.is_static or self.mass <= 0:
            return
        self.set_velocity(
            self._velocity + (impulse / self.mass)
        )

    def apply_force(self, force: Vector2D, delta_time: float) -> None:
        if self.is_static or self.mass <= 0:
            return
        acceleration = force / self.mass
        self.set_velocity(
            self._velocity + acceleration * delta_time
        )

    def move_position(self, position: Vector2D, speed: float = None) -> None:
        self._move_target = position
        self._move_speed = speed

    def translate(self, delta: Vector2D) -> None:
        if not self.enabled:
            return

        new_position = Vector2D(
            self.position.x + delta.x,
            self.position.y + delta.y,
        )
        if self._freeze_position[0]:
            new_position.x = self.position.x
        if self._freeze_position[1]:
            new_position.y = self.position.y

        self.transform.position = new_position

    def integrate(self, delta_time: float) -> None:
        if self._is_static or not self._enabled:
            return

        if self._move_target is not None:
            direction = self._move_target - self.transform.position
            distance = direction.magnitude()
            if distance > 0.0:
                if self._move_speed is not None and self._move_speed > 0.0:
                    move_distance = min(distance, self._move_speed * delta_time)
                else:
                    move_distance = distance
                move_vector = direction.normalized() * move_distance
                self._velocity = self._apply_linear_constraints(move_vector / delta_time)
            else:
                self._move_target = None
                self._velocity = Vector2D(0.0, 0.0)

        if self._friction > 0.0:
            damping = max(0.0, 1.0 - self._friction * delta_time)
            self._velocity = self._apply_linear_constraints(self._velocity * damping)
            if abs(self._velocity.x) < 1e-4 and abs(self._velocity.y) < 1e-4:
                self._velocity = Vector2D.zero()

        self.translate(self._velocity * delta_time)

    def freeze_position(self, *, x: Optional[bool] = None, y: Optional[bool] = None) -> None:
        if x is not None:
            self._freeze_position[0] = bool(x)
            if x:
                self._velocity = Vector2D(0.0, self._velocity.y)
        if y is not None:
            self._freeze_position[1] = bool(y)
            if y:
                self._velocity = Vector2D(self._velocity.x, 0.0)

    def _apply_linear_constraints(self, vec: Vector2D) -> Vector2D:
        return Vector2D(
            0.0 if self._freeze_position[0] else vec.x,
            0.0 if self._freeze_position[1] else vec.y,
        )
