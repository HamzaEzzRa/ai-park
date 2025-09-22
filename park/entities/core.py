from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from park.internal.collider import Collider
from park.internal.math import Rect, Vector2D
from park.internal.rigidbody import RigidBody
from park.internal.sprite import Sprite
from park.internal.transform import Transform

if TYPE_CHECKING:
    from park.simulation import Simulation

class BaseEntity:
    def __init__(self, simulation: Simulation, position: Vector2D):
        self.id = uuid4()
        self.simulation = simulation

        self._transform = Transform(position)
        self._current_step = 0

    def update(self):
        self._current_step += 1

    def attach_component(self, component):
        self.transform.attach_component(component)
        setattr(component, "entity", self)

        if isinstance(component, Collider):
            component.on_collision_enter = self.on_collision_enter
            component.on_collision_stay = self.on_collision_stay
            component.on_collision_exit = self.on_collision_exit

    def get_component(self, cls):
        return self.transform.get_component(cls)

    def get_components(self, cls):
        return self.transform.get_components(cls)

    def delete(self):
        self.transform.delete()

    def on_collision_enter(self, other: Collider) -> None:
        pass

    def on_collision_stay(self, other: Collider) -> None:
        pass

    def on_collision_exit(self, other: Collider) -> None:
        pass

    @property
    def transform(self) -> Transform:
        return self._transform

    @property
    def sprite(self) -> Optional[Sprite]:
        return self.get_component(Sprite)

    @property
    def collider(self) -> Optional[Collider]:
        return self.get_component(Collider)

    @property
    def rigidbody(self) -> Optional[RigidBody]:
        return self.get_component(RigidBody)

    @property
    def bounds(self) -> Optional[Rect]:
        if self.collider:
            return self.collider.bounds
        elif self.sprite:
            size = self.sprite.size
            half_size = size // 2
            x_min = self.transform.position.x - half_size.x
            y_min = self.transform.position.y - half_size.y
            return Rect(x_min, y_min, size.x, size.y)

        return None
