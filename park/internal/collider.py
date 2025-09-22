from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Union

from park.internal.layers import CollisionLayer
from park.internal.math import Rect, Vector2D

if TYPE_CHECKING:
    from park.internal.transform import Transform


class Collider(ABC):
    def __init__(
        self,
        offset: Vector2D = Vector2D(0.0, 0.0),
        size: Vector2D = Vector2D(1.0, 1.0),
        layer_bits: int | CollisionLayer = CollisionLayer.DEFAULT,
        mask_bits: int | CollisionLayer = CollisionLayer.ALL_BITS,
    ):
        self._enabled = True
        self._transform = None
        self.set_offset(offset)
        self.set_size(size)

        # Bitmask-based collision filtering
        # Default: belongs to DEFAULT and collides with all bits
        self._layer_bits: int = int(layer_bits) or CollisionLayer.DEFAULT
        self._mask_bits: int = int(mask_bits) or CollisionLayer.ALL_BITS

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def transform(self) -> Transform:
        return self._transform

    @transform.setter
    def transform(self, transform: Transform) -> None:
        self._transform = transform

    @property
    def layer_bits(self) -> int:
        return self._layer_bits

    def set_layer_bits(self, bits: int) -> None:
        self._layer_bits = int(bits) if bits is not None else 0

    @property
    def mask_bits(self) -> int:
        return self._mask_bits

    def set_mask_bits(self, bits: int) -> None:
        # If None, default to collide with nothing
        self._mask_bits = int(bits) if bits is not None else 0

    # Convenience helpers using CollisionLayer flags
    def set_layers(self, *layers: CollisionLayer | int | Iterable[Union[CollisionLayer, int]]) -> None:
        self._layer_bits = CollisionLayer.mask_of(*layers)

    def set_collides_with(self, *layers: CollisionLayer | int | Iterable[Union[CollisionLayer, int]]) -> None:
        self._mask_bits = CollisionLayer.mask_of(*layers)

    @property
    def offset(self) -> Vector2D:
        return self._offset

    def set_offset(self, offset: Vector2D) -> None:
        self._offset = offset.copy()

    @property
    def size(self) -> Vector2D:
        return self._size

    def set_size(self, size: Vector2D) -> None:
        self._size = size.copy()

    @property
    def center(self) -> Vector2D:
        pos = self._transform.position
        return Vector2D(pos.x + self._offset.x, pos.y + self._offset.y)

    @property
    def half_extents(self) -> Vector2D:
        return Vector2D(self._size.x / 2.0, self._size.y / 2.0)

    def on_collision_enter(self, other: "Collider") -> None:
        pass

    def on_collision_stay(self, other: "Collider") -> None:
        pass

    def on_collision_exit(self, other: "Collider") -> None:
        pass

    @abstractmethod
    def bounds(self) -> Rect:
        """Axis-aligned bounds in world space."""

    @abstractmethod
    def contains(self, point: Vector2D) -> bool:
        """Return True if the world-space point lies inside the collider."""

    def intersects(self, other: "Collider") -> bool:
        return self.bounds().intersects(other.bounds())


class BoxCollider(Collider):
    def __init__(
        self,
        offset: Vector2D = Vector2D(0.0, 0.0),
        size: Vector2D = Vector2D(1.0, 1.0),
        layer_bits: int | CollisionLayer = CollisionLayer.DEFAULT,
        mask_bits: int | CollisionLayer = CollisionLayer.ALL_BITS,
    ):
        resolved_size = Vector2D(abs(size.x), abs(size.y))
        super().__init__(
            offset=offset,
            size=resolved_size,
            layer_bits=layer_bits,
            mask_bits=mask_bits
        )

    def bounds(self) -> Rect:
        return Rect(
            x=self.center.x - self.half_extents.x,
            y=self.center.y - self.half_extents.y,
            width=self._size.x,
            height=self._size.y,
        )

    def contains(self, point: Vector2D) -> bool:
        min_x = self.center.x - self.half_extents.x
        max_x = self.center.x + self.half_extents.x
        min_y = self.center.y - self.half_extents.y
        max_y = self.center.y + self.half_extents.y
        return min_x <= point.x <= max_x and min_y <= point.y <= max_y

    def intersects(self, other: Collider) -> bool:
        if isinstance(other, BoxCollider):
            return (
                self.center.x - self.half_extents.x < other.center.x + other.half_extents.x
                and self.center.x + self.half_extents.x > other.center.x - other.half_extents.x
                and self.center.y - self.half_extents.y < other.center.y + other.half_extents.y
                and self.center.y + self.half_extents.y > other.center.y - other.half_extents.y
            )

        return super().intersects(other)
