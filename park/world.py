from enum import Enum

from park.internal.collider import BoxCollider
from park.internal.layers import CollisionLayer
from park.internal.math import Vector2D
from park.internal.rigidbody import RigidBody
from park.internal.transform import Transform


class World:
    class SpaceType(Enum):
        CONTINUOUS = "continuous"
        GRID = "grid"

    def __init__(
        self,
        width: int,
        height: int,
        cell_size: float = 1.0,
        space_type: "World.SpaceType" = SpaceType.CONTINUOUS
    ):
        self.width = width
        self.height = height
        self.space_type = space_type
        self.cell_size = cell_size

        self._transform = Transform(position=Vector2D(0.0, 0.0))

        self._colliders = [
            self._transform.attach_component(
                BoxCollider(
                    offset=Vector2D(width / 2, -height / 2),
                    size=Vector2D(2 * width, height),
                    layer_bits=CollisionLayer.ENVIRONMENT
                )
            ),
            self._transform.attach_component(
                BoxCollider(
                    offset=Vector2D(width * 1.5, height / 2),
                    size=Vector2D(width, 2 * height),
                    layer_bits=CollisionLayer.ENVIRONMENT
                )
            ),
            self._transform.attach_component(
                BoxCollider(
                    offset=Vector2D(width / 2, height * 1.5),
                    size=Vector2D(2 * width, height),
                    layer_bits=CollisionLayer.ENVIRONMENT
                )
            ),
            self._transform.attach_component(
                BoxCollider(
                    offset=Vector2D(-width / 2, height / 2),
                    size=Vector2D(width, 2 *height),
                    layer_bits=CollisionLayer.ENVIRONMENT
                )
            ),
        ]
        self._rigidbody = self._transform.attach_component(RigidBody(is_static=True))

    def snap(self, v: Vector2D) -> Vector2D:
        """Snap a vector to the world's grid if in GRID space, otherwise return unchanged."""
        if self.space_type == World.SpaceType.GRID:
            gx = round(v.x / self.cell_size) * self.cell_size
            gy = round(v.y / self.cell_size) * self.cell_size
            return Vector2D(gx, gy)
        return v
