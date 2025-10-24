from park.internal.collider import BoxCollider
from park.internal.layers import CollisionLayer
from park.internal.math import Vector2D
from park.internal.rigidbody import RigidBody
from park.internal.transform import Transform
from park.logic.grid import Grid2D


class World:
    def __init__(
        self,
        width: int,
        height: int,
        cell_size: float = 1.0,
    ):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = Grid2D(
            width,
            height,
            cell_size=cell_size // 2,
            mask_bits=CollisionLayer.ALL_BITS & ~CollisionLayer.ENVIRONMENT
        )

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
                    size=Vector2D(width, 2 * height),
                    layer_bits=CollisionLayer.ENVIRONMENT
                )
            ),
        ]
        self._rigidbody = self._transform.attach_component(RigidBody(is_static=True))

    def update(self):
        self.grid.update_nodes()
