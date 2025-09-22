from park.internal.collider import Collider
from park.internal.math import Vector2D
from park.internal.physics import Physics
from park.internal.rigidbody import RigidBody


class Transform:
    def __init__(
        self,
        position: Vector2D = Vector2D(0, 0),
        scale: Vector2D = Vector2D(1, 1),
    ):
        self.position = position
        self.scale = scale

        self._components = { Transform: [self] }

    def attach_component(self, component) -> None:
        if component.__class__ == Transform:
            raise ValueError("Transform component already attached.")

        if not component.__class__ in self._components:
            self._components[component.__class__] = [component]
        else:
            self._components[component.__class__].append(component)
        setattr(component, "transform", self)

        # Track in physics system when attached
        if isinstance(component, Collider):
            Physics.add_collider(component)
        if isinstance(component, RigidBody):
            Physics.add_rigidbody(component)

    def get_component(self, cls):
        if cls in self._components:
            return self._components[cls][0]

        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            if subclass in self._components:
                return self._components[subclass][0]

        return None

    def get_components(self, cls):
        results = []
        if cls in self._components:
            results.extend(self._components[cls])

        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            if subclass in self._components:
                results.extend(self._components[subclass])

        return results

    def delete(self):
        for key in list(self._components.keys()):
            for component in self._components[key]:
                if component != self:
                    if isinstance(component, Collider):
                        Physics.remove_collider(component)
                    if isinstance(component, RigidBody):
                        Physics.remove_rigidbody(component)
                    del component
                    del self._components[key]

        del self._components[Transform][0]
        del self._components[Transform]

    def translate(self, delta: Vector2D) -> None:
        self.position = self.position + delta

    def set_position(self, position: Vector2D) -> None:
        self.position = position.copy()

    def set_scale(self, scale: Vector2D) -> None:
        self.scale = scale.copy()

    def copy(self) -> "Transform":
        return Transform(self.position.copy(), self.scale.copy())

    def __repr__(self) -> str:
        return f"Transform(position={self.position}, scale={self.scale})"
