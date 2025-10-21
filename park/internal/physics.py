from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from park.internal.collider import CollisionLayer
from park.internal.math import Rect, Vector2D
from park.internal.rigidbody import RigidBody

if TYPE_CHECKING:
    from park.internal.collider import Collider


class Physics:
    _colliders: List["Collider"] = []
    _rigidbodies: List["RigidBody"] = []
    # Track collision state per-collider so multi-collider entities work
    _collision_states: Dict["Collider", set[int]] = {}

    @classmethod
    def add_rigidbody(cls, rigidbody: "RigidBody") -> None:
        if rigidbody not in cls._rigidbodies:
            cls._rigidbodies.append(rigidbody)

    @classmethod
    def remove_rigidbody(cls, rigidbody: "RigidBody") -> None:
        if rigidbody in cls._rigidbodies:
            cls._rigidbodies.remove(rigidbody)
        # No direct collision-state cleanup here; tracked per collider

    @classmethod
    def add_collider(cls, collider: "Collider") -> None:
        if collider not in cls._colliders:
            cls._colliders.append(collider)
        cls._collision_states.setdefault(collider, set())

    @classmethod
    def remove_collider(cls, collider: "Collider") -> None:
        if collider in cls._colliders:
            cls._colliders.remove(collider)
        cls._collision_states.pop(collider, None)

    @classmethod
    def clear(cls) -> None:
        cls._rigidbodies.clear()
        cls._collision_states.clear()
        cls._colliders.clear()

    @classmethod
    def bodies(cls) -> Iterable["RigidBody"]:
        return tuple(cls._rigidbodies)

    @classmethod
    def step(cls, delta_time: float) -> None:
        # Integrate velocities
        for body in cls._rigidbodies:
            body.integrate(delta_time)

        # Resolve collisions (collider-vs-collider)
        current_contacts = set()
        active_colliders = tuple(cls.active_colliders())
        for a, b in combinations(active_colliders, 2):
            # Skip colliders on the same Transform to avoid self-collision
            if getattr(a, "transform", None) is getattr(b, "transform", None):
                continue
            # Layer/mask filtering (bitmask-based)
            if not cls._can_collide(a, b):
                continue
            key = cls._pair_key(a, b)
            collision = cls._compute_collision(a, b)
            if collision is None:
                cls._mark_exit(a, b, key)
                continue

            normal, penetration = collision
            current_contacts.add(key)
            cls._handle_collision_events(a, b, key)
            cls._resolve_collision(a, b, normal, penetration)

        cls._cleanup_exits(current_contacts)

    @classmethod
    def check_rect(cls, rect: Rect, mask_bits: int | CollisionLayer) -> List["Collider"]:
        hits = []
        for col in cls.active_colliders():
            if not CollisionLayer.can_collide(
                col.layer_bits,
                col.mask_bits,
                CollisionLayer.ALL_BITS,
                mask_bits
            ):
                continue
            if rect.intersects(col.bounds()):
                hits.append(col)
        return hits

    @classmethod
    def active_colliders(cls) -> Iterable["Collider"]:
        for col in cls._colliders:
            if getattr(col, "enabled", False) and getattr(col, "transform", None) is not None:
                yield col

    @staticmethod
    def _compute_collision(a: "Collider", b: "Collider") -> Optional[Tuple[Vector2D, float]]:
        bounds_a = a.bounds()
        bounds_b = b.bounds()

        ax = bounds_a.x + bounds_a.width * 0.5
        ay = bounds_a.y + bounds_a.height * 0.5
        bx = bounds_b.x + bounds_b.width * 0.5
        by = bounds_b.y + bounds_b.height * 0.5

        overlap_x = bounds_a.width * 0.5 + bounds_b.width * 0.5 - abs(ax - bx)
        if overlap_x <= 0:
            return None

        overlap_y = bounds_a.height * 0.5 + bounds_b.height * 0.5 - abs(ay - by)
        if overlap_y <= 0:
            return None

        if overlap_x < overlap_y:
            normal = Vector2D(1.0, 0.0) if ax < bx else Vector2D(-1.0, 0.0)
            penetration = overlap_x
        else:
            normal = Vector2D(0.0, 1.0) if ay < by else Vector2D(0.0, -1.0)
            penetration = overlap_y

        return normal, penetration

    @staticmethod
    def _resolve_collision(a: "Collider", b: "Collider", normal: Vector2D, penetration: float) -> None:
        # Fetch associated rigidbodies (if any)
        transform_a = a.transform
        transform_b = b.transform
        if transform_a is not None:
            rb_a: RigidBody = transform_a.get_component(RigidBody)
        if transform_b is not None:
            rb_b: RigidBody = transform_b.get_component(RigidBody)

        # Derive inverse masses and velocities
        inv_mass_a = 0.0 if rb_a is None else rb_a.inverse_mass
        inv_mass_b = 0.0 if rb_b is None else rb_b.inverse_mass
        inv_mass_sum = inv_mass_a + inv_mass_b
        if inv_mass_sum == 0:
            return

        # Positional correction
        correction = normal * (penetration / inv_mass_sum)
        if rb_a is not None and inv_mass_a > 0 and not rb_a.is_static:
            correction_a = rb_a._apply_linear_constraints(correction * -inv_mass_a)
            rb_a.translate(correction_a)
        if rb_b is not None and inv_mass_b > 0 and not rb_b.is_static:
            correction_b = rb_b._apply_linear_constraints(correction * inv_mass_b)
            rb_b.translate(correction_b)

        # Resolve velocities
        va = Vector2D.zero() if rb_a is None else rb_a.velocity
        vb = Vector2D.zero() if rb_b is None else rb_b.velocity
        relative_velocity = vb - va
        vel_along_normal = relative_velocity.dot(normal)
        if vel_along_normal > 0:
            return

        bounce_a = 0.0 if rb_a is None else rb_a.bounciness
        bounce_b = 0.0 if rb_b is None else rb_b.bounciness
        bounciness = max(bounce_a, bounce_b)
        impulse_mag = -(1 + bounciness) * vel_along_normal / inv_mass_sum
        impulse = normal * impulse_mag

        if rb_a is not None and inv_mass_a > 0 and not rb_a.is_static:
            delta_va = rb_a._apply_linear_constraints(impulse * inv_mass_a)
            rb_a.set_velocity(va - delta_va)
        if rb_b is not None and inv_mass_b > 0 and not rb_b.is_static:
            delta_vb = rb_b._apply_linear_constraints(impulse * inv_mass_b)
            rb_b.set_velocity(vb + delta_vb)

    @classmethod
    def _pair_key(cls, a: "Collider", b: "Collider") -> int:
        return hash((min(id(a), id(b)), max(id(a), id(b))))

    @classmethod
    def _handle_collision_events(
        cls,
        a: "Collider",
        b: "Collider",
        key: int,
    ) -> None:
        seen_a = key in cls._collision_states.get(a, set())
        seen_b = key in cls._collision_states.get(b, set())

        if not seen_a:
            a.on_collision_enter(b)
            cls._collision_states.setdefault(a, set()).add(key)
        else:
            a.on_collision_stay(b)

        if not seen_b:
            b.on_collision_enter(a)
            cls._collision_states.setdefault(b, set()).add(key)
        else:
            b.on_collision_stay(a)

    @classmethod
    def _mark_exit(cls, a: "Collider", b: "Collider", key: int) -> None:
        states_a = cls._collision_states.get(a)
        states_b = cls._collision_states.get(b)
        removed = False

        if states_a and key in states_a:
            states_a.remove(key)
            a.on_collision_exit(b)
            removed = True
        if states_b and key in states_b:
            states_b.remove(key)
            b.on_collision_exit(a)
            removed = True

        if removed and states_a is not None and not states_a:
            cls._collision_states.pop(a, None)
        if removed and states_b is not None and not states_b:
            cls._collision_states.pop(b, None)

    @classmethod
    def _cleanup_exits(cls, current_contacts: set[int]) -> None:
        to_remove = []
        for col, keys in cls._collision_states.items():
            expired = keys - current_contacts
            for key in expired:
                other = cls._find_collider_by_key(key, exclude=col)
                if other is None:
                    continue
                col.on_collision_exit(other)
            keys.intersection_update(current_contacts)
            if not keys:
                to_remove.append(col)
        for col in to_remove:
            cls._collision_states.pop(col, None)

    @classmethod
    def _find_collider_by_key(cls, key: int, exclude: "Collider") -> Optional["Collider"]:
        for col in cls._colliders:
            if col is exclude:
                continue
            pair = cls._pair_key(col, exclude)
            if pair == key:
                return col
        return None

    @classmethod
    def _can_collide(cls, a: "Collider", b: "Collider") -> bool:
        return CollisionLayer.can_collide(
            a.layer_bits,
            a.mask_bits,
            b.layer_bits,
            b.mask_bits
        )
