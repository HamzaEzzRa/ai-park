from __future__ import annotations

from enum import IntFlag
from typing import Iterable, Union


class CollisionLayer(IntFlag):
    """Bit flags used for collider layer/mask filtering."""

    DEFAULT = 1 << 0
    ROBOT = 1 << 1
    VISITOR = 1 << 2
    RIDE = 1 << 3
    ENVIRONMENT = 1 << 4

    ALL_BITS = 0xFFFFFFFF
    NONE = 0x00000000

    @classmethod
    def mask_of(cls, *layers: Union[CollisionLayer, int, Iterable[Union[CollisionLayer, int]]]) -> int:
        """Build a bit mask from one or more layers."""
        bits = 0
        for item in layers:
            if isinstance(item, (list, tuple, set)):
                for sub in item:
                    bits |= int(sub)
            else:
                bits |= int(item)
        return int(bits)

    @classmethod
    def can_collide(cls, layer_bits_a: int, mask_bits_a: int, layer_bits_b: int, mask_bits_b: int) -> bool:
        """Symmetric layer/mask check for collisions."""
        return (mask_bits_a & layer_bits_b) != 0 and (mask_bits_b & layer_bits_a) != 0
