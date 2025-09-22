from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from park.internal.math import Vector2D

from ipywidgets import Image


class SpriteShape(Enum):
    CIRCLE = "circle"
    RECT = "rect"
    GROUP = "group"


class Sprite:
    def __init__(
        self,
        size: Vector2D,
        color: str = "#ffffff",
        shape: SpriteShape = SpriteShape.RECT,
        image_path: Optional[str] = None,  # Path to image asset on disk, falls back to primitive shape if None
        data: Dict[str, Any] = {}
    ):
        self.size: Vector2D = size.copy()
        self.color: str = color
        self.shape: SpriteShape = shape
        self.image_path: Optional[str] = image_path
        self.data: Dict[str, Any] = data
        self._image: Optional["Image"] = None
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def half_size(self) -> Vector2D:
        return Vector2D(self.size.x * 0.5, self.size.y * 0.5)

    def set_size(self, size: Vector2D) -> None:
        self.size = size.copy()

    def ensure_image(self) -> Optional["Image"]:
        if self.image_path is None:
            return None
        if self._image is not None:
            return self._image
        if Image is None:
            return None
        try:
            self._image = Image.from_file(self.image_path)
            return self._image
        except Exception:
            # Keep _image as None for primitive fallback
            return None

    def set_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)
