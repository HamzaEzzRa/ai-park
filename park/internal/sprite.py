from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from ipycanvas import Canvas
from PIL import Image as PILImage
from PIL.Image import Image

from park.internal.math import Vector2D


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
        self.primitive_canvas: Optional["Canvas"] = None

        self._image: Optional["Image"] = None  # just to get image width and height
        self._image_array: Optional[np.ndarray] = None
        self._canvas: Optional["Canvas"] = None
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def half_size(self) -> Vector2D:
        return Vector2D(self.size.x * 0.5, self.size.y * 0.5)

    @property
    def image(self) -> Optional["Image"]:
        return self._ensure_image()

    @property
    def image_size(self) -> Optional[Vector2D]:
        if self.image is None:
            return None
        return Vector2D(
            float(self._image.width),
            float(self._image.height)
        )

    @property
    def image_aspect_ratio(self) -> Optional[float]:
        if self.image is None:
            return None
        return float(self._image.width) / float(self._image.height)

    @property
    def image_array_size(self) -> Optional[Vector2D]:
        if self._image_array is None:
            return None
        return Vector2D(
            float(self._image_array.shape[1]),
            float(self._image_array.shape[0])
        )

    def ensure_canvas(
        self,
        scale: float = 1.0,
        rotation_angle: float = 0.0,
        reset=False
    ) -> Optional["Canvas"]:
        if not reset and self._canvas is not None:
            return self._canvas

        image_array = self._ensure_image_array(
            scale=scale,
            rotation_angle=rotation_angle,
            reset=reset
        )
        if image_array is None:
            return None

        image_array_size = self.image_array_size
        canvas: Canvas = Canvas(
            width=int(image_array_size.x),
            height=int(image_array_size.y)
        )
        canvas.put_image_data(image_array, 0, 0)
        self._canvas = canvas
        return self._canvas

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_size(self, size: Vector2D) -> None:
        self.size = size.copy()

    def set_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    def copy(self) -> "Sprite":
        new_sprite = Sprite(
            size=self.size,
            color=self.color,
            shape=self.shape,
            image_path=self.image_path,
            data=self.data.copy()
        )
        new_sprite._image = self._image
        new_sprite._canvas = self._canvas
        new_sprite.set_enabled(self.enabled)
        return new_sprite

    def primitive_canvas_to_image(self) -> Optional["Image"]:
        if self.primitive_canvas is None:
            return None

        self.primitive_canvas.sync_image_data = True
        data = np.asarray(
            self.primitive_canvas.get_image_data(
                0,
                0,
                self.primitive_canvas.width,
                self.primitive_canvas.height
            ),
            dtype=np.uint8
        )
        image = PILImage.fromarray(data, mode="RGBA")
        return image

    def _ensure_image(self) -> Optional["Image"]:
        if self.image_path is None:
            return None
        if self._image is not None:
            return self._image

        try:
            self._image = PILImage.open(self.image_path)
            return self._image
        except Exception as e:
            # Keep _image as None for primitive fallback
            self._image = None
            return None

    def _ensure_image_array(
        self,
        scale: float = 1.0,
        rotation_angle: float = 0.0,
        reset=False,
    ) -> Optional[np.ndarray]:
        if not reset and self._image_array is not None:
            return self._image_array

        image = self.image
        if image is None:
            return None

        target_size = self.image_size.copy()
        target_size.x = max(1.0, target_size.x * scale)
        target_size.y = max(1.0, target_size.y * scale)

        if scale != 1.0:
            image = self.image.resize(
                (int(target_size.x), int(target_size.y)),
                PILImage.Resampling.BILINEAR
            )
        if rotation_angle != 0.0:
            image = image.rotate(
                rotation_angle,
                expand=True,
                resample=PILImage.Resampling.BICUBIC
            )
        self._image_array = np.array(image, dtype=np.uint8)
        return self._image_array
