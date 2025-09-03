from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ipycanvas import Canvas, MultiCanvas


@dataclass(frozen=True)
class Colors:
    background = "#2c2c31"
    minor_grid: str = "#3a3a40"
    major_grid: str = "#4a4a52"
    label: str = "#ffffff"

    attraction: str = "#4cc19a"
    robot: str = "#78b7ff"
    visitor: str = "#ff9f9f"


class Camera:
    def __init__(self, width, height, world_width, world_height, min_zoom=1, max_zoom=3):
        self.width = width
        self.height = height
        self.world_width = world_width
        self.world_height = world_height
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

        self.zoom = 2
        self.x = 0
        self.y = 0

    def set_position(self, x, y):
        view_width, view_height = self.get_viewport_size()
        new_x = max(0, min(x, self.world_width - view_width))
        new_y = max(0, min(y, self.world_height - view_height))

        step = 1.0 / self.zoom
        self.x = round(new_x / step) * step
        self.y = round(new_y / step) * step

    def pan(self, dx, dy):
        self.set_position(self.x + dx, self.y + dy)

    def get_viewport_size(self):
        return int(self.width / self.zoom), int(self.height / self.zoom)


class Renderer:
    LAYERS = (
        "background",
        "grid",
        "attraction",
        "robot",
        "visitor"
    )

    def __init__(self, camera: Camera, tile_pixels: int = 32, tile_grouping: int = 5):
        self.cam = camera
        self.tile_pixels = tile_pixels
        self.tile_grouping = max(2, tile_grouping)
        self.mcanvas = MultiCanvas(
            len(Renderer.LAYERS),
            width=camera.width,
            height=camera.height
        )
        self._front = {
            name: self.mcanvas[i]
            for i, name in enumerate(Renderer.LAYERS)
        }
        for layer in self._front.values():
            layer.pixel_ratio = 1 

        # Double buffering
        self._back = {
            name: Canvas(
                width=camera.width,
                height=camera.height
            ) for name in Renderer.LAYERS
        }
        for layer in self._back.values():
            layer.pixel_ratio = 1

        self._dirty = {
            name: False
            for name in Renderer.LAYERS
        }
        self._render_call = {
            "background": self._render_background,
            "grid": self._render_grid,
            "attractions": self._render_attractions,
            "robots": self._render_robots,
            "visitors": self._render_visitors
        }

        self._last_cam_transform = (
            camera.x,
            camera.y,
            camera.zoom
        )

        self.update_draw(force=True)

    @property
    def widget(self):
        return self.mcanvas

    def mark_all_dirty(self):
        for name in self._dirty:
            self._dirty[name] = True

    def resize(self, width: int, height: int):
        # resize layers and camera
        self.mcanvas.width = width
        self.mcanvas.height = height
        self._back = {
            name: Canvas(
                width=width,
                height=height
            ) for name in self.LAYERS
        }
        for layer in self._back.values():
            layer.pixel_ratio = 1

        self.cam.width = width
        self.cam.height = height
        self.mark_all_dirty()

    def screen_to_world(self, x: float, y: float):
        wx = (x - self.cam.x) / self.cam.zoom
        wy = (y - self.cam.y) / self.cam.zoom
        return wx, wy

    def world_to_screen(self, x: float, y: float):
        return (
            (x - self.cam.x) * self.cam.zoom,
            (y - self.cam.y) * self.cam.zoom
        )

    def zoom_at(self, x, y, zoom_delta, sensitivity=0.0015):
        wx, wy = self.screen_to_world(x, y)

        # delta_y > 0 -> zoom out
        factor = 1.0 - (zoom_delta * sensitivity)
        if factor <= 0:
            return

        new_s = min(max(self.cam.zoom * factor, self.cam.min_zoom), self.cam.max_zoom)
        factor = new_s / self.cam.zoom
        self.cam.zoom = new_s

        # Adjust translation so the focus is on (wx, wy)
        self.cam.x = x - wx * self.cam.zoom
        self.cam.y = y - wy * self.cam.zoom

    def update_draw(self, force: bool = False):
        cam_transform = (
            self.cam.x,
            self.cam.y,
            self.cam.zoom
        )
        if force or cam_transform != self._last_cam_transform:
            self._last_cam_transform = cam_transform
            self._dirty["grid"] = True

        for name in self._dirty:
            if force or self._dirty[name]:
                if name in self._render_call:
                    self._render_call[name](
                        self._back[name]
                    )

                self._blit(name)
                self._dirty[name] = False

    def _clear(self, canvas: Canvas, color=None):
        canvas.clear()
        if color:
            canvas.fill_style = color
            canvas.fill_rect(
                0,
                0,
                self.cam.width,
                self.cam.height
            )

    def _blit(self, name: str):
        front = self._front[name]
        back = self._back[name]
        front.clear()
        front.draw_image(back, 0, 0)

    def _render_background(self, canvas: Canvas):
        self._clear(canvas, Colors.background)

    def _render_grid(self, canvas: Canvas):
        self._clear(canvas, None)

        cam = self.cam
        tile_pixels = self.tile_pixels
        vw, vh = cam.get_viewport_size()
        x0 = cam.x - (cam.x % tile_pixels)
        y0 = cam.y - (cam.y % tile_pixels)

        canvas.stroke_style = Colors.minor_grid
        canvas.line_width = 1

        x = x0
        while x <= cam.x + vw:
            sx = (x - cam.x) * cam.zoom
            canvas.begin_path()
            canvas.move_to(sx, 0)
            canvas.line_to(sx, cam.height)
            canvas.stroke()
            x += tile_pixels

        y = y0
        while y <= cam.y + vh:
            sy = (y - cam.y) * cam.zoom
            canvas.begin_path()
            canvas.move_to(0, sy)
            canvas.line_to(cam.width, sy)
            canvas.stroke()
            y += tile_pixels

        grouping = self.tile_grouping
        first_major_x = (
            x0 - ((x0 // tile_pixels) % grouping)
            * tile_pixels
        )
        first_major_y = (
            y0 - ((y0 // tile_pixels) % grouping)
            * tile_pixels
        )

        canvas.stroke_style = Colors.major_grid
        canvas.line_width = 2

        x = first_major_x
        while x <= cam.x + vw:
            sx = (x - cam.x) * cam.zoom
            canvas.begin_path()
            canvas.move_to(sx, 0)
            canvas.line_to(sx, cam.height)
            canvas.stroke()
            x += tile_pixels * grouping

        y = first_major_y
        while y <= cam.y + vh:
            sy = (y - cam.y) * cam.zoom
            canvas.begin_path()
            canvas.move_to(0, sy)
            canvas.line_to(cam.width, sy)
            canvas.stroke()
            y += tile_pixels * grouping

    def _render_attractions(self, canvas: Canvas):
        pass

    def _render_robots(self, canvas: Canvas):
        pass

    def _render_visitors(self, canvas: Canvas):
        pass
