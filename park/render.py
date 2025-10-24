from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict

from ipycanvas import Canvas, MultiCanvas, hold_canvas

from park.entities.ride import Ride
from park.entities.robot import Robot
from park.entities.visitor import Visitor
from park.internal.math import Rect, Vector2D
from park.internal.physics import Physics
from park.internal.sprite import SpriteShape
from park.logic.grid import Grid2D
from park.stats import Curve

if TYPE_CHECKING:
    from park.entities.core import BaseEntity
    from park.logic.node import Node
    from park.simulation import Simulation


@dataclass(frozen=True)
class Colors:
    background = "#2c2c31"
    minor_grid: str = "#3a3a40"
    major_grid: str = "#4a4a52"
    label: str = "#ffffff"
    node_free: str = "rgba(120, 180, 255, 0.18)"
    node_occupied: str = "rgba(255, 90, 90, 0.35)"

    ride: str = "#d276df"
    robot: str = "#78b7ff"
    visitor: str = "#ff9f9f"
    collider: str = "#f4c542"
    queue: str = "#2bff004e"
    charger: str = "#16c542"

    ride_progress_bg: str = "#4f5158"
    ride_loading: str = "#16c542"
    ride_unloading: str = "#ff5f5f"

    selection_circle: str = "rgba(22, 197, 66, 0.65)"
    robot_waypoint: str = "rgba(255, 255, 255, 0.35)"


class Camera:
    def __init__(self, width, height, world_width, world_height, min_zoom=1, max_zoom=3):
        self.width = width
        self.height = height
        self.world_width = world_width
        self.world_height = world_height
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

        self.zoom = min_zoom
        self.x = 0
        self.y = 0
        # Optional snapping of zoom to predefined levels (e.g., [0.5, 1, 2, 3])
        self.zoom_steps = None

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

    def set_zoom(self, z: float):
        z = max(self.min_zoom, min(z, self.max_zoom))
        if self.zoom_steps:
            # snap to nearest step
            z = min(self.zoom_steps, key=lambda s: abs(s - z))
        self.zoom = z

    def set_zoom_steps(self, steps):
        """Set allowed zoom levels. Pass None to disable snapping."""
        if steps is None:
            self.zoom_steps = None
            return
        self.zoom_steps = sorted(float(s) for s in steps if s > 0)
        if self.zoom_steps:
            # Ensure range includes steps
            self.min_zoom = min(self.min_zoom, self.zoom_steps[0])
            self.max_zoom = max(self.max_zoom, self.zoom_steps[-1])

    def fit_to_world(self, padding: int = 0, lock: bool = True):
        """Scale so the entire world fits in the canvas.

        - padding: pixels to leave around edges (applied on both sides)
        - lock: if True, sets min_zoom=max_zoom=scale to disable interactive zoom
        """
        avail_w = max(1, self.width - 2 * padding)
        avail_h = max(1, self.height - 2 * padding)
        sx = avail_w / max(1e-6, self.world_width)
        sy = avail_h / max(1e-6, self.world_height)
        scale = min(sx, sy)
        if lock:
            self.min_zoom = self.max_zoom = scale
        self.set_zoom(scale)
        self.set_position(0, 0)


class Renderer:
    class TooltipRenderLayer(Enum):
        BACKGROUND = "background"
        CONTENT = "content"

    class SimRenderLayer(Enum):
        BACKGROUND = "background"
        GRID = "grid"
        GRID_DEBUG = "grid_debug"
        QUEUES = "queues"
        CHARGERS = "chargers"
        RIDES = "rides"
        ROBOTS = "robots"
        VISITORS = "visitors"
        COLLIDERS = "colliders"

    class StatRenderLayer(Enum):
        BACKGROUND = "background"
        CURVES = "curves"

    def __init__(
        self,
        sim: Simulation,
        camera: Camera,
        pixel_scale: int,
        tile_grouping: int = 0,
    ):
        self.sim = sim
        self.cam = camera
        self.pixel_scale = pixel_scale
        self.tile_grouping = tile_grouping if tile_grouping >= 2 else 0

        self.curves: Dict[str, Curve] = {}

        self.sim_canvas: MultiCanvas = MultiCanvas(
            len(Renderer.SimRenderLayer),
            width=camera.width,
            height=camera.height,
        )
        self.sim_canvas.layout.width = f"{camera.width}px"
        self.sim_canvas.layout.height = f"{camera.height}px"
        self.sim_canvas.layout.border = "2px solid #444444"

        self._tooltip_width = 200
        self.tooltip_canvas: MultiCanvas = MultiCanvas(
            len(Renderer.TooltipRenderLayer),
            width=self._tooltip_width,
            height=camera.height
        )
        self.tooltip_canvas.layout.width = f"{self._tooltip_width}px"
        self.tooltip_canvas.layout.height = f"{camera.height}px"

        self._stats_width = 170
        self.stats_canvas: MultiCanvas = MultiCanvas(
            len(Renderer.StatRenderLayer),
            width=self._stats_width,
            height=camera.height
        )
        self.stats_canvas.layout.width = f"{self._stats_width}px"
        self.stats_canvas.layout.height = f"{camera.height}px"

        self._front = {
            name: self.sim_canvas[i]
            for i, name in enumerate(Renderer.SimRenderLayer)
        }
        self._front.update({
            name: self.tooltip_canvas[i]
            for i, name in enumerate(Renderer.TooltipRenderLayer)
        })
        self._front.update({
            name: self.stats_canvas[i]
            for i, name in enumerate(Renderer.StatRenderLayer)
        })
        # Disable smoothing for blits/draws
        for layer in self._front.values():
            try:
                layer.image_smoothing_enabled = False
            except Exception:
                pass

        # Double buffering
        self._back = {
            name: Canvas(
                width=camera.width,
                height=camera.height
            ) for name in Renderer.SimRenderLayer
        }
        self._back.update({
            name: Canvas(
                width=self._tooltip_width,
                height=camera.height
            ) for name in Renderer.TooltipRenderLayer
        })
        self._back.update({
            name: Canvas(
                width=self._stats_width,
                height=camera.height
            ) for name in Renderer.StatRenderLayer
        })
        for layer in self._back.values():
            try:
                layer.image_smoothing_enabled = False
            except Exception:
                pass

        self._dirty = {
            name: False
            for name in (
                list(Renderer.SimRenderLayer)
                + list(Renderer.TooltipRenderLayer)
                + list(Renderer.StatRenderLayer)
            )
        }
        self._render_call = {
            Renderer.SimRenderLayer.BACKGROUND: self._render_background,
            Renderer.SimRenderLayer.GRID: self._render_grid,
            Renderer.SimRenderLayer.GRID_DEBUG: self._render_debug_grid,
            Renderer.SimRenderLayer.CHARGERS: self._render_chargers,
            Renderer.SimRenderLayer.RIDES: self._render_rides,
            Renderer.SimRenderLayer.ROBOTS: self._render_robots,
            Renderer.SimRenderLayer.VISITORS: self._render_visitors,
            Renderer.SimRenderLayer.QUEUES: self._render_queues,
            Renderer.SimRenderLayer.COLLIDERS: self._render_colliders,

            Renderer.TooltipRenderLayer.BACKGROUND: self._render_tooltip_background,
            Renderer.TooltipRenderLayer.CONTENT: self._render_tooltip_content,
            Renderer.StatRenderLayer.BACKGROUND: self._render_stats_background,
            Renderer.StatRenderLayer.CURVES: self._render_stats_curves,
        }

        self._last_cam_transform = (
            camera.x,
            camera.y,
            camera.zoom
        )

        self._show_debug_grids = False
        self._debug_grid_force_full = False
        self._show_queues = False
        self._show_colliders = False
        self._show_stats_curves = False
        self.update_draw(force=True)

    @property
    def sim_widget(self):
        return self.sim_canvas

    @property
    def tooltip_widget(self):
        return self.tooltip_canvas

    @property
    def stats_widget(self):
        return self.stats_canvas

    def mark_all_dirty(self):
        for name in self._dirty:
            self._dirty[name] = True

    def set_pixel_scale(self, scale: float):
        """Set the world grid step used for rendering and glyph sizing.

        This affects grid spacing and the size of rides/robots/visitors relative
        to world units. Larger values make everything appear larger on screen
        for a fixed camera zoom.
        """
        s = max(0.1, float(scale))
        if s != self.pixel_scale:
            self.pixel_scale = s
            self.mark_all_dirty()

    def set_debug_grids_visible(self, enabled: bool):
        if enabled != self._show_debug_grids:
            self._show_debug_grids = enabled
            self._debug_grid_force_full = enabled
            self._dirty[Renderer.SimRenderLayer.GRID_DEBUG] = True

    def set_queues_visible(self, enabled: bool):
        if enabled != self._show_queues:
            self._show_queues = enabled
            self._dirty[Renderer.SimRenderLayer.QUEUES] = True

    def set_colliders_visible(self, enabled: bool):
        if enabled != self._show_colliders:
            self._show_colliders = enabled
            self._dirty[Renderer.SimRenderLayer.COLLIDERS] = True

    def set_stats_curves_visible(self, enabled: bool):
        if enabled != self._show_stats_curves:
            self._show_stats_curves = enabled
            self._dirty[Renderer.StatRenderLayer.CURVES] = True

    def add_curve(self, curve: Curve):
        if curve.name in self.curves:
            raise ValueError(f"Curve with name '{curve.name}' already exists")
        self.curves[curve.name] = curve
        idx = len(self.curves) - 1
        curve.set_rect(self._get_curve_rect(idx))
        self._dirty[Renderer.StatRenderLayer.CURVES] = True

    def update_curve(self, name: str, value: float):
        if not name in self.curves:
            return
        self.curves[name].append(value)
        self._dirty[Renderer.StatRenderLayer.CURVES] = True

    def clear_curves(self):
        for curve in self.curves.values():
            curve.clear()
        self._dirty[Renderer.StatRenderLayer.CURVES] = True

    def resize(self, width: int, height: int):
        # resize layers and camera
        self.sim_canvas.width = width
        self.sim_canvas.height = height
        self._back = {
            name: Canvas(
                width=width,
                height=height
            ) for name in Renderer.SimRenderLayer
        }

        self.cam.width = width
        self.cam.height = height
        for idx, curve in enumerate(self.curves.values()):
            curve.set_rect(self._get_curve_rect(idx))
            self._dirty[Renderer.StatRenderLayer.CURVES] = True

        self.mark_all_dirty()

    def screen_to_world(self, point, y=None) -> Vector2D:
        x, y = Renderer._extract_xy(point, y)
        wx = x / self.cam.zoom + self.cam.x
        wy = y / self.cam.zoom + self.cam.y
        return Vector2D(wx, wy)

    def world_to_screen(self, point, y=None) -> Vector2D:
        x, y = Renderer._extract_xy(point, y)
        sx = (x - self.cam.x) * self.cam.zoom
        sy = (y - self.cam.y) * self.cam.zoom
        return Vector2D(sx, sy)

    def zoom_at(self, point, y=None, zoom_delta=None, sensitivity=0.0015):
        # Support both zoom_at(Vector2D, delta) and zoom_at(x, y, delta)
        if zoom_delta is None and not isinstance(point, (int, float)):
            zoom_delta = y
            x, y = Renderer._extract_xy(point)
        else:
            if zoom_delta is None:
                raise TypeError("zoom_delta must be provided")
            x, y = Renderer._extract_xy(point, y)
        stw = self.screen_to_world(x, y)

        # delta_y > 0 -> zoom out
        factor = 1.0 - (zoom_delta * sensitivity)
        if factor <= 0:
            return

        new_s = self.cam.zoom * factor
        self.cam.set_zoom(new_s)
        factor = self.cam.zoom / max(1e-6, new_s)

        # Adjust translation so the focus is on (wx, wy)
        self.cam.x = x - stw.x * self.cam.zoom
        self.cam.y = y - stw.y * self.cam.zoom

    def select_at_point(self, point: Vector2D, clear: bool):
        world_point = self.screen_to_world(point)
        if clear:
            for robot in Robot.get_tooltip_robots():
                robot.set_tooltip_visible(False)
            for visitor in Visitor.get_tooltip_visitors():
                visitor.set_tooltip_visible(False)
            for ride in Ride.get_tooltip_rides():
                ride.set_tooltip_visible(False)

        for robot in self.sim.robots:
            if not robot.sprite.enabled and not robot.collider.enabled:
                continue
            if (
                robot.bounds is not None
                and robot.bounds.contains(world_point)
            ):
                robot.set_tooltip_visible(True)
                return
        for visitor in self.sim.visitors:
            if not visitor.sprite.enabled and not visitor.collider.enabled:
                continue
            if (
                visitor.bounds is not None
                and visitor.bounds.contains(world_point)
            ):
                visitor.set_tooltip_visible(True)
                return
        for ride in self.sim.rides:
            if not ride.sprite.enabled and not ride.collider.enabled:
                continue
            if (
                ride.bounds is not None
                and ride.bounds.contains(world_point)
            ):
                ride.set_tooltip_visible(True)
                return

    def update_draw(self, force: bool = False):
        cam_transform = (
            self.cam.x,
            self.cam.y,
            self.cam.zoom
        )
        if force or cam_transform != self._last_cam_transform:
            self._last_cam_transform = cam_transform
            self._dirty[Renderer.SimRenderLayer.GRID] = True
            self._dirty[Renderer.SimRenderLayer.CHARGERS] = True
            if self._show_debug_grids:
                self._debug_grid_force_full = True

        self._dirty[Renderer.SimRenderLayer.RIDES] = True
        self._dirty[Renderer.SimRenderLayer.ROBOTS] = True
        self._dirty[Renderer.SimRenderLayer.VISITORS] = True        

        if self._show_debug_grids:
            need_debug = self._debug_grid_force_full
            if not need_debug:
                for grid in Grid2D._grids:
                    if grid.has_dirty_nodes():
                        need_debug = True
                        break
            if need_debug:
                self._dirty[Renderer.SimRenderLayer.GRID_DEBUG] = True

        if self._show_colliders:
            self._dirty[Renderer.SimRenderLayer.COLLIDERS] = True

        self._dirty[Renderer.TooltipRenderLayer.CONTENT] = True

        for layer in self._dirty:
            if force or self._dirty[layer]:
                if layer in self._render_call:
                    self._render_call[layer](
                        self._back[layer]
                    )

                self._blit(layer)
                self._dirty[layer] = False

    def _clear(self, canvas: Canvas, color=None):
        canvas.clear()
        if color:
            canvas.fill_style = color
            canvas.fill_rect(
                0,
                0,
                canvas.width,
                canvas.height,
            )

    def _blit(self, name: str):
        front: Canvas = self._front[name]
        back: Canvas = self._back[name]
        with hold_canvas(front):  # keep clear/draw atomic to prevent flicker
            front.clear()
            try:
                front.image_smoothing_enabled = False
            except Exception:
                pass
            front.draw_image(back, 0, 0)

    @staticmethod
    def _hash01(key: str) -> float:
        """Deterministic 0..1 noise from string key (FNV-1a)."""
        h = 2166136261
        for b in key.encode('utf-8'):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        return (h & 0xFFFFFFFF) / 4294967296.0

    @staticmethod
    def _extract_xy(point, maybe_y=None) -> tuple[float, float]:
        if maybe_y is not None:
            return float(point), float(maybe_y)
        if isinstance(point, Vector2D):
            return point.x, point.y
        if isinstance(point, (tuple, list)) and len(point) == 2:
            return float(point[0]), float(point[1])
        raise TypeError("Expected Vector2D or (x, y) pair")

    def _draw_sprite(self, canvas: Canvas, entity: BaseEntity) -> bool:
        sprite = entity.sprite
        if sprite is None or not sprite.enabled:
            return False

        center = self.world_to_screen(entity.transform.position)
        zoom = self.cam.zoom
        width = sprite.size.x * zoom
        height = sprite.size.y * zoom
        if width <= 0 or height <= 0:
            return False

        _buffer = sprite.ensure_canvas()
        if _buffer is not None:
            canvas.draw_image(
                _buffer,
                center.x - width / 2.0,
                center.y - height / 2.0,
                width,
                height,
            )
            return True

        if sprite.shape == SpriteShape.GROUP:
            self._draw_group_sprite(canvas, entity, center, width, height, sprite)
            return True

        if sprite.shape == SpriteShape.CIRCLE:
            radius = max(width, height) / 2.0
            canvas.fill_style = sprite.color
            canvas.begin_path()
            canvas.move_to(center.x + radius, center.y)
            canvas.arc(center.x, center.y, radius, 0, 2 * math.pi)
            canvas.fill()
            canvas.stroke_style = Colors.label
            canvas.line_width = 0.5
            canvas.begin_path()
            canvas.move_to(center.x + radius, center.y)
            canvas.arc(center.x, center.y, radius, 0, 2 * math.pi)
            canvas.stroke()
            return True

        if sprite.shape == SpriteShape.RECT:
            x0 = center.x - width / 2.0
            y0 = center.y - height / 2.0
            canvas.fill_style = sprite.color
            canvas.fill_rect(x0, y0, width, height)
            canvas.stroke_style = Colors.label
            canvas.line_width = 1
            canvas.stroke_rect(x0, y0, width, height)
            return True

        return False

    def _draw_group_sprite(self, canvas: Canvas, entity, center: Vector2D, width: float, height: float, sprite) -> None:
        members = max(1, int(sprite.data.get("members", getattr(entity, "group_size", 1))))
        cols = max(1, int(math.ceil(math.sqrt(members))))
        rows = int(math.ceil(members / cols))

        radius = min(width / (cols * 2.0), height / (rows * 2.0))
        if radius <= 0:
            return

        x0 = center.x - width / 2.0
        y0 = center.y - height / 2.0
        jitter_amp = 0.5 * radius

        centers = []
        for k in range(members):
            r_idx = k // cols
            c_idx = k % cols
            cell_w = width / cols
            cell_h = height / rows
            cx = x0 + (c_idx + 0.5) * cell_w
            cy = y0 + (r_idx + 0.5) * cell_h
            vid = str(getattr(entity, "id", "v"))
            jx = (Renderer._hash01(f"{vid}:{k}:x") - 0.5) * 2 * jitter_amp
            jy = (Renderer._hash01(f"{vid}:{k}:y") - 0.5) * 2 * jitter_amp
            cx = min(max(cx + jx, x0 + radius), x0 + width - radius)
            cy = min(max(cy + jy, y0 + radius), y0 + height - radius)
            centers.append((cx, cy))

        canvas.fill_style = sprite.color
        canvas.begin_path()
        for cx, cy in centers:
            canvas.move_to(cx + radius, cy)
            canvas.arc(cx, cy, radius, 0, 2 * math.pi)
        canvas.fill()

        canvas.stroke_style = Colors.label
        canvas.line_width = 0.5
        canvas.begin_path()
        for cx, cy in centers:
            canvas.move_to(cx + radius, cy)
            canvas.arc(cx, cy, radius, 0, 2 * math.pi)
        canvas.stroke()

    def _render_background(self, canvas: Canvas):
        self._clear(canvas, Colors.background)

    def _render_grid(self, canvas: Canvas):
        self._clear(canvas, None)

        cam = self.cam
        pixel_scale = self.pixel_scale
        vw, vh = cam.get_viewport_size()
        x0 = cam.x - (cam.x % pixel_scale)
        y0 = cam.y - (cam.y % pixel_scale)

        with hold_canvas(canvas):
            canvas.stroke_style = Colors.minor_grid
            # Keep line width at 1 logical pixel; ipycanvas scales by pixel_ratio
            canvas.line_width = 1

            # Batch minor grid lines with 0.5px offsets for crispness
            canvas.begin_path()
            x = x0
            while x <= cam.x + vw:
                sx = (x - cam.x) * cam.zoom + 0.5
                canvas.move_to(sx, 0)
                canvas.line_to(sx, cam.height)
                x += pixel_scale
            y = y0
            while y <= cam.y + vh:
                sy = (y - cam.y) * cam.zoom + 0.5
                canvas.move_to(0, sy)
                canvas.line_to(cam.width, sy)
                y += pixel_scale
            canvas.stroke()

            if self.tile_grouping >= 2:
                grouping = self.tile_grouping
                first_major_x = (
                    x0 - ((x0 // pixel_scale) % grouping)
                    * pixel_scale
                )
                first_major_y = (
                    y0 - ((y0 // pixel_scale) % grouping)
                    * pixel_scale
                )

                canvas.stroke_style = Colors.major_grid
                canvas.line_width = 4

                canvas.begin_path()
                x = first_major_x + pixel_scale * grouping
                while x <= cam.x + vw:
                    sx = (x - cam.x) * cam.zoom + 0.5
                    canvas.move_to(sx, 0)
                    canvas.line_to(sx, cam.height)
                    x += pixel_scale * grouping

                y = first_major_y + pixel_scale * grouping
                while y <= cam.y + vh:
                    sy = (y - cam.y) * cam.zoom + 0.5
                    canvas.move_to(0, sy)
                    canvas.line_to(cam.width, sy)
                    y += pixel_scale * grouping
                canvas.stroke()

    def _render_debug_grid(self, canvas: Canvas):
        if not self._show_debug_grids or not Grid2D._grids:
            self._clear(canvas, None)
            self._debug_grid_force_full = False
            return

        cam = self.cam
        zoom = cam.zoom
        view_w, view_h = cam.get_viewport_size()
        view_x0 = cam.x
        view_y0 = cam.y
        view_x1 = view_x0 + view_w
        view_y1 = view_y0 + view_h
        if self._debug_grid_force_full:
            self._clear(canvas, None)
            with hold_canvas(canvas):
                canvas.line_width = 1
                canvas.stroke_style = Colors.minor_grid
                for grid in Grid2D._grids:
                    for row in grid.nodes:
                        for node in row:
                            left = node.center.x - node.radius
                            top = node.center.y - node.radius
                            right = left + node.diameter
                            bottom = top + node.diameter
                            if right < view_x0 or left > view_x1 or bottom < view_y0 or top > view_y1:
                                continue
                            screen_pos = self.world_to_screen(left, top)
                            width = node.diameter * zoom
                            height = node.diameter * zoom
                            canvas.fill_style = Colors.node_occupied if node.occupants else Colors.node_free
                            canvas.fill_rect(screen_pos.x, screen_pos.y, width, height)
                            canvas.stroke_rect(screen_pos.x, screen_pos.y, width, height)
            self._debug_grid_force_full = False
            for grid in Grid2D._grids:
                if grid.has_dirty_nodes():
                    grid.consume_dirty_nodes()
            return

        updates: list[tuple[Grid2D, list[tuple[int, int, "Node"]]]] = []
        for grid in Grid2D._grids:
            dirty_nodes = grid.consume_dirty_nodes()
            if dirty_nodes:
                updates.append((grid, dirty_nodes))
        if not updates:
            return

        with hold_canvas(canvas):
            canvas.line_width = 1
            canvas.stroke_style = Colors.minor_grid
            for _, dirty_nodes in updates:
                for node in dirty_nodes:
                    left = node.center.x - node.radius
                    top = node.center.y - node.radius
                    right = left + node.diameter
                    bottom = top + node.diameter
                    if right < view_x0 or left > view_x1 or bottom < view_y0 or top > view_y1:
                        continue

                    screen_pos = self.world_to_screen(left, top)
                    width = node.diameter * zoom
                    height = node.diameter * zoom
                    canvas.clear_rect(screen_pos.x, screen_pos.y, width, height)

                    canvas.fill_style = Colors.node_occupied if node.occupants else Colors.node_free
                    canvas.fill_rect(screen_pos.x, screen_pos.y, width, height)
                    canvas.stroke_rect(screen_pos.x, screen_pos.y, width, height)

    def _render_chargers(self, canvas: Canvas):
        self._clear(canvas, None)
        with hold_canvas(canvas):
            for charger in self.sim.chargers:
                self._draw_sprite(canvas, charger)

    def _render_rides(self, canvas: Canvas):
        self._clear(canvas, None)
        with hold_canvas(canvas):
            for ride in self.sim.rides:
                drawn = self._draw_sprite(canvas, ride)
                if not drawn:
                    continue

                if ride.tooltip_visible:
                    # draw selection circle
                    center = self.world_to_screen(ride.transform.position)
                    radius = (max(ride.sprite.size.x, ride.sprite.size.y) * self.cam.zoom)
                    self._draw_selection_circle(
                        canvas,
                        center,
                        radius,
                        Colors.selection_circle
                    )

                sprite = ride.sprite
                center = self.world_to_screen(ride.transform.position)
                run_state = ride.run_state

                if run_state in (
                    ride.RunState.LOADING,
                    ride.RunState.RUNNING,
                    ride.RunState.UNLOADING
                ):
                    member_count = ride.riders.member_count
                    capacity = ride.capacity
                    center = self.world_to_screen(ride.transform.position)
                    radius = (max(ride.sprite.size.x, ride.sprite.size.y) * self.cam.zoom) / 3.0
                    if radius <= 0:
                        continue
                    thickness = max(2.0, 0.15 * radius)
                    inner_radius = max(0.0, radius - thickness)
                    start = -math.pi / 2
                    if run_state == ride.RunState.LOADING:
                        fill_frac = min(1.0, member_count / max(1, capacity))
                        fill_style = Colors.ride_loading
                    elif run_state == ride.RunState.UNLOADING:
                        fill_frac = min(1.0, member_count / max(1, capacity))
                        fill_style = Colors.ride_unloading
                    else:
                        start += self.sim.current_step * 0.1
                        fill_frac = 0.9
                        fill_style = Colors.ride_loading
                    if fill_frac > 0:
                        if run_state != ride.RunState.RUNNING:
                            canvas.begin_path()
                            canvas.arc(center.x, center.y, radius, 0, 2 * math.pi)
                            canvas.arc(center.x, center.y, inner_radius, 0, 2 * math.pi, True)
                            canvas.close_path()
                            canvas.fill_style = Colors.ride_progress_bg
                            canvas.fill()

                        canvas.begin_path()
                        canvas.move_to(center.x, center.y)
                        canvas.arc(center.x, center.y, radius, start, start + 2 * math.pi * fill_frac)
                        canvas.arc(center.x, center.y, inner_radius, start + 2 * math.pi * fill_frac, start, True)
                        canvas.close_path()
                        canvas.fill_style = fill_style
                        canvas.fill()

                label = None
                if sprite is not None:
                    label = sprite.data.get("label") or getattr(ride, "name", None)
                    offset = (sprite.size.y * self.cam.zoom)
                else:
                    label = getattr(ride, "name", None)
                    offset = (0.3 * self.pixel_scale) * self.cam.zoom

                if label:
                    canvas.fill_style = Colors.label
                    font_px = max(6, int(0.3 * self.pixel_scale * self.cam.zoom))
                    canvas.font = f"{font_px}px Arial"
                    canvas.text_align = "center"
                    canvas.fill_text(label, center.x, center.y - offset)

    def _render_robots(self, canvas: Canvas):
        self._clear(canvas, None)
        with hold_canvas(canvas):
            for robot in self.sim.robots:
                self._draw_sprite(canvas, robot)

                if robot.tooltip_visible:
                    # draw selection circle
                    center = self.world_to_screen(robot.transform.position)
                    radius = (max(robot.sprite.size.x, robot.sprite.size.y) * self.cam.zoom)
                    self._draw_selection_circle(
                        canvas,
                        center,
                        radius,
                        Colors.selection_circle
                    )

                    if robot.current_plan is not None:
                        # draw waypoints as little hollow dots with a line to each
                        waypoints = robot.current_plan.remaining_waypoints
                        if waypoints:
                            canvas.fill_style = Colors.robot_waypoint
                            canvas.stroke_style = Colors.label
                            canvas.line_width = 1
                            canvas.begin_path()
                            canvas.move_to(center.x, center.y)
                            for wp in waypoints:
                                wp_screen = self.world_to_screen(wp)
                                canvas.line_to(wp_screen.x, wp_screen.y)
                                canvas.move_to(wp_screen.x, wp_screen.y)
                                canvas.arc(wp_screen.x, wp_screen.y, max(2.0, 0.2 * radius), 0, 2 * math.pi)
                                canvas.move_to(wp_screen.x, wp_screen.y)
                            canvas.stroke()
                            canvas.fill()

    def _render_visitors(self, canvas: Canvas):
        self._clear(canvas, None)
        with hold_canvas(canvas):
            for visitor in self.sim.visitors:
                self._draw_sprite(canvas, visitor)

                if visitor.tooltip_visible:
                    # draw selection circle
                    center = self.world_to_screen(visitor.transform.position)
                    radius = (max(visitor.sprite.size.x, visitor.sprite.size.y) * self.cam.zoom)
                    self._draw_selection_circle(
                        canvas,
                        center,
                        radius,
                        Colors.selection_circle
                    )

    def _render_queues(self, canvas: Canvas):
        self._clear(canvas, None)
        if not self._show_queues:
            return

        queues = [self.sim.entrance_queue, self.sim.exit_queue]
        for ride in self.sim.rides:
            queues.append(ride.entrance_queue)
            queues.append(ride.exit_queue)
        for charger in self.sim.chargers:
            queues.append(charger.charge_queue)
        with hold_canvas(canvas):
            for queue in queues:
                # draw a rectangle from the queue's tail to its head, like a carpet
                if queue.head is None or queue.tail is None:
                    continue
                head = self.world_to_screen(queue.head)
                tail = self.world_to_screen(queue.tail)
                width = abs(head.x - tail.x)
                height = abs(head.y - tail.y)
                if width == 0:
                    width = max(2.0, 16.0 * self.cam.zoom)
                    head.x -= width / 2.0
                    tail.x -= width / 2.0
                if height == 0:
                    height = max(2.0, 16.0 * self.cam.zoom)
                    head.y -= height / 2.0
                    tail.y -= height / 2.0
                canvas.fill_style = Colors.queue
                canvas.fill_rect(
                    min(head.x, tail.x),
                    min(head.y, tail.y),
                    width,
                    height,
                )

    def _render_colliders(self, canvas: Canvas):
        self._clear(canvas, None)
        if not self._show_colliders:
            return

        zoom = self.cam.zoom
        stroke_width = max(1.0, 1.0 * zoom)

        with hold_canvas(canvas):
            canvas.stroke_style = Colors.collider
            canvas.line_width = stroke_width

            for collider in Physics._colliders:
                if collider is None or not collider.enabled:
                    continue
                rect: Rect = collider.bounds()
                if rect.width <= 0 or rect.height <= 0:
                    continue
                top_left = self.world_to_screen(Vector2D(rect.x, rect.y))
                canvas.stroke_rect(
                    top_left.x,
                    top_left.y,
                    rect.width * zoom,
                    rect.height * zoom,
                )

    def _render_tooltip_background(self, canvas: Canvas):
        self._clear(canvas, Colors.background)

    def _render_tooltip_content(self, canvas: Canvas):
        self._clear(canvas, None)
        selected_robot = next((r for r in self.sim.robots if r.tooltip_visible), None)
        selected_visitor = next((v for v in self.sim.visitors if v.tooltip_visible), None)
        selected_ride = next((rd for rd in self.sim.rides if rd.tooltip_visible), None)
        if (
            selected_robot is None
            and selected_visitor is None
            and selected_ride is None
        ):
            return

        with hold_canvas(canvas):
            if selected_robot is not None:
                lines = [
                    f"Robot #{str(selected_robot.id)[:8]}",
                    f"State: {selected_robot.state.value}",
                    f"Ride: {selected_robot.target_ride.name if selected_robot.target_ride else '--'}",
                    f"Health: {selected_robot.health_percentage:.2f}%",
                    f"Battery: {selected_robot.battery_percentage:.2f}%",
                ]
            elif selected_visitor is not None:
                lines = [
                    f"Visitor #{str(selected_visitor.id)[:8]}",
                    f"State: {selected_visitor.state.value}",
                    f"Ride: {selected_visitor.assigned_ride.name if selected_visitor.assigned_ride else '--'}",
                    f"Satisfaction: {selected_visitor.satisfaction:.2f}",
                    f"Completed Rides: {selected_visitor.completed_rides}/{selected_visitor.desired_rides}",
                    # f"Queue Time: {selected_visitor.time_in_queues}",
                    # f"Ride Time: {selected_visitor.time_in_rides}",
                    # f"Queue: {selected_visitor.queue_ref.name if selected_visitor.queue_ref else '--'}",
                ]
            elif selected_ride is not None:
                lines = [
                    f"Ride: {selected_ride.name}",
                    # f"Operation State: {selected_ride.operational_state.value}",
                    f"Run State: {selected_ride.run_state.value}",
                    f"Ticket Price: ${selected_ride.entry_price:.2f}",
                    f"Riders: {selected_ride.riders.member_count}/{selected_ride.capacity}",
                    # f"Timer: {selected_ride.timer}/{selected_ride.duration}",
                    # f"Entrance: {selected_ride.entrance_queue.name}",
                    # f"Exit: {selected_ride.exit_queue.name}",
                ]

            pad = 10.0
            box_width = canvas.width - pad * 2.0
            box_height = len(lines) * 18.0 + 24.0
            box_x = pad
            box_y = pad

            canvas.fill_style = "rgba(28, 29, 36, 0.92)"
            canvas.fill_rect(box_x, box_y, box_width, box_height)

            canvas.stroke_style = Colors.label
            canvas.line_width = 1.0
            canvas.stroke_rect(box_x, box_y, box_width, box_height)

            canvas.fill_style = Colors.label
            canvas.font = "13px monospace"
            canvas.text_align = "left"
            line_height = 18.0
            text_y = box_y + 24.0
            for line in lines:
                canvas.fill_text(line, box_x + 12.0, text_y)
                text_y += line_height

    def _render_stats_background(self, canvas: Canvas):
        self._clear(canvas, Colors.background)

    def _render_stats_curves(self, canvas: Canvas):
        if not self._dirty[Renderer.StatRenderLayer.CURVES]:
            return

        canvas.clear()
        if not self._show_stats_curves:
            return

        for idx, curve in enumerate(self.curves.values()):
            curve.set_rect(self._get_curve_rect(idx))
            curve.render(canvas)

    def _get_curve_rect(self, idx: int) -> Rect:
        width = 150
        height = 100
        top = 10.0
        gap = 10.0

        current_x = self._stats_width - width - gap
        current_y = top + (height + gap) * idx
        return Rect(current_x, current_y, width, height)

    def _draw_selection_circle(self, canvas: Canvas, center: Vector2D, radius: float, color: str):
        if radius <= 0:
            return
        canvas.stroke_style = color
        canvas.line_width = max(2.0, 0.15 * radius)
        canvas.begin_path()
        canvas.move_to(center.x + radius, center.y)
        canvas.arc(center.x, center.y, radius, 0, 2 * math.pi)
        canvas.stroke()
