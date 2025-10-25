from __future__ import annotations

import asyncio
import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ipycanvas import Canvas, hold_canvas
from ipyevents import Event

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.fsm.core import Machine, MachineError
from park.internal.collider import BoxCollider
from park.internal.layers import CollisionLayer
from park.internal.math import Vector2D
from park.internal.physics import Physics
from park.internal.rigidbody import RigidBody
from park.internal.sprite import Sprite, SpriteShape
from park.internal.transform import Transform


@dataclass
class Car:
    transform: Transform
    speed: float
    direction: int
    lane_index: int
    rigidbody: Optional[RigidBody]

    @property
    def sprite(self) -> Sprite:
        return self.transform.get_component(Sprite)

    @property
    def collider(self) -> BoxCollider:
        return self.transform.get_component(BoxCollider)

    def delete(self) -> None:
        self.transform.delete()


class LaneType(Enum):
    TRAFFIC = "traffic"
    OBSTACLES = "obstacles"


class DuckGame:
    """
    Tiny “cross the road” playground driven by the students' FSM.

    Parameters
    ----------
    fsm: Machine
        The finite-state machine already configured by the students.
    trigger_map: dict[str, str]
        Maps keyboard keys (e.g. "ArrowUp") to FSM trigger names.
    release_trigger: str | None
        Trigger fired once no movement key is held (useful for AUCUN_BOUTON).
    goal_trigger / hit_trigger: str | None
        Trigger names to fire on victory or collision.
    starting_lane_count: int
        Number of lanes present at level 0.
    """

    def __init__(
        self,
        fsm: Machine,
        *,
        trigger_map: Optional[Dict[str, str]] = None,
        release_trigger: Optional[str] = "aucun_bouton",
        goal_trigger: Optional[str] = "arrive",
        hit_trigger: Optional[str] = "collision",
        arena_size: Tuple[int, int] = (480, 360),
        starting_lane_count: int = 4,
        lane_increase_interval: int = 3,
        rng_seed: Optional[int] = None,
        starting_max_obstacles_per_lane: Optional[int] = None,
        obstacle_increase_interval: int = 3,
    ) -> None:
        self.fsm = fsm
        self.fsm.reset()

        self.trigger_map = trigger_map or {
            "ArrowUp": "fleche_haut",
            "ArrowLeft": "fleche_gauche",
            "ArrowRight": "fleche_droite",
        }
        self.release_trigger = release_trigger
        self.goal_trigger = goal_trigger
        self.hit_trigger = hit_trigger

        self.width, self.height = arena_size
        self.starting_lane_count = max(1, int(starting_lane_count))
        self.lane_increase_interval = max(1, int(lane_increase_interval))
        self.road_margin = 48.0
        self.dt = 1 / 60.0

        self.level = 0
        self.score = 0
        self.high_score = 0

        self.rng = random.Random(rng_seed)
        self._car_colors = ["#ff6b6b", "#4ecdc4", "#ffbe76", "#686de0", "#d2dae2", "#fd9644"]
        self.starting_max_obstacles_per_lane = (
            max(1, int(starting_max_obstacles_per_lane))
            if starting_max_obstacles_per_lane is not None else None
        )
        self.obstacle_increase_interval = max(1, int(obstacle_increase_interval))
        self._current_max_obstacles: Optional[int] = self.starting_max_obstacles_per_lane

        self.canvas: Canvas = Canvas(width=self.width, height=self.height)
        self.canvas.layout.border = "2px solid #999"
        self.canvas.layout.width = f"{self.width}px"
        self.canvas.layout.height = f"{self.height}px"

        self._keys_down: set[str] = set()
        self._key_repeat_timers: Dict[str, float] = {}
        self._key_repeat_interval = 0.18
        
        self._logic_task: Optional[asyncio.Task] = None
        self._draw_task: Optional[asyncio.Task] = None

        self._last_state: Optional[str] = None
        self._waiting_for_restart = False
        self._pending_reset = False
        self._prompt_visible = False

        self._grid_cols: int = 0
        self._col_positions: Sequence[float] = ()
        self._row_positions: Sequence[float] = ()

        self._traffic_lanes: List[Dict[str, Any]] = []
        self._active_cars: List[Car] = []

        self._obstacle_layout: List[List[int]] = []
        self._obstacle_transforms: List[List[Transform]] = []
        self._obstacle_sprites: List[List[Sprite]] = []

        self.lane_count: int = 0
        self.lane_types: List[LaneType] = []

        self.duck_transform: Optional[Transform] = None
        self.duck_sprite: Optional[Sprite] = None
        self.duck_collider: Optional[BoxCollider] = None
        self._col_index = 0
        self._row_index = 0

        self._car_sprites: List[Sprite] = [
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_black.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_blue.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_gray.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_green.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_red.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_sky.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_white_red.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_white.png",
            ),
            Sprite(
                size=Vector2D(32.0, 32.0),
                image_path="./images/vehicle_yellow.png",
            ),
        ]

        self._setup_level(initial=True)
        self._bind_events()
        self._draw()

        try:
            self.canvas.focus()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Level construction
    # ------------------------------------------------------------------ #
    def _setup_level(self, *, initial: bool = False) -> None:
        """(Re)build the entire world for the current level."""
        self._cleanup_cars()
        self._clear_obstacles()

        self.lane_count = self.starting_lane_count + (self.level // self.lane_increase_interval)
        self._generate_lane_types()
        self._init_grid()
        self._build_duck_transform()
        self._prepare_traffic_lanes()
        self._update_current_max_obstacles()
        self._randomize_obstacles()

        # spawn one car per traffic lane to kick-start motion
        for lane in self._traffic_lanes:
            self._spawn_car(lane)

        self._keys_down.clear()
        self._key_repeat_timers.clear()
        self._waiting_for_restart = False if initial else self._waiting_for_restart
        self._apply_duck_position()
        self.fsm.reset()

    def _generate_lane_types(self) -> None:
        lane_types: List[LaneType] = []
        traffic_count = 0
        last_type: Optional[LaneType] = None

        for idx in range(self.lane_count):
            remaining = self.lane_count - idx
            if remaining == 1 and traffic_count == 0:
                choice = LaneType.TRAFFIC
            else:
                if last_type is LaneType.OBSTACLES:
                    choice = LaneType.TRAFFIC
                else:
                    choice = LaneType.TRAFFIC if self.rng.random() < 0.6 else LaneType.OBSTACLES

            lane_types.append(choice)
            last_type = choice
            if choice is LaneType.TRAFFIC:
                traffic_count += 1

        if lane_types:
            lane_types[0] = LaneType.TRAFFIC

        if traffic_count == 0:
            idx = self.rng.randrange(len(lane_types))
            lane_types[idx] = LaneType.TRAFFIC

        self.lane_types = lane_types

    def _init_grid(self) -> None:
        self._grid_cols = 9
        cell_w = self.width / self._grid_cols
        self._col_positions = [(i + 0.5) * cell_w for i in range(self._grid_cols)]

        road_height = self.height - 2 * self.road_margin
        lane_height = road_height / max(1, self.lane_count)

        rows: List[float] = [self.height - self.road_margin * 0.5]
        for lane_idx in range(self.lane_count):
            center = self.road_margin + lane_height * (self.lane_count - lane_idx - 0.5)
            rows.append(center)
        rows.append(self.road_margin * 0.5)
        self._row_positions = rows

        self._col_index = self._grid_cols // 2
        self._row_index = 0

    def _build_duck_transform(self) -> None:
        if self.duck_transform is not None:
            self.duck_transform.delete()

        start_pos = Vector2D(
            self._col_positions[self._col_index],
            self._row_positions[self._row_index]
        )
        self.duck_transform = Transform(position=start_pos)
        sprite = Sprite(
            size=Vector2D(19.0, 32.0),
            color="#f5c542",
            shape=SpriteShape.RECT,
            image_path="./images/duck_sprite.png"
        )
        sprite.ensure_canvas(reset=True)
        self.duck_sprite = sprite
        sprite_size = sprite.image_array_size or sprite.size
        self.duck_collider = BoxCollider(size=sprite_size)
        self.duck_transform.attach_component(self.duck_sprite)
        self.duck_transform.attach_component(self.duck_collider)

        self.duck_rigidbody = RigidBody(mass=0.1, friction=0.1, bounciness=0.0)
        self.duck_transform.attach_component(self.duck_rigidbody)
        self.duck_rigidbody.set_velocity(Vector2D.zero())

    def _prepare_traffic_lanes(self) -> None:
        self._traffic_lanes = []
        road_height = self.height - 2 * self.road_margin
        lane_height = road_height / max(1, self.lane_count)

        for lane_idx, lane_type in enumerate(self.lane_types):
            if lane_type is not LaneType.TRAFFIC:
                continue
            direction = 1 if lane_idx % 2 == 0 else -1
            y = self._row_positions[lane_idx + 1]

            speed_min = 80.0 + self.level * 10.0
            speed_max = speed_min + 60.0

            base_interval = max(0.5, 2.4 - 0.18 * (self.level + lane_idx * 0.3))
            spawn_range = (base_interval * 0.6, base_interval * 1.3)

            width_min = (self.width / self._grid_cols) * 0.9
            width_max = (self.width / self._grid_cols) * 1.8
            car_height = lane_height * 0.65

            lane_info = {
                "index": lane_idx,
                "direction": direction,
                "y": y,
                "speed_range": (speed_min, speed_max),
                "spawn_range": spawn_range,
                "spawn_timer": self.rng.uniform(*spawn_range),
                "size_height": car_height,
                "size_width_range": (width_min, width_max),
            }
            self._traffic_lanes.append(lane_info)

        self._active_cars = []

    def _update_current_max_obstacles(self) -> None:
        if self.starting_max_obstacles_per_lane is None:
            self._current_max_obstacles = None
            return
        increment = self.level // self.obstacle_increase_interval
        self._current_max_obstacles = self.starting_max_obstacles_per_lane + increment

    def _randomize_obstacles(self) -> None:
        self._clear_obstacles()

        road_height = self.height - 2 * self.road_margin
        lane_height = road_height / max(1, self.lane_count)
        size = Vector2D(
            (self.width / self._grid_cols) * 0.75,
            lane_height * 0.55,
        )

        self._obstacle_layout = []
        self._obstacle_transforms = []
        self._obstacle_sprites = []

        for lane_idx, lane_type in enumerate(self.lane_types):
            blocked: List[int] = []
            transforms: List[Transform] = []
            sprites: List[Sprite] = []

            if lane_type is LaneType.OBSTACLES:
                available_cols = list(range(self._grid_cols))
                max_spacing_limited = (self._grid_cols + 1) // 2
                max_allowed = max_spacing_limited
                if self._current_max_obstacles is not None:
                    max_allowed = min(max_allowed, self._current_max_obstacles)
                block_count = self.rng.randint(1, max_allowed)

                self.rng.shuffle(available_cols)
                chosen: List[int] = []
                for col in available_cols:
                    if all(abs(col - existing) > 1 for existing in chosen):
                        chosen.append(col)
                    if len(chosen) == block_count:
                        break
                chosen.sort()

                for col in chosen:
                    pos = Vector2D(
                        self._col_positions[col],
                        self._row_positions[lane_idx + 1]
                    )
                    transform = Transform(position=pos)
                    sprite = Sprite(
                        size=size,
                        color="#8e8268",
                        shape=SpriteShape.RECT,
                        image_path="./images/road_barrier.png"
                    )
                    transform.attach_component(sprite)
                    transforms.append(transform)
                    sprites.append(sprite)
                blocked = chosen

            self._obstacle_layout.append(blocked)
            self._obstacle_transforms.append(transforms)
            self._obstacle_sprites.append(sprites)

    # ------------------------------------------------------------------ #
    # Entity cleanup helpers
    # ------------------------------------------------------------------ #
    def _cleanup_cars(self) -> None:
        for car in getattr(self, "_active_cars", []):
            if car.rigidbody:
                car.rigidbody.set_velocity(Vector2D.zero())
            car.delete()
        self._active_cars = []

    def _clear_obstacles(self) -> None:
        for transforms in getattr(self, "_obstacle_transforms", []):
            for transform in transforms:
                transform.delete()
        self._obstacle_layout = []
        self._obstacle_transforms = []
        self._obstacle_sprites = []

    # ------------------------------------------------------------------ #
    # Input binding
    # ------------------------------------------------------------------ #
    def _bind_events(self) -> None:
        self._event = Event(
            source=self.canvas,
            watched_events=["keydown", "keyup"],
            prevent_default_action=True,
            stop_propagation=True,
        )
        self._event.on_dom_event(self._handle_dom_event)

    # ------------------------------------------------------------------ #
    # Game loop & rendering
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._logic_task is None or self._logic_task.done():
            self._logic_task = asyncio.create_task(self._logic_loop())
        if self._draw_task is None or self._draw_task.done():
            self._draw_task = asyncio.create_task(self._draw_loop())
        try:
            self.canvas.focus()
        except Exception:
            pass

    def stop(self) -> None:
        if self._logic_task and not self._logic_task.done():
            self._logic_task.cancel()
        self._logic_task = None
        if self._draw_task and not self._draw_task.done():
            self._draw_task.cancel()
        self._draw_task = None

    async def _logic_loop(self) -> None:
        try:
            while True:
                self._update()
                await asyncio.sleep(self.dt)
        except asyncio.CancelledError:
            pass

    async def _draw_loop(self) -> None:
        try:
            while True:
                self._draw()
                await asyncio.sleep(self.dt)
        except asyncio.CancelledError:
            pass

    def _update(self) -> None:
        if self._waiting_for_restart:
            return

        self._update_input_repeat()

        # spawn new cars per lane
        for lane in self._traffic_lanes:
            lane["spawn_timer"] -= self.dt
            if lane["spawn_timer"] <= 0.0:
                self._spawn_car(lane)

        for car in self._active_cars:
            velocity = Vector2D(car.speed * car.direction, 0.0)
            if car.rigidbody:
                car.rigidbody.set_velocity(velocity)
                car.transform.set_position(Vector2D(
                    car.transform.position.x,
                    self._row_positions[car.lane_index + 1]
                ))
            else:
                car.transform.translate(velocity * self.dt)

        Physics.step(self.dt)

        for car in list(self._active_cars):
            sprite = car.sprite
            half_width = sprite.size.x * 0.5
            x = car.transform.position.x

            out_right = car.direction > 0 and x - half_width > self.width + 40
            out_left = car.direction < 0 and x + half_width < -40
            if out_right or out_left:
                self._remove_car(car)

        self._check_collisions()
        self._check_goal()
        self._on_state_change()

    def _spawn_car(self, lane: Dict[str, Any]) -> None:
        width = self.rng.uniform(*lane["size_width_range"])
        height = lane["size_height"]
        size = Vector2D(width, height)

        direction = lane["direction"]
        start_x = -size.x if direction > 0 else self.width + size.x
        y = lane["y"]

        transform = Transform(Vector2D(start_x, y))
        sprite = self.rng.choice(self._car_sprites).copy()
        # image is front facing, rotate canvas to right if direction > 0 and left if direction < 0
        sprite.ensure_canvas(
            rotation_angle=90.0 * (-1 if direction > 0 else 1),
            reset=True
        )

        sprite_size = sprite.image_array_size or size
        sprite.set_size(sprite_size)
        collider = BoxCollider(size=sprite_size)
        collider.set_layer_bits(CollisionLayer.ROBOT)
        collider.set_mask_bits(CollisionLayer.ALL_BITS)

        transform.attach_component(sprite)
        transform.attach_component(collider)

        speed = self.rng.uniform(*lane["speed_range"])
        rigidbody = RigidBody(
            mass=1.0,
            friction=0.05,
            bounciness=0.0,
            freeze_position=[False, True]  # freeze y position (lane)
        )
        transform.attach_component(rigidbody)
        rigidbody.set_velocity(Vector2D(speed * direction, 0.0))
        car = Car(
            transform=transform,
            speed=speed,
            direction=direction,
            lane_index=lane["index"],
            rigidbody=rigidbody,
        )
        self._active_cars.append(car)

        lane["spawn_timer"] = self.rng.uniform(*lane["spawn_range"])

    def _remove_car(self, car: Car) -> None:
        if car in self._active_cars:
            self._active_cars.remove(car)
            if car.rigidbody:
                car.rigidbody.set_velocity(Vector2D.zero())
            car.delete()

    def _update_input_repeat(self) -> None:
        if self._waiting_for_restart or not self._keys_down:
            return
        for key in list(self._keys_down):
            if key not in self.trigger_map:
                continue
            timer = self._key_repeat_timers.get(key, self._key_repeat_interval)
            timer -= self.dt
            if timer <= 0.0:
                moved = self._handle_move_key(key)
                timer = self._key_repeat_interval
                if self._waiting_for_restart:
                    self._key_repeat_timers.pop(key, None)
                    self._keys_down.clear()
                    break
            self._key_repeat_timers[key] = timer

    def _apply_duck_position(self) -> None:
        if self.duck_transform is None:
            return
        pos = Vector2D(
            self._col_positions[self._col_index],
            self._row_positions[self._row_index]
        )
        self.duck_transform.set_position(pos)

    def _draw(self) -> None:
        with hold_canvas(self.canvas):
            self.canvas.clear()

            self.canvas.fill_style = "#75b798"
            self.canvas.fill_rect(0, 0, self.width, self.height)

            road_top = self.road_margin
            road_height = self.height - 2 * self.road_margin
            self.canvas.fill_style = "#2d3436"
            self.canvas.fill_rect(0, road_top, self.width, road_height)

            lane_height = road_height / max(1, self.lane_count)
            self.canvas.stroke_style = "#dfe6e9"
            self.canvas.set_line_dash([8, 12])
            for lane_idx in range(1, self.lane_count):
                y = road_top + lane_height * lane_idx
                self.canvas.begin_path()
                self.canvas.move_to(0, y)
                self.canvas.line_to(self.width, y)
                self.canvas.stroke()
            self.canvas.set_line_dash([])

            for car in self._active_cars:
                pos = car.transform.position
                sprite = car.sprite
                canvas = sprite.ensure_canvas(reset=False)
                if canvas:
                    self.canvas.draw_image(
                        canvas,
                        pos.x - sprite.size.x / 2,
                        pos.y - sprite.size.y / 2
                    )
                else:
                    self.canvas.fill_style = sprite.color
                    self.canvas.fill_rect(
                        pos.x - sprite.size.x / 2,
                        pos.y - sprite.size.y / 2,
                        sprite.size.x,
                        sprite.size.y,
                    )

            for transforms, sprites in zip(self._obstacle_transforms, self._obstacle_sprites):
                for transform, sprite in zip(transforms, sprites):
                    canvas = sprite.ensure_canvas(reset=False)
                    pos = transform.position
                    if canvas:
                        # keep sprite height but scale width to fit
                        image_size = sprite.image_array_size or sprite.size
                        final_size = Vector2D(
                            image_size.x * (sprite.size.y / image_size.y),
                            sprite.size.y
                        )
                        self.canvas.draw_image(
                            canvas,
                            pos.x - final_size.x / 2,
                            pos.y - final_size.y / 2,
                            final_size.x,
                            final_size.y
                        )
                    else:
                        self.canvas.fill_style = sprite.color
                        self.canvas.fill_rect(
                            pos.x - sprite.size.x / 2,
                            pos.y - sprite.size.y / 2,
                            sprite.size.x,
                            sprite.size.y,
                        )

            if self.duck_transform and self.duck_sprite:
                pos = self.duck_transform.position
                sprite = self.duck_sprite
                canvas = sprite.ensure_canvas(reset=False)
                if canvas:
                    self.canvas.draw_image(
                        canvas,
                        pos.x - sprite.size.x / 2,
                        pos.y - sprite.size.y / 2
                    )
                else:
                    self.canvas.fill_style = sprite.color
                    self.canvas.fill_rect(
                        pos.x - sprite.size.x / 2,
                        pos.y - sprite.size.y / 2,
                        sprite.size.x,
                        sprite.size.y,
                    )

            state_name = self.fsm.current_state.name if self.fsm.current_state else "--"
            self.canvas.fill_style = "#000"
            self.canvas.font = "14px monospace"
            self.canvas.fill_text(f"État: {state_name}", 10, 18)
            self.canvas.fill_text(f"Score: {self.score}", 10, 36)
            self.canvas.fill_text(f"Highscore: {self.high_score}", self.width - 115, 18)
            self.canvas.fill_text(f"Niveau: {self.level + 1}", self.width - 115, 36)

            if self._prompt_visible:
                self.canvas.fill_style = "rgba(0, 0, 0, 0.55)"
                self.canvas.fill_rect(0, self.height / 2 - 30, self.width, 60)
                self.canvas.fill_style = "#ffffff"
                self.canvas.font = "20px monospace"
                self.canvas.text_align = "center"
                self.canvas.fill_text("Appuyer sur ENTRER pour recommencer", self.width / 2, self.height / 2 + 7)
                self.canvas.text_align = "left"

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _handle_dom_event(self, event: Dict[str, Any]) -> None:
        key = event.get("key")
        if key is None:
            return

        etype = event.get("type")
        if etype == "keydown" and self._waiting_for_restart:
            if key == "Enter" and not event.get("repeat", False):
                self._resume_after_pause()
            return

        if etype == "keydown":
            if key in self.trigger_map and not self._waiting_for_restart:
                if not self._keys_down and key not in self._keys_down:
                    self._keys_down.add(key)
                    moved = self._handle_move_key(key)
                    self._key_repeat_timers[key] = self._key_repeat_interval
            return
        elif etype == "keyup":
            if key in self._keys_down:
                self._keys_down.remove(key)
            self._key_repeat_timers.pop(key, None)
            if not self._keys_down and self.release_trigger and not self._waiting_for_restart:
                self._emit_trigger(self.release_trigger)

    def _handle_move_key(self, key: str) -> bool:
        dx = 0
        dy = 0
        if key == "ArrowUp":
            dy = 1
        elif key == "ArrowLeft":
            dx = -1
        elif key == "ArrowRight":
            dx = 1
        else:
            return False

        moved = self._move_duck(dx, dy)
        if moved and dy > 0 and not self._waiting_for_restart:
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
        if moved:
            self._emit_trigger(self.trigger_map.get(key))
        return moved

    def _move_duck(self, dx: int, dy: int) -> bool:
        new_col = self._col_index + dx
        new_row = self._row_index + dy
        if not (0 <= new_col < self._grid_cols):
            return False
        if not (0 <= new_row < len(self._row_positions)):
            return False

        self._col_index = new_col
        self._row_index = new_row
        self._apply_duck_position()

        if 0 < self._row_index < len(self._row_positions) - 1:
            lane_idx = self._row_index - 1
            if (
                self.lane_types[lane_idx] is LaneType.OBSTACLES
                and self._col_index in self._obstacle_layout[lane_idx]
            ):
                self._handle_duck_hit()
                return False
        return True

    def _emit_trigger(self, trigger: Optional[str | Enum]) -> None:
        if trigger is None:
            return
        name = trigger.name.lower() if isinstance(trigger, Enum) else str(trigger)
        handler = getattr(self.fsm, name, None)
        try:
            if callable(handler):
                handler()
            else:
                self.fsm.trigger(name)
        except MachineError:
            pass

    # ------------------------------------------------------------------ #
    # Game rules
    # ------------------------------------------------------------------ #
    def _check_collisions(self) -> None:
        if self._waiting_for_restart or self.duck_collider is None:
            return

        duck_bounds = self.duck_collider.bounds()
        for car in self._active_cars:
            if duck_bounds.intersects(car.collider.bounds()):
                self._handle_duck_hit()
                return

    def _check_goal(self) -> None:
        if self._row_index >= len(self._row_positions) - 1:
            if self.goal_trigger:
                self._emit_trigger(self.goal_trigger)
            self._advance_level()

    def _advance_level(self) -> None:
        self.level += 1
        self._setup_level()

    def _handle_duck_hit(self) -> None:
        if self.hit_trigger:
            self._emit_trigger(self.hit_trigger)
        self.high_score = max(self.high_score, self.score)
        self.score = 0
        self.level = 0
        self._waiting_for_restart = True
        self._pending_reset = True
        self._prompt_visible = True
        self._keys_down.clear()
        self._key_repeat_timers.clear()
        if self.release_trigger:
            self._emit_trigger(self.release_trigger)

    def _resume_after_pause(self) -> None:
        self._waiting_for_restart = False
        self._prompt_visible = False
        self._keys_down.clear()
        self._key_repeat_timers.clear()
        if self._pending_reset:
            self._pending_reset = False
            self._setup_level()
        else:
            self._apply_duck_position()

    # ------------------------------------------------------------------ #
    # FSM monitoring
    # ------------------------------------------------------------------ #
    def _on_state_change(self) -> None:
        current = self.fsm.current_state.name if self.fsm.current_state else None
        if current != self._last_state:
            self._last_state = current
