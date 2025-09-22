from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ipycanvas import Canvas, hold_canvas

from park.internal.math import Rect


class Curve:
    def __init__(
        self,
        name: str,
        max_samples: int,
        line_color: str = "#4da3ff",
        background_color: str = "rgba(31,31,37,0.82)",
        border_color: Optional[str] = None,
        grid_color: str = "#8f8f8f",
        text_color: str = "#f0f4ff",
        value_color: str = "#d9efff",
        font: str = "9px 'Fira Mono', monospace",
        title_font: str = "10px 'Fira Mono', monospace",
        value_format: str = "{:.2f}",
        auto_scale: bool = True,
        floor: float = 0.0,
        ceil: float = 1.0,
    ) -> None:
        self.name = name
        self.line_color = line_color
        self.background_color = background_color
        self.border_color = border_color or line_color
        self.grid_color = grid_color
        self.text_color = text_color
        self.value_color = value_color
        self.font = font
        self.title_font = title_font
        self.value_format = value_format
        self.max_samples = max_samples
        self.auto_scale = auto_scale
        self.floor = floor
        self.ceil = ceil

        self.rect: Optional[Rect] = None
        self.samples: List[float] = []

    def append(self, value: float) -> None:
        self.samples.append(float(value))
        self._trim()

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.append(float(value))

    def clear(self) -> None:
        self.samples.clear()

    def set_rect(self, rect: Rect) -> None:
        self.rect = rect

    def is_empty(self) -> bool:
        return not self.samples

    def render(self, canvas: Canvas) -> None:
        if self.rect is None:
            raise ValueError("Curve render rect not set")

        x0 = self.rect.x
        y0 = self.rect.y
        width = self.rect.width
        height = self.rect.height

        pad_top = 28.0
        pad_bottom = 8.0
        pad_left = 8.0
        pad_right = 8.0

        plot_width = max(1.0, width - pad_left - pad_right)
        plot_height = max(1.0, height - pad_top - pad_bottom)
        plot_x0 = x0 + pad_left
        plot_y0 = y0 + pad_top

        with hold_canvas(canvas):
            canvas.fill_style = self.background_color
            canvas.fill_rect(x0, y0, width, height)

            canvas.stroke_style = self.border_color
            canvas.line_width = 1
            canvas.stroke_rect(x0, y0, width, height)

            # Title and current value
            canvas.text_align = "left"
            canvas.text_baseline = "top"
            canvas.font = self.title_font
            canvas.fill_style = self.text_color
            if self.name:
                canvas.fill_text(self.name, x0 + pad_left, y0 + 6)

            canvas.text_align = "right"
            latest_text = "--"
            if self.samples:
                latest_text = self.value_format.format(self.samples[-1])
            canvas.fill_style = self.value_color
            canvas.fill_text(latest_text, x0 + width - pad_right, y0 + 6)

            canvas.text_align = "left"
            canvas.fill_style = self.grid_color
            canvas.font = self.font

            vmin, vmax = self._value_range()
            span = max(vmax - vmin, 1e-9)

            # Grid lines and labels
            h_steps = 4
            v_steps = 4
            canvas.stroke_style = self.grid_color
            canvas.line_width = 1

            canvas.text_align = "right"
            canvas.text_baseline = "middle"
            for i in range(h_steps + 1):
                t = i / max(h_steps, 1)
                y = plot_y0 + plot_height - t * plot_height
                canvas.begin_path()
                canvas.move_to(plot_x0, y)
                canvas.line_to(plot_x0 + plot_width, y)
                canvas.stroke()
                if not self.samples:
                    continue

                value = vmin + t * span
                label = self.value_format.format(value)
                if label == "-0":
                    label = "0"
                canvas.fill_text(label, plot_x0 + plot_width - 2, y - 4)

            canvas.text_align = "center"
            canvas.text_baseline = "top"
            for j in range(1, v_steps):
                t = j / max(v_steps, 1)
                x = plot_x0 + t * plot_width
                canvas.begin_path()
                canvas.move_to(x, plot_y0 - 8)
                canvas.line_to(x, plot_y0 + plot_height)
                canvas.stroke()

            if not self.samples:
                return

            # Draw the curve
            canvas.text_align = "left"
            canvas.text_baseline = "alphabetic"
            canvas.stroke_style = self.line_color
            canvas.line_width = 2
            step_x = plot_width / max(len(self.samples) - 1, 1)
            canvas.begin_path()
            for idx, value in enumerate(self.samples):
                norm = (value - vmin) / span
                px = plot_x0 + idx * step_x
                py = plot_y0 + plot_height - norm * plot_height
                if idx == 0:
                    canvas.move_to(px, py)
                else:
                    canvas.line_to(px, py)
            canvas.stroke()

    def _value_range(self) -> Tuple[float, float]:
        if self.auto_scale and self.samples:
            vmin = min(self.samples)
            vmax = max(self.samples)
            if vmin == vmax:
                padding = abs(vmin) * 0.05 or 0.05
                vmin -= padding
                vmax += padding
            return vmin, vmax
        return self.floor, self.ceil

    def _trim(self) -> None:
        if self.max_samples and len(self.samples) > self.max_samples:
            excess = len(self.samples) - self.max_samples
            if excess > 0:
                del self.samples[:excess]
