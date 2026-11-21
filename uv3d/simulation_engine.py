"""Purely virtual UV3D material-jetting timeline.

All positions and speeds are illustrative. No real PrintExp movement, ink
passes, carriage geometry or curing parameters are assumed or controlled.
The scan path is a conceptual left/right serpentine over each raster.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


TIMELINE_TICKS = 100_000


@dataclass(frozen=True)
class SimulationFrame:
    layer_index: int              # zero-based, inclusive
    layer_count: int
    layer_progress: float         # printed fraction of this layer by travel
    completed_height_mm: float    # fully completed layers only
    current_top_mm: float
    head_position_mm: tuple[float, float, float]
    row_index: int                # zero-based printed scan row
    row_fraction: float           # 0..1 along that row
    rows_per_layer: int
    elapsed_virtual_s: float
    total_virtual_s: float
    timeline_fraction: float
    completed: bool


class PrintSimulation:
    """Independent movement and playback speeds with an arbitrary seek position."""

    def __init__(self, manifest: dict, move_speed_mm_s: float = 100.0,
                 scan_pitch_mm: float = 2.0, head_clearance_mm: float = 2.0):
        layers = manifest.get("layers", [])
        if not layers:
            raise ValueError("A completed layer-mask job is required for simulation.")
        self.layers = layers
        self.count = len(layers)
        info = manifest["settings"]
        self.layer_mm = float(info["layer_mm"])
        self.dpi = int(info["dpi"])
        if self.dpi <= 0 or self.layer_mm <= 0:
            raise ValueError("Invalid job layer settings.")
        margin = float(manifest.get("effective_raster_margin_mm",
                                    info.get("margin_mm", 0.0)))
        bounds = manifest["model_bounds_mm"]
        self.min_x = float(bounds[0][0]) - margin
        self.max_y = float(bounds[1][1]) + margin
        w_px, h_px = manifest["raster_size_px"]
        self.width = float(w_px) * 25.4 / self.dpi
        self.depth = float(h_px) * 25.4 / self.dpi
        self.min_y = self.max_y - self.depth
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("Invalid job raster size.")
        self.scan_pitch_mm = float(scan_pitch_mm)
        self.head_clearance_mm = float(head_clearance_mm)
        if self.scan_pitch_mm <= 0 or self.head_clearance_mm < 0:
            raise ValueError("Invalid simulation geometry.")
        self.rows = max(1, math.ceil(self.depth / self.scan_pitch_mm))
        self.row_step = self.depth / self.rows
        self.path_length_mm = self.rows * self.width + (self.rows - 1) * self.row_step
        self.layer_change_seconds = 0.25  # visualization only
        self.move_speed_mm_s = 100.0
        self.set_move_speed(move_speed_mm_s)

    def set_move_speed(self, speed_mm_s: float) -> None:
        if not math.isfinite(speed_mm_s) or not 1.0 <= speed_mm_s <= 500.0:
            raise ValueError("Move speed must be between 1 and 500 mm/s.")
        self.move_speed_mm_s = float(speed_mm_s)

    @property
    def layer_seconds(self) -> float:
        return self.path_length_mm / self.move_speed_mm_s + self.layer_change_seconds

    @property
    def total_seconds(self) -> float:
        return self.count * self.layer_seconds

    def get_frame(self, timeline_fraction: float) -> SimulationFrame:
        if not math.isfinite(timeline_fraction):
            raise ValueError("Timeline position must be finite.")
        p = max(0.0, min(1.0, float(timeline_fraction)))
        total = self.total_seconds
        elapsed = p * total
        complete = p >= 1.0
        # Timeline fractions like 2/5 can round a few ULPs below a layer
        # boundary. Treat exact layer-step positions as the next layer.
        layer = min(self.count - 1,
                    int(math.floor(elapsed / self.layer_seconds + 1e-9)))
        offset = self.layer_seconds if complete else elapsed - layer * self.layer_seconds
        move_seconds = self.path_length_mm / self.move_speed_mm_s
        distance = min(self.path_length_mm, max(0.0, offset * self.move_speed_mm_s))
        progress = distance / self.path_length_mm
        # Each raster row scans across X, then advances in Y at the row end.
        block = self.width + self.row_step
        row = min(self.rows - 1, int(distance / block))
        within = max(0.0, distance - row * block)
        across = min(1.0, within / self.width)
        direction_right = (row % 2 == 0)
        x = self.min_x + (across if direction_right else 1 - across) * self.width
        if within > self.width and row < self.rows - 1:
            y = self.max_y - (row + .5 + (within - self.width) / self.row_step) * self.row_step
        else:
            y = self.max_y - (row + .5) * self.row_step
        height = float(self.layers[layer]["target_stack_height_mm"])
        completed_height = (
            height if complete else
            (0.0 if layer == 0 else
             float(self.layers[layer - 1]["target_stack_height_mm"])))
        return SimulationFrame(
            layer_index=layer, layer_count=self.count,
            layer_progress=progress, completed_height_mm=completed_height,
            current_top_mm=height,
            head_position_mm=(float(x), float(y), height + self.head_clearance_mm),
            row_index=row, row_fraction=across,
            rows_per_layer=self.rows,
            elapsed_virtual_s=elapsed, total_virtual_s=total,
            timeline_fraction=p, completed=complete,
        )

    def step_layer(self, timeline_fraction: float, amount: int) -> float:
        layer = self.get_frame(timeline_fraction).layer_index
        return max(0.0, min(1.0, (layer + amount) / self.count))

    def position_from_ticks(self, ticks: int) -> SimulationFrame:
        return self.get_frame(int(ticks) / TIMELINE_TICKS)


def format_sim_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
