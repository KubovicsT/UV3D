"""Deterministic seekable *estimated* Epson i3200-U1 material-jet timeline.

No printer data, motion commands, real acceleration or UV curing: scan speed,
travel speed, passes, print width and head physical envelope are assumptions
for a useful offline preview. Full return motions are explicit segments.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math

TIMELINE_TICKS = 100_000


@dataclass(frozen=True)
class PrintHeadSpec:
    name: str = "Epson i3200-U1"
    body_x_mm: float = 69.1
    body_y_mm: float = 59.5
    body_z_mm: float = 35.6
    swath_mm: float = 33.8


@dataclass(frozen=True)
class MotionSegment:
    kind: str                     # print, travel, stepover, return, z_move
    layer_index: int
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    speed_mm_s: float
    duration_s: float
    band_index: int
    pass_index: int
    printing: bool


@dataclass(frozen=True)
class SimulationFrame:
    layer_index: int
    layer_count: int
    layer_progress: float
    completed_height_mm: float
    current_top_mm: float
    head_position_mm: tuple[float, float, float]
    row_index: int
    row_fraction: float
    rows_per_layer: int
    elapsed_virtual_s: float
    total_virtual_s: float
    timeline_fraction: float
    completed: bool
    motion_kind: str = "print"
    printing: bool = False
    progress_percent: float = 0.0
    elapsed_print_s: float = 0.0
    remaining_virtual_s: float = 0.0
    pass_index: int = 0
    scan_direction: int = 1             # +1 scans left->right, -1 right->left
    passes_completed: int = 0          # completed printing passes in current band


class PrintSimulation:
    """Continuous motion: print across X, move between passes/bands, return,
    then change Z. Both speeds modify actual segment durations. Binary-searched
    timeline seeking remains fast even over 1000+ layers.
    """

    def __init__(
        self, manifest: dict, move_speed_mm_s: float = 600.0,
        scan_pitch_mm: float | None = None, head_clearance_mm: float = 2.0,
        print_speed_mm_s: float = 300.0, passes: int = 6,
        overscan_mm: float = 10.0, head_spec: PrintHeadSpec | None = None,
        print_mode: str = "bidirectional",
    ):
        self.layers = manifest.get("layers", [])
        if not self.layers:
            raise ValueError("A completed layer-mask job is required for simulation.")
        self.count = len(self.layers)
        info = manifest["settings"]
        self.layer_mm = float(info["layer_mm"])
        self.dpi = int(info["dpi"])
        if self.dpi <= 0 or self.layer_mm <= 0:
            raise ValueError("Invalid job layer settings.")
        margin = float(manifest.get("effective_raster_margin_mm", info.get("margin_mm", 0)))
        bounds = manifest["model_bounds_mm"]
        self.min_x = float(bounds[0][0]) - margin
        self.max_y = float(bounds[1][1]) + margin
        w_px, h_px = manifest["raster_size_px"]
        self.width = float(w_px) * 25.4 / self.dpi
        self.depth = float(h_px) * 25.4 / self.dpi
        self.min_y = self.max_y - self.depth
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("Invalid job raster size.")
        self.head_spec = head_spec or PrintHeadSpec()
        self.scan_pitch_mm = float(
            scan_pitch_mm if scan_pitch_mm is not None else self.head_spec.swath_mm)
        self.head_clearance_mm = float(head_clearance_mm)
        self.overscan_mm = float(overscan_mm)
        if (not math.isfinite(self.scan_pitch_mm) or self.scan_pitch_mm <= 0
                or not math.isfinite(self.head_clearance_mm) or self.head_clearance_mm < 0
                or not math.isfinite(self.overscan_mm) or self.overscan_mm < 0):
            raise ValueError("Invalid simulation geometry.")
        self.rows = max(1, math.ceil(self.depth / self.scan_pitch_mm))
        self.row_step = self.depth / self.rows
        self.passes = self._validate_passes(passes)
        self.print_mode = self._validate_print_mode(print_mode)
        self.print_speed_mm_s = self._validate_speed(print_speed_mm_s)
        self.move_speed_mm_s = self._validate_speed(move_speed_mm_s)
        self.segments: list[MotionSegment] = []
        self._segment_end_s: list[float] = []
        self._segment_print_end_s: list[float] = []
        self._layer_start_s: list[float] = []
        self._layer_end_s: list[float] = []
        self._layer_print_time_s: list[float] = []
        self._build_path()

    @staticmethod
    def _validate_speed(value: float) -> float:
        if not math.isfinite(value) or not 1 <= value <= 3000:
            raise ValueError("Printhead and movement speed must be 1–3000 mm/s.")
        return float(value)

    @staticmethod
    def _validate_passes(value: int) -> int:
        if isinstance(value, bool) or int(value) != value or not 1 <= int(value) <= 32:
            raise ValueError("Passes per band must be 1–32.")
        return int(value)

    @staticmethod
    def _validate_print_mode(mode: str) -> str:
        if mode not in ("bidirectional", "unidirectional"):
            raise ValueError("Printing direction must be bidirectional or unidirectional.")
        return mode

    def set_print_speed(self, value: float):
        self.print_speed_mm_s = self._validate_speed(value)
        self._build_path()

    def set_move_speed(self, value: float):
        self.move_speed_mm_s = self._validate_speed(value)
        self._build_path()

    def set_passes(self, value: int):
        self.passes = self._validate_passes(value)
        self._build_path()

    def set_print_mode(self, value: str):
        self.print_mode = self._validate_print_mode(value)
        self._build_path()

    def configure(self, print_speed_mm_s: float, move_speed_mm_s: float,
                  passes: int, print_mode: str | None = None):
        """Validate and recompute the timeline just once for a UI change."""
        printing = self._validate_speed(print_speed_mm_s)
        travel = self._validate_speed(move_speed_mm_s)
        passes = self._validate_passes(passes)
        mode = self._validate_print_mode(print_mode if print_mode is not None
                                         else self.print_mode)
        self.print_mode = mode
        self.print_speed_mm_s = printing
        self.move_speed_mm_s = travel
        self.passes = passes
        self._build_path()

    @property
    def layer_seconds(self):
        return self._layer_end_s[0]

    @property
    def total_seconds(self):
        return self._segment_end_s[-1]

    @property
    def total_print_seconds(self):
        return self._segment_print_end_s[-1]

    @property
    def total_movement_seconds(self):
        return self.total_seconds - self.total_print_seconds

    def _build_path(self):
        self.segments = []
        self._segment_end_s = []
        self._segment_print_end_s = []
        self._layer_start_s = []
        self._layer_end_s = []
        self._layer_print_time_s = []
        elapsed = 0.0
        elapsed_print = 0.0
        left = self.min_x - self.overscan_mm
        right = self.min_x + self.width + self.overscan_mm
        y_first = self.max_y - self.row_step * 0.5
        last_xyz = (left, y_first,
                    float(self.layers[0]["target_stack_height_mm"]) + self.head_clearance_mm)

        def add(kind, layer, dest, band, pidx):
            nonlocal elapsed, elapsed_print, last_xyz
            dist = math.dist(last_xyz, dest)
            speed = self.print_speed_mm_s if kind == "print" else self.move_speed_mm_s
            duration = dist / speed
            # No zero-length return segment needed; every meaningful move has
            # an explicit continuous start and end and positive duration.
            if duration > 1e-12:
                seg = MotionSegment(kind, layer, last_xyz, dest, speed,
                                    duration, band, pidx, kind == "print")
                self.segments.append(seg)
                elapsed += duration
                if kind == "print":
                    elapsed_print += duration
                self._segment_end_s.append(elapsed)
                self._segment_print_end_s.append(elapsed_print)
            last_xyz = dest

        for layer in range(self.count):
            self._layer_start_s.append(elapsed)
            height = float(self.layers[layer]["target_stack_height_mm"])
            z = height + self.head_clearance_mm
            print_seconds = 0.0
            for band in range(self.rows):
                y = self.max_y - (band + .5) * self.row_step
                for pass_index in range(self.passes):
                    # Bidirectional: alternate print direction EVERY pass,
                    # including the transition between adjacent swath bands.
                    # Unidirectional: every pass prints left->right; the
                    # carriage must travel back without printing in between.
                    forward = (self.print_mode == "unidirectional"
                               or (band * self.passes + pass_index) % 2 == 0)
                    start_x = left if forward else right
                    target_x = right if forward else left
                    if (abs(last_xyz[0] - start_x) > 1e-9
                            or abs(last_xyz[1] - y) > 1e-9):
                        add("stepover" if band else "travel", layer,
                            (start_x, y, z), band, pass_index)
                    before = elapsed
                    add("print", layer, (target_x, y, z), band, pass_index)
                    print_seconds += elapsed - before
            self._layer_print_time_s.append(print_seconds)
            # Explicit return at the non-printing movement speed, even for
            # final layer. Next layer starts exactly at this return position.
            add("return", layer, (left, y_first, z), self.rows - 1, self.passes - 1)
            if layer + 1 < self.count:
                next_z = (float(self.layers[layer + 1]["target_stack_height_mm"])
                          + self.head_clearance_mm)
                add("z_move", layer, (left, y_first, next_z),
                    self.rows - 1, self.passes - 1)
            self._layer_end_s.append(elapsed)
        if not self.segments:
            raise ValueError("Print path has no movement.")

    def get_frame(self, timeline_fraction: float) -> SimulationFrame:
        if not math.isfinite(timeline_fraction):
            raise ValueError("Timeline position must be finite.")
        fraction = max(0.0, min(1.0, float(timeline_fraction)))
        total = self.total_seconds
        t = fraction * total
        complete = fraction >= 1.0
        ix = min(len(self.segments) - 1,
                 bisect_right(self._segment_end_s, t + 1e-9))
        segment = self.segments[ix]
        t_begin = self._segment_end_s[ix-1] if ix else 0.0
        alpha = max(0.0, min(1.0,
                            (t - t_begin) / segment.duration_s))
        xyz = tuple(a + alpha * (b - a) for a, b in
                    zip(segment.start_xyz, segment.end_xyz))
        layer = segment.layer_index
        start = self._layer_start_s[layer]
        end = self._layer_end_s[layer]
        layer_progress = max(0.0, min(1.0, (t-start) / max(1e-9, end-start)))
        height = float(self.layers[layer]["target_stack_height_mm"])
        below = 0.0 if layer == 0 else float(
            self.layers[layer-1]["target_stack_height_mm"])
        if complete:
            below = height
            layer_progress = 1.0
        if segment.printing:
            direction = 1 if segment.end_xyz[0] > segment.start_xyz[0] else -1
            # Head X is referenced to the ACTUAL raster bounds (not
            # overscan distance), so the same physical pixel is revealed
            # exactly when the printhead reaches it in either direction.
            raster_left = self.min_x
            if direction == 1:
                scan_fraction = (xyz[0] - raster_left) / self.width
            else:
                scan_fraction = (raster_left + self.width - xyz[0]) / self.width
            scan_fraction = max(0.0, min(1.0, scan_fraction))
            completed_passes = segment.pass_index
        else:
            # A travel/stepover precedes the next pass. Return and Z moves
            # follow the LAST pass; no more deposition occurs there.
            direction = 1
            scan_fraction = 0.0
            completed_passes = (self.passes if segment.kind in ("return", "z_move")
                                else segment.pass_index)
        return SimulationFrame(
            layer_index=layer, layer_count=self.count,
            layer_progress=layer_progress,
            completed_height_mm=below,
            current_top_mm=height,
            head_position_mm=xyz,
            row_index=segment.band_index,
            row_fraction=scan_fraction,
            rows_per_layer=self.rows,
            elapsed_virtual_s=t,
            total_virtual_s=total,
            timeline_fraction=fraction,
            completed=complete,
            motion_kind=segment.kind,
            printing=segment.printing,
            progress_percent=100 * fraction,
            elapsed_print_s=(self._segment_print_end_s[ix] -
                (segment.duration_s * (1.0 - alpha) if segment.printing else 0.0)),
            remaining_virtual_s=max(0.0, total-t),
            pass_index=segment.pass_index,
            scan_direction=direction,
            passes_completed=completed_passes,
        )

    def step_layer(self, timeline_fraction: float, amount: int) -> float:
        layer = self.get_frame(timeline_fraction).layer_index
        target = min(self.count, max(0, layer+int(amount)))
        if target == self.count:
            return 1.0
        return self._layer_start_s[target] / self.total_seconds

    def position_from_ticks(self, ticks: int) -> SimulationFrame:
        return self.get_frame(int(ticks) / TIMELINE_TICKS)


def format_sim_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
