"""Deterministic seekable *estimated* Epson i3200-U1 material-jet timeline.

No printer data, motion commands, real acceleration or UV curing: scan speed,
travel speed, passes, print width and head physical envelope are assumptions
for a useful offline preview. Full return motions are explicit segments.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math

from .gantry import GantryConfig, HeadConfig, default_gantry

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
    lamp_index: int = -1
    curing: bool = False


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
    model_scan_fraction: float = 0.0
    support_scan_fraction: float = 0.0
    lamp_active: bool = False
    active_lamp_index: int = -1
    curing: bool = False
    model_row_shift: float = 0.0
    support_row_shift: float = 0.0
    raster_band_scale: float = 1.0


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
        gantry_config: GantryConfig | None = None,
    ):
        self.layers = manifest.get("layers", [])
        self._body_material = manifest.get("settings", {}).get(
            "body_material", "white")
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
        # Core engine callers without a gantry explicitly configured retain
        # a single-head / no-UV calculation. The GUI loads the user's
        # configured 3-head + lamp arrangement from its folder-local settings.
        self.gantry_config = (
            GantryConfig.from_dict(gantry_config.as_dict()) if gantry_config
            else GantryConfig(
                heads=[HeadConfig("i3200-U1", "color")], lamps=[]))
        self.gantry_config.validate()
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
        self._model_head = self.gantry_config.heads[0]
        self._support_head = self.gantry_config.heads[0]
        self._min_head_dy = 0.0
        self._gantry_top_y = self.max_y
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

    def set_gantry_config(self, gantry_config: GantryConfig):
        """Change offsets/lamps and rebuild both route and estimated time."""
        config = GantryConfig.from_dict(gantry_config.as_dict())
        self.gantry_config = config
        self._build_path()

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
        active_heads = [h for h in self.gantry_config.heads if h.enabled]
        active_lamps = [l for l in self.gantry_config.lamps if l.enabled]
        self._model_head = next(
            (h for h in active_heads if h.material == self._body_material),
            active_heads[0])
        self._support_head = next(
            (h for h in active_heads if h.material == "support"),
            active_heads[0])
        self._min_head_dy = min(h.offset_y_mm for h in active_heads)
        max_dy = max(h.offset_y_mm for h in active_heads)
        # Enlarge the Y sweep if heads are staggered along Y. A head behind
        # the reference printhead must still reach both raster edges.
        gantry_depth = self.depth + max_dy - self._min_head_dy
        self.rows = max(1, math.ceil(gantry_depth / self.scan_pitch_mm))
        self.row_step = gantry_depth / self.rows
        self._gantry_top_y = self.max_y - self._min_head_dy
        # Head offsets determine how far the gantry itself must travel for
        # every installed material head to cover the full XY raster.
        left = (self.min_x - max(h.offset_x_mm for h in active_heads)
                - self.overscan_mm)
        right = (self.min_x + self.width
                 - min(h.offset_x_mm for h in active_heads)
                 + self.overscan_mm)
        y_first = self._gantry_top_y - self.row_step * 0.5
        last_xyz = (left, y_first,
                    float(self.layers[0]["target_stack_height_mm"]) + self.head_clearance_mm)

        def add(kind, layer, dest, band, pidx, lamp_index=-1):
            nonlocal elapsed, elapsed_print, last_xyz
            dist = math.dist(last_xyz, dest)
            speed = self.print_speed_mm_s if kind == "print" else self.move_speed_mm_s
            duration = dist / speed
            # No zero-length return segment needed; every meaningful move has
            # an explicit continuous start and end and positive duration.
            if duration > 1e-12:
                seg = MotionSegment(
                    kind, layer, last_xyz, dest, speed, duration,
                    band, pidx, kind == "print", lamp_index,
                    kind == "cure")
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
                y = self._gantry_top_y - (band + .5) * self.row_step
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
            # Every enabled UV lamp must physically cross the current layer
            # at least once for each requested cure sweep. The printhead,
            # lamps and beam all travel together; the lamp cannot magically
            # illuminate an area beyond its configured footprint.
            for _ in range(self.gantry_config.cure_passes):
                for lamp_index, lamp in enumerate(self.gantry_config.lamps):
                    if not lamp.enabled:
                        continue
                    lamp_rows = max(1, math.ceil(self.depth / lamp.depth_mm))
                    lamp_pitch = self.depth / lamp_rows
                    lamp_start_x = (
                        self.min_x - lamp.offset_x_mm -
                        lamp.width_mm * .5 - self.overscan_mm)
                    lamp_end_x = (
                        self.min_x + self.width - lamp.offset_x_mm +
                        lamp.width_mm * .5 + self.overscan_mm)
                    for cure_row in range(lamp_rows):
                        cure_y = (
                            self.max_y - (cure_row + .5) * lamp_pitch -
                            lamp.offset_y_mm)
                        # The nearest sweep endpoint minimizes dead travel.
                        if abs(last_xyz[0] - lamp_start_x) <= abs(
                                last_xyz[0] - lamp_end_x):
                            start_x, end_x = lamp_start_x, lamp_end_x
                        else:
                            start_x, end_x = lamp_end_x, lamp_start_x
                        add("uv_travel", layer, (start_x, cure_y, z),
                            self.rows - 1, self.passes, lamp_index)
                        add("cure", layer, (end_x, cure_y, z),
                            self.rows - 1, self.passes, lamp_index)
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
            raster_left = self.min_x
            def scan_for(head):
                physical_x = xyz[0] + head.offset_x_mm
                if direction > 0:
                    p = (physical_x - raster_left) / self.width
                else:
                    p = (raster_left + self.width - physical_x) / self.width
                return max(0.0, min(1.0, p))

            model_fraction = scan_for(self._model_head)
            support_fraction = scan_for(self._support_head)
            scan_fraction = model_fraction
            completed_passes = segment.pass_index
        else:
            # A travel/stepover precedes the next pass. Return and Z moves
            # follow the LAST pass; no more deposition occurs there.
            direction = 1
            scan_fraction = 0.0
            completed_passes = (
                self.passes if segment.kind in (
                    "return", "z_move", "uv_travel", "cure")
                else segment.pass_index)
            model_fraction = support_fraction = scan_fraction
        if segment.curing:
            # Material deposition is finished before UV-only movement.
            completed_passes = self.passes
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
            model_scan_fraction=model_fraction,
            support_scan_fraction=support_fraction,
            lamp_active=(segment.printing or segment.curing),
            active_lamp_index=segment.lamp_index,
            curing=segment.curing,
            model_row_shift=(self._model_head.offset_y_mm -
                             self._min_head_dy) / self.row_step,
            support_row_shift=(self._support_head.offset_y_mm -
                               self._min_head_dy) / self.row_step,
            raster_band_scale=self.depth / self.row_step,
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
