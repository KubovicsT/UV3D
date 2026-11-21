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

# A lamp large enough to cover a full printed swath may cure while the
# carriage is stationary. This is an illustrative hold, NOT calibrated dose.
UV_STATIONARY_CURE_SECONDS = 0.1


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class PrintHeadSpec:
    name: str = "Epson i3200-U1"
    body_x_mm: float = 69.1
    body_y_mm: float = 59.5
    body_z_mm: float = 35.6
    swath_mm: float = 33.8


@dataclass(frozen=True)
class MotionSegment:
    kind: str                     # print, travel, uv_followthrough, cure, return, z_move
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
    """Model a shared gantry whose *trailing* UV lamp cures during printing.

    Forward scans use the lamp to the left of the printheads and reverse
    scans the lamp to their right. If the lamp covers the active swath, UV
    runs during deposition, with only a short follow-through if its
    footprint has not yet reached the raster's far edge. Otherwise one
    post-print cure sweep is scheduled. Every segment is continuous in XYZ.
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
        self._scan_pitch_override = scan_pitch_mm
        self.scan_pitch_mm = float(
            scan_pitch_mm if scan_pitch_mm is not None
            else min(h.swath_mm for h in self.gantry_config.heads if h.enabled))
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
        # The narrowest enabled head sets a conservative gantry band step.
        # User-edited physical swath widths therefore change the virtual
        # number of passes and ETA. Explicit test/special simulation pitch
        # overrides take precedence.
        if self._scan_pitch_override is None:
            self.scan_pitch_mm = min(h.swath_mm for h in active_heads)
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

        def add(kind, layer, dest, band, pidx, lamp_index=-1,
                stationary_hold_s=0.0, lamp_on=False):
            nonlocal elapsed, elapsed_print, last_xyz
            dist = math.dist(last_xyz, dest)
            speed = self.print_speed_mm_s if kind == "print" else self.move_speed_mm_s
            # When a lamp footprint fully covers the just-printed strip at
            # the current position, do not force a fictitious UV scan.
            # Instead keep the gantry still and the lamp illuminated briefly.
            duration = dist / speed if dist > 1e-12 else stationary_hold_s
            if dist <= 1e-12 and duration > 0.0:
                speed = 0.0
            if duration > 1e-12:
                seg = MotionSegment(
                    kind, layer, last_xyz, dest, speed, duration,
                    band, pidx, kind == "print", lamp_index,
                    kind in ("cure", "uv_followthrough") or lamp_on)
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
                    # UV lamp selection is based on PHYSICAL relative X
                    # placement, not on the direction of the current row.
                    # The lamp behind ALL active heads is the only one lit.
                    # This prevents the leading lamp curing ink before the
                    # printheads have deposited it.
                    printed_top = min(
                        self.max_y,
                        max(y + head.offset_y_mm + head.swath_mm * .5
                            for head in active_heads))
                    printed_bottom = max(
                        self.min_y,
                        min(y + head.offset_y_mm - head.swath_mm * .5
                            for head in active_heads))
                    candidates = [
                        (i, lamp)
                        for i, lamp in enumerate(self.gantry_config.lamps)
                        if lamp.enabled and (
                            lamp.offset_x_mm < min(h.offset_x_mm for h in active_heads)
                            if forward else
                            lamp.offset_x_mm > max(h.offset_x_mm for h in active_heads))
                    ]
                    if not candidates:
                        # If only a leading/central lamp is installed, do
                        # not illuminate it during printing. Use ONE lamp
                        # in a later dedicated curing sweep instead.
                        candidates = [
                            (i, lamp) for i, lamp in enumerate(
                                self.gantry_config.lamps) if lamp.enabled
                        ]

                    def lamp_rank(candidate):
                        _, lamp = candidate
                        covers_y = (
                            y + lamp.offset_y_mm + lamp.depth_mm*.5 >=
                                printed_top - 1e-7
                            and y + lamp.offset_y_mm - lamp.depth_mm*.5 <=
                                printed_bottom + 1e-7)
                        if forward:
                            tail = max(
                                0.0, self.min_x + self.width -
                                (target_x + lamp.offset_x_mm + lamp.width_mm*.5))
                        else:
                            tail = max(
                                0.0,
                                target_x + lamp.offset_x_mm -
                                lamp.width_mm*.5 - self.min_x)
                        # Prefer a lamp that fits the swath, and the one
                        # needing least extra non-print travel.
                        return (not covers_y, tail, abs(lamp.offset_x_mm))

                    selected = min(candidates, key=lamp_rank) if candidates else None
                    selected_i = selected[0] if selected else -1
                    selected_lamp = selected[1] if selected else None
                    is_trailing = selected_lamp is not None and (
                        selected_lamp.offset_x_mm <
                            min(h.offset_x_mm for h in active_heads)
                        if forward else
                        selected_lamp.offset_x_mm >
                            max(h.offset_x_mm for h in active_heads))
                    fits_y = (
                        selected_lamp is not None and
                        y + selected_lamp.offset_y_mm +
                            selected_lamp.depth_mm*.5 >= printed_top - 1e-7 and
                        y + selected_lamp.offset_y_mm -
                            selected_lamp.depth_mm*.5 <= printed_bottom + 1e-7)
                    inline_uv = (
                        printed_top > printed_bottom and is_trailing and fits_y)
                    before = elapsed
                    add("print", layer, (target_x, y, z), band, pass_index,
                        selected_i if inline_uv else -1, lamp_on=inline_uv)
                    print_seconds += elapsed - before

                    if inline_uv:
                        # The lamp footprint trails the nozzles. Carry the
                        # head further only if the trailing edge would
                        # otherwise leave the farthest ink uncured.
                        if forward:
                            far_x = max(
                                last_xyz[0],
                                self.min_x + self.width -
                                selected_lamp.offset_x_mm -
                                selected_lamp.width_mm*.5)
                        else:
                            far_x = min(
                                last_xyz[0],
                                self.min_x - selected_lamp.offset_x_mm +
                                selected_lamp.width_mm*.5)
                        add("uv_followthrough", layer, (far_x, y, z),
                            band, pass_index, selected_i)

                    # The first UV exposure is already completed DURING
                    # printing in the inline case. Additional configured
                    # sweeps remain possible (cure_passes > 1). Otherwise,
                    # use a separate post-print sweep of ONE selected lamp.
                    extra_sweeps = (
                        self.gantry_config.cure_passes - int(inline_uv))
                    if printed_top > printed_bottom and selected_lamp:
                        for _ in range(extra_sweeps):
                            lamp_index, lamp = selected_i, selected_lamp
                            # A narrow lamp requires more than one
                            # coverage sweep *of the printed strip*.
                            # No fictitious whole-layer UV pass is added.
                            lamp_rows = max(
                                1, math.ceil((printed_top - printed_bottom)
                                             / lamp.depth_mm))
                            lamp_pitch = (
                                (printed_top - printed_bottom) / lamp_rows)
                            # The UV rectangle covers points from
                            # lamp_center +/- half its illuminated size.
                            # Crucially UV curing has NO nozzle overscan.
                            # Earlier code reversed these +/- signs and
                            # added overscan, forcing the entire carriage
                            # unnecessarily far outside the printed model.
                            #
                            # If the lamp is wider than the whole raster,
                            # many gantry positions already fully cover
                            # its X span, and no X sweep is necessary.
                            wide_x = lamp.width_mm >= self.width
                            if wide_x:
                                possible_x_min = (
                                    self.min_x + self.width -
                                    lamp.width_mm * .5)
                                possible_x_max = (
                                    self.min_x + lamp.width_mm * .5)
                            else:
                                left_center_x = (
                                    self.min_x + lamp.width_mm * .5)
                                right_center_x = (
                                    self.min_x + self.width -
                                    lamp.width_mm * .5)
                            for cure_row in range(lamp_rows):
                                # Rows split only the JUST-PRINTED band.
                                # Keep the lamp as near as possible to
                                # the present gantry Y position when its
                                # depth already covers the whole band.
                                row_top = printed_top - cure_row * lamp_pitch
                                row_bottom = printed_top - (
                                    cure_row + 1) * lamp_pitch
                                lamp_y = _clamp(
                                    last_xyz[1] + lamp.offset_y_mm,
                                    row_top - lamp.depth_mm * .5,
                                    row_bottom + lamp.depth_mm * .5)
                                cure_y = lamp_y - lamp.offset_y_mm
                                if wide_x:
                                    lamp_x = _clamp(
                                        last_xyz[0] + lamp.offset_x_mm,
                                        possible_x_min, possible_x_max)
                                    cure_x = lamp_x - lamp.offset_x_mm
                                    # No X travel at all when the current
                                    # footprint already covers everything.
                                    # Move Y only when truly required.
                                    add("uv_travel", layer,
                                        (cure_x, cure_y, z), band, pass_index,
                                        lamp_index)
                                    add("cure", layer,
                                        (cure_x, cure_y, z), band, pass_index,
                                        lamp_index,
                                        stationary_hold_s=UV_STATIONARY_CURE_SECONDS)
                                else:
                                    lamp_start_x = (
                                        left_center_x - lamp.offset_x_mm)
                                    lamp_end_x = (
                                        right_center_x - lamp.offset_x_mm)
                                    if abs(last_xyz[0] - lamp_start_x) <= abs(
                                            last_xyz[0] - lamp_end_x):
                                        start_x, end_x = lamp_start_x, lamp_end_x
                                    else:
                                        start_x, end_x = lamp_end_x, lamp_start_x
                                    add("uv_travel", layer,
                                        (start_x, cure_y, z), band, pass_index,
                                        lamp_index)
                                    add("cure", layer,
                                        (end_x, cure_y, z), band, pass_index,
                                        lamp_index)
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
            # Non-printing motion must NOT deposit any new material.
            # Cure sweeps occur immediately AFTER one print pass, so only
            # that pass has completed; never show the whole layer as printed
            # while the first UV cure is moving.
            direction = 1
            scan_fraction = 0.0
            completed_passes = (
                self.passes if segment.kind in ("return", "z_move")
                else segment.pass_index + 1 if segment.kind in
                ("uv_travel", "cure", "uv_followthrough")
                else segment.pass_index)
            model_fraction = support_fraction = scan_fraction
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
            # During printing only the trailing mounted lamp lights up;
            # follow-through and any extra UV sweeps keep the same lamp.
            lamp_active=segment.curing,
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
