"""UV curing must follow each print pass before another ink pass."""
import math

import pytest

from uv3d.gantry import GantryConfig, HeadConfig, LampConfig
from uv3d.simulation_engine import PrintSimulation


def example_job():
    return {
        "settings": {"dpi": 254, "layer_mm": .2,
                     "margin_mm": 0, "body_material": "white"},
        "model_bounds_mm": [[0, 0, 0], [20, 30, .4]],
        "raster_size_px": [200, 300],
        "layers": [{"target_stack_height_mm": .2},
                   {"target_stack_height_mm": .4}]
    }


def config(lamp_depth=45, swath=33.8, cure_passes=1):
    return GantryConfig(
        heads=[HeadConfig("White model", "white", swath_mm=swath)],
        # Centered lamp is neither trailing side; it must cure in its
        # separate post-print movement in either direction. This fixture
        # specifically verifies the FALLBACK when inline curing is unsafe.
        lamps=[LampConfig("UV", 0, 0, 20, lamp_depth)],
        cure_passes=cure_passes)


def frame_at_midpoint(plan, index):
    s = plan.segments[index]
    t0 = plan._segment_end_s[index - 1] if index else 0
    return plan.get_frame((t0 + .5 * s.duration_s) / plan.total_seconds)


@pytest.mark.parametrize("mode", ["bidirectional", "unidirectional"])
@pytest.mark.parametrize("passes", [1, 2, 3])
def test_each_print_pass_immediately_followed_by_uv_sweep(mode, passes):
    plan = PrintSimulation(example_job(), print_mode=mode,
                           passes=passes, gantry_config=config())
    for layer in range(plan.count):
        order = [(i, s) for i, s in enumerate(plan.segments)
                 if s.layer_index == layer and s.kind in ("print", "cure")]
        assert [s.kind for _, s in order] == ["print", "cure"] * passes
        assert [s.pass_index for _, s in order if s.kind == "cure"] == list(range(passes))
        for i, s in order:
            if s.kind == "cure":
                frame = frame_at_midpoint(plan, i)
                assert frame.curing and frame.lamp_active
                assert frame.passes_completed == s.pass_index + 1
                assert frame.row_index == s.band_index
                assert not frame.printing
            else:
                frame = frame_at_midpoint(plan, i)
                assert frame.printing and not frame.lamp_active
    for a, b in zip(plan.segments, plan.segments[1:]):
        assert a.end_xyz == pytest.approx(b.start_xyz)


def test_first_uv_pass_does_not_mark_two_pass_layer_finished():
    plan = PrintSimulation(example_job(), passes=2,
                           gantry_config=config())
    index = next(i for i, s in enumerate(plan.segments)
                 if s.kind == "cure")
    frame = frame_at_midpoint(plan, index)
    assert frame.passes_completed == 1
    assert frame.row_index == 0
    assert not frame.printing
    assert frame.lamp_active
    later_print = next(s for s in plan.segments[index + 1:]
                       if s.kind == "print")
    assert later_print.pass_index == 1
    assert later_print.layer_index == 0


@pytest.mark.parametrize("lamp_depth,rows_per_band", [(40,1), (11,2), (5,3)])
def test_narrow_lamps_only_cure_just_printed_band(lamp_depth, rows_per_band):
    plan = PrintSimulation(example_job(), passes=2,
                           gantry_config=config(lamp_depth,swath=15))
    assert plan.rows == 2
    for layer in range(plan.count):
        for band in range(plan.rows):
            curing = [s for s in plan.segments if s.kind == "cure"
                      and s.layer_index == layer and s.band_index == band]
            assert len(curing) == 2 * rows_per_band
            # A lamp can cover the full printed X span at its existing
            # position and dwell, requiring no horizontal movement.
            assert all(s.speed_mm_s in (0, plan.move_speed_mm_s)
                       for s in curing)
            assert all(s.duration_s > 0 for s in curing)
            assert all(s.pass_index in (0,1) for s in curing)


def test_curing_passes_repeat_per_print_pass():
    plan = PrintSimulation(example_job(), passes=2,
                           gantry_config=config(cure_passes=2))
    for layer in range(plan.count):
        order = [s.kind for s in plan.segments
                 if s.layer_index == layer and s.kind in ("print","cure")]
        assert order == ["print","cure","cure",
                         "print","cure","cure"]
