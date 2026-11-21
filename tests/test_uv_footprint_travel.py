"""UV lamp footprint optimization: oversized lamps must not send carriage away."""
import pytest

from uv3d.gantry import GantryConfig, HeadConfig, LampConfig
from uv3d.simulation_engine import (
    PrintSimulation, UV_STATIONARY_CURE_SECONDS)
from uv3d.simulation_viewport import uv_footprints


def job(width=20, depth=30, layers=2):
    return {
        "settings": {"dpi": 254, "layer_mm": .2,
                     "margin_mm": 0.0, "body_material": "white"},
        "model_bounds_mm": [[0, 0, 0],
                            [width, depth, layers*.2]],
        "raster_size_px": [width*10, depth*10],
        "layers": [{"target_stack_height_mm": (i+1)*.2}
                   for i in range(layers)],
    }


def midpoint(plan, index):
    segment=plan.segments[index]
    before=plan._segment_end_s[index-1] if index else 0.0
    return plan.get_frame(
        (before+segment.duration_s/2)/plan.total_seconds)


@pytest.mark.parametrize("print_mode", ["bidirectional", "unidirectional"])
@pytest.mark.parametrize("passes", [1, 2, 3])
def test_oversized_aligned_uv_lamp_cures_in_place_without_side_travel(
        print_mode, passes):
    """An oversized lamp already covers the raster at the print endpoint.

    UV must turn on and briefly dwell THERE: no needless carriage trip
    to the side, no head teleport, and no rescan of full raster.
    """
    config=GantryConfig(
        heads=[HeadConfig("white", "white", swath_mm=33.8)],
        lamps=[LampConfig("large UV", 0, 0, 100, 80)])
    plan=PrintSimulation(job(), print_mode=print_mode, passes=passes,
                         gantry_config=config)
    for layer in range(plan.count):
        filtered=[(i,s) for i,s in enumerate(plan.segments)
                  if s.layer_index==layer and s.kind in
                  ("print","uv_travel","cure")]
        assert [s.kind for _,s in filtered] == ["print","cure"]*passes
        for idx,(i,s) in enumerate(filtered):
            if s.kind=="cure":
                previous=filtered[idx-1][1]
                assert previous.kind=="print"
                assert s.start_xyz == pytest.approx(previous.end_xyz)
                assert s.end_xyz == pytest.approx(previous.end_xyz)
                assert s.duration_s == pytest.approx(
                    UV_STATIONARY_CURE_SECONDS)
                assert s.speed_mm_s == 0
                frame=midpoint(plan,i)
                assert frame.curing and not frame.printing
                assert frame.head_position_mm == pytest.approx(s.start_xyz)
                assert frame.passes_completed == s.pass_index+1
                footprints=uv_footprints(frame,config,0,0,20,30)
                assert footprints
                assert footprints[0][1:] == pytest.approx((0,0,20,30))
    assert all(a.end_xyz==pytest.approx(b.start_xyz)
               for a,b in zip(plan.segments,plan.segments[1:]))


def test_oversized_lamp_repositions_only_if_its_actual_offset_requires_it():
    config=GantryConfig(
        heads=[HeadConfig("main","white")],
        lamps=[LampConfig("offset lamp", 200, 0, 100, 80)])
    plan=PrintSimulation(job(), passes=1, gantry_config=config)
    segments=plan.segments
    first_print=next(i for i,s in enumerate(segments) if s.kind=="print")
    travel=segments[first_print+1]
    cure=segments[first_print+2]
    assert travel.kind=="uv_travel"
    assert cure.kind=="cure"
    assert travel.start_xyz==pytest.approx(segments[first_print].end_xyz)
    assert travel.end_xyz==pytest.approx(cure.start_xyz)
    assert cure.start_xyz==pytest.approx(cure.end_xyz)
    # Position chosen is the nearest feasible location that covers raster.
    frame=midpoint(plan,first_print+2)
    lit=uv_footprints(frame,config,0,0,20,30)
    assert lit and lit[0][1:] == pytest.approx((0,0,20,30))


def test_narrow_lamp_only_sweeps_needed_distance_without_overscan():
    config=GantryConfig(
        heads=[HeadConfig("main","white")],
        lamps=[LampConfig("narrow lamp", 0, 0, 10, 45)])
    plan=PrintSimulation(job(),passes=2,gantry_config=config)
    cures=[s for s in plan.segments if s.kind=="cure"]
    assert len(cures)==2*plan.count
    # 20-mm printed width with a 10-mm-wide UV footprint only needs
    # 10 mm of lamp center travel, not 20 mm plus head overscan.
    assert all(abs(s.end_xyz[0]-s.start_xyz[0])==pytest.approx(10)
               for s in cures)
    assert all({s.start_xyz[0],s.end_xyz[0]}=={5.0,15.0}
               for s in cures)


def test_oversized_lamp_stays_aligned_in_y_without_reposition():
    config=GantryConfig(
        heads=[HeadConfig("main","white")],
        lamps=[LampConfig("big lamp", 0,0, 100,100)])
    plan=PrintSimulation(job(depth=30),passes=1,
                         gantry_config=config)
    for i,s in enumerate(plan.segments):
        if s.kind=="cure":
            earlier=plan.segments[i-1]
            assert earlier.kind=="print"
            assert s.start_xyz[1]==pytest.approx(earlier.end_xyz[1])
            assert s.end_xyz[1]==pytest.approx(earlier.end_xyz[1])
