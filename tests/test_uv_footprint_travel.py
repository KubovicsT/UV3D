"""UV lamp footprint optimization: oversized lamps must not send carriage away."""
import pytest

from uv3d.gantry import GantryConfig, HeadConfig, LampConfig
from uv3d.simulation_engine import PrintSimulation
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
def test_oversized_aligned_uv_lamp_fully_sweeps_every_printed_pixel(
        print_mode, passes):
    """Even a very wide lamp must traverse its COMPLETE illuminated width.

    A stationary UV pause gives nonuniform exposure compared with a
    continuously moving lamp. Verify every layer gets a full footprint
    sweep with uninterrupted carriage motion and correct duration.
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
        # The dedicated full-footprint sweep starts outside the raster.
        # Repositioning is explicit and the lamp is OFF during that move.
        assert [s.kind for _,s in filtered] == (
            ["print","uv_travel","cure"]*passes)
        for idx,(i,s) in enumerate(filtered):
            if s.kind=="cure":
                previous=filtered[idx-1][1]
                assert previous.kind=="uv_travel"
                assert not previous.curing
                assert previous.end_xyz == pytest.approx(s.start_xyz)
                # The full 100-mm lamp footprint must pass from just
                # outside X=0 to just outside X=20: 120 mm of travel.
                assert abs(s.end_xyz[0]-s.start_xyz[0]) == pytest.approx(120)
                assert s.speed_mm_s == plan.move_speed_mm_s
                assert s.duration_s == pytest.approx(
                    120 / plan.move_speed_mm_s)
                assert {s.start_xyz[0],s.end_xyz[0]} == {-50, 70}
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
    # Center of lamp travels the raster width PLUS the lamp width,
    # regardless of its mounting offset, to fully traverse the print.
    assert abs(cure.end_xyz[0]-cure.start_xyz[0])==pytest.approx(120)
    assert {cure.start_xyz[0],cure.end_xyz[0]}=={-250,-130}
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
    # A 10-mm UV lamp on a 20-mm printed raster needs a 30-mm sweep:
    # lead edge starts at X=0 and trail edge clears X=20.
    assert all(abs(s.end_xyz[0]-s.start_xyz[0])==pytest.approx(30)
               for s in cures)
    assert all({s.start_xyz[0],s.end_xyz[0]}=={-5.0,25.0}
               for s in cures)


def test_oversized_lamp_stays_aligned_in_y_without_reposition():
    config=GantryConfig(
        heads=[HeadConfig("main","white")],
        lamps=[LampConfig("big lamp", 0,0, 100,100)])
    plan=PrintSimulation(job(depth=30),passes=1,
                         gantry_config=config)
    for i,s in enumerate(plan.segments):
        if s.kind=="cure":
            # The lamp may first travel to the full-footprint sweep start.
            earlier=next(p for p in reversed(plan.segments[:i])
                         if p.kind=="print")
            assert s.start_xyz[1]==pytest.approx(earlier.end_xyz[1])
            assert s.end_xyz[1]==pytest.approx(earlier.end_xyz[1])
