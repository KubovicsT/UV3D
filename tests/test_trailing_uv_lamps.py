"""Two-sided UV lamps must cure BEHIND the active printheads, never ahead."""
import pytest

from uv3d.gantry import GantryConfig, HeadConfig, LampConfig
from uv3d.simulation_engine import PrintSimulation
from uv3d.simulation_viewport import uv_footprints


def job(width=20, depth=30, layers=2):
    return {
        "settings": {"dpi": 254, "layer_mm": .2, "margin_mm": 0,
                     "body_material": "white"},
        "model_bounds_mm": [[0,0,0],[width,depth,layers*.2]],
        "raster_size_px": [width*10,depth*10],
        "layers": [{"target_stack_height_mm":(i+1)*.2}
                   for i in range(layers)],
    }


def midpoint(plan, i):
    seg = plan.segments[i]
    start = plan._segment_end_s[i-1] if i else 0
    return plan.get_frame((start + seg.duration_s*.5) / plan.total_seconds)


@pytest.mark.parametrize("passes", [2,3,4])
def test_bidirectional_lamps_switch_sides_and_cure_during_printing(passes):
    cfg = GantryConfig(
        heads=[HeadConfig("white", "white")],
        lamps=[
            LampConfig("left trailing", -25, 0, 40, 55),
            LampConfig("right trailing", 25, 0, 40, 55),
        ], cure_passes=1)
    plan = PrintSimulation(job(), passes=passes, print_mode="bidirectional",
                           gantry_config=cfg)
    prints = [(i,s) for i,s in enumerate(plan.segments)
              if s.kind == "print" and s.layer_index == 0]
    assert len(prints) == passes
    assert not any(s.kind in ("cure","uv_travel","uv_followthrough")
                   for s in plan.segments)
    for n, (i,s) in enumerate(prints):
        assert s.printing and s.curing
        assert s.lamp_index == n % 2
        assert s.end_xyz[0] > s.start_xyz[0] if n%2 == 0 else (
            s.end_xyz[0] < s.start_xyz[0])
        frame = midpoint(plan, i)
        assert frame.printing and frame.curing and frame.lamp_active
        assert frame.active_lamp_index == n%2
        footprints = uv_footprints(frame, cfg, 0, 0, 20, 30)
        assert len(footprints) == 1
        assert footprints[0][0] == n % 2
    assert all(a.end_xyz == pytest.approx(b.start_xyz)
               for a,b in zip(plan.segments,plan.segments[1:]))


def test_two_trailing_lamps_are_faster_than_single_leading_side_lamp():
    heads = [HeadConfig("white", "white")]
    left = LampConfig("left", -25, 0, 40, 55)
    right = LampConfig("right", 25, 0, 40, 55)
    plan_both = PrintSimulation(
        job(), passes=4, gantry_config=GantryConfig(heads, [left,right]))
    plan_only_left = PrintSimulation(
        job(), passes=4, gantry_config=GantryConfig(heads, [left]))
    assert plan_both.total_seconds < plan_only_left.total_seconds
    assert not any(s.kind=="cure" for s in plan_both.segments)
    assert any(s.kind=="cure" for s in plan_only_left.segments)


@pytest.mark.parametrize("mode", ["bidirectional","unidirectional"])
def test_one_direction_printing_always_uses_same_trailing_lamp(mode):
    cfg = GantryConfig(
        heads=[HeadConfig("head", "white")],
        lamps=[LampConfig("left",-25,0,40,55),
               LampConfig("right",25,0,40,55)])
    plan=PrintSimulation(job(),passes=3,print_mode=mode,gantry_config=cfg)
    prints=[s for s in plan.segments if s.kind=="print"]
    if mode == "unidirectional":
        assert all(s.lamp_index==0 and s.curing for s in prints)
        assert all(s.end_xyz[0]>s.start_xyz[0] for s in prints)
        assert all(not s.curing for s in plan.segments
                   if s.kind not in ("print","cure","uv_followthrough"))
    else:
        assert [s.lamp_index for s in prints[:3]] == [0,1,0]


def test_distant_trailing_lamp_only_extends_curing_to_far_raster_edge():
    cfg=GantryConfig(
        heads=[HeadConfig("white","white")],
        lamps=[LampConfig("left distant",-75,0,40,55),
               LampConfig("right distant",75,0,40,55)])
    plan=PrintSimulation(job(),passes=2,gantry_config=cfg)
    first_print=next(i for i,s in enumerate(plan.segments) if s.kind=="print")
    second_print=next(i for i,s in enumerate(plan.segments[first_print+1:],
                           start=first_print+1) if s.kind=="print")
    for i in (first_print,second_print):
        printing=plan.segments[i]
        follow=plan.segments[i+1]
        assert follow.kind=="uv_followthrough"
        assert follow.curing and not follow.printing
        assert follow.lamp_index==printing.lamp_index
        assert follow.start_xyz==pytest.approx(printing.end_xyz)
        assert follow.duration_s>0
        assert follow.speed_mm_s==plan.move_speed_mm_s
        # Check near the END of the follow-through: until the lamp reaches
        # the raster, the early part of a distant trailing move may be dark.
        t=plan._segment_end_s[i+1]-min(1e-7,follow.duration_s*.001)
        final=plan.get_frame(t/plan.total_seconds)
        assert len(uv_footprints(final,cfg,0,0,20,30))==1
    assert all(a.end_xyz==pytest.approx(b.start_xyz)
               for a,b in zip(plan.segments,plan.segments[1:]))


def test_lamp_with_wrong_y_coverage_uses_postpass_cure_not_premature_light():
    cfg=GantryConfig(
        heads=[HeadConfig("white","white")],
        lamps=[LampConfig("left",-25,40,40,10),
               LampConfig("right",25,40,40,10)])
    plan=PrintSimulation(job(),passes=2,gantry_config=cfg)
    prints=[s for s in plan.segments if s.kind=="print"]
    assert all(not s.curing for s in prints)
    assert all(s.lamp_index==-1 for s in prints)
    cures=[s for s in plan.segments if s.kind=="cure"]
    assert cures
    assert {s.lamp_index for s in cures}=={0,1}
    assert all(s.kind!="uv_followthrough" for s in plan.segments)



def test_default_three_head_two_lamp_gantry_uses_only_trailing_side():
    from uv3d.gantry import default_gantry
    config = default_gantry()
    plan = PrintSimulation(job(), passes=2, gantry_config=config)
    scans = [(i,s) for i,s in enumerate(plan.segments)
             if s.kind=="print" and s.layer_index==0]
    assert len(scans)==2
    assert [s.lamp_index for _,s in scans]==[0,1]
    assert all(s.curing for _,s in scans)
    for i,s in scans:
        frame=midpoint(plan,i)
        assert frame.active_lamp_index==s.lamp_index
        assert frame.lamp_active
        # With these illustrative default offsets, illumination may still
        # be outside the print at mid-scan. The selected lamp must never be
        # the leading-side lamp (even before overlap begins).
        assert frame.printing
    assert all(a.end_xyz==pytest.approx(b.start_xyz)
               for a,b in zip(plan.segments,plan.segments[1:]))


def test_lamps_physically_larger_than_printhead_use_short_followthrough_not_full_cure_sweep():
    config = GantryConfig(
        heads=[HeadConfig("white","white")],
        lamps=[LampConfig("large left",-80,0,120,90),
               LampConfig("large right",80,0,120,90)])
    plan = PrintSimulation(job(), passes=2, gantry_config=config)
    scans=[(i,s) for i,s in enumerate(plan.segments)
           if s.kind=="print" and s.layer_index==0]
    assert [s.lamp_index for i,s in scans]==[0,1]
    assert all(s.curing for _,s in scans)
    assert not any(s.kind=="cure" for s in plan.segments)
    follow=[s for s in plan.segments if s.kind=="uv_followthrough"]
    assert len(follow)==plan.count*2
    # The light is 120 mm wide, larger than the 69.1-mm head body.
    # Each print endpoint is only 10 mm short of fully curing a 20-mm
    # raster, NOT a full 20-mm UV scan or a carriage jump to the side.
    assert all(abs(s.end_xyz[0]-s.start_xyz[0])==pytest.approx(10)
               for s in follow)
