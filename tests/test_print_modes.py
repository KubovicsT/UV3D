"""Simulation scan direction must agree with physical printhead X movement."""
import math
import pytest
from PySide6.QtWidgets import QApplication

from uv3d.simulation_engine import PrintSimulation
from uv3d.simulation_widget import SimulationWidget
from uv3d.simulation_viewport import SimulationViewport, SIM_LAYER_FRAGMENT


@pytest.fixture
def sample_job():
    return {
        "settings": {"layer_mm": 0.2, "dpi": 254,
                     "margin_mm": 0.0, "body_material": "white"},
        "model_bounds_mm": [[0,0,0],[20,20,.4]],
        "raster_size_px": [200,200],
        "layers": [{"target_stack_height_mm":.2},
                   {"target_stack_height_mm":.4}]
    }


def middle(plan, segment_index):
    segment=plan.segments[segment_index]
    first=plan._segment_end_s[segment_index-1] if segment_index else 0
    return plan.get_frame((first+segment.duration_s*.5)/plan.total_seconds)


def test_bidirectional_scans_are_truly_reversed_and_paint_follows_motion(sample_job):
    plan=PrintSimulation(sample_job, print_mode="bidirectional",
                         passes=3, scan_pitch_mm=8, overscan_mm=3)
    scan_indices=[i for i,s in enumerate(plan.segments)
                  if s.kind=="print" and s.layer_index==0]
    frames=[middle(plan,i) for i in scan_indices]
    # Three passes: forward, reverse, forward, then next row is reverse.
    assert [f.scan_direction for f in frames[:4]]==[1,-1,1,-1]
    assert all(f.printing for f in frames)
    for i,frame in zip(scan_indices,frames):
        seg=plan.segments[i]
        assert frame.head_position_mm[0] == pytest.approx(
            (seg.start_xyz[0]+seg.end_xyz[0])*.5)
        assert frame.model_scan_fraction == pytest.approx(.5)
        assert frame.support_scan_fraction == pytest.approx(.5)
        assert 0 <= frame.passes_completed <= 3
        assert seg.speed_mm_s == plan.print_speed_mm_s


def test_unidirectional_scans_print_forward_and_return_without_ink(sample_job):
    plan=PrintSimulation(sample_job, print_mode="unidirectional",
                         passes=3, scan_pitch_mm=8)
    printing=[s for s in plan.segments if s.kind=="print"]
    assert printing and all(s.end_xyz[0]>s.start_xyz[0] for s in printing)
    reverse=[(i,s) for i,s in enumerate(plan.segments)
             if s.end_xyz[0]<s.start_xyz[0]]
    assert reverse
    assert all(not s.printing for _,s in reverse)
    assert all(s.speed_mm_s==plan.move_speed_mm_s for _,s in reverse)
    for i,s in reverse:
        frame=middle(plan,i)
        assert not frame.printing
    bi=PrintSimulation(sample_job, print_mode="bidirectional",
                       passes=3, scan_pitch_mm=8)
    assert plan.total_seconds > bi.total_seconds
    with pytest.raises(ValueError, match="direction"):
        plan.set_print_mode("bothways")


def test_colors_and_opacity_can_change_without_reloading_model(qtbot=None):
    app=QApplication.instance() or QApplication([])
    viewport=SimulationViewport()
    viewport.set_head_opacity(.24)
    viewport.set_current_layer_colors((251,112,27),(10,200,251))
    assert viewport._head_alpha==pytest.approx(.24)
    assert viewport._current_model_rgb[0]==pytest.approx(251/255)
    assert viewport._current_support_rgb[2]==pytest.approx(251/255)
    assert "uCurrentModelTint" in SIM_LAYER_FRAGMENT
    assert "uCurrentSupportTint" in SIM_LAYER_FRAGMENT
    assert "uScanDirection" in SIM_LAYER_FRAGMENT
    viewport.close()
