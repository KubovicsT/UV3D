"""Uniform geometric UV exposure requires the lamp's BACK edge to clear.

For constant scan speed, each point on the model must spend precisely one
full illuminated-footprint width under the active trailing lamp, including
both the first and last pixels. These are geometric/timing assertions, not
a simulation of dose, optical intensity, acceleration or photochemistry.
"""
import pytest

from uv3d.gantry import GantryConfig, HeadConfig, LampConfig
from uv3d.simulation_engine import PrintSimulation, _full_uv_sweep_x


def manifest():
    return {
        "settings": {"dpi": 254, "layer_mm": 0.2,
                     "margin_mm": 0, "body_material": "white"},
        "model_bounds_mm": [[0,0,0],[20,30,.2]],
        "raster_size_px": [200,300],
        "layers": [{"target_stack_height_mm":.2}],
    }


@pytest.mark.parametrize("lamp_width,offset", [(40,25), (120,80)])
@pytest.mark.parametrize("mode", ["bidirectional", "unidirectional"])
def test_every_pixel_gets_full_footprint_and_constant_speed(lamp_width, offset, mode):
    config = GantryConfig(
        heads=[HeadConfig("model", "white")],
        lamps=[
            LampConfig("UV left", -offset, 0, lamp_width, 55),
            LampConfig("UV right", offset, 0, lamp_width, 55),
        ])
    plan = PrintSimulation(manifest(), passes=2,
                           print_mode=mode, gantry_config=config,
                           print_speed_mm_s=240, move_speed_mm_s=720)
    prints = [(i,s) for i,s in enumerate(plan.segments) if s.kind=="print"]
    assert len(prints) == 2
    for i,printing in prints:
        assert printing.curing and printing.lamp_index >= 0
        lamp = config.lamps[printing.lamp_index]
        follow = plan.segments[i+1]
        assert follow.kind == "uv_followthrough"
        assert follow.curing and not follow.printing
        assert follow.lamp_index == printing.lamp_index
        assert follow.speed_mm_s == printing.speed_mm_s == 240
        assert follow.start_xyz == pytest.approx(printing.end_xyz)
        head_start = printing.start_xyz[0]
        lamp_start = head_start + lamp.offset_x_mm
        lamp_finish = follow.end_xyz[0] + lamp.offset_x_mm
        left, right = sorted((lamp_start, lamp_finish))

        # The complete illuminated rectangle, not just its leading edge,
        # traverses EACH point from the first through the final raster pixel.
        for x in (0.0, 0.5, 5.0, 10.0, 15.0, 19.5, 20.0):
            half = lamp.width_mm/2
            # Travel distance of illuminated lamp center while its
            # footprint overlaps a fixed model pixel x.
            irradiated_distance = max(
                0.0, min(right,x+half)-max(left,x-half))
            irradiated_seconds = irradiated_distance/printing.speed_mm_s
            assert irradiated_seconds == pytest.approx(
                lamp.width_mm/240, abs=1e-8)
        if printing.end_xyz[0]>printing.start_xyz[0]:
            assert lamp_start + lamp.width_mm/2 <= 0
            assert lamp_finish - lamp.width_mm/2 >= 20
        else:
            assert lamp_start - lamp.width_mm/2 >= 20
            assert lamp_finish + lamp.width_mm/2 <= 0


@pytest.mark.parametrize("lamp_width", [10, 40, 100])
@pytest.mark.parametrize("offset", [-80, 0, 80])
def test_dedicated_lamp_sweep_moves_entire_footprint_past_raster(lamp_width,offset):
    """Position planning for non-trailing or extra post-print UV sweeps."""
    lamp = LampConfig("UV", offset, 0, lamp_width, 50)
    start,end = _full_uv_sweep_x(0,20,lamp)
    assert start + offset + lamp_width/2 == pytest.approx(0)
    assert end + offset - lamp_width/2 == pytest.approx(20)
    assert end-start == pytest.approx(20+lamp_width)
