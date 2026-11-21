"""Gantry hardware-layout settings remain virtual: validate geometry and ETA."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from uv3d.gantry import (
    GantryConfig, HeadConfig, LampConfig, default_gantry,
    gantry_file, load_gantry, save_gantry,
)
from uv3d.gantry_setup_widget import GantrySetupWidget
from uv3d.simulation_engine import PrintSimulation
from uv3d.simulation_viewport import uv_footprints


@pytest.fixture
def example_job():
    return {
        "settings": {"dpi": 254, "layer_mm": .2, "margin_mm": 0,
                     "body_material": "white"},
        "model_bounds_mm": [[0, 0, 0], [20, 30, .6]],
        "raster_size_px": [200, 300],
        "layers": [
            {"target_stack_height_mm": .2},
            {"target_stack_height_mm": .4},
            {"target_stack_height_mm": .6},
        ],
    }


def test_head_assignments_and_lamp_geometry_are_serializable_and_folder_local(tmp_path):
    gantry = default_gantry()
    assert len(gantry.heads) == 3
    assert {h.material for h in gantry.heads} == {"color", "white", "varnish"}
    assert all(h.swath_mm == 33.8 for h in gantry.heads)
    assert len(gantry.lamps) == 2
    path = save_gantry(tmp_path, gantry)
    assert path == gantry_file(tmp_path)
    assert "workspace" in str(path)
    assert "machine" in str(path)
    assert load_gantry(tmp_path) == gantry
    gantry.heads.append(HeadConfig("Soluble support nozzle", "support", 245, 6))
    save_gantry(tmp_path, gantry)
    assert load_gantry(tmp_path).heads[-1].material == "support"


def test_lamp_sweeps_add_duration_cover_all_xy_and_obey_travel_speed(example_job):
    config = GantryConfig(
        heads=[HeadConfig("Main head", "white")],
        lamps=[LampConfig("UV test lamp", offset_x_mm=65,
                          offset_y_mm=9, width_mm=10, depth_mm=11)],
    )
    without = PrintSimulation(example_job, print_speed_mm_s=250,
                              move_speed_mm_s=500, passes=2,
                              gantry_config=GantryConfig(
                                  heads=config.heads, lamps=[]))
    with_lamp = PrintSimulation(example_job, print_speed_mm_s=250,
                                move_speed_mm_s=500, passes=2,
                                gantry_config=config)
    cure = [s for s in with_lamp.segments if s.kind == "cure"]
    assert len(cure) == 3 * math.ceil(30 / 11) * config.cure_passes
    assert with_lamp.total_seconds > without.total_seconds
    assert all(s.curing and not s.printing for s in cure)
    assert all(s.speed_mm_s == 500 for s in cure)
    for before, after in zip(with_lamp.segments, with_lamp.segments[1:]):
        assert before.end_xyz == pytest.approx(after.start_xyz)
    assert min(s.start_xyz[0] for s in cure) <= -65-5
    assert max(s.end_xyz[0] for s in cure) >= 20-65+5


def test_lamp_position_size_count_and_cure_passes_affect_route(example_job):
    config = GantryConfig(
        heads=[HeadConfig("Head", "color")],
        lamps=[LampConfig("UV", 0, 0, 20, 30)])
    plan = PrintSimulation(example_job, gantry_config=config)
    first = plan.total_seconds
    moved = GantryConfig(heads=config.heads,
                         lamps=[LampConfig("UV", -125, 6, 32, 10)])
    plan.set_gantry_config(moved)
    assert plan.total_seconds > first
    first_cure_duration = sum(s.duration_s for s in plan.segments
                              if s.kind == "cure")
    moved.cure_passes = 2
    plan.set_gantry_config(moved)
    assert plan.total_seconds > first
    assert sum(s.duration_s for s in plan.segments
               if s.kind == "cure") == pytest.approx(first_cure_duration * 2)


def test_lamps_move_with_gantry_and_illuminate_layer_only_during_active_segments(example_job):
    from uv3d.gantry import GantryConfig
    config = GantryConfig(
        heads=[HeadConfig("H", "white")],
        lamps=[LampConfig("L", 0, 0, 10, 10)])
    plan = PrintSimulation(example_job, gantry_config=config)
    cure_i = next(i for i,s in enumerate(plan.segments) if s.kind == "cure")
    seg = plan.segments[cure_i]
    previous = plan._segment_end_s[cure_i-1] if cure_i else 0.0
    frame = plan.get_frame((previous + seg.duration_s/2) / plan.total_seconds)
    assert frame.curing
    assert frame.lamp_active
    assert frame.active_lamp_index == 0
    assert not frame.printing
    regions = uv_footprints(frame, config, 0, 0, 20, 30)
    assert len(regions) == 1
    _, x0, y0, x1, y1 = regions[0]
    assert 0 <= x0 < x1 <= 20
    assert 0 <= y0 < y1 <= 30
    travel_i = next(i for i,s in enumerate(plan.segments) if s.kind == "uv_travel")
    seg = plan.segments[travel_i]
    prev = plan._segment_end_s[travel_i-1] if travel_i else 0.0
    inactive = plan.get_frame((prev + seg.duration_s/2)/plan.total_seconds)
    assert not inactive.lamp_active
    assert not uv_footprints(inactive, config, 0, 0, 20, 30)


def test_gantry_ui_edit_material_lamp_size_and_persist_without_reloading(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = GantrySetupWidget(tmp_path)
    observed=[]
    widget.configurationChanged.connect(observed.append)
    assert widget.head_table.rowCount() == 3
    assert widget.lamp_table.rowCount() == 2
    widget.head_table.cellWidget(0, 1).setCurrentIndex(3)
    assert load_gantry(tmp_path).heads[0].material == "support"
    widget.lamp_table.cellWidget(0, 1).setValue(-150)
    widget.lamp_table.cellWidget(0, 3).setValue(25)
    assert load_gantry(tmp_path).lamps[0].offset_x_mm == -150
    assert load_gantry(tmp_path).lamps[0].width_mm == 25
    widget.cure_passes.setValue(3)
    assert load_gantry(tmp_path).cure_passes == 3
    widget.add_head()
    assert len(load_gantry(tmp_path).heads) == 4
    widget.add_lamp()
    assert len(load_gantry(tmp_path).lamps) == 3
    assert observed
    widget.close()
