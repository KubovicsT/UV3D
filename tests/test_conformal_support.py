"""Physical column occupancy invariants for experimental inkjet support masks."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from uv3d.core import SliceSettings, _slice_model_legacy, slice_model
from uv3d.support import (
    effective_raster_margin, generate_support_masks, make_casing_kernel,
    radius_pixels, support_layer,
)


def _masks(job):
    data = json.loads(job.read_text(encoding="utf-8"))
    folder = job.parent
    part_key = data["settings"]["body_material"]
    model = [cv2.imread(str(folder / layer[part_key]), cv2.IMREAD_GRAYSCALE)
             for layer in data["layers"]]
    support = [cv2.imread(str(folder / layer["support"]), cv2.IMREAD_GRAYSCALE)
               for layer in data["layers"]]
    assert all(x is not None for x in model + support)
    return data, model, support


def test_one_column_to_bed_for_all_overhangs_and_no_support_inside_part():
    # A model that grows to the right above its original base. Every added
    # top pixel must have model/support material directly below it.
    lower = np.zeros((22, 30), dtype=np.uint8)
    lower[6:16, 4:10] = 255
    middle = np.zeros_like(lower)
    middle[6:16, 4:10] = 255
    upper = np.zeros_like(lower)
    upper[4:18, 4:22] = 255
    solids = [lower, middle, upper]
    envelope = np.zeros_like(lower)
    support = [None] * 3
    for index in range(2, -1, -1):
        support[index], envelope = support_layer(solids[index], envelope, None)
    for index in range(3):
        assert not np.any((support[index] > 0) & (solids[index] > 0))
        if index:
            earlier = (solids[index - 1] > 0) | (support[index - 1] > 0)
            current = (solids[index] > 0) | (support[index] > 0)
            assert np.all(~current | earlier)
    assert support[0][10, 20] == 255  # floating right-hand cap supported to bed
    assert support[0][10, 7] == 0      # left-hand stem is solid part


def test_casing_dilation_adds_adjustable_support_only_outside_solid():
    point = np.zeros((30, 30), dtype=np.uint8)
    point[15, 15] = 255
    empty = np.zeros_like(point)
    thin, _ = support_layer(point, empty, make_casing_kernel(0))
    thick, _ = support_layer(point, empty, make_casing_kernel(3))
    assert not np.any(thin)
    assert cv2.countNonZero(thick) > cv2.countNonZero(thin)
    assert thick[15, 15] == 0
    assert thick[15, 17] == 255
    assert radius_pixels(0.6, 300) == 7
    assert effective_raster_margin(SliceSettings(
        support_casing_mm=4, margin_mm=1, dpi=100)) > 4


@pytest.mark.parametrize("fn", [slice_model, _slice_model_legacy])
def test_export_solid_body_conformal_support_and_savings(tmp_path, fn):
    # Offset stacked boxes: upper cap projects beyond lower stem.
    stem = trimesh.creation.box(extents=[3, 3, 2])
    cap = trimesh.creation.box(extents=[10, 7, 1])
    stem.apply_translation([0, 0, 1])
    cap.apply_translation([3, 0, 2.5])
    combined = trimesh.util.concatenate([stem, cap])
    source = tmp_path / "overhang.stl"
    combined.export(source)
    stages = []
    events = []
    job = fn(
        source, tmp_path / "export",
        SliceSettings(target_height_mm=3, layer_mm=.5, dpi=120,
                      support_casing_mm=.6, support_enabled=True),
        on_stage=stages.append,
        on_progress=lambda current, total: events.append((current, total)))
    meta, model, support = _masks(job)
    assert len(meta["layers"]) == 6
    assert events[-1] == (12, 12)
    assert len(stages) == 2
    assert meta["solid_volume"] is True
    assert len(list(job.parent.glob("*_support.png"))) == 6
    assert meta["storage_plan"]["png_files"] == 24
    assert meta["support"]["material"].startswith("UNASSIGNED")
    stats = meta["support"]["statistics"]
    assert stats["support_volume_mm3"] > 0
    assert 0 < stats["saving_vs_cube_percent"] < 100
    # Stem is a filled solid part, not a perimeter-only wall.
    # The base stem occupies x=0,y=0; the full raster is centered on the
    # *combined* overhanging model, so raster center is not stem center.
    x_mm, y_mm = 0.0, 0.0
    extent = meta["model_bounds_mm"]
    ppm = meta["settings"]["dpi"] / 25.4
    margin = meta["effective_raster_margin_mm"]
    col = round((x_mm - extent[0][0] + margin) * ppm)
    row = round((extent[1][1] - y_mm + margin) * ppm)
    assert model[0][row, col] == 255
    # Support never overlaps the solid model.
    for part, sacrificial in zip(model, support):
        assert not np.any((part > 0) & (sacrificial > 0))
    # Each higher material pixel has backing in the previous layer.
    for k in range(1, len(model)):
        occupied_here = (model[k] > 0) | (support[k] > 0)
        occupied_below = (model[k - 1] > 0) | (support[k - 1] > 0)
        assert np.all(~occupied_here | occupied_below)


def test_export_without_support_keeps_only_original_three_channels(tmp_path):
    src = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[3, 3, 1]).export(src)
    job = slice_model(
        src, tmp_path / "no_support",
        SliceSettings(layer_mm=.5, dpi=100, support_enabled=False))
    data = json.loads(job.read_text(encoding="utf-8"))
    assert not data["support"]["enabled"]
    assert data["storage_plan"]["png_files"] == 6
    assert not list(job.parent.glob("*_support.png"))
    assert not any("support" in layer for layer in data["layers"])


def test_casing_is_not_clipped_when_bigger_than_user_margin(tmp_path):
    src = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[3, 3, 1]).export(src)
    job = slice_model(
        src, tmp_path / "room",
        SliceSettings(layer_mm=1, dpi=100,
                      margin_mm=.2, support_casing_mm=2,
                      support_enabled=True))
    meta, part, support = _masks(job)
    assert meta["effective_raster_margin_mm"] > 2
    assert np.any(support[0])
    # Outer casing doesn't accidentally get clipped against the raster edges.
    border = support[0]
    assert not np.any(border[0, :])
    assert not np.any(border[-1, :])
    assert not np.any(border[:, 0])
    assert not np.any(border[:, -1])


def test_bad_support_settings_rejected():
    with pytest.raises(ValueError, match="Support casing"):
        SliceSettings(support_casing_mm=6).validate()
    with pytest.raises(ValueError, match="Support enabled"):
        SliceSettings(support_enabled="yes").validate()
