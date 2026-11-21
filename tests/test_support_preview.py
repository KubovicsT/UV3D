"""Check sparse 3D wireframe uses model coordinates and correct support pixels."""
import json

import numpy as np
import trimesh

from uv3d.core import SliceSettings, slice_model
from uv3d.support_preview import build_support_wireframe


def test_wireframe_of_conformal_support_is_bounded_and_in_model_coordinates(tmp_path):
    cube = trimesh.creation.box(extents=(5, 6, 2))
    src = tmp_path / "cube.stl"
    cube.export(src)
    result = slice_model(
        src, tmp_path / "job",
        SliceSettings(layer_mm=.2, dpi=100, support_casing_mm=1))
    lines = build_support_wireframe(result, max_levels=5, max_vertices=2000)
    assert lines.ndim == 2 and lines.shape[1] == 3
    assert len(lines) > 0
    assert len(lines) % 2 == 0
    assert len(lines) <= 2000
    assert lines[:, 0].min() < -2.4
    assert lines[:, 0].max() > 2.4
    assert lines[:, 2].min() >= 0
    assert lines[:, 2].max() <= 2.0
    meta = json.loads(result.read_text(encoding="utf-8"))
    assert len(meta["layers"]) == 10


def test_no_support_means_no_3d_preview(tmp_path):
    cube = trimesh.creation.box(extents=(4, 4, 1))
    src = tmp_path / "cube.stl"
    cube.export(src)
    result = slice_model(src, tmp_path / "job",
                         SliceSettings(layer_mm=.5, dpi=72, support_enabled=False))
    lines = build_support_wireframe(result)
    assert isinstance(lines, np.ndarray)
    assert lines.shape == (0, 3)
