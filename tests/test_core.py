import json

import numpy as np
import pytest
from PIL import Image
import trimesh

from uv3d.core import SliceSettings, slice_model


def test_box_slicing(tmp_path):
    cube = trimesh.creation.box(extents=[10, 8, 1])
    source = tmp_path / "box.stl"
    cube.export(source)
    result = slice_model(source, tmp_path / "export", SliceSettings(layer_mm=0.25, dpi=100))
    data = json.loads(result.read_text())
    assert len(data["layers"]) == 4
    assert data["format"] == "UV3D_OFFLINE_JOB_V1"
    mask = Image.open(tmp_path / "export/00001_white.png")
    assert np.asarray(mask).max() == 255
    varnish = Image.open(tmp_path / "export/00001_varnish.png")
    assert np.asarray(varnish).max() == 0
    color = Image.open(tmp_path / "export/00001_color.png")
    assert np.asarray(color)[:, :, 3].max() == 255


def test_hollow_tube_cross_section(tmp_path):
    tube = trimesh.creation.annulus(r_min=2, r_max=5, height=2)
    src = tmp_path / "annulus.stl"
    tube.export(src)
    result = slice_model(src, tmp_path / "out", SliceSettings(layer_mm=1, dpi=100))
    mask = Image.open(result.parent / "00001_white.png")
    assert mask.getpixel((mask.width // 2, mask.height // 2)) == 0
    assert np.asarray(mask).max() == 255


def test_oversize_rejected(tmp_path):
    cube = trimesh.creation.box(extents=[610, 10, 2])
    source = tmp_path / "huge.stl"
    cube.export(source)
    with pytest.raises(ValueError, match="Raster|bed"):
        slice_model(source, tmp_path / "out", SliceSettings(dpi=72))


def test_export_folder_cannot_be_reused(tmp_path):
    src = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[1, 1, 1]).export(src)
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "x.txt").write_text("existing")
    with pytest.raises(ValueError, match="empty"):
        slice_model(src, dest, SliceSettings(layer_mm=1, dpi=72))
