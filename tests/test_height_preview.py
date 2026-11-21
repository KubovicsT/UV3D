import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from uv3d.core import SliceSettings, prepare_mesh, slice_model
from uv3d.viewport import clip_triangle_below


def test_height_uniformly_scales_xyz_without_mutating_input(tmp_path):
    original = trimesh.creation.box(extents=(10, 8, 2))
    source_dims = original.extents.copy()
    scaled = prepare_mesh(original, 5)
    np.testing.assert_allclose(original.extents, source_dims)
    np.testing.assert_allclose(scaled.extents, [25, 20, 5])
    assert scaled.bounds[0, 2] == pytest.approx(0.0)

    path = tmp_path / "original.stl"
    original.export(path)
    job = slice_model(path, tmp_path / "layers", SliceSettings(
        target_height_mm=5, layer_mm=1, dpi=72))
    manifest = json.loads(job.read_text())
    assert len(manifest["layers"]) == 5
    np.testing.assert_allclose(manifest["original_dimensions_mm"], source_dims)
    np.testing.assert_allclose(manifest["scaled_dimensions_mm"], [25, 20, 5])
    assert manifest["uniform_scale_factor"] == pytest.approx(2.5)
    img = Image.open(tmp_path / "layers" / "00003_white.png")
    # Original 10 mm width scaled to 25 mm -> 31 mm with the margins.
    assert img.width == int(np.ceil(31 * 72 / 25.4))


def test_height_invalid_or_outside_printer_limit():
    mesh = trimesh.creation.box(extents=(4, 4, 4))
    with pytest.raises(ValueError, match="Target height"):
        prepare_mesh(mesh, 0.001)
    with pytest.raises(ValueError, match="Target height"):
        SliceSettings(target_height_mm=181).validate()


def test_triangle_clipping_exact():
    tri = np.array([[0., 0., 0.], [2., 0., 2.], [0., 2., 0.]])
    clipped = clip_triangle_below(tri, 1.0)
    assert len(clipped) == 4
    assert clipped[:, 2].max() <= 1.000001
    assert np.count_nonzero(np.isclose(clipped[:, 2], 1.0)) == 2


def test_gui_height_and_slider_before_export(tmp_path):
    from PySide6.QtWidgets import QApplication
    from uv3d.app import MainWindow

    app = QApplication.instance() or QApplication([])
    mesh = trimesh.creation.box(extents=(5, 4, 2))
    src = tmp_path / "mesh.stl"
    mesh.export(src)
    gui = MainWindow()
    gui.load_model_file(src)
    assert gui.viewport.mesh is not None
    assert gui.slice_slider.isEnabled()
    gui.model_height.setValue(4)
    np.testing.assert_allclose(gui.preview_mesh.extents, [10, 8, 4])
    assert gui.layer_select.maximum() == 80
    gui.slice_slider.setValue(20)
    assert gui.layer_select.value() == 20
    assert gui.viewport.slice_z == pytest.approx((20 - 0.5) * 0.05)
    assert gui.job_data is None
    gui.close()
