import json
import time

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


def process_until(app, predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert predicate(), "Background task did not complete within test timeout"


def test_gui_height_and_slider_before_export(tmp_path):
    from PySide6.QtWidgets import QApplication
    from uv3d.app import MainWindow

    app = QApplication.instance() or QApplication([])
    mesh = trimesh.creation.box(extents=(5, 4, 2))
    src = tmp_path / "mesh.stl"
    mesh.export(src)
    gui = MainWindow()
    gui.load_model_file(src)
    process_until(app, lambda: gui.import_thread is None and gui.source_mesh is not None)
    assert gui.viewport.mesh is not None
    assert gui.slice_slider.isEnabled()
    gui.model_height.setValue(4)
    np.testing.assert_allclose(gui.preview_mesh.extents, [10, 8, 4])
    assert gui.layer_select.maximum() == 80
    gui.slice_slider.setValue(20)
    assert gui.layer_select.value() == 20
    assert gui.viewport.slice_z == pytest.approx((20 - 0.5) * 0.05)
    assert gui.job_data is None
    gui.viewport.resize(660, 440)
    gui.viewport.show()
    gui.viewport._rebuild_cap()
    process_until(app, lambda: bool(gui.viewport._cap_polys), seconds=15)
    image = gui.viewport.grab().toImage()
    assert image.width() > 0 and image.height() > 0
    gui.close()


def test_large_file_open_is_asynchronous_and_resize_reuses_preview(tmp_path):
    from PySide6.QtWidgets import QApplication
    from uv3d.app import MainWindow

    app = QApplication.instance() or QApplication([])
    dense = trimesh.creation.icosphere(subdivisions=5)
    src = tmp_path / "dense.stl"
    dense.export(src)
    gui = MainWindow()
    start = time.monotonic()
    gui.load_model_file(src)
    assert time.monotonic() - start < 1.5, "File loading blocked the user interface"
    assert gui.import_thread is not None
    process_until(app, lambda: gui.import_thread is None and gui.source_mesh is not None)
    first_preview = gui.source_preview
    assert gui.viewport.original_mesh is gui.source_mesh
    t0 = time.monotonic()
    gui.model_height.setValue(12)
    assert time.monotonic() - t0 < 1.0, "Height change rebuilt the full mesh"
    assert gui.source_preview is first_preview
    assert gui.viewport.triangles.shape[0] <= gui.viewport.PREVIEW_TRIANGLES
    gui.close()


def test_detailed_model_preview_keeps_surface_and_colors():
    from uv3d.viewport import build_preview_geometry

    sphere = trimesh.creation.icosphere(subdivisions=4)
    sphere.visual.face_colors = [80, 145, 220, 255]
    triangles, normals, colors = build_preview_geometry(sphere, max_faces=750)
    assert 100 < len(triangles) <= 750
    assert len(normals) == len(triangles) == len(colors)
    assert triangles.shape[1:] == (3, 3)
    assert np.allclose(colors[:, 0], 80)
    assert np.max(triangles[:, 2]) > 0.85
    assert np.min(triangles[:, 2]) < -0.85
