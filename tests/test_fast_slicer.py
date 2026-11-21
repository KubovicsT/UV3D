import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from uv3d.core import SliceSettings, slice_model
from uv3d.fast_slicer import _contours_from_segments
from uv3d.gpu_viewport import prepare_gpu_geometry, VERTEX_DTYPE


def test_full_resolution_index_buffer_does_not_decimate():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    sphere.visual.face_colors = [110, 150, 220, 255]
    vertices, indices, vertex_count, index_count = prepare_gpu_geometry(sphere)
    assert vertex_count == len(sphere.vertices)
    assert index_count == len(sphere.faces) * 3
    assert len(indices) == len(sphere.faces) * 3 * 4
    assert len(vertices) == len(sphere.vertices) * VERTEX_DTYPE.itemsize


def test_different_face_colors_preserved_without_downsampling():
    cube = trimesh.creation.box()
    color = np.tile([200, 20, 10, 255], (len(cube.faces), 1)).astype(np.uint8)
    color[:len(cube.faces) // 2] = [20, 20, 200, 255]
    cube.visual.face_colors = color
    payload, indices, vertex_count, index_count = prepare_gpu_geometry(cube)
    assert indices is None
    assert vertex_count == len(cube.faces) * 3
    assert index_count == 0
    arr = np.frombuffer(payload, dtype=VERTEX_DTYPE)
    assert (arr["color"][:3, :3] == color[0, :3]).all()
    assert (arr["color"][-3:, :3] == color[-1, :3]).all()


def test_fast_slice_keeps_hollow_centers_and_layer_count(tmp_path):
    tube = trimesh.creation.annulus(r_min=2, r_max=6, height=2)
    src = tmp_path / "tube.stl"
    tube.export(src)
    job = slice_model(
        src, tmp_path / "out", SliceSettings(
            target_height_mm=3, layer_mm=.5, dpi=100))
    info = json.loads(job.read_text(encoding="utf-8"))
    assert len(info["layers"]) == 6
    assert info["scaled_dimensions_mm"][2] == pytest.approx(3)
    mask = np.asarray(Image.open(job.parent / "00003_white.png"))
    assert mask[mask.shape[0] // 2, mask.shape[1] // 2] == 0
    assert mask.max() == 255
    alpha = np.asarray(Image.open(job.parent / "00003_color.png"))[:, :, 3]
    assert alpha.max() > 0
    assert not np.any(alpha[mask == 0])


def test_fast_slice_preserves_rgb_channels(tmp_path):
    cube = trimesh.creation.box(extents=[3, 3, 1])
    cube.visual.face_colors = [225, 65, 35, 255]
    src = tmp_path / "rgb.ply"
    cube.export(src)
    job = slice_model(src, tmp_path / "out", SliceSettings(
        layer_mm=.5, dpi=140, shell_mm=.8))
    color = np.asarray(Image.open(job.parent / "00001_color.png"))
    painted = color[color[:, :, 3] > 0]
    assert len(painted) > 0
    assert float(np.mean(painted[:, 0])) > float(np.mean(painted[:, 2]))


def test_segment_contour_stitch_rejects_open_edges():
    segs = np.array([
        [[0, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 1, 0]],
    ], dtype=float)
    assert _contours_from_segments(segs) is None
