"""Regression for the Ania 60.02-mm STL rejected above 350M layer-pixels."""
from __future__ import annotations

import json
import shutil

import pytest
import trimesh

from uv3d.core import SliceSettings, _slice_model_legacy, slice_model
from uv3d.job_storage import calculate_job_plan, ensure_disk_space


def test_reported_user_job_is_allowed_by_storage_plan():
    # Dimensions and parameters from the 2026-11-21 UV3D screenshot.
    settings = SliceSettings(
        target_height_mm=60.021, layer_mm=.05, dpi=300, margin_mm=3.0)
    width = __import__("math").ceil((44.52 + 2 * settings.margin_mm) * settings.dpi / 25.4)
    height = __import__("math").ceil((44.87 + 2 * settings.margin_mm) * settings.dpi / 25.4)
    layers = __import__("math").ceil(settings.target_height_mm / settings.layer_mm)
    plan = calculate_job_plan(width, height, layers)
    assert layers == 1201
    assert plan.total_layer_pixels > 350_000_000
    assert plan.png_files == 3 * 1201
    assert plan.raw_channels_bytes == plan.total_layer_pixels * 6


def test_preflight_disk_space_does_not_require_total_uncompressed_job_bytes(tmp_path, monkeypatch):
    # A large multi-layer job can compress well and is streamed to disk one
    # layer at a time. Reject only genuinely critically low free space.
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: shutil._ntuple_diskusage(10**12, 10**12 - 10**9, 10**9))
    plan = calculate_job_plan(597, 601, 1201)
    assert plan.total_layer_pixels > 350_000_000
    free = ensure_disk_space(tmp_path / "jobs" / "not_created_yet", plan.pixels_per_layer)
    assert free == 10**9


def test_low_disk_space_stops_before_output_directory_created(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: shutil._ntuple_diskusage(10**9, 10**9 - 128 * 1024**2,
                                                             128 * 1024**2))
    src = tmp_path / "box.stl"
    trimesh.creation.box(extents=[2, 2, 1]).export(src)
    output = tmp_path / "job"
    with pytest.raises(OSError, match="Not enough free disk space"):
        slice_model(src, output, SliceSettings(layer_mm=.5, dpi=72))
    assert not output.exists()


@pytest.mark.parametrize("compatibility", [False, True])
def test_disk_becoming_full_stops_cleanly_and_keeps_partial_layers(tmp_path, monkeypatch, compatibility):
    src = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[3, 3, 3]).export(src)
    import uv3d.core as core
    import uv3d.fast_slicer as fast
    original = ensure_disk_space
    observed = []

    def guarded(out, pixels):
        observed.append(pixels)
        if len(observed) >= 4:
            raise OSError("Not enough free disk space: injected test")
        return original(out, pixels)

    monkeypatch.setattr(core if compatibility else fast,
                        "ensure_disk_space", guarded)
    target = tmp_path / "partially_written"
    fn = _slice_model_legacy if compatibility else slice_model
    with pytest.raises(OSError, match="Not enough free disk space"):
        fn(src, target, SliceSettings(layer_mm=.5, dpi=72))
    assert len(observed) == 4
    assert target.is_dir()
    assert list(target.glob("*.png"))
    assert not (target / "job.json").exists()


def test_plan_callback_and_manifest_written_on_success(tmp_path):
    src = tmp_path / "cube.stl"
    trimesh.creation.box(extents=[4, 4, 1]).export(src)
    plans = []
    result = slice_model(src, tmp_path / "job",
                         SliceSettings(layer_mm=.5, dpi=72),
                         on_plan=plans.append)
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert len(plans) == 1
    assert manifest["storage_plan"] == plans[0].as_dict()
    assert manifest["storage_plan"]["png_files"] == 6
