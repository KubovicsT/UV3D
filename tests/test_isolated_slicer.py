"""Headless tests for separate-process slicer and user-matched camera controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from uv3d.core import SliceSettings
from uv3d.gui_slicer import SlicerController
from uv3d.navigation import PITCH_LIMIT, ORBIT_PER_PIXEL, PRESETS, orbit, zoom, preset


def until(app, predicate, deadline=30):
    stop = time.monotonic() + deadline
    while time.monotonic() < stop and not predicate():
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert predicate(), "Worker process failed to finish"


def test_worker_success_and_progress_log(tmp_path):
    app = QApplication.instance() or QApplication([])
    model = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(3, 4, 1)).export(model)
    worker = SlicerController(tmp_path)
    result = []
    failure = []
    progress = []
    worker.succeeded.connect(result.append)
    worker.failed.connect(lambda reason, log: failure.append((reason, log)))
    worker.progress.connect(lambda current, total, seconds: progress.append((current, total)))
    assert worker.start(model, SliceSettings(layer_mm=.5, dpi=72))
    assert worker.running
    # QProcess must not block the UI.
    until(app, lambda: not worker.running)
    assert not failure
    assert result and Path(result[0]).exists()
    manifest = json.loads(Path(result[0]).read_text(encoding="utf-8"))
    assert len(manifest["layers"]) == 2
    assert progress and progress[-1] == (2, 2)
    assert worker.log_path and "SUCCESS" in worker.log_path.read_text(encoding="utf-8")


def test_worker_failure_is_reported_without_terminating_gui(tmp_path):
    app = QApplication.instance() or QApplication([])
    missing_model = tmp_path / "missing_file.stl"
    worker = SlicerController(tmp_path)
    failure = []
    worker.failed.connect(lambda reason, log: failure.append((reason, log)))
    assert worker.start(missing_model, SliceSettings(layer_mm=.5, dpi=72))
    until(app, lambda: not worker.running)
    assert failure
    assert "missing_file.stl" in failure[0][0] or "No such file" in failure[0][0]
    assert Path(failure[0][1]).is_file()
    assert "FAILED:" in Path(failure[0][1]).read_text(encoding="utf-8")
    assert not worker.running
    # QApplication still responds to events and can start another job.
    app.processEvents()


def test_supp_camera_controls_and_presets():
    yaw, pitch = preset("Isometric")
    changed_yaw, changed_pitch = orbit(yaw, pitch, 30, -15)
    assert changed_yaw == pytest.approx(yaw - 30 * ORBIT_PER_PIXEL)
    assert changed_pitch > pitch
    assert orbit(0, 0, 0, -1e6)[1] == PITCH_LIMIT
    assert orbit(0, 0, 0, 1e6)[1] == -PITCH_LIMIT
    assert zoom(1, 1) > 1
    assert zoom(1, -1) < 1
    assert set(PRESETS) == {"Isometric", "Top", "Front", "Right"}


def test_compatibility_worker_can_retry_after_accelerated_failure(tmp_path):
    app = QApplication.instance() or QApplication([])
    src = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(4, 4, 1)).export(src)
    worker = SlicerController(tmp_path)
    succeeded = []
    failures = []
    worker.succeeded.connect(succeeded.append)
    worker.failed.connect(lambda reason, log: failures.append((reason, log)))
    assert worker.start(src, SliceSettings(layer_mm=1, dpi=72),
                        compatibility_mode=True)
    until(app, lambda: not worker.running)
    assert not failures
    assert succeeded and Path(succeeded[0]).is_file()
    assert "Mode: compatibility" in worker.log_path.read_text(encoding="utf-8")
