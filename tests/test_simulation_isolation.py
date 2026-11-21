"""Tests for native-crash-resistant 3D simulation OS process boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest
from PySide6.QtWidgets import QApplication

from uv3d.simulation_process_panel import SimulationProcessPanel


def build_job(root: Path):
    directory = root / "workspace" / "jobs" / "testjob"
    directory.mkdir(parents=True)
    manifest = {
        "settings": {"layer_mm": .5, "dpi": 96, "margin_mm": 1,
                     "body_material": "white"},
        "model_bounds_mm": [[0,0,0],[2,2,1]],
        "scaled_dimensions_mm": [2,2,1],
        "raster_size_px": [18,18],
        "layers": [
            {"target_stack_height_mm": .5},
            {"target_stack_height_mm": 1.0},
        ],
    }
    path = directory / "job.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def wait_for(app, callback, timeout=40):
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        app.processEvents()
        if callback():
            return
        time.sleep(.025)
    raise AssertionError("Isolated simulator did not exit within test timeout.")


@pytest.mark.parametrize("abort", [False, True])
def test_simulation_child_can_exit_or_crash_without_main_gui_crash(tmp_path, monkeypatch, abort):
    app = QApplication.instance() or QApplication([])
    # The child Qt process opens an offscreen window without OpenGL, then
    # either closes normally or simulates an abrupt native termination.
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("UV3D_SIMULATION_TEST_EXIT_MS", "250")
    monkeypatch.setenv("UV3D_SIMULATION_TEST_ABORT", "1" if abort else "0")
    job = build_job(tmp_path)
    main_tab = SimulationProcessPanel(tmp_path, gpu_enabled=False)
    main_tab.load_job(job)
    assert main_tab.job_path == job
    assert not main_tab.busy
    main_tab.launch()
    assert main_tab.busy
    wait_for(app, lambda: not main_tab.busy)
    assert main_tab.launch_button.isEnabled()  # retry available
    assert main_tab.process is None
    report = tmp_path / "workspace" / "logs" / "uv3d-simulation-child.log"
    assert report.is_file()
    assert str(job) in report.read_text(encoding="utf-8")
    if abort:
        assert "crashed" in main_tab.status.text().lower()
        # Native Windows Qt faults may preempt the deliberate os._exit(91)
        # with an access violation (0xC0000005). Regardless of *why* the
        # child dies, the parent must survive and report its actual code.
        assert "Windows code 0x" in main_tab.status.text()
        assert "Crash report:" in main_tab.log_label.text()
        # The parent Qt interpreter is alive, and the child can be relaunched.
        main_tab.launch()
        assert main_tab.busy
        wait_for(app, lambda: not main_tab.busy)
    else:
        assert "closed" in main_tab.status.text().lower(), (
            report.read_text(encoding="utf-8") +
            "\n--- native log ---\n" +
            (tmp_path/"workspace"/"logs"/"uv3d-simulation-native.log").read_text(
                encoding="utf-8"))
    app.processEvents()
    main_tab.close()


def test_simulation_panel_is_not_gpu_widget_and_does_not_spawn_on_export(tmp_path):
    app = QApplication.instance() or QApplication([])
    job = build_job(tmp_path)
    panel = SimulationProcessPanel(tmp_path)
    panel.load_job(job)
    assert panel.process is None
    assert not panel.busy
    assert panel.launch_button.isEnabled()
    # A completed mask export never starts an expensive graphics process.
    panel.pause()
    assert panel.process is None
    panel.close()
