"""Regression: original Windows failure logged QBasicTimer::stop wrong-thread.

The simulator used QObject.moveToThread + QObject.deleteLater chains.
Check the replacement QThread.run workers have main-thread affinity,
deliver results to the GUI thread, then shut down without Qt warnings.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from uv3d.simulation_widget import SimulationAssetsThread, LayerTextureThread


def make_job(tmp_path):
    job_dir = tmp_path / "jobs" / "test"
    job_dir.mkdir(parents=True)
    model = np.zeros((10, 20), np.uint8)
    model[2:8, 3:15] = 255
    color = np.zeros((10, 20, 4), np.uint8)
    color[2:8, 3:15] = (40, 80, 180, 255)
    for key, bitmap in (("white", model), ("color", color)):
        ok, png = cv2.imencode(".png", bitmap)
        assert ok
        (job_dir / f"00001_{key}.png").write_bytes(png.tobytes())
    manifest = {
        "settings": {"layer_mm": 1.0, "dpi": 72, "margin_mm": 1,
                     "body_material": "white"},
        "model_bounds_mm": [[0,0,0],[3,2,1]],
        "scaled_dimensions_mm": [3,2,1],
        "raster_size_px": [20,10],
        "support": {"enabled": False},
        "layers": [{"index": 1, "white":"00001_white.png",
                    "color":"00001_color.png", "target_stack_height_mm":1.0}],
    }
    job = job_dir / "job.json"
    job.write_text(json.dumps(manifest), encoding="utf-8")
    return job, manifest


class Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.threads = []
        self.errors = []

    @Slot(str, int, object)
    def texture_ready(self, path, index, arrays):
        self.threads.append(QThread.currentThread())
        self.messages.append((path,index,arrays))

    @Slot(str, object, object, object)
    def assets_ready(self, path, mesh, payload, surface):
        self.threads.append(QThread.currentThread())
        self.messages.append((path,mesh,payload,surface))

    @Slot(str)
    @Slot(str, int, str)
    def failed(self, *error):
        self.errors.append(error)


def spin_until(app, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Qt worker thread did not finish.")


def test_texture_and_assets_callbacks_are_queued_to_gui_thread(tmp_path):
    app = QApplication.instance() or QApplication([])
    job, manifest = make_job(tmp_path)
    receiver = Receiver()
    messages = []

    def capture(kind, context, message):
        if "QBasicTimer" in message or "different thread" in message:
            messages.append(message)

    original = qInstallMessageHandler(capture)
    try:
        for _ in range(12):
            worker = LayerTextureThread(job, 0)
            assert worker.thread() == app.thread()
            finished = []
            worker.ready.connect(receiver.texture_ready, Qt.QueuedConnection)
            worker.failed.connect(receiver.failed, Qt.QueuedConnection)
            worker.finished.connect(lambda finished=finished: finished.append(True),
                                    Qt.QueuedConnection)
            worker.start()
            spin_until(app, lambda: bool(finished))
            spin_until(app, lambda: len(receiver.messages) >= _+1)
            assert not worker.isRunning()
            assert not receiver.errors
            assert receiver.threads[-1] == app.thread()
            worker.deleteLater()
            app.processEvents()

        # No mesh loading is needed when the calling application already has
        # GPU geometry; the assets worker still builds support preview info.
        asset = SimulationAssetsThread(job, manifest, tmp_path,
                                       source_mesh="model", gpu_payload=b"mesh")
        assert asset.thread() == app.thread()
        done = []
        asset.ready.connect(receiver.assets_ready, Qt.QueuedConnection)
        asset.failed.connect(receiver.failed, Qt.QueuedConnection)
        asset.finished.connect(lambda: done.append(True), Qt.QueuedConnection)
        asset.start()
        spin_until(app, lambda: bool(done))
        spin_until(app, lambda: len(receiver.messages) == 13)
        assert receiver.messages[-1][1] == "model"
        assert receiver.threads[-1] == app.thread()
        assert not receiver.errors
        asset.deleteLater()
        app.processEvents()
    finally:
        qInstallMessageHandler(original)
    assert not messages, "Qt emitted cross-thread timer-affinity warnings: "+repr(messages)
