"""UV3D 3D simulation tab: playback of already-generated, offline layer masks.

A virtual scanhead follows a conceptual serpentine scan; this is a visualization
rather than a calibrated Gongzheng print strategy. No printer communication.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import weakref

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from .simulation_engine import (
    PrintSimulation, TIMELINE_TICKS, format_sim_time,
)
from .support_surface import build_support_surface

GPU_AVAILABLE = (sys.platform == "win32" and
                 os.environ.get("QT_QPA_PLATFORM", "").lower() not in ("offscreen", "minimal"))
if GPU_AVAILABLE:
    from .simulation_viewport import SimulationViewport
else:
    SimulationViewport = None


def read_mask(path: Path, grayscale: bool) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)  # Works with Unicode Windows paths.
    image = cv2.imdecode(
        raw, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not decode layer mask: {path}")
    if not grayscale:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def read_layer_textures(job_path: str | Path, layer_index: int,
                        max_side: int = 1024):
    """Load only ONE layer's material masks; shrink textures for GPU viewing.

    Source PNG masks are left intact. Pixel-perfect export is not modified.
    """
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    layer = job["layers"][layer_index]
    body = read_mask(job_path.parent / layer[job["settings"]["body_material"]], True)
    color = read_mask(job_path.parent / layer["color"], False)
    support_file = layer.get("support")
    support = (read_mask(job_path.parent / support_file, True)
               if support_file else np.zeros_like(body))
    if color.shape[:2] != body.shape or support.shape != body.shape:
        raise ValueError("Inconsistent layer mask dimensions.")
    h,w = body.shape
    factor = min(1.0, max_side / max(w,h))
    if factor < 1.0:
        target = (max(1,round(w*factor)), max(1,round(h*factor)))
        body = cv2.resize(body,target,interpolation=cv2.INTER_NEAREST)
        color = cv2.resize(color,target,interpolation=cv2.INTER_NEAREST)
        support = cv2.resize(support,target,interpolation=cv2.INTER_NEAREST)
    return body,color,support


def discover_model_source(job_path: Path, manifest: dict, root: Path) -> Path | None:
    """Recover the original file path from task request or newer manifests."""
    candidates = [manifest.get("source_path")]
    task = root / "workspace" / "tasks" / (job_path.parent.name + ".json")
    if task.is_file():
        try:
            candidates.append(json.loads(task.read_text(encoding="utf-8")).get("source"))
        except (ValueError, OSError):
            pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


class SimulationAssetsWorker(QObject):
    ready = Signal(str, object, object, object)
    failed = Signal(str)

    def __init__(self, job_path, manifest, root, source_mesh=None, gpu_payload=None):
        super().__init__()
        self.job_path = str(job_path)
        self.manifest = manifest
        self.root = Path(root)
        self.mesh = source_mesh
        self.payload = gpu_payload

    def run(self):
        try:
            mesh, payload = self.mesh, self.payload
            if GPU_AVAILABLE and (mesh is None or payload is None):
                source = discover_model_source(Path(self.job_path), self.manifest, self.root)
                if source is None:
                    raise FileNotFoundError(
                        "The original 3D model is unavailable. Import it in the 3D slicer "
                        "or keep the original source path when opening an older saved job.")
                from .core import load_mesh
                from .gpu_viewport import prepare_gpu_geometry
                mesh, _ = load_mesh(source)
                payload = prepare_gpu_geometry(mesh)
            surface = build_support_surface(self.job_path)
            self.ready.emit(self.job_path, mesh, payload, surface)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class LayerTextureWorker(QObject):
    ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, job_path, index):
        super().__init__()
        self.job_path = str(job_path)
        self.index = int(index)

    def run(self):
        try:
            self.ready.emit(
                self.index,
                read_layer_textures(self.job_path, self.index))
        except Exception as exc:
            self.failed.emit(self.index, f"{type(exc).__name__}: {exc}")


class SimulationWidget(QWidget):
    """A complete, seekable print-player with separate speed and style controls."""

    def __init__(self, root: Path, parent=None, gpu_enabled=None):
        super().__init__(parent)
        self.root = Path(root)
        if gpu_enabled is None:
            gpu_enabled = GPU_AVAILABLE
        self.gpu_enabled = bool(gpu_enabled and GPU_AVAILABLE)
        self.job_path = None
        self.manifest = None
        self.plan = None
        self.elapsed = 0.0
        self.playing = False
        self._last_clock = None
        self._loaded_texture_layer = -1
        self._wanted_texture_layer = -1
        self._asset_thread = self._asset_worker = None
        self._texture_thread = self._texture_worker = None
        self._deferred_job = None
        self._pending_model = None
        self._scene_ready = False
        self._support_color = QColor("#a86edc")
        self._model_color = QColor("#c8c4b5")

        layout = QVBoxLayout(self)
        warning = QLabel(
            "OFFLINE VISUALIZATION ONLY — printhead motion and timing are conceptual; "
            "not FlexiPRINT/PrintExp commands or real print passes.")
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#ba760d;font-weight:600;")
        layout.addWidget(warning)
        row = QHBoxLayout()
        layout.addLayout(row,stretch=1)

        if self.gpu_enabled:
            self.viewport = SimulationViewport(self)
            self.viewport.opengl_failed.connect(self._display_gl_error)
            row.addWidget(self.viewport,stretch=1)
        else:
            self.viewport = None
            self.fallback = QLabel("3D simulation requires Windows with OpenGL 3.3.\n"
                                   "Timeline and exported-layer masks remain available.")
            self.fallback.setAlignment(Qt.AlignCenter)
            self.fallback.setMinimumSize(320,260)
            self.fallback.setStyleSheet("background:#34383e;color:#e5e6e9;")
            row.addWidget(self.fallback,stretch=1)

        sidebar = QVBoxLayout()
        row.addLayout(sidebar)
        self.open_job_button = QPushButton("Open existing job.json…")
        self.open_job_button.clicked.connect(self.open_job)
        sidebar.addWidget(self.open_job_button)
        playback_group = QGroupBox("Playback")
        group_layout = QVBoxLayout(playback_group)
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        group_layout.addWidget(self.play_button)
        nav = QHBoxLayout()
        prev_btn = QPushButton("◀ Layer")
        next_btn = QPushButton("Layer ▶")
        owner = weakref.ref(self)
        def step_previous(checked=False, ref=owner):
            widget = ref()
            if widget is not None:
                widget.step_layer(-1)
        def step_next(checked=False, ref=owner):
            widget = ref()
            if widget is not None:
                widget.step_layer(1)
        prev_btn.clicked.connect(step_previous)
        next_btn.clicked.connect(step_next)
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        group_layout.addLayout(nav)
        replay_btn = QPushButton("Replay from start")
        replay_btn.clicked.connect(self.replay)
        group_layout.addWidget(replay_btn)
        sidebar.addWidget(playback_group)

        speeds = QGroupBox("Simulation speed")
        speeds_layout = QVBoxLayout(speeds)
        speeds_layout.addWidget(QLabel("Printhead movement (mm/s)"))
        self.move_speed = QDoubleSpinBox()
        self.move_speed.setRange(1,500)
        self.move_speed.setDecimals(1)
        self.move_speed.setValue(100)
        self.move_speed.setSuffix(" mm/s")
        self.move_speed.valueChanged.connect(self.speed_changed)
        speeds_layout.addWidget(self.move_speed)
        speeds_layout.addWidget(QLabel("Replay speed (× virtual time)"))
        self.replay_speed = QDoubleSpinBox()
        self.replay_speed.setRange(.1,1000)
        self.replay_speed.setDecimals(1)
        self.replay_speed.setSingleStep(5)
        self.replay_speed.setValue(25)
        self.replay_speed.setSuffix("×")
        speeds_layout.addWidget(self.replay_speed)
        sidebar.addWidget(speeds)

        support_group = QGroupBox("Material appearance")
        support_layout = QVBoxLayout(support_group)
        self.support_color_button = QPushButton("Support color: purple")
        self.support_color_button.clicked.connect(self.pick_support_color)
        support_layout.addWidget(self.support_color_button)
        support_layout.addWidget(QLabel("Support opacity"))
        self.support_opacity = QSlider(Qt.Horizontal)
        self.support_opacity.setRange(0,100)
        self.support_opacity.setValue(32)
        self.support_opacity.valueChanged.connect(self.opacity_changed)
        support_layout.addWidget(self.support_opacity)
        self.opacity_readout = QLabel("32%")
        support_layout.addWidget(self.opacity_readout)
        self.model_color_button = QPushButton("Uncolored model tint")
        self.model_color_button.clicked.connect(self.pick_model_color)
        self.model_color_button.setToolTip(
            "STL has no standard surface texture. For colored OBJ/PLY/GLB files, "
            "the source vertex/face colors are preserved instead.")
        support_layout.addWidget(self.model_color_button)
        self.future_checkbox = QCheckBox("Show unprinted model silhouette")
        self.future_checkbox.setChecked(False)
        self.future_checkbox.toggled.connect(self.show_future_changed)
        support_layout.addWidget(self.future_checkbox)
        sidebar.addWidget(support_group)

        if self.viewport is not None:
            cameras = QHBoxLayout()
            for name in ("Isometric","Top","Front","Right"):
                btn=QPushButton(name)
                ref=weakref.ref(self.viewport)
                btn.clicked.connect(
                    lambda checked=False, view=name, vp=ref:
                    vp().set_view_preset(view) if vp() is not None else None)
                cameras.addWidget(btn)
            layout.addLayout(cameras)
            nav_label=QLabel("RMB orbit · MMB pan · mouse wheel zoom · original model colors where available")
            nav_label.setStyleSheet("color:gray;")
            layout.addWidget(nav_label)

        timeline_row = QHBoxLayout()
        self.time_label = QLabel("00:00:00")
        timeline_row.addWidget(self.time_label)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0,TIMELINE_TICKS)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self.timeline_changed)
        timeline_row.addWidget(self.timeline,stretch=1)
        self.duration_label = QLabel("00:00:00")
        timeline_row.addWidget(self.duration_label)
        layout.addLayout(timeline_row)
        self.detail = QLabel("Generate layer masks or open a completed job.json to simulate.")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._texture_delay = QTimer(self)
        self._texture_delay.setSingleShot(True)
        self._texture_delay.setInterval(35)
        self._texture_delay.timeout.connect(self._start_texture_load)

    @property
    def busy(self):
        return (self._asset_thread is not None
                or self._texture_thread is not None)

    def _display_gl_error(self, problem):
        self.detail.setText("GPU simulation renderer unavailable: " + problem)

    def open_job(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose completed UV3D job.json",
            str(self.root / "workspace" / "jobs"), "UV3D jobs (job.json)")
        if filename:
            self.load_job(filename)

    def load_job(self, job_path, source_mesh=None, gpu_geometry=None):
        job_path = Path(job_path).resolve()
        if self._asset_thread is not None:
            self._deferred_job = (job_path, source_mesh, gpu_geometry)
            self.pause()
            self.detail.setText("Finishing preview loading before switching jobs…")
            return
        try:
            manifest = json.loads(job_path.read_text(encoding="utf-8"))
            if not manifest.get("layers"):
                raise ValueError("This job contains no completed layers.")
            plan = PrintSimulation(
                manifest,move_speed_mm_s=self.move_speed.value())
        except Exception as exc:
            self.detail.setText(f"Cannot open simulation job: {exc}")
            return
        self.pause()
        self.manifest = manifest
        self.plan = plan
        self.job_path = job_path
        self._scene_ready = False
        if self.viewport is not None:
            # Do not draw the previous model while a different job is loading.
            self.viewport.frame = None
            self.viewport.mesh = None
            self.viewport.update()
        self.elapsed = 0.0
        self._loaded_texture_layer = -1
        self._wanted_texture_layer = -1
        self.timeline.setEnabled(True)
        self.play_button.setEnabled(True)
        self.duration_label.setText(format_sim_time(plan.total_seconds))
        self.detail.setText(f"Preparing 3D simulation for {plan.count:,} exported layers…")
        if self.viewport is None:
            self._update_frame()
            return
        # The imported mesh is reused when it matches this export.
        if (source_mesh is not None and gpu_geometry is not None
                and source_mesh.extents[2] > 0):
            model_name = manifest.get("source_name", "")
            source_name = getattr(source_mesh, "metadata", {}).get("file_name", "")
            # Existing import/export flow supplies matching geometry directly.
            self._pending_model = (source_mesh,gpu_geometry)
        else:
            self._pending_model = (None,None)
        thread = QThread(self)
        worker = SimulationAssetsWorker(
            job_path,manifest,self.root,*self._pending_model)
        self._asset_thread,self._asset_worker = thread,worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self._assets_ready)
        worker.failed.connect(self._assets_failed)
        worker.ready.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._asset_thread_finished)
        thread.start()
        self._update_frame()

    def _assets_ready(self, job_path, mesh, payload, surface):
        if self.job_path is None or str(self.job_path) != job_path:
            return
        if self.viewport is not None and mesh is not None and payload is not None:
            self.viewport.set_simulation_scene(mesh,payload,self.manifest)
            self.viewport.set_support_surface(surface.triangles)
            self._scene_ready = True
            self.viewport.set_support_tint(
                self._support_color.getRgb()[:3],self.support_opacity.value()/100)
            # Use a fallback color for STL, whose standard format has no
            # stored texture or required per-face colors.
            uncolored = self.manifest.get("source_name","").lower().endswith(".stl")
            self.viewport.set_model_tint(self._model_color.getRgb()[:3],
                                         fallback_only=uncolored)
            self.viewport.set_show_future(self.future_checkbox.isChecked())
        self.detail.setText(
            f"Ready: {self.plan.count:,} layers · "
            f"support preview {surface.sampled_levels} sampled levels. "
            "Virtual head path; NOT a calibrated printer motion.")
        self._update_frame()
        self._request_texture(self.plan.get_frame(0).layer_index)

    def _assets_failed(self, problem):
        self.detail.setText("3D preview failed: " + problem)
        # Timeline still works; a missing model cannot silently become a
        # different colored object.

    def _asset_thread_finished(self):
        self._asset_thread = self._asset_worker = None
        if self._deferred_job is not None:
            args = self._deferred_job
            self._deferred_job = None
            self.load_job(*args)

    def _request_texture(self,index):
        if self.viewport is None or self.job_path is None:
            return
        self._wanted_texture_layer = int(index)
        if self._loaded_texture_layer == index:
            return
        self._texture_delay.start()

    def _start_texture_load(self):
        if self._texture_thread is not None or self.job_path is None:
            return
        index = self._wanted_texture_layer
        if index < 0 or index == self._loaded_texture_layer:
            return
        thread = QThread(self)
        worker = LayerTextureWorker(self.job_path,index)
        self._texture_thread,self._texture_worker = thread,worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self._texture_ready)
        worker.failed.connect(self._texture_failed)
        worker.ready.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._texture_thread_finished)
        thread.start()

    def _texture_ready(self,index,arrays):
        if self.viewport is None or self.plan is None:
            return
        # Avoid stale layers when the user scrubs the timeline rapidly.
        if index == self._wanted_texture_layer:
            self._loaded_texture_layer = index
            self.viewport.set_layer_bitmaps(index,*arrays)

    def _texture_failed(self,index,problem):
        self.detail.setText(f"Layer {index+1:,} preview error: {problem}")

    def _texture_thread_finished(self):
        self._texture_thread = self._texture_worker = None
        if self._loaded_texture_layer != self._wanted_texture_layer:
            self._texture_delay.start(10)

    def _update_frame(self):
        if self.plan is None:
            return
        frame = self.plan.get_frame(self.elapsed / self.plan.total_seconds)
        self.timeline.blockSignals(True)
        self.timeline.setValue(round(frame.timeline_fraction * TIMELINE_TICKS))
        self.timeline.blockSignals(False)
        self.time_label.setText(format_sim_time(self.elapsed))
        self.duration_label.setText(format_sim_time(self.plan.total_seconds))
        hx,hy,hz = frame.head_position_mm
        self.detail.setText(
            f"Printing layer {frame.layer_index+1:,}/{frame.layer_count:,} "
            f"({frame.layer_progress*100:.1f}% of layer). "
            f"Printhead X {hx:.1f}, Y {hy:.1f}, Z {hz:.2f} mm "
            f"· {frame.rows_per_layer} simulated scan rows.")
        if self.viewport is not None and self._scene_ready:
            self.viewport.set_frame(frame)
            if frame.layer_index != self._wanted_texture_layer:
                self._request_texture(frame.layer_index)

    def timeline_changed(self,ticks):
        if self.plan is None:
            return
        self.elapsed = (ticks / TIMELINE_TICKS) * self.plan.total_seconds
        self._last_clock = time.monotonic()
        self._update_frame()

    def speed_changed(self,value):
        if self.plan is None:
            return
        normalized = self.elapsed / self.plan.total_seconds
        self.plan.set_move_speed(float(value))
        self.elapsed = normalized * self.plan.total_seconds
        self._last_clock = time.monotonic()
        self._update_frame()

    def toggle_play(self):
        if self.plan is None:
            return
        if self.playing:
            self.pause()
            return
        if self.elapsed >= self.plan.total_seconds:
            self.elapsed = 0
        self.playing = True
        self.play_button.setText("⏸ Pause")
        self._last_clock = time.monotonic()
        self.timer.start()

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.play_button.setText("▶ Play")
        self._last_clock = None

    def replay(self):
        if self.plan is None:
            return
        self.pause()
        self.elapsed = 0.0
        self._update_frame()
        self.toggle_play()

    def step_layer(self, amount: int):
        if self.plan is None:
            return
        self.pause()
        frame = self.plan.get_frame(self.elapsed/self.plan.total_seconds)
        target = min(self.plan.count, max(0, frame.layer_index + int(amount)))
        self.elapsed = (target / self.plan.count) * self.plan.total_seconds
        self._update_frame()

    def _tick(self):
        if not self.playing or self.plan is None:
            return
        now = time.monotonic()
        elapsed_real = max(0.0, min(.5, now-(self._last_clock or now)))
        self._last_clock = now
        self.elapsed = min(self.plan.total_seconds,
                           self.elapsed + elapsed_real * self.replay_speed.value())
        self._update_frame()
        if self.elapsed >= self.plan.total_seconds:
            self.pause()

    def pick_support_color(self):
        picked = QColorDialog.getColor(self._support_color,self,"Support material color")
        if picked.isValid():
            self._support_color = picked
            self.support_color_button.setText("Support: "+picked.name())
            self.opacity_changed(self.support_opacity.value())

    def opacity_changed(self,value):
        self.opacity_readout.setText(f"{value}%")
        if self.viewport is not None:
            self.viewport.set_support_tint(self._support_color.getRgb()[:3],value/100)

    def pick_model_color(self):
        picked = QColorDialog.getColor(self._model_color,self,"Fallback model color (uncolored STL)")
        if picked.isValid():
            self._model_color = picked
            if self.viewport is not None and self.manifest is not None:
                uncolored=self.manifest.get("source_name","").lower().endswith(".stl")
                self.viewport.set_model_tint(picked.getRgb()[:3], fallback_only=uncolored)

    def show_future_changed(self,enabled):
        if self.viewport is not None:
            self.viewport.set_show_future(enabled)
