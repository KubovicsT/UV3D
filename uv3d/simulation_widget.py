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
from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from .simulation_engine import (
    PrintSimulation, TIMELINE_TICKS, format_sim_time,
)
from .support_surface import build_support_surface
from .gantry import GantryConfig, load_gantry

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


class SimulationAssetsThread(QThread):
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


class LayerTextureThread(QThread):
    ready = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self, job_path, index):
        super().__init__()
        self.job_path = str(job_path)
        self.index = int(index)

    def run(self):
        try:
            self.ready.emit(
                self.job_path, self.index,
                read_layer_textures(self.job_path, self.index))
        except Exception as exc:
            self.failed.emit(self.job_path, self.index, f"{type(exc).__name__}: {exc}")


class SimulationWidget(QWidget):
    """A complete, seekable print-player with separate speed and style controls."""

    def __init__(self, root: Path, parent=None, gpu_enabled=None):
        super().__init__(parent)
        self.root = Path(root)
        self._gantry_config = load_gantry(self.root)
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
        self._asset_thread = None
        self._texture_thread = None
        self._deferred_job = None
        self._pending_model = None
        self._scene_ready = False
        self._close_when_idle = False
        self._support_color = QColor("#a86edc")
        self._model_color = QColor("#c8c4b5")
        self._current_model_color = QColor("#ffad2e")
        self._current_support_color = QColor("#1aebcf")

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
        speeds_layout.addWidget(QLabel("Printhead speed — printing (mm/s)"))
        self.print_speed = QDoubleSpinBox()
        self.print_speed.setRange(1, 3000)
        self.print_speed.setDecimals(1)
        self.print_speed.setValue(300)
        self.print_speed.setSuffix(" mm/s")
        self.print_speed.valueChanged.connect(self.speed_changed)
        speeds_layout.addWidget(self.print_speed)
        speeds_layout.addWidget(QLabel("Movement speed — non-print travel (mm/s)"))
        self.move_speed = QDoubleSpinBox()
        self.move_speed.setRange(1, 3000)
        self.move_speed.setDecimals(1)
        self.move_speed.setValue(600)
        self.move_speed.setSuffix(" mm/s")
        self.move_speed.valueChanged.connect(self.speed_changed)
        speeds_layout.addWidget(self.move_speed)
        speeds_layout.addWidget(QLabel("Print passes per 33.8 mm band (estimate)"))
        self.print_passes = QSpinBox()
        self.print_passes.setRange(1, 32)
        self.print_passes.setValue(6)
        self.print_passes.setToolTip(
            "6 is an illustrative default; this is not read from the printer RIP.")
        self.print_passes.valueChanged.connect(self.speed_changed)
        speeds_layout.addWidget(self.print_passes)
        speeds_layout.addWidget(QLabel("Printing direction"))
        self.print_mode = QComboBox()
        self.print_mode.addItem("Bidirectional — prints in both directions",
                                "bidirectional")
        self.print_mode.addItem("One direction — prints left to right",
                                "unidirectional")
        self.print_mode.setToolTip(
            "Bidirectional prints on alternating left/right scans. "
            "One direction prints only left-to-right; every return is non-printing "
            "travel at Movement speed.")
        self.print_mode.currentIndexChanged.connect(self.speed_changed)
        speeds_layout.addWidget(self.print_mode)
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
        support_layout.addWidget(QLabel("Current printing layer — highlight colors"))
        self.current_model_button = QPushButton("Printing model: orange")
        self.current_model_button.setToolTip(
            "Color only the actively printed model layer; completed model keeps "
            "its original appearance.")
        self.current_model_button.clicked.connect(self.pick_current_model_color)
        support_layout.addWidget(self.current_model_button)
        self.current_support_button = QPushButton("Printing support: cyan")
        self.current_support_button.setToolTip(
            "Color only the actively printed support layer. Completed support "
            "keeps the normal support color and opacity.")
        self.current_support_button.clicked.connect(self.pick_current_support_color)
        support_layout.addWidget(self.current_support_button)
        support_layout.addWidget(QLabel("Printhead opacity"))
        self.head_opacity = QSlider(Qt.Horizontal)
        self.head_opacity.setRange(0, 100)
        self.head_opacity.setValue(65)
        self.head_opacity.valueChanged.connect(self.head_opacity_changed)
        support_layout.addWidget(self.head_opacity)
        self.head_opacity_readout = QLabel("65%")
        support_layout.addWidget(self.head_opacity_readout)
        self.future_checkbox = QCheckBox("Show unprinted model silhouette")
        self.future_checkbox.setChecked(True)
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
        self.timeline.sliderPressed.connect(self.timeline_pressed)
        self.timeline.sliderReleased.connect(self.timeline_released)
        timeline_row.addWidget(self.timeline,stretch=1)
        self.duration_label = QLabel("00:00:00")
        timeline_row.addWidget(self.duration_label)
        layout.addLayout(timeline_row)
        self.estimate_label = QLabel("Estimated total: -- · Elapsed: -- · Remaining: -- · Progress: 0.0%")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("font-weight:600;")
        layout.addWidget(self.estimate_label)
        self.detail = QLabel("Generate layer masks or open a completed job.json to simulate.")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self._texture_delay = QTimer(self)
        self._texture_delay.setSingleShot(True)
        # File decoding and GPU texture uploads must not follow every mouse
        # movement. Coalesce fast seeks into one final requested layer.
        self._texture_delay.setInterval(180)
        self._texture_delay.timeout.connect(self._start_texture_load)
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setInterval(33)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.timeout.connect(self._flush_scrub_frame)
        self._scrub_pending = False

    @property
    def busy(self):
        return (self._asset_thread is not None
                or self._texture_thread is not None)

    def _display_gl_error(self, problem):
        # Qt/OpenGL failures can occur only after the first timeline seek.
        # Keep a persistent traceback rather than silently losing the reason.
        try:
            directory = self.root / "workspace" / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            with (directory / "uv3d-simulation.log").open(
                    "a", encoding="utf-8") as stream:
                stream.write(f"\n=== {datetime.now().isoformat()} ===\n"
                             f"Timeline {self.timeline.value()}, "
                             f"Layer {self._wanted_texture_layer+1}:\n{problem}\n")
        except Exception:
            pass
        self.pause()
        self.detail.setText(
            "Simulation renderer error (the app remains open). "
            "See workspace/logs/uv3d-simulation.log — " + problem.splitlines()[0])

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
                manifest, move_speed_mm_s=self.move_speed.value(),
                print_speed_mm_s=self.print_speed.value(),
                passes=self.print_passes.value(),
                print_mode=self.print_mode.currentData(),
                gantry_config=self._gantry_config)
        except Exception as exc:
            self.detail.setText(f"Cannot open simulation job: {exc}")
            return
        self.pause()
        self.manifest = manifest
        self.plan = plan
        self.job_path = job_path
        self._scene_ready = False
        self._texture_delay.stop()
        self._scrub_timer.stop()
        self._scrub_pending = False
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
        # QThread objects live on the GUI thread; only their run() methods
        # execute in worker threads. This avoids moving/deleting timer-owning
        # QObject workers across threads on Windows.
        thread = SimulationAssetsThread(
            job_path, manifest, self.root, *self._pending_model)
        self._asset_thread = thread
        thread.ready.connect(self._assets_ready, Qt.QueuedConnection)
        thread.failed.connect(self._assets_failed, Qt.QueuedConnection)
        thread.finished.connect(self._asset_thread_finished, Qt.QueuedConnection)
        thread.start()
        self._update_frame()

    @Slot(str, object, object, object)
    def _assets_ready(self, job_path, mesh, payload, surface):
        if self.job_path is None or str(self.job_path) != job_path:
            return
        if self.viewport is not None and mesh is not None and payload is not None:
            self.viewport.set_simulation_scene(mesh,payload,self.manifest)
            self.viewport.set_support_surface(surface.triangles)
            self.viewport.set_gantry_config(self._gantry_config)
            self._scene_ready = True
            self.viewport.set_simulation_passes(self.plan.passes)
            self.viewport.set_head_opacity(self.head_opacity.value() / 100)
            self._sync_current_layer_colors()
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

    @Slot(str)
    def _assets_failed(self, problem):
        self.detail.setText("3D preview failed: " + problem)
        # Timeline still works; a missing model cannot silently become a
        # different colored object.

    @Slot()
    def _asset_thread_finished(self):
        self._asset_thread.deleteLater()
        self._asset_thread = None
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
            return
        if self._deferred_job is not None:
            args = self._deferred_job
            self._deferred_job = None
            self.load_job(*args)

    def _request_texture(self,index):
        if self.viewport is None or self.job_path is None:
            return
        if self.timeline.isSliderDown():
            # The geometry can seek immediately, but do not open hundreds of
            # PNGs as the mouse crosses hundreds of layers in one drag.
            return
        self._wanted_texture_layer = int(index)
        if self._loaded_texture_layer == index:
            return
        self._texture_delay.start()

    def _start_texture_load(self):
        if (self.timeline.isSliderDown() or self._texture_thread is not None
                or self.job_path is None or not self._scene_ready):
            return
        index = self._wanted_texture_layer
        if index < 0 or index == self._loaded_texture_layer:
            return
        thread = LayerTextureThread(self.job_path, index)
        self._texture_thread = thread
        thread.ready.connect(self._texture_ready, Qt.QueuedConnection)
        thread.failed.connect(self._texture_failed, Qt.QueuedConnection)
        thread.finished.connect(self._texture_thread_finished, Qt.QueuedConnection)
        thread.start()

    @Slot(str, int, object)
    def _texture_ready(self,job_path,index,arrays):
        if (self.viewport is None or self.plan is None or
                self.job_path is None or str(self.job_path) != job_path):
            return
        # Avoid stale layers when the user scrubs the timeline rapidly.
        if index == self._wanted_texture_layer:
            self._loaded_texture_layer = index
            self._log_simulation_checkpoint("texture-upload-request")
            self.viewport.set_layer_bitmaps(index,*arrays)

    @Slot(str, int, str)
    def _texture_failed(self,job_path,index,problem):
        if self.job_path is not None and str(self.job_path) == job_path:
            self.detail.setText(f"Layer {index+1:,} preview error: {problem}")

    @Slot()
    def _texture_thread_finished(self):
        self._texture_thread.deleteLater()
        self._texture_thread = None
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
            return
        if (not self.timeline.isSliderDown()
                and self._loaded_texture_layer != self._wanted_texture_layer):
            self._texture_delay.start(100)

    def _update_frame(self, *, update_viewport=True):
        if self.plan is None:
            return
        frame = self.plan.get_frame(self.elapsed / self.plan.total_seconds)
        self.timeline.blockSignals(True)
        self.timeline.setValue(round(frame.timeline_fraction * TIMELINE_TICKS))
        self.timeline.blockSignals(False)
        self.time_label.setText(format_sim_time(self.elapsed))
        self.duration_label.setText(format_sim_time(self.plan.total_seconds))
        self.estimate_label.setText(
            f"Estimated total: {format_sim_time(self.plan.total_seconds)}  ·  "
            f"Elapsed: {format_sim_time(self.elapsed)}  ·  "
            f"Remaining: {format_sim_time(frame.remaining_virtual_s)}  ·  "
            f"Progress: {frame.progress_percent:.1f}%")
        hx,hy,hz = frame.head_position_mm
        self.detail.setText(
            f"Layer {frame.layer_index+1:,}/{frame.layer_count:,} · "
            f"{frame.motion_kind.upper()} · "
            f"{frame.layer_progress*100:.1f}% through layer. "
            f"Head X {hx:.1f}, Y {hy:.1f}, Z {hz:.2f} mm · "
            f"Band {frame.row_index+1}/{frame.rows_per_layer}, "
            f"pass {frame.pass_index+1}/{self.plan.passes} "
            f"({self.print_mode.currentText().split(' — ')[0]}). "
            f"Carriage: {sum(h.enabled for h in self._gantry_config.heads)} heads, "
            f"{sum(l.enabled for l in self._gantry_config.lamps)} UV lamps. "
            f"{'Curing immediately after this pass' if frame.curing else 'Deposition/travel'}; "
            "UV footprint and time are simulation estimates, not calibrated.")
        if update_viewport and self.viewport is not None and self._scene_ready:
            self.viewport.set_frame(frame)
            if frame.layer_index != self._wanted_texture_layer:
                self._request_texture(frame.layer_index)

    def _log_simulation_checkpoint(self, stage):
        # A single flushed, human-readable record survives even if Windows
        # forcibly terminates the viewer before Python can display an error.
        # Write only on drag start/end and GPU texture handoff, not every tick.
        if self.job_path is None:
            return
        try:
            from datetime import datetime
            folder = self.root / "workspace" / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "uv3d-simulation-last-seek.log").open(
                    "a", encoding="utf-8", buffering=1) as stream:
                stream.write(
                    f"{datetime.now().isoformat(timespec='milliseconds')} "
                    f"{stage} | tick={self.timeline.value()} "
                    f"layer={self._wanted_texture_layer+1} "
                    f"loaded={self._loaded_texture_layer+1} "
                    f"asset_loading={self._asset_thread is not None} "
                    f"texture_loading={self._texture_thread is not None}\n"
                )
        except OSError:
            pass

    def timeline_pressed(self):
        self._log_simulation_checkpoint("drag-start")
        # Pause playback to avoid timer races with user-controlled seeking.
        self.pause()
        self._texture_delay.stop()
        self._scrub_timer.stop()
        self._scrub_pending = False

    def timeline_released(self):
        self._log_simulation_checkpoint("drag-release")
        # Only decode the final layer after the user finishes the drag.
        self._scrub_timer.stop()
        self._scrub_pending = False
        self._update_frame()
        if self.plan is not None:
            frame = self.plan.get_frame(self.elapsed / self.plan.total_seconds)
            self._request_texture(frame.layer_index)
            # The last selected layer takes precedence over any stale loader.
            if self._texture_thread is None:
                self._texture_delay.start(20)

    def _flush_scrub_frame(self):
        if self._scrub_pending and self.plan is not None:
            self._scrub_pending = False
            self._update_frame()

    def timeline_changed(self,ticks):
        if self.plan is None:
            return
        self.elapsed = (ticks / TIMELINE_TICKS) * self.plan.total_seconds
        self._last_clock = time.monotonic()
        if self.timeline.isSliderDown():
            # Labels update immediately, but GPU redraws are limited to ~30Hz.
            # We deliberately avoid layer PNG decoding until release.
            self._update_frame(update_viewport=False)
            self._scrub_pending = True
            if not self._scrub_timer.isActive():
                self._scrub_timer.start()
        else:
            self._update_frame()

    def set_gantry_config(self, config: GantryConfig):
        """Update entire virtual carriage, UV cure route and estimated time."""
        self._gantry_config = GantryConfig.from_dict(config.as_dict())
        if self.plan is not None:
            fraction = self.elapsed / self.plan.total_seconds
            self.plan.set_gantry_config(self._gantry_config)
            self.elapsed = fraction * self.plan.total_seconds
            self._last_clock = time.monotonic()
        if self.viewport is not None:
            self.viewport.set_gantry_config(self._gantry_config)
        self._update_frame()

    def speed_changed(self,value):
        if self.plan is None:
            return
        normalized = self.elapsed / self.plan.total_seconds
        self.plan.configure(
            self.print_speed.value(), self.move_speed.value(),
            self.print_passes.value(), print_mode=self.print_mode.currentData())
        if self.viewport is not None and self._scene_ready:
            self.viewport.set_simulation_passes(self.plan.passes)
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
        target_fraction = self.plan.step_layer(
            self.elapsed / self.plan.total_seconds, amount)
        self.elapsed = target_fraction * self.plan.total_seconds
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

    def head_opacity_changed(self, value):
        self.head_opacity_readout.setText(f"{value}%")
        if self.viewport is not None:
            self.viewport.set_head_opacity(value / 100)

    def _sync_current_layer_colors(self):
        if self.viewport is not None:
            self.viewport.set_current_layer_colors(
                self._current_model_color.getRgb()[:3],
                self._current_support_color.getRgb()[:3])

    def pick_current_model_color(self):
        picked = QColorDialog.getColor(
            self._current_model_color, self, "Current printing model color")
        if picked.isValid():
            self._current_model_color = picked
            self.current_model_button.setText("Printing model: " + picked.name())
            self._sync_current_layer_colors()

    def pick_current_support_color(self):
        picked = QColorDialog.getColor(
            self._current_support_color, self, "Current printing support color")
        if picked.isValid():
            self._current_support_color = picked
            self.current_support_button.setText("Printing support: " + picked.name())
            self._sync_current_layer_colors()

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


    def closeEvent(self, event):
        """Wait for IO/mesh Qt workers before destroying their QThread objects.

        Destroying a running QThread is a native Qt fatal error and used to
        terminate the whole application silently. In a separate child process
        it still matters for clean shutdown and complete diagnostic logs.
        """
        self.pause()
        self._texture_delay.stop()
        self._scrub_timer.stop()
        if self.busy:
            self._close_when_idle = True
            self.detail.setText("Closing simulation after active image load finishes...")
            event.ignore()
            return
        self._close_when_idle = False
        event.accept()
