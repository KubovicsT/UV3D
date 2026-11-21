"""UV3D desktop GUI. OFFLINE ONLY — not connected to any printer hardware."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import weakref
from datetime import datetime

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)
from .core import MachineProfile, SliceSettings, load_mesh
from .gui_slicer import SlicerController
from .support_preview import build_support_wireframe
from .simulation_process_panel import SimulationProcessPanel
from .diagnostics import install_gui_diagnostics
from .viewport import ModelViewport as SoftwareModelViewport, build_preview_geometry

# Interactive Windows sessions render the ORIGINAL mesh on the GPU.
# Headless CI and environments without an OpenGL display use the existing
# software viewport, which deliberately remains a low-detail fallback.
if sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM", "").lower() not in ("offscreen", "minimal"):
    from .gpu_viewport import GpuModelViewport as ModelViewport, prepare_gpu_geometry
else:
    ModelViewport = SoftwareModelViewport
    prepare_gpu_geometry = None

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"


class ImportWorker(QObject):
    """Load and simplify full-resolution files without blocking Qt's event loop."""
    stage = Signal(str)
    finished = Signal(object, object, object, object, object)
    error = Signal(str)

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def run(self):
        try:
            self.stage.emit("Loading and checking 3D geometry...")
            begin = time.perf_counter()
            mesh, warnings = load_mesh(self.filename)
            read_seconds = time.perf_counter() - begin
            if getattr(ModelViewport, "FULL_RESOLUTION", False):
                self.stage.emit(f"Preparing full-resolution GPU model ({len(mesh.faces):,} triangles)...")
            else:
                self.stage.emit(f"Preparing software preview ({len(mesh.faces):,} source triangles)...")
            begin = time.perf_counter()
            if getattr(ModelViewport, "FULL_RESOLUTION", False):
                preview = prepare_gpu_geometry(mesh)  # Original vertices/faces, no simplification.
            else:
                preview = build_preview_geometry(
                    mesh, max_faces=ModelViewport.PREVIEW_TRIANGLES)
            preview_seconds = time.perf_counter() - begin
            self.finished.emit(self.filename, mesh, preview, warnings,
                               (read_seconds, preview_seconds))
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class SupportPreviewWorker(QObject):
    ready = Signal(str, object)
    error = Signal(str)

    def __init__(self, job_path: str | Path):
        super().__init__()
        self.job_path = str(job_path)

    def run(self):
        try:
            lines = build_support_wireframe(self.job_path)
            self.ready.emit(self.job_path, lines)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV3D — 3D Slicer / Gongzheng H6090EI (offline)")
        self.resize(1330, 850)
        self.mesh_path: Path | None = None
        self.source_mesh = None
        self.preview_mesh = None
        self.job_path: Path | None = None
        self.job_data: dict | None = None
        self.worker_thread = None
        self.worker = None
        self.slicer = SlicerController(ROOT, self)
        self.slicer.running_changed.connect(self.on_slicer_running_changed)
        self.slicer.progress.connect(self.on_slicer_progress)
        self.slicer.stage.connect(self.on_slicer_stage)
        self.slicer.succeeded.connect(self.export_finished)
        self.slicer.failed.connect(self.on_slicer_failed)
        self.import_thread = None
        self.import_worker = None
        self.support_preview_thread = None
        self.support_preview_worker = None
        self.source_preview = None
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        (WORKSPACE / "jobs").mkdir(exist_ok=True)
        tabs = QTabWidget()
        tabs.addTab(self.build_slicer(), "3D slicer")
        tabs.addTab(self.build_simulator(), "Print queue — simulation")
        tabs.addTab(self.build_machine(), "Machine / calibration")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("OFFLINE — printer communication disabled")
        update = QPushButton("Update from GitHub main")
        update.clicked.connect(self.update_app)
        self.statusBar().addPermanentWidget(update)

    def build_slicer(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split)

        left = QWidget()
        controls = QVBoxLayout(left)
        open_button = QPushButton("Open 3D model…")
        self.open_button = open_button
        open_button.clicked.connect(self.open_model)
        controls.addWidget(open_button)
        self.model_info = QLabel("No model loaded.\nSTL, OBJ, GLB, GLTF, 3MF, PLY")
        self.model_info.setWordWrap(True)
        controls.addWidget(self.model_info)

        settings = QGroupBox("Model dimensions and slice settings")
        form = QFormLayout(settings)
        self.model_height = QDoubleSpinBox()
        self.model_height.setDecimals(3)
        self.model_height.setRange(0.05, 180.0)
        self.model_height.setSingleStep(1.0)
        self.model_height.setSuffix(" mm")
        self.model_height.setKeyboardTracking(False)
        self.model_height.setEnabled(False)
        self.model_height.valueChanged.connect(self.on_height_changed)
        form.addRow("Model height (uniform scaling)", self.model_height)

        self.layer = QDoubleSpinBox()
        self.layer.setDecimals(3)
        self.layer.setRange(0.01, 1.0)
        self.layer.setSingleStep(0.01)
        self.layer.setValue(0.05)
        self.layer.setSuffix(" mm")
        self.layer.setKeyboardTracking(False)
        self.layer.valueChanged.connect(self.refresh_layer_range)
        form.addRow("Target layer thickness", self.layer)

        self.layer_count = QLabel("—")
        form.addRow("Preview layer count", self.layer_count)

        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        form.addRow("Raster DPI", self.dpi)
        self.shell = QDoubleSpinBox()
        self.shell.setRange(0.05, 5.0)
        self.shell.setSingleStep(0.1)
        self.shell.setValue(0.3)
        form.addRow("Boundary color width (mm)", self.shell)
        self.body = QComboBox()
        self.body.addItems(["white", "varnish"])
        form.addRow("Solid model fill material", self.body)
        self.support_enabled = QCheckBox("Fill all overhang columns to the build plate")
        self.support_enabled.setChecked(True)
        self.support_enabled.setToolTip(
            "Every XY position occupied by model or casing at a higher layer is filled beneath it, all the way down to the bed. This is SOLID support, not a thin shell.")
        form.addRow("Conformal support", self.support_enabled)
        self.support_casing = QDoubleSpinBox()
        self.support_casing.setDecimals(2)
        self.support_casing.setRange(0, 5)
        self.support_casing.setSingleStep(0.2)
        self.support_casing.setSuffix(" mm")
        self.support_casing.setValue(0.6)
        self.support_casing.setKeyboardTracking(False)
        self.support_casing.setToolTip(
            "Extra radial material around the model. 0 mm supports only vertical overhang columns.")
        form.addRow("Support casing thickness", self.support_casing)
        support_warning = QLabel(
            "SUPPORT is a separate proposed material channel; white/varnish "
            "ink is NOT assumed water-soluble. Trapped cavities may be inaccessible.")
        support_warning.setWordWrap(True)
        form.addRow(support_warning)
        self.x_pos = QDoubleSpinBox()
        self.x_pos.setRange(0, 600)
        self.x_pos.setValue(10)
        form.addRow("Raster offset X (mm)", self.x_pos)
        self.y_pos = QDoubleSpinBox()
        self.y_pos.setRange(0, 900)
        self.y_pos.setValue(10)
        form.addRow("Raster offset Y (mm)", self.y_pos)
        controls.addWidget(settings)
        self.support_enabled.toggled.connect(self.invalidate_export)
        self.support_casing.valueChanged.connect(self.invalidate_export)
        self.support_enabled.toggled.connect(self.support_casing.setEnabled)
        for ctl in (self.dpi, self.shell, self.body, self.x_pos, self.y_pos):
            if isinstance(ctl, QComboBox):
                ctl.currentTextChanged.connect(self.invalidate_export)
            else:
                ctl.valueChanged.connect(self.invalidate_export)

        self.compatibility_mode = QCheckBox(
            "Compatibility slicing (slower; alternative to accelerated OpenCV)")
        self.compatibility_mode.setToolTip(
            "Try this if a model repeatedly crashes the accelerated slicer.")
        controls.addWidget(self.compatibility_mode)
        self.export_button = QPushButton("Generate offline layer masks")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.start_export)
        controls.addWidget(self.export_button)
        self.cancel_export_button = QPushButton('Cancel layer generation')
        self.cancel_export_button.setEnabled(False)
        self.cancel_export_button.clicked.connect(self.cancel_export)
        controls.addWidget(self.cancel_export_button)
        self.progress = QProgressBar()
        controls.addWidget(self.progress)
        self.output_label = QLabel("No job exported")
        self.output_label.setWordWrap(True)
        controls.addWidget(self.output_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setPlaceholderText("Warnings and export details appear here.")
        controls.addWidget(self.log, stretch=1)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        top = QHBoxLayout()
        title = QLabel(
            "3D build preview — full original resolution (GPU)"
            if getattr(ModelViewport, "FULL_RESOLUTION", False)
            else "3D build preview — software fallback"
        )
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        top.addWidget(title)
        top.addStretch(1)
        self.slice_checkbox = QCheckBox("Show layers / ghost silhouette")
        self.slice_checkbox.setChecked(True)
        self.slice_checkbox.toggled.connect(self.on_slice_mode_changed)
        top.addWidget(self.slice_checkbox)
        self.show_support = QCheckBox("Solid support (wireframe)")
        self.show_support.setChecked(True)
        self.show_support.setEnabled(False)
        self.show_support.setToolTip(
            "Preview purple outlines of the SOLID support volume generated under all overhangs. The viewer samples outlines; the full masks are filled, including areas below upper ledges.")
        top.addWidget(self.show_support)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        top.addWidget(reset)
        right_layout.addLayout(top)

        view_row = QHBoxLayout()
        self.viewport = ModelViewport()
        self.show_support.toggled.connect(self.viewport.set_show_support_preview)
        if hasattr(self.viewport, "opengl_failed"):
            self.viewport.opengl_failed.connect(self.report_opengl_failure)
        view_row.addWidget(self.viewport, stretch=1)
        slider_box = QVBoxLayout()
        top_label = QLabel("Top")
        top_label.setAlignment(Qt.AlignCenter)
        slider_box.addWidget(top_label)
        self.slice_slider = QSlider(Qt.Vertical)
        self.slice_slider.setRange(1, 1)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slider_changed)
        slider_box.addWidget(self.slice_slider, stretch=1, alignment=Qt.AlignHCenter)
        bottom_label = QLabel("Bed")
        bottom_label.setAlignment(Qt.AlignCenter)
        slider_box.addWidget(bottom_label)
        view_row.addLayout(slider_box)
        right_layout.addLayout(view_row, stretch=6)

        camera_buttons = QHBoxLayout()
        viewport_ref = weakref.ref(self.viewport)
        for name in ('Isometric', 'Top', 'Front', 'Right', 'Fit model'):
            button = QPushButton(name)
            # Avoid a signal -> lambda -> MainWindow reference cycle. That
            # could keep native Qt widgets alive after QApplication exits.
            def go_to_view(checked=False, view=name, ref=viewport_ref):
                target = ref()
                if target is not None:
                    target.set_view_preset(view)
            button.clicked.connect(go_to_view)
            camera_buttons.addWidget(button)
        camera_buttons.addStretch(1)
        right_layout.addLayout(camera_buttons)
        controls_hint = QLabel('RMB orbit   ·   MMB pan   ·   Wheel zoom   ·   LMB selection')
        controls_hint.setStyleSheet('color: #aeb9c6;')
        right_layout.addWidget(controls_hint)

        layer_nav = QHBoxLayout()
        self.layer_select = QSpinBox()
        self.layer_select.setRange(1, 1)
        self.layer_select.setEnabled(False)
        self.layer_select.valueChanged.connect(self.on_spin_changed)
        self.slice_position = QLabel("Import a model to preview layers")
        layer_nav.addWidget(QLabel("Layer"))
        layer_nav.addWidget(self.layer_select)
        layer_nav.addWidget(self.slice_position, stretch=1)
        right_layout.addLayout(layer_nav)

        bottom = QHBoxLayout()
        self.preview_label = QLabel("Export masks to preview channel images")
        self.preview_label.setMinimumSize(180, 100)
        self.preview_label.setMaximumHeight(125)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background: #18212d; color: #b0bfcf;")
        bottom.addWidget(self.preview_label, stretch=1)
        channel_column = QVBoxLayout()
        channel_column.addWidget(QLabel("Exported mask channel"))
        self.channel = QComboBox()
        self.channel.addItems(["combined", "white", "color", "varnish", "support"])
        self.channel.currentTextChanged.connect(self.show_slice)
        channel_column.addWidget(self.channel)
        channel_column.addWidget(QLabel("Preview only — not a printer-ready RIP job."))
        channel_column.addStretch(1)
        bottom.addLayout(channel_column)
        right_layout.addLayout(bottom, stretch=1)
        split.addWidget(right)
        split.setSizes([370, 960])
        return panel

    def report_opengl_failure(self, problem: str):
        self.log.appendPlainText("OpenGL viewer initialization failed: " + problem)
        self.statusBar().showMessage("GPU viewer unavailable: check OpenGL drivers")
        QMessageBox.warning(
            self, "GPU viewer unavailable",
            "UV3D could not start OpenGL 3.3. Update your graphics driver or "
            "run the software fallback.\n\n" + problem)

    def reset_view(self):
        self.viewport.reset_camera()

    def build_simulator(self):
        self.simulator = SimulationProcessPanel(
            ROOT, self, gpu_enabled=getattr(ModelViewport, "FULL_RESOLUTION", False))
        return self.simulator

    def build_machine(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        profile = MachineProfile()
        info = QLabel(
            f"Machine profile: {profile.name}\n"
            f"Bed (nominal): {profile.bed_x_mm:.0f} × {profile.bed_y_mm:.0f} mm\n"
            f"Head count: {profile.heads}\n"
            f"Observed Z scale: {profile.z_pulses_per_mm:.6f} motor pulses/mm\n"
            f"Factory manual step: {profile.z_manual_increment_mm} mm\n"
            "\nHardware communication: DISABLED\n\n"
            "Z pulse calculations are theoretical commands; mechanical accuracy and UV cured height "
            "have not been measured. Laser, collision alarm, travel limits, ink, color channels and "
            "3D supports require machine-specific validation.\n\n"
            "The program cannot generate PrintExp .prt files. Rendered masks must not be sent "
            "directly to the printer or interpreted as completed ink calibration."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        update = QPushButton("Update UV3D from main (closes and restarts app)")
        update.clicked.connect(self.update_app)
        layout.addWidget(update)
        layout.addStretch(1)
        return panel

    def open_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose 3D model", str(WORKSPACE),
            "3D meshes (*.stl *.obj *.glb *.gltf *.3mf *.ply)")
        if filename:
            self.load_model_file(filename)

    def load_model_file(self, filename: str | Path):
        """Start an asynchronous import; returns immediately, keeping UI usable."""
        if self.slicer.running or self.import_thread is not None:
            QMessageBox.warning(self, "Busy", "Finish the current export/import before opening another model.")
            return
        self.open_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.model_info.setText(f"Loading model:\n{Path(filename).name}\nYou can still use the window.")
        self.statusBar().showMessage("Loading 3D model in the background...")
        self.progress.setRange(0, 0)
        self.log.appendPlainText(f"Import started: {filename}")
        thread = QThread(self)
        worker = ImportWorker(str(filename))
        self.import_thread = thread
        self.import_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.stage.connect(self.import_stage)
        worker.finished.connect(self.import_finished)
        worker.error.connect(self.import_failed)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.import_thread_finished)
        thread.start()

    def import_stage(self, stage: str):
        self.model_info.setText(stage)
        self.statusBar().showMessage(stage)

    def import_finished(self, filename, mesh, preview_geometry, warnings, durations):
        self.mesh_path = Path(filename)
        self.source_mesh = mesh
        self.source_preview = preview_geometry
        self.model_height.blockSignals(True)
        self.model_height.setValue(min(max(float(mesh.extents[2]), 0.05), 180.0))
        self.model_height.setEnabled(True)
        self.model_height.blockSignals(False)
        self.viewport.set_scene(
            mesh, preview_geometry, self.model_height.value(), reset_camera=True)
        self.on_height_changed(reset_layer=True)
        self.export_button.setEnabled(True)
        # This progress bar is for MASK GENERATION, not for model import.
        # Previously import completion set it to 100% even when no masks
        # existed, making a successfully opened STL look like a finished job.
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.statusBar().showMessage("3D model loaded; masks not yet generated")
        self.log.appendPlainText(
            f"Import complete: {len(mesh.faces):,} faces. "
            f"Geometry {durations[0]:.2f} s, preview {durations[1]:.2f} s.")
        for warning in warnings:
            self.log.appendPlainText("WARNING: " + warning)

    def import_failed(self, problem: str):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.statusBar().showMessage("Failed to import 3D model")
        self.log.appendPlainText("ERROR importing model: " + problem)
        QMessageBox.critical(self, "Model import failed", problem)

    def import_thread_finished(self):
        self.import_thread = None
        self.import_worker = None
        self.open_button.setEnabled(True)
        self.export_button.setEnabled(self.mesh_path is not None and not self.slicer.running)

    def on_height_changed(self, *_, reset_camera=False, reset_layer=False):
        if self.source_mesh is None:
            return
        try:
            if self.viewport.original_mesh is not self.source_mesh:
                self.viewport.set_scene(
                    self.source_mesh, self.source_preview,
                    self.model_height.value(), reset_camera=reset_camera)
            else:
                self.viewport.set_model_height(self.model_height.value())
            self.preview_mesh = self.viewport.mesh
        except Exception as exc:
            self.log.appendPlainText("Unable to resize model: " + str(exc))
            return
        orig = self.source_mesh.extents
        dims = self.preview_mesh.extents
        scale = dims[2] / orig[2]
        self.model_info.setText(
            f"{self.mesh_path.name if self.mesh_path else 'Model'}\n"
            f"Original: {orig[0]:.2f} × {orig[1]:.2f} × {orig[2]:.2f} mm\n"
            f"Resized: {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm\n"
            f"Uniform scale: {scale:.4f}×    Faces: {len(self.source_mesh.faces):,}\n"
            f"Watertight: {self.source_mesh.is_watertight}"
        )
        self.refresh_layer_range(reset_index=reset_layer)

    def refresh_layer_range(self, *_, reset_index=False):
        if self.preview_mesh is None:
            return
        n = max(1, math.ceil(float(self.preview_mesh.extents[2]) / self.layer.value()))
        self.layer_count.setText(f"{n:,}")
        previous = n if reset_index else self.layer_select.value()
        self.layer_select.blockSignals(True)
        self.slice_slider.blockSignals(True)
        self.layer_select.setRange(1, n)
        self.slice_slider.setRange(1, n)
        self.layer_select.setValue(min(previous if self.layer_select.isEnabled() else n, n))
        self.slice_slider.setValue(self.layer_select.value())
        self.slice_slider.setEnabled(True)
        self.layer_select.setEnabled(True)
        self.layer_select.blockSignals(False)
        self.slice_slider.blockSignals(False)
        self.on_layer_changed(self.layer_select.value())
        self.invalidate_export()

    def on_slice_mode_changed(self, checked):
        self.viewport.set_slice_mode(bool(checked))
        self.slice_slider.setEnabled(self.preview_mesh is not None and bool(checked))
        self.layer_select.setEnabled(self.preview_mesh is not None and bool(checked))
        self.on_layer_changed(self.layer_select.value())

    def on_slider_changed(self, index: int):
        if self.layer_select.value() != index:
            self.layer_select.setValue(index)
        else:
            self.on_layer_changed(index)

    def on_spin_changed(self, index: int):
        if self.slice_slider.value() != index:
            self.slice_slider.setValue(index)
        else:
            self.on_layer_changed(index)

    def on_layer_changed(self, index: int):
        if self.preview_mesh is None:
            return
        max_z = float(self.preview_mesh.extents[2])
        z = min(max_z, (index - 0.5) * self.layer.value())
        self.slice_position.setText(f"{index:,} / {self.layer_select.maximum():,}   ·   Z = {z:.3f} mm")
        self.viewport.set_slice_mode(self.slice_checkbox.isChecked())
        self.viewport.set_layer_z(z)
        self.show_slice()

    def invalidate_export(self, *_):
        if self.job_data is None:
            return
        self.job_data = None
        self.job_path = None
        self.viewport.set_support_preview_lines(None)
        self.viewport.set_show_support_preview(False)
        self.show_support.setEnabled(False)
        self.output_label.setText("Model/settings changed — regenerate masks.")
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Export masks to preview updated channels")
        self.simulator.pause()
        # A previously exported simulation remains available for comparison.
        # Changing current slicer settings does not silently change old masks.

    def start_export(self):
        if not self.mesh_path or self.slicer.running or self.import_thread is not None:
            return
        settings = SliceSettings(
            layer_mm=self.layer.value(), target_height_mm=self.model_height.value(),
            dpi=self.dpi.value(), shell_mm=self.shell.value(),
            body_material=self.body.currentText(),
            support_enabled=self.support_enabled.isChecked(),
            support_casing_mm=self.support_casing.value(),
            placement_x_mm=self.x_pos.value(), placement_y_mm=self.y_pos.value()
        )
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.slicer.start(
            self.mesh_path, settings,
            compatibility_mode=self.compatibility_mode.isChecked())

    def on_slicer_running_changed(self, running):
        self.export_button.setEnabled(not running and self.mesh_path is not None)
        self.cancel_export_button.setEnabled(running)
        self.compatibility_mode.setEnabled(not running)
        self.open_button.setEnabled(not running and self.import_thread is None)
        for control in (self.model_height, self.layer, self.dpi, self.shell,
                        self.body, self.x_pos, self.y_pos, self.support_enabled):
            control.setEnabled(not running and self.mesh_path is not None)
        self.support_casing.setEnabled(not running and self.support_enabled.isChecked())

    def on_slicer_progress(self, current, total, seconds):
        total = max(1, total)
        self.progress.setValue(round(100 * current / total))
        self.statusBar().showMessage(
            f'Generating layer {current:,}/{total:,} ({seconds:.1f}s)')

    def on_slicer_stage(self, stage):
        self.log.appendPlainText(stage)
        self.statusBar().showMessage(stage)

    def on_slicer_failed(self, problem, log_path):
        self.progress.setValue(0)
        self.log.appendPlainText(f'Layer generation failed: {problem}')
        self.log.appendPlainText(f'Diagnostic log: {log_path}')
        self.statusBar().showMessage('Layer generation failed; see diagnostic log')
        QMessageBox.warning(self, 'UV3D layer generation failed',
                            f'{problem}\n\nCrash report: {log_path}\n'
                            'The app remains open. Incomplete output is retained.\n'
                            'Try Compatibility slicing if the model fails again.')

    def cancel_export(self):
        if self.slicer.running and QMessageBox.question(
            self, 'Cancel layer generation',
            'Stop the worker? Incomplete layers will be preserved for diagnosis.') == QMessageBox.Yes:
            self.slicer.cancel()

    def export_finished(self, job_path: str):
        self.job_path = Path(job_path)
        self.job_data = json.loads(self.job_path.read_text(encoding="utf-8"))
        n = len(self.job_data["layers"])
        self.show_slice()
        summary = f"{n:,} solid model layers exported to:\n{self.job_path.parent}"
        support = self.job_data.get("support", {})
        if support.get("enabled"):
            stats = support.get("statistics", {})
            summary += (
                f"\nConformal support: {stats.get('support_volume_mm3', 0) / 1000:.1f} cm³"
                f" | {stats.get('saving_vs_cube_percent', 0):.1f}% less than rectangular enclosure"
            )
        self.output_label.setText(summary)
        self.log.appendPlainText(f"Exported {n} OFFLINE layers, scaled height {self.job_data['scaled_dimensions_mm'][2]:.3f} mm.")
        self.statusBar().showMessage("Offline layer masks generated successfully")
        for warning in self.job_data["warnings"]:
            self.log.appendPlainText("WARNING: " + warning)
        self.simulator.load_job(self.job_path, self.source_mesh, self.source_preview)
        self.log.appendPlainText("Simulation prepared: open Print queue — simulation tab.")
        self.viewport.set_support_preview_lines(None)
        support = self.job_data.get("support", {})
        if support.get("enabled"):
            self.show_support.setEnabled(
                bool(getattr(ModelViewport, "FULL_RESOLUTION", False)))
            if self.show_support.isEnabled():
                self.start_support_preview(self.job_path)

    def export_failed(self, problem: str):
        self.log.appendPlainText("ERROR: " + problem)
        QMessageBox.warning(self, "Slice export failed", problem)

    def start_support_preview(self, job_path: Path):
        if self.support_preview_thread is not None:
            return
        thread = QThread(self)
        worker = SupportPreviewWorker(job_path)
        self.support_preview_thread = thread
        self.support_preview_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(self.support_preview_ready)
        worker.error.connect(self.support_preview_failed)
        worker.ready.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.support_preview_thread_finished)
        self.log.appendPlainText("Preparing sparse 3D support wireframe in background...")
        thread.start()

    def support_preview_ready(self, path, lines):
        if self.job_path is not None and str(self.job_path) == path:
            self.viewport.set_support_preview_lines(lines)
            self.viewport.set_show_support_preview(self.show_support.isChecked())
            self.log.appendPlainText(
                f"3D support wireframe ready: {len(lines) // 2:,} line segments.")

    def support_preview_failed(self, problem):
        self.log.appendPlainText("3D support wireframe unavailable: " + problem)

    def support_preview_thread_finished(self):
        self.support_preview_thread = None
        self.support_preview_worker = None

    def show_slice(self, *_):
        if not self.job_data or not self.job_path:
            return
        idx = min(self.layer_select.value() - 1, len(self.job_data["layers"]) - 1)
        layer = self.job_data["layers"][idx]
        channel = self.channel.currentText()
        root = self.job_path.parent
        try:
            if channel in ("combined", "support"):
                # Colors are UI preview ONLY. The raw support PNG remains an
                # unmapped, separate grayscale material channel.
                from PIL import Image
                from PySide6.QtGui import QImage
                part_name = self.job_data["settings"]["body_material"]
                model = Image.open(root / layer[part_name]).convert("L")
                preview = Image.new("RGBA", model.size, (25, 33, 46, 255))
                support_file = layer.get("support")
                if support_file:
                    support_mask = Image.open(root / support_file).convert("L")
                    preview.paste((153, 106, 206, 255), (0, 0), support_mask)
                if channel == "combined":
                    preview.paste((226, 232, 239, 255), (0, 0), model)
                    color = Image.open(root / layer["color"]).convert("RGBA")
                    preview.alpha_composite(color)
                rgba = preview.tobytes()
                qim = QImage(rgba, preview.width, preview.height,
                             preview.width * 4, QImage.Format_RGBA8888).copy()
                image = QPixmap.fromImage(qim)
            else:
                image = QPixmap(str(root / layer[channel]))
            if image.isNull():
                raise ValueError("Could not decode mask image")
            self.preview_label.setPixmap(
                image.scaled(330, 165, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as exc:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(f"Preview unavailable: {exc}")

    def closeEvent(self, event):
        # The viewer is a separate process. Closing UV3D terminates that child
        # first; it must never prevent closing the main application.
        self.simulator.shutdown()
        if (self.support_preview_thread is not None
                and self.support_preview_thread.isRunning()):
            QMessageBox.information(self, "Support preview preparing",
                                    "Let the support wireframe finish loading before closing.")
            event.ignore()
            return
        if self.import_thread is not None and self.import_thread.isRunning():
            QMessageBox.warning(self, "Import in progress", "Wait for the import to finish before closing UV3D.")
            event.ignore()
            return
        if self.slicer.running or (self.worker_thread is not None and self.worker_thread.isRunning()):
            QMessageBox.warning(self, "Export in progress", "Finish exporting layers before closing UV3D.")
            event.ignore()
            return
        event.accept()

    def update_app(self):
        script = ROOT / "Update-UV3D.ps1"
        if not script.is_file() or sys.platform != "win32":
            QMessageBox.warning(self, "Update unavailable", "Windows updater script not found.")
            return
        if self.slicer.running or self.import_thread is not None or self.simulator.busy:
            QMessageBox.warning(self, "Busy", "Finish model import/export/simulation loading before updating.")
            return
        if QMessageBox.question(self, "Update from main",
                                "UV3D will close, fast-forward from GitHub/main, check dependencies "
                                "and reopen. Continue?") != QMessageBox.Yes:
            return
        # Display a separate visible PowerShell console rather than silently
        # detaching the updater. -NoExit retains the final status/errors.
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        try:
            subprocess.Popen(["powershell.exe", "-NoProfile", "-NoExit",
                              "-ExecutionPolicy", "Bypass", "-File",
                              str(script), "-WaitForPid", str(os.getpid())],
                             cwd=str(ROOT), creationflags=flags, close_fds=True)
        except Exception as exc:
            QMessageBox.critical(self, "Update launch error", str(exc))
            return
        QApplication.quit()


def main():
    install_gui_diagnostics(WORKSPACE)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
