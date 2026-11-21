"""UV3D desktop GUI. OFFLINE ONLY — not connected to any printer hardware."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)
from .core import MachineProfile, SliceSettings, load_mesh, prepare_mesh, slice_model
from .viewport import ModelViewport

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"


class SliceWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, src: Path, dst: Path, settings: SliceSettings):
        super().__init__()
        self.src, self.dst, self.settings = src, dst, settings

    def run(self):
        try:
            result = slice_model(self.src, self.dst, self.settings,
                                 lambda current, total: self.progress.emit(current, total))
            self.finished.emit(str(result))
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
        self.sim_position = 0
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.simulation_tick)
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
        form.addRow("Body mask", self.body)
        self.x_pos = QDoubleSpinBox()
        self.x_pos.setRange(0, 600)
        self.x_pos.setValue(10)
        form.addRow("Raster offset X (mm)", self.x_pos)
        self.y_pos = QDoubleSpinBox()
        self.y_pos.setRange(0, 900)
        self.y_pos.setValue(10)
        form.addRow("Raster offset Y (mm)", self.y_pos)
        controls.addWidget(settings)
        for ctl in (self.dpi, self.shell, self.body, self.x_pos, self.y_pos):
            if isinstance(ctl, QComboBox):
                ctl.currentTextChanged.connect(self.invalidate_export)
            else:
                ctl.valueChanged.connect(self.invalidate_export)

        self.export_button = QPushButton("Generate offline layer masks")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.start_export)
        controls.addWidget(self.export_button)
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
        title = QLabel("3D build preview")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        top.addWidget(title)
        top.addStretch(1)
        self.slice_checkbox = QCheckBox("Show layers / ghost silhouette")
        self.slice_checkbox.setChecked(True)
        self.slice_checkbox.toggled.connect(self.on_slice_mode_changed)
        top.addWidget(self.slice_checkbox)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        top.addWidget(reset)
        right_layout.addLayout(top)

        view_row = QHBoxLayout()
        self.viewport = ModelViewport()
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
        self.channel.addItems(["color", "white", "varnish"])
        self.channel.currentTextChanged.connect(self.show_slice)
        channel_column.addWidget(self.channel)
        channel_column.addWidget(QLabel("Preview only — not a printer-ready RIP job."))
        channel_column.addStretch(1)
        bottom.addLayout(channel_column)
        right_layout.addLayout(bottom, stretch=1)
        split.addWidget(right)
        split.setSizes([370, 960])
        return panel

    def reset_view(self):
        self.viewport.reset_camera()

    def build_simulator(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel("SIMULATION ONLY — does not connect to PrintExp, FlexiPRINT, or motors.")
        label.setStyleSheet("font-weight: bold; color: #b56700;")
        layout.addWidget(label)
        self.sim_detail = QLabel("Generate a job first.")
        layout.addWidget(self.sim_detail)
        self.sim_progress = QProgressBar()
        layout.addWidget(self.sim_progress)
        buttons = QHBoxLayout()
        start = QPushButton("Start offline simulation")
        stop = QPushButton("Stop simulation")
        start.clicked.connect(self.start_simulation)
        stop.clicked.connect(self.sim_timer.stop)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        layout.addLayout(buttons)
        layout.addStretch(1)
        return panel

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
        """Public entry point shared by Open dialog and GUI smoke tests."""
        if self.worker_thread is not None:
            QMessageBox.warning(self, "Busy", "Finish generating layers before importing a new model.")
            return
        try:
            mesh, warnings = load_mesh(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Model import failed", str(exc))
            return
        self.mesh_path = Path(filename)
        self.source_mesh = mesh
        self.model_height.blockSignals(True)
        # Actual model height can be greater than the tested nominal clearance.
        # Clamp only the editor target; never silently export the original geometry.
        self.model_height.setValue(min(max(float(mesh.extents[2]), 0.05), 180.0))
        self.model_height.setEnabled(True)
        self.model_height.blockSignals(False)
        self.on_height_changed(reset_camera=True, reset_layer=True)
        self.export_button.setEnabled(True)
        self.log.appendPlainText(f"Imported: {filename}")
        for warning in warnings:
            self.log.appendPlainText("WARNING: " + warning)

    def on_height_changed(self, *_, reset_camera=False, reset_layer=False):
        if self.source_mesh is None:
            return
        target = self.model_height.value()
        try:
            self.preview_mesh = prepare_mesh(self.source_mesh, target)
            self.viewport.set_mesh(self.preview_mesh, reset_camera=reset_camera)
        except Exception as exc:
            self.log.appendPlainText("Unable to resize mesh: " + str(exc))
            return
        orig = self.source_mesh.extents
        dims = self.preview_mesh.extents
        scale = dims[2] / orig[2]
        self.model_info.setText(
            f"{self.mesh_path.name if self.mesh_path else 'Model'}\n"
            f"Original: {orig[0]:.2f} × {orig[1]:.2f} × {orig[2]:.2f} mm\n"
            f"Resized: {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm\n"
            f"Uniform scale: {scale:.4f}×    Faces: {len(self.preview_mesh.faces):,}\n"
            f"Watertight: {self.preview_mesh.is_watertight}"
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
        self.output_label.setText("Model/settings changed — regenerate masks.")
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Export masks to preview updated channels")
        self.sim_timer.stop()
        self.sim_detail.setText("Previous job invalidated by model/settings change.")

    def start_export(self):
        if not self.mesh_path or self.worker_thread is not None:
            return
        settings = SliceSettings(
            layer_mm=self.layer.value(), target_height_mm=self.model_height.value(),
            dpi=self.dpi.value(), shell_mm=self.shell.value(),
            body_material=self.body.currentText(),
            placement_x_mm=self.x_pos.value(), placement_y_mm=self.y_pos.value()
        )
        job_name = self.mesh_path.stem[:50] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = WORKSPACE / "jobs" / job_name
        self.export_button.setEnabled(False)
        for control in (self.model_height, self.layer, self.dpi, self.shell, self.body,
                        self.x_pos, self.y_pos):
            control.setEnabled(False)
        self.progress.setValue(0)
        self.log.appendPlainText(f"Generating scaled offline export: {out}")
        thread = QThread(self)
        worker = SliceWorker(self.mesh_path, out, settings)
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(lambda current, total: self.progress.setValue(round(100 * current / total)))
        worker.finished.connect(self.export_finished)
        worker.error.connect(self.export_failed)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.thread_finished)
        self.worker_thread = thread
        thread.start()

    def thread_finished(self):
        self.worker_thread = None
        self.worker = None
        for control in (self.model_height, self.layer, self.dpi, self.shell, self.body,
                        self.x_pos, self.y_pos):
            control.setEnabled(self.mesh_path is not None)
        self.export_button.setEnabled(self.mesh_path is not None)

    def export_finished(self, job_path: str):
        self.job_path = Path(job_path)
        self.job_data = json.loads(self.job_path.read_text(encoding="utf-8"))
        n = len(self.job_data["layers"])
        self.show_slice()
        self.output_label.setText(f"{n:,} scaled layers exported to:\n{self.job_path.parent}")
        self.log.appendPlainText(f"Exported {n} OFFLINE layers, scaled height {self.job_data['scaled_dimensions_mm'][2]:.3f} mm.")
        for warning in self.job_data["warnings"]:
            self.log.appendPlainText("WARNING: " + warning)
        self.sim_detail.setText(f"Loaded offline job: {n} layers")

    def export_failed(self, problem: str):
        self.log.appendPlainText("ERROR: " + problem)
        QMessageBox.warning(self, "Slice export failed", problem)

    def show_slice(self, *_):
        if not self.job_data or not self.job_path:
            return
        idx = min(self.layer_select.value() - 1, len(self.job_data["layers"]) - 1)
        layer = self.job_data["layers"][idx]
        image = QPixmap(str(self.job_path.parent / layer[self.channel.currentText()]))
        if image.isNull():
            self.preview_label.setText("Unable to load preview")
            return
        self.preview_label.setPixmap(image.scaled(260, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_simulation(self):
        if not self.job_data:
            QMessageBox.information(self, "No job", "Generate an offline job before simulating.")
            return
        self.sim_position = 0
        self.sim_progress.setValue(0)
        self.sim_timer.start(100)

    def simulation_tick(self):
        total = len(self.job_data["layers"])
        self.sim_position += 1
        self.sim_progress.setValue(round(100 * self.sim_position / total))
        self.sim_detail.setText(f"Simulated layer {self.sim_position}/{total}; no machine commands issued.")
        if self.sim_position >= total:
            self.sim_timer.stop()

    def closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Export in progress", "Finish exporting layers before closing UV3D.")
            event.ignore()
            return
        event.accept()

    def update_app(self):
        script = ROOT / "Update-UV3D.ps1"
        if not script.is_file() or sys.platform != "win32":
            QMessageBox.warning(self, "Update unavailable", "Windows updater script not found.")
            return
        if self.worker_thread is not None:
            QMessageBox.warning(self, "Busy", "Finish layer generation before updating.")
            return
        if QMessageBox.question(self, "Update from main",
                                "UV3D will close, fast-forward from GitHub/main, check dependencies "
                                "and reopen. Continue?") != QMessageBox.Yes:
            return
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                              str(script), "-WaitForPid", str(os.getpid())],
                             cwd=str(ROOT), creationflags=flags, close_fds=True)
        except Exception as exc:
            QMessageBox.critical(self, "Update launch error", str(exc))
            return
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
