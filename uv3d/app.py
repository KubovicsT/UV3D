"""UV3D desktop GUI. OFFLINE ONLY: no hardware control in this version."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)
from .core import MachineProfile, SliceSettings, load_mesh, slice_model

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
        self.setWindowTitle("UV3D — Offline Slicer / Gongzheng H6090EI")
        self.resize(1200, 780)
        self.mesh_path: Path | None = None
        self.job_path: Path | None = None
        self.job_data: dict | None = None
        self.worker_thread = None
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
        settings = QGroupBox("Slice settings (unverified material targets)")
        form = QFormLayout(settings)
        self.layer = QDoubleSpinBox()
        self.layer.setDecimals(3)
        self.layer.setRange(0.01, 1.0)
        self.layer.setSingleStep(0.01)
        self.layer.setValue(0.05)
        form.addRow("Layer thickness (mm)", self.layer)
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
        self.preview_label = QLabel("Generate a job to preview slices")
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview_label)
        right_layout.addWidget(scroll)
        navigation = QHBoxLayout()
        self.layer_select = QSpinBox()
        self.layer_select.setEnabled(False)
        self.layer_select.valueChanged.connect(self.show_slice)
        self.channel = QComboBox()
        self.channel.addItems(["color", "white", "varnish"])
        self.channel.currentTextChanged.connect(self.show_slice)
        navigation.addWidget(QLabel("Layer"))
        navigation.addWidget(self.layer_select)
        navigation.addWidget(QLabel("Channel"))
        navigation.addWidget(self.channel)
        right_layout.addLayout(navigation)
        warning = QLabel("Preview masks are NOT PrintExp .prt files or calibrated spot-color channels.")
        warning.setWordWrap(True)
        right_layout.addWidget(warning)
        split.addWidget(right)
        split.setSizes([450, 700])
        return panel

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
        if not filename:
            return
        try:
            mesh, warnings = load_mesh(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Model import failed", str(exc))
            return
        self.mesh_path = Path(filename)
        dims = mesh.extents
        self.model_info.setText(f"{self.mesh_path.name}\n"
                                f"Dimensions: {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm\n"
                                f"Triangles: {len(mesh.faces):,}; watertight: {mesh.is_watertight}")
        self.export_button.setEnabled(True)
        self.log.appendPlainText(f"Imported: {filename}")
        for warning in warnings:
            self.log.appendPlainText("WARNING: " + warning)

    def start_export(self):
        if not self.mesh_path or self.worker_thread is not None:
            return
        settings = SliceSettings(layer_mm=self.layer.value(), dpi=self.dpi.value(),
                                 shell_mm=self.shell.value(), body_material=self.body.currentText(),
                                 placement_x_mm=self.x_pos.value(), placement_y_mm=self.y_pos.value())
        job_name = self.mesh_path.stem[:50] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = WORKSPACE / "jobs" / job_name
        self.export_button.setEnabled(False)
        self.progress.setValue(0)
        self.log.appendPlainText(f"Generating offline export: {out}")
        thread = QThread(self)
        worker = SliceWorker(self.mesh_path, out, settings)
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
        self.export_button.setEnabled(self.mesh_path is not None)

    def export_finished(self, job_path: str):
        self.job_path = Path(job_path)
        self.job_data = json.loads(self.job_path.read_text(encoding="utf-8"))
        n = len(self.job_data["layers"])
        self.layer_select.setRange(1, n)
        self.layer_select.setEnabled(True)
        self.layer_select.setValue(1)
        self.show_slice()
        self.output_label.setText(f"{n} layers exported to:\n{self.job_path.parent}")
        self.log.appendPlainText(f"Completed {n} OFFLINE layers.")
        for warning in self.job_data["warnings"]:
            self.log.appendPlainText("WARNING: " + warning)
        self.sim_detail.setText(f"Loaded offline job: {n} layers")

    def export_failed(self, problem: str):
        self.log.appendPlainText("ERROR: " + problem)
        QMessageBox.warning(self, "Slice export failed", problem)

    def show_slice(self, *_):
        if not self.job_data or not self.job_path:
            return
        idx = self.layer_select.value() - 1
        layer = self.job_data["layers"][idx]
        file = self.job_path.parent / layer[self.channel.currentText()]
        image = QPixmap(str(file))
        if image.isNull():
            self.preview_label.setText("Unable to load preview")
            return
        self.preview_label.setPixmap(image.scaled(700, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
