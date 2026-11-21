"""Main UV3D simulation tab: supervise a completely separate 3D Qt process.

A PyOpenGL/Qt/graphics-driver access violation cannot be caught reliably in
the same Python interpreter. A process boundary is the only reliable way to
keep the slicer and model editor alive if the simulation process dies.
The complete existing 3D SimulationWidget runs in the child window.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QMessageBox
)


class SimulationProcessPanel(QWidget):
    def __init__(self, root: Path, parent=None, gpu_enabled=True):
        super().__init__(parent)
        self.root = Path(root).resolve()
        self.gpu_enabled = gpu_enabled
        self.job_path: Path | None = None
        self.process: QProcess | None = None
        self._running_job: Path | None = None
        self._log_path: Path | None = None
        self._stderr = ""
        self._stdout = ""
        self._shutting_down = False
        self._expected_exit = False

        layout = QVBoxLayout(self)
        label = QLabel("3D PRINT SIMULATION — crash-isolated viewer")
        label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(label)
        info = QLabel(
            "The interactive 3D simulation runs in its own window to protect "
            "UV3D from crashes in OpenGL, graphics drivers, or native Qt code. "
            "Play, timeline, printhead speed, replay speed, support color and "
            "transparency all work in the separate simulation window.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.job_label = QLabel("No completed layer-mask job selected.")
        self.job_label.setWordWrap(True)
        layout.addWidget(self.job_label)
        row = QHBoxLayout()
        self.launch_button = QPushButton("Open 3D simulation")
        self.launch_button.setEnabled(False)
        self.launch_button.clicked.connect(self.launch)
        row.addWidget(self.launch_button)
        self.stop_button = QPushButton("Close simulation")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        self.safe_mode = QCheckBox("Troubleshooting mode: disable GPU rendering")
        self.safe_mode.setToolTip(
            "Start the simulation timeline without an OpenGL viewport. "
            "If this does not crash but 3D mode does, the problem is in the GPU/Qt path.")
        layout.addWidget(self.safe_mode)
        self.status = QLabel("Open a completed job, or generate masks on the 3D slicer tab.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.log_label = QLabel("")
        self.log_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log_label.setWordWrap(True)
        layout.addWidget(self.log_label)
        layout.addStretch()

    @property
    def busy(self):
        # Keep UV3D update/close safe while a child is active.
        return self.process is not None

    def pause(self):
        # Compatibility with the old embedded SimulationWidget API.
        # The separate viewer has its own pause control.
        pass

    def load_job(self, job_path, source_mesh=None, gpu_geometry=None):
        # Never transfer giant meshes over IPC: reopen the source file in the
        # child using source_path in the exported job manifest/task request.
        path = Path(job_path).resolve()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            layers = manifest.get("layers", [])
            if not layers:
                raise ValueError("This job has no exported masks.")
        except Exception as exc:
            self.job_path = None
            self.job_label.setText(f"Cannot load simulation job: {exc}")
            self.launch_button.setEnabled(False)
            return
        self.job_path = path
        self.job_label.setText(f"{len(layers):,} layers — {path}")
        self.launch_button.setEnabled(True)
        self.status.setText(
            "Ready. Open the simulation in its own window; UV3D stays available.")
        # Do not reopen a running simulation with a different job implicitly.

    def launch(self):
        if self.job_path is None:
            QMessageBox.information(self, "No simulation job", "Export layer masks first.")
            return
        if self.process is not None:
            self.status.setText(
                "A simulation is already running. Close it before opening another job.")
            return
        logs = self.root / "workspace" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._log_path = logs / "uv3d-simulation-child.log"
        self._running_job = self.job_path
        self._shutting_down = False
        self._expected_exit = False
        self._stderr = ""
        self._stdout = ""
        env = QProcessEnvironment.systemEnvironment()
        source_root = str(Path(__file__).resolve().parent.parent)
        path = env.value("PYTHONPATH", "")
        env.insert("PYTHONPATH", source_root + (os.pathsep + path if path else ""))
        process = QProcess(self)
        process.setWorkingDirectory(str(self.root))
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardError.connect(self._read_stderr)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.errorOccurred.connect(self._error)
        process.finished.connect(self._finished)
        self.process = process
        self.launch_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.safe_mode.setEnabled(False)
        self.log_label.setText(f"Crash log: {self._log_path}")
        self.status.setText("Starting isolated 3D simulation...")
        self._record(f"\n=== Launch {datetime.now().isoformat()} ===\n"
                     f"Job: {self.job_path}\n"
                     f"Troubleshooting mode: {self.safe_mode.isChecked()}\n")
        arguments = ["-m", "uv3d.simulation_runner", "--job", str(self.job_path),
                     "--root", str(self.root)]
        if self.safe_mode.isChecked() or not self.gpu_enabled:
            arguments.append("--no-gpu")
        process.start(sys.executable, arguments)

    def _record(self, text):
        if self._log_path is not None:
            try:
                with self._log_path.open("a", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
            except OSError:
                pass

    def _read_stderr(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode(
            "utf-8", errors="replace")
        self._stderr = (self._stderr + data)[-8192:]
        if data:
            self._record(data)

    def _read_stdout(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        self._stdout = (self._stdout + data)[-2048:]
        if data:
            self._record(data)

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self.status.setText("Failed to start simulation. See crash log.")
            self._record(f"QProcess failed to start: {error}\n")
            # Windows may omit finished after a failed process launch.
            QTimer.singleShot(0, self._cleanup_failed_start)

    def _cleanup_failed_start(self):
        if self.process is not None and self.process.state() == QProcess.NotRunning:
            self._finish_process(-1, QProcess.CrashExit)

    def _finished(self, code, status):
        self._read_stderr()
        self._read_stdout()
        self._finish_process(code, status)

    def _finish_process(self, code, status):
        process = self.process
        if process is None:
            return
        self.process = None
        process.deleteLater()
        self.launch_button.setEnabled(self.job_path is not None)
        self.stop_button.setEnabled(False)
        self.safe_mode.setEnabled(True)
        try:
            abnormal = status != QProcess.NormalExit or int(code) != 0
        except Exception:
            abnormal = True
        if abnormal and not self._expected_exit:
            # Windows native exit codes may be negative signed equivalents.
            unsigned = int(code) & 0xFFFFFFFF
            message = (
                f"3D simulation crashed (exit {code}; Windows code 0x{unsigned:08X}). "
                "UV3D itself is safe. Try Troubleshooting mode; share the crash log.")
            self.status.setText(message)
            self._record(message + "\n")
            self.log_label.setText(f"Crash report: {self._log_path}")
        else:
            self.status.setText("Simulation closed. UV3D remains open.")
            self._record(f"Simulation ended (code={code}, status={status}).\n")
        self._expected_exit = False

    def stop(self):
        if self.process is None:
            return
        self._expected_exit = True
        self.process.terminate()
        QTimer.singleShot(
            3000,
            lambda process=self.process:
            process.kill() if process is not None and
            process.state() != QProcess.NotRunning else None)
        self.status.setText("Closing simulation...")

    def shutdown(self):
        """Synchronously stop child before the main Qt widgets are destroyed."""
        process = self.process
        if process is None:
            return
        self._expected_exit = True
        process.terminate()
        if not process.waitForFinished(3000):
            process.kill()
            process.waitForFinished(3000)
        self._read_stderr()
        self._read_stdout()
        if self.process is not None:
            self._finish_process(process.exitCode(), process.exitStatus())
