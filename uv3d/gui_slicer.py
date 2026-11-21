"""GUI-side controller for crash-isolated layer-mask generation.

Native slicer exceptions (e.g. C++ faults) cannot be caught reliably by
Python within the same application process. QProcess isolates that risk.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .core import SliceSettings


class SlicerController(QObject):
    progress = Signal(int, int, float)
    stage = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str, str)
    running_changed = Signal(bool)

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = Path(root).resolve()
        self.process: QProcess | None = None
        self.log_path: Path | None = None
        self.request_path: Path | None = None
        self._output_dir: Path | None = None
        self._stream = ""
        self._stderr = ""
        self._result = ""
        self._error = ""

    @property
    def running(self) -> bool:
        return self.process is not None

    def start(self, source: Path, settings: SliceSettings, compatibility_mode: bool = False) -> bool:
        if self.running:
            return False
        settings.validate()
        workspace = self.root / "workspace"
        task_dir = workspace / "tasks"
        log_dir = workspace / "logs"
        for directory in (task_dir, log_dir, workspace / "jobs"):
            directory.mkdir(parents=True, exist_ok=True)
        name = source.stem[:48] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.request_path = task_dir / (name + ".json")
        self._output_dir = workspace / "jobs" / name
        self.log_path = log_dir / (name + ".log")
        request = {"source": str(Path(source).resolve()),
                   "output": str(self._output_dir),
                   "settings": asdict(settings),
                   "compatibility_mode": bool(compatibility_mode),
                   "log_dir": str(log_dir)}
        self.request_path.write_text(json.dumps(request, indent=2),
                                     encoding="utf-8")
        self._stream = self._stderr = self._result = self._error = ""
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(self.root))
        # Embedded Python uses its local ._pth; system Python used by CI/dev
        # needs a known source root so the isolated worker can import UV3D.
        env = QProcessEnvironment.systemEnvironment()
        source_root = str(Path(__file__).resolve().parent.parent)
        existing = env.value("PYTHONPATH", "")
        import os
        env.insert("PYTHONPATH", source_root + (os.pathsep + existing if existing else ""))
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self._stdout_ready)
        process.readyReadStandardError.connect(self._stderr_ready)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._process_finished)
        self.running_changed.emit(True)
        self.stage.emit("Starting isolated mask generator...")
        process.start(sys.executable, ["-m", "uv3d.slicer_process",
                                       "--request", str(self.request_path)])
        return True

    def _stdout_ready(self):
        if self.process is None:
            return
        self._stream += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        while "\n" in self._stream:
            line, self._stream = self._stream.split("\n", 1)
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except ValueError:
                self.stage.emit("Worker: " + line[:400])
                continue
            kind = data.get("type")
            if kind == "progress":
                self.progress.emit(int(data.get("current", 0)),
                                   int(data.get("total", 1)),
                                   float(data.get("elapsed_seconds", 0)))
            elif kind == "stage":
                self.stage.emit(str(data.get("message", "")))
            elif kind == "error":
                self._error = str(data.get("message", "Unknown error"))
                self.stage.emit("Worker error: " + self._error)
            elif kind == "finished":
                self._result = str(data.get("job", ""))

    def _stderr_ready(self):
        if self.process is not None:
            self._stderr = (self._stderr
                            + bytes(self.process.readAllStandardError()).decode(
                                "utf-8", errors="replace"))[-12000:]

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._error = "Failed to start Python layer worker"
            # Some Qt platforms don't emit finished after FailedToStart.
            QTimer.singleShot(0, self._failed_start)

    def _failed_start(self):
        process = self.process
        if process is not None and process.state() == QProcess.NotRunning:
            self._finalize(False, -1, "Process failed to start")

    def _process_finished(self, exit_code, exit_status):
        self._stdout_ready()
        self._stderr_ready()
        ok = (exit_code == 0 and exit_status == QProcess.NormalExit
              and self._result and Path(self._result).is_file())
        self._finalize(ok, exit_code, str(exit_status))

    def _finalize(self, ok, exit_code, status):
        process = self.process
        if process is None:
            return
        self.process = None
        process.deleteLater()
        self.running_changed.emit(False)
        if ok:
            self.succeeded.emit(self._result)
            return
        reason = self._error or (
            f"Native layer generator exited (code {exit_code}; status {status}).")
        if self._stderr:
            reason += "\nRecent worker output:\n" + self._stderr[-1200:]
        log = str(self.log_path) if self.log_path else "(unavailable)"
        self.failed.emit(reason, log)

    def cancel(self):
        process = self.process
        if process is None:
            return
        self._error = "Layer generation cancelled by the operator."
        process.terminate()
        QTimer.singleShot(
            2500, lambda p=process:
                p.kill() if p.state() != QProcess.NotRunning else None)
