"""Crash-isolated interactive 3D simulator process.

Run with bundled UV3D Python:
    python -m uv3d.simulation_runner --job .../job.json --root X:/UV3D

A native OpenGL/graphics-driver/Qt failure terminates only this subprocess.
The main UV3D window supervises the process and reports its exit status.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import faulthandler
import os
from pathlib import Path
import sys
import traceback

_LOG_HANDLE = None


def install_crash_log(root: Path):
    global _LOG_HANDLE
    folder = root / "workspace" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "uv3d-simulation-native.log"
    _LOG_HANDLE = log.open("a", encoding="utf-8", buffering=1)
    _LOG_HANDLE.write(
        f"\n=== Simulation launched {datetime.now().isoformat()} "
        f"PID={os.getpid()} ===\n")
    try:
        faulthandler.enable(_LOG_HANDLE, all_threads=True)
    except (RuntimeError, ValueError) as exc:
        _LOG_HANDLE.write(f"Fatal handler unavailable: {exc}\n")
    original = sys.excepthook

    def uncaught(exc_type, error, tb):
        _LOG_HANDLE.write(
            "UNCAUGHT:\n" + "".join(traceback.format_exception(exc_type,error,tb)))
        original(exc_type, error, tb)

    sys.excepthook = uncaught
    return log


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--no-gpu", action="store_true",
                        help="Use timeline and job data without any OpenGL widget")
    opts = parser.parse_args(argv)
    root = Path(opts.root).resolve()
    log = install_crash_log(root)
    # This is deliberately BEFORE importing Qt and OpenGL; if Qt itself
    # aborts on graphics setup, faulthandler and the parent monitor can log it.
    _LOG_HANDLE.write(f"Job: {opts.job}\nGPU mode: {not opts.no_gpu}\n")
    _LOG_HANDLE.flush()

    from PySide6.QtCore import Qt, QTimer, qInstallMessageHandler
    from PySide6.QtWidgets import QApplication
    from .simulation_widget import SimulationWidget

    # Qt warnings (especially QBasicTimer cross-thread and QObject ownership
    # diagnostics) are native-side signals: they otherwise disappear when the
    # child exits before Python's exception hook can run. Keep this callback
    # strongly referenced for the entire lifetime of the process.
    import threading
    def qt_message_handler(kind, context, message):
        location = (f"{context.file}:{context.line}" if context.file else "Qt")
        line = (f"{datetime.now().isoformat(timespec='milliseconds')} "
                f"[thread={threading.get_ident()}] {location} "
                f"Qt({kind}): {message}\n")
        try:
            _LOG_HANDLE.write(line)
            _LOG_HANDLE.flush()
        except Exception:
            pass
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass

    previous_handler = qInstallMessageHandler(qt_message_handler)
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    widget = SimulationWidget(root, gpu_enabled=not opts.no_gpu)
    widget.setWindowTitle("UV3D — 3D print simulation (isolated)")
    widget.resize(1420, 860)
    widget.show()
    widget.load_job(opts.job)
    # Controlled process-exit probes are used ONLY by CI to verify that
    # a sudden child process death does not terminate the parent UV3D GUI.
    test_exit = os.environ.get("UV3D_SIMULATION_TEST_EXIT_MS")
    if test_exit is not None:
        delay = max(1, int(test_exit))
        if os.environ.get("UV3D_SIMULATION_TEST_ABORT") == "1":
            QTimer.singleShot(delay, lambda: os._exit(91))
        else:
            QTimer.singleShot(delay, app.quit)
    _LOG_HANDLE.write("Simulation window opened\n")
    _LOG_HANDLE.flush()
    exit_code = int(app.exec())
    _LOG_HANDLE.write(f"Simulation GUI exited normally: {exit_code}\n")
    _LOG_HANDLE.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
