"""Crash diagnostics for UV3D desktop, separate from mask-worker logs."""
from __future__ import annotations

import faulthandler
import sys
import traceback
from pathlib import Path
from datetime import datetime

_FD = None
_ORIGINAL_HOOK = sys.excepthook


def install_gui_diagnostics(workspace: Path) -> Path:
    """Enable best-effort fatal stack dumps and explicit Python error dialogs."""
    global _FD
    directory = Path(workspace) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "uv3d-gui-crash.log"
    if _FD is None:
        _FD = path.open("a", encoding="utf-8", buffering=1)
        _FD.write(f"\n=== UV3D started {datetime.now().isoformat()} ===\n")
        try:
            faulthandler.enable(file=_FD, all_threads=True)
        except (RuntimeError, ValueError):
            pass

    def handle(exc_type, value, tb):
        message = "".join(traceback.format_exception(exc_type, value, tb))
        if _FD:
            _FD.write("UNHANDLED GUI EXCEPTION:\n" + message + "\n")
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                QMessageBox.critical(None, "UV3D unexpected error",
                                     f"{exc_type.__name__}: {value}\n\n"
                                     f"Full traceback:\n{path}")
        except Exception:
            pass
        # Never swallow the original traceback.
        _ORIGINAL_HOOK(exc_type, value, tb)

    sys.excepthook = handle
    return path
