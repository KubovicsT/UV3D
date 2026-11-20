import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from uv3d.app import MainWindow


def test_gui_starts_without_printer():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert "UV3D" in window.windowTitle()
    assert window.job_data is None
    assert window.worker_thread is None
    window.close()
