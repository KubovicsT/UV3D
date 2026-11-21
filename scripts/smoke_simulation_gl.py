"""Exercise the live Qt OpenGL print simulator under Xvfb/Mesa.

This test intentionally draws support, body, color and the virtual printhead,
then rapidly seeks between exported layers. It catches real driver-side
GL_INVALID_OPERATION/value errors that pure unit tests cannot reproduce.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_OPENGL", "desktop")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from uv3d.gpu_viewport import prepare_gpu_geometry
from uv3d.simulation_engine import PrintSimulation
from uv3d.simulation_viewport import SimulationViewport


def main():
    app = QApplication.instance() or QApplication([])
    mesh = trimesh.creation.box(extents=(6.0, 5.0, 2.0))
    mesh.visual.face_colors = [202, 112, 66, 255]
    layers = [
        {"target_stack_height_mm": .5 * (i+1)} for i in range(4)
    ]
    manifest = {
        "settings": {"layer_mm": .5, "dpi": 100, "margin_mm": 2.0,
                     "body_material": "white"},
        "effective_raster_margin_mm": 2.0,
        "raster_size_px": [42, 36],
        "model_bounds_mm": [[-3,-2.5,0],[3,2.5,2]],
        "scaled_dimensions_mm": [6,5,2],
        "layers": layers,
    }
    errors = []
    viewport = SimulationViewport()
    viewport.opengl_failed.connect(errors.append)
    viewport.resize(640, 480)
    viewport.show()
    for _ in range(8):
        app.processEvents()
    if not viewport._gpu_ready:
        raise RuntimeError("No OpenGL 3.3 core context: " + repr(errors))
    viewport.set_simulation_scene(mesh, prepare_gpu_geometry(mesh), manifest)
    # A small 3D support wall, 2 triangles.
    wall = np.array([
        [3,-2,0],[3,2,0],[3,2,2],
        [3,-2,0],[3,2,2],[3,-2,2],
    ], dtype=np.float32)
    viewport.set_support_surface(wall)
    viewport.set_support_tint((165,112,220),.28)
    viewport.set_model_tint((220,185,155))
    viewport.set_show_future(True)
    plan = PrintSimulation(manifest)
    for n in range(180):
        frame = plan.get_frame((n*73 % 180)/179)
        body = np.zeros((36,42),dtype=np.uint8)
        body[7:28,10:30] = 255
        color = np.zeros((36,42,4),dtype=np.uint8)
        color[7:28,10:30] = [235,105,55,255]
        support = np.zeros((36,42),dtype=np.uint8)
        support[7:28,5:10] = 255
        viewport.set_layer_bitmaps(frame.layer_index,body,color,support)
        viewport.set_frame(frame)
        app.processEvents()
        if errors:
            raise RuntimeError("OpenGL playback error:\n" + "\n".join(errors))
    viewport.close()
    app.processEvents()
    print("Real OpenGL simulation rendering and 180 non-sequential seeks: PASS")


if __name__ == "__main__":
    main()
