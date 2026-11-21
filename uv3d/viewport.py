"""Interactive slicer-style 3D viewport, software-rendered with Qt.

Uses QPainter for portable rendering on Windows systems without additional
graphics drivers/packages. Preview triangles are sampled for responsiveness;
cross-sections are computed from the FULL transformed mesh.
"""
from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import numpy as np
import trimesh
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .core import _face_colors, _polygons_from_segments


def clip_triangle_below(triangle: np.ndarray, z: float) -> np.ndarray:
    """Clip a 3D triangle to the half-space z <= cutoff."""
    result = []
    for i in range(3):
        a, b = triangle[i - 1], triangle[i]
        a_ok, b_ok = a[2] <= z, b[2] <= z
        if a_ok != b_ok:
            t = (z - a[2]) / (b[2] - a[2])
            result.append(a + t * (b - a))
        if b_ok:
            result.append(b)
    if not result:
        return np.empty((0, 3), dtype=float)
    return np.asarray(result, dtype=float)


def build_preview_geometry(mesh: trimesh.Trimesh, max_faces: int = 3000):
    """Return connected low-detail surface triangles, normals and source colors.

    Quantized-vertex clustering keeps neighboring faces joined. This is
    deliberately preview-only: full geometry is used for cross sections/export.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = _face_colors(mesh)
    if len(faces) <= max_faces:
        return vertices[faces], np.asarray(mesh.face_normals), colors

    extent = float(np.max(mesh.extents))
    bounds_min = mesh.bounds[0]

    def cluster(cell_size):
        cells = np.floor((vertices - bounds_min) / cell_size).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        centers = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
        np.add.at(centers, inverse, vertices)
        centers /= np.maximum(1, np.bincount(inverse))[:, None]
        new_faces = inverse[faces]
        keep = ((new_faces[:, 0] != new_faces[:, 1])
                & (new_faces[:, 0] != new_faces[:, 2])
                & (new_faces[:, 1] != new_faces[:, 2]))
        kept = np.flatnonzero(keep)
        if not len(kept):
            return None
        # Remove duplicate geometric triangles while retaining the corresponding
        # source face ID (to keep colors mapped to the right part of the model).
        _, ids = np.unique(np.sort(new_faces[kept], axis=1), axis=0,
                           return_index=True)
        selected = kept[ids]
        return centers[new_faces[selected]], colors[selected]

    lo = max(extent / 2500, 1e-8)
    hi = max(extent / 3, 1e-7)
    best = None
    # More cells = more detail. Find the highest usable triangle count.
    for _ in range(5):
        step = math.sqrt(lo * hi)
        candidate = cluster(step)
        count = len(candidate[0]) if candidate else 0
        if count > max_faces:
            lo = step
        else:
            hi = step
            if candidate and (best is None or count > len(best[0])):
                best = candidate
        if best is not None and len(best[0]) >= 0.91 * max_faces:
            break
    if best is None:
        # This is an emergency fallback only, never used for typical closed meshes.
        ids = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        return vertices[faces[ids]], np.asarray(mesh.face_normals)[ids], colors[ids]

    triangles, rgb = best
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    cross /= np.maximum(norms[:, None], 1e-12)
    return triangles, cross, rgb


class ModelViewport(QWidget):
    """Orbits with left drag, pans with right drag and zooms with wheel."""

    PREVIEW_TRIANGLES = 3000
    cap_ready = Signal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 320)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.mesh = None  # Lightweight bounding-box proxy, never millions of faces.
        self.original_mesh = None
        self._base_preview = None
        self._base_min_z = 0.0
        self._scale = 1.0
        self._cap_revision = 0
        self._cap_running = False
        self._cap_pending = False
        self.cap_ready.connect(self._finish_cap)
        self.triangles = None
        self.normals = None
        self.face_colors = None
        self.yaw = math.radians(37)
        self.pitch = math.radians(27)
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=float)
        self._last_pos = None
        self._drag_button = None
        self.slice_mode = False
        self.slice_z = 0.0
        self._cap_polys = []
        self._cap_timer = QTimer(self)
        self._cap_timer.setSingleShot(True)
        self._cap_timer.setInterval(250)
        self._cap_timer.timeout.connect(self._rebuild_cap)

    def set_scene(self, source_mesh: trimesh.Trimesh,
                  preview_geometry: tuple, target_height_mm: float,
                  reset_camera: bool = False) -> None:
        """Install an already simplified preview without costly GUI-thread work."""
        self.original_mesh = source_mesh
        self._base_preview = preview_geometry
        self._base_min_z = float(source_mesh.bounds[0, 2])
        self._cap_revision += 1
        self._cap_timer.stop()
        self._cap_polys = []
        self.set_model_height(target_height_mm)
        if reset_camera:
            self.reset_camera()
        self.update()

    def set_model_height(self, target_height_mm: float) -> None:
        """Reuse the same preview triangles for height changes (O(3000))."""
        if self.original_mesh is None or self._base_preview is None:
            return
        source = self.original_mesh
        original_height = float(source.extents[2])
        if not 0.05 <= target_height_mm <= 180.0 or original_height <= 0:
            raise ValueError("Invalid model height")
        self._scale = float(target_height_mm / original_height)
        base, normals, colors = self._base_preview
        self.triangles = np.asarray(base, dtype=float) * self._scale
        self.triangles[:, :, 2] -= self._base_min_z * self._scale
        self.normals = normals
        self.face_colors = colors
        raw_bounds = np.asarray(source.bounds, dtype=float)
        bounds = raw_bounds * self._scale
        bounds[:, 2] -= self._base_min_z * self._scale
        self.mesh = SimpleNamespace(
            bounds=bounds,
            extents=bounds[1] - bounds[0],
        )
        self._cap_revision += 1
        self._cap_polys = []
        self.set_layer_z(min(self.slice_z, float(target_height_mm)))
        if self.slice_mode:
            self._cap_timer.start()
        self.update()

    def reset_camera(self):
        self.yaw = math.radians(37)
        self.pitch = math.radians(27)
        self.zoom = 1.0
        self.pan[:] = 0
        self.update()

    def set_slice_mode(self, active: bool) -> None:
        active = bool(active)
        if self.slice_mode == active:
            return
        self.slice_mode = active
        self._cap_revision += 1
        self._cap_polys = []
        if self.slice_mode:
            self._cap_timer.start()
        else:
            self._cap_timer.stop()
        self.update()

    def set_layer_z(self, z_mm: float) -> None:
        if self.mesh is None:
            self.slice_z = 0
            return
        height = float(self.mesh.extents[2])
        new_z = float(np.clip(z_mm, 0.0, height))
        if abs(new_z - self.slice_z) < 1e-9:
            return
        self.slice_z = new_z
        self._cap_revision += 1
        self._cap_polys = []
        if self.slice_mode:
            self._cap_timer.start()
        self.update()

    def _rebuild_cap(self):
        """Full-mesh cap calculation runs off the GUI thread and is debounced.

        At most one calculation runs at a time. Further slider changes only
        replace the desired layer; outdated results are discarded.
        """
        if self.original_mesh is None or not self.slice_mode:
            return
        if self._cap_running:
            self._cap_pending = True
            return
        self._cap_running = True
        self._cap_pending = False
        revision = self._cap_revision
        source = self.original_mesh
        scale = self._scale
        z = self._base_min_z + self.slice_z / scale
        z = min(max(z, float(source.bounds[0, 2]) + 1e-7),
                float(source.bounds[1, 2]) - 1e-7)
        # Local bindings keep source mesh alive until background task completes.
        def compute():
            try:
                segments, _ = trimesh.intersections.mesh_plane(
                    source, [0, 0, 1], [0, 0, z], return_faces=True)
                polygons = _polygons_from_segments(segments)
                self.cap_ready.emit(revision, polygons)
            except Exception:
                self.cap_ready.emit(revision, [])

        threading.Thread(target=compute, name="UV3D-cap-preview", daemon=True).start()

    def _finish_cap(self, revision: int, polygons):
        self._cap_running = False
        if revision == self._cap_revision and self.slice_mode:
            self._cap_polys = polygons
            self.update()
        if (self._cap_pending or revision != self._cap_revision) and self.slice_mode:
            self._cap_pending = False
            self._cap_timer.start(15)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._last_pos = event.position()
            self._drag_button = event.button()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        if self._drag_button == Qt.LeftButton:
            self.yaw += delta.x() * 0.010
            self.pitch = float(np.clip(self.pitch + delta.y() * 0.007, -1.35, 1.35))
        else:
            self.pan += np.array([delta.x(), delta.y()], dtype=float)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._last_pos = self._drag_button = None

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.zoom = float(np.clip(self.zoom * 1.14 ** steps, 0.10, 30.0))
        self.update()
        event.accept()

    def _project(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.mesh is None:
            return np.zeros((len(pts), 3), dtype=float)
        dims = self.mesh.extents
        bmin, bmax = self.mesh.bounds
        mid = (bmin + bmax) * 0.5
        middle = np.array([mid[0], mid[1], float(dims[2]) * 0.48])
        xyz = pts - middle
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        sideways = cy * xyz[:, 0] - sy * xyz[:, 1]
        into = sy * xyz[:, 0] + cy * xyz[:, 1]
        vertical = sp * into - cp * xyz[:, 2]
        depth = cp * into + sp * xyz[:, 2]
        max_dim = max(float(max(dims[:2])), float(dims[2]) * 1.45, 1e-4)
        scale = min(self.width(), self.height()) * 0.63 / max_dim * self.zoom
        return np.column_stack((
            self.width() * 0.5 + self.pan[0] + sideways * scale,
            self.height() * 0.53 + self.pan[1] + vertical * scale,
            depth,
        ))

    @staticmethod
    def _polygon(points):
        return QPolygonF([QPointF(float(x), float(y)) for x, y in points[:, :2]])

    def _draw_grid(self, painter):
        if self.mesh is None:
            return
        b0, b1 = self.mesh.bounds
        dx, dy = self.mesh.extents[:2]
        pad = max(2.0, float(max(dx, dy)) * .23)
        xmin, xmax = float(b0[0] - pad), float(b1[0] + pad)
        ymin, ymax = float(b0[1] - pad), float(b1[1] + pad)
        corners = np.array([[xmin, ymin, 0], [xmax, ymin, 0],
                            [xmax, ymax, 0], [xmin, ymax, 0]])
        projected = self._project(corners)
        painter.setPen(QPen(QColor("#425469"), 1.2))
        painter.setBrush(QColor("#202d3c"))
        painter.drawPolygon(self._polygon(projected))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#324355"), 0.8))
        for step in range(1, 10):
            t = step / 10
            x = xmin + t * (xmax - xmin)
            y = ymin + t * (ymax - ymin)
            for endpoints in (
                [[x, ymin, 0], [x, ymax, 0]],
                [[xmin, y, 0], [xmax, y, 0]]
            ):
                a, b = self._project(endpoints)
                painter.drawLine(QPointF(*a[:2]), QPointF(*b[:2]))

    def _paint_faces(self, painter, projected, cutoff=None, ghost=False):
        if self.triangles is None:
            return
        tri = self.triangles
        # Use the view direction in world coordinates for culling and shading.
        direction = np.array([math.sin(self.yaw) * math.cos(self.pitch),
                              math.cos(self.yaw) * math.cos(self.pitch),
                              math.sin(self.pitch)])
        visible = self.normals @ direction > 0.0
        if not np.any(visible):
            visible = np.ones(len(tri), dtype=bool)
        indices = np.flatnonzero(visible)
        depth = projected[indices, :, 2].mean(axis=1)
        painter.setPen(Qt.NoPen)
        for idx in indices[np.argsort(depth)]:
            face = tri[idx]
            if cutoff is not None:
                if face[:, 2].min() > cutoff:
                    continue
                if face[:, 2].max() > cutoff:
                    face = clip_triangle_below(face, cutoff)
                    if len(face) < 3:
                        continue
            pts = projected[idx] if len(face) == 3 and (cutoff is None or tri[idx, :, 2].max() <= cutoff) else self._project(face)
            if ghost:
                fill = QColor(136, 162, 187, 39)
            else:
                light = max(0.0, float(np.dot(self.normals[idx], np.array([0.34, 0.48, 0.81]))))
                brightness = .55 + .45 * light
                rgb = (self.face_colors[idx].astype(float) * brightness).astype(int)
                fill = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
            painter.setBrush(fill)
            painter.drawPolygon(self._polygon(pts))

    def _draw_cut_plane(self, painter):
        if self.mesh is None:
            return
        b0, b1 = self.mesh.bounds
        pad = max(0.5, max(self.mesh.extents[:2]) * 0.08)
        a, b, z = b0[:2] - pad, b1[:2] + pad, self.slice_z
        xy = np.array([[a[0], a[1], z], [b[0], a[1], z],
                       [b[0], b[1], z], [a[0], b[1], z]])
        painter.setBrush(QColor(27, 179, 211, 19))
        painter.setPen(QPen(QColor(38, 204, 229, 95), 1.5, Qt.DashLine))
        painter.drawPolygon(self._polygon(self._project(xy)))

    def _draw_slice_cap(self, painter):
        if not self._cap_polys:
            return
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        for poly in self._cap_polys:
            for ring in [poly.exterior, *poly.interiors]:
                z = np.full(len(ring.coords), self.slice_z)
                xy = np.asarray(ring.coords) * self._scale
                pts = np.column_stack((xy, z))
                path.addPolygon(self._polygon(self._project(pts)))
                path.closeSubpath()
        painter.setPen(QPen(QColor(255, 191, 79, 225), 1.4))
        painter.setBrush(QColor(255, 177, 59, 165))
        painter.drawPath(path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#131b27"))
        if self.mesh is None:
            painter.setPen(QColor("#a8bbd0"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a model to explore it in 3D")
            painter.end()
            return
        self._draw_grid(painter)
        projected = self._project(self.triangles.reshape(-1, 3)).reshape(-1, 3, 3)
        if self.slice_mode:
            self._paint_faces(painter, projected, ghost=True)
            self._draw_cut_plane(painter)
            self._paint_faces(painter, projected, cutoff=self.slice_z)
            self._draw_slice_cap(painter)
        else:
            self._paint_faces(painter, projected)
        painter.setPen(QColor("#bed3e5"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(QRectF(14, 12, self.width() - 26, 30),
                         "LMB orbit  •  RMB/MMB pan  •  Wheel zoom")
        if self.slice_mode:
            painter.setPen(QColor("#f9c06c"))
            painter.drawText(QRectF(14, self.height()-40, self.width()-25, 26),
                             f"SLICE  |  Z {self.slice_z:.3f} mm  |  Amber: selected layer; gray: full model")
        painter.end()
