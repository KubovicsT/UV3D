"""Full-resolution GPU model viewer for Windows/OpenGL 3.3.

No decimation. Uploads original triangles into GPU buffers; uses a shader for
camera, scaling, coloring and per-layer clipping. No printer communication.
"""
from __future__ import annotations

import ctypes
import math
import threading
from types import SimpleNamespace

import numpy as np
import trimesh
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL
from OpenGL.GL import shaders
from .navigation import (orbit as supp_orbit, zoom as supp_zoom,
                         preset as camera_preset, pan_focal_delta)

VERTEX_DTYPE = np.dtype([("position", "<f4", (3,)), ("color", "u1", (4,))])

VERTEX_SHADER = r"""
#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec4 aColor;
uniform float uScale;
uniform float uBaseZ;
uniform vec3 uCenter;
uniform vec2 uViewport;
uniform vec2 uPan;
uniform float uYaw;
uniform float uPitch;
uniform float uPixelsPerUnit;
uniform float uMaxDim;
out vec3 vWorld;
out vec4 vColor;
void main() {
    vec3 p = aPosition * uScale;
    p.z -= uBaseZ * uScale;
    vec3 xyz = p - uCenter;
    float cy = cos(uYaw), sy = sin(uYaw);
    float cp = cos(uPitch), sp = sin(uPitch);
    float side = cy * xyz.x - sy * xyz.y;
    float into = sy * xyz.x + cy * xyz.y;
    float vertical = sp * into - cp * xyz.z;
    float depth = cp * into + sp * xyz.z;
    gl_Position = vec4(
       2.0 * (side * uPixelsPerUnit + uPan.x) / uViewport.x,
      -2.0 * (vertical * uPixelsPerUnit + uPan.y) / uViewport.y,
      -depth / (uMaxDim * 2.0), 1.0
    );
    vWorld = p;
    vColor = aColor;
}
"""

FRAGMENT_SHADER = r"""
#version 330 core
in vec3 vWorld;
in vec4 vColor;
out vec4 outColor;
uniform bool uSlice;
uniform float uCutZ;
uniform int uMode;
uniform vec4 uSupportTint;
uniform vec3 uFallbackModelTint;
uniform bool uUseFallbackModelTint;
void main() {
    if ((uMode == 0 || uMode == 5) && uSlice && vWorld.z > uCutZ + 0.00001)
        discard;
    if (uMode == 1) {
        outColor = vec4(0.43, 0.58, 0.74, 0.16);
    } else if (uMode == 3) {
        outColor = vec4(1.0, 0.67, 0.19, 1.0);
    } else if (uMode == 4) {
        outColor = vColor;
    } else if (uMode == 5) {
        outColor = uSupportTint;
    } else if (uMode == 6) {
        // Simulation printhead only: retain head face shading with user alpha.
        outColor = vec4(vColor.rgb, vColor.a * uSupportTint.a);
    } else {
        vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        float light = 0.52 + 0.48 * abs(dot(N, normalize(vec3(0.3, 0.5, 0.8))));
        vec3 baseColor = uUseFallbackModelTint ? uFallbackModelTint : vColor.rgb;
        outColor = vec4(baseColor * light, 1.0);
    }
}
"""


def prepare_gpu_geometry(mesh: trimesh.Trimesh):
    """Build GPU vertex and optional index buffers from ALL original faces.

    No face reduction occurs. Interleaved 16-byte vertices; indexed drawing for
    uniform/vertex colors, expanded triangles for differing per-face colors.
    Executed in ImportWorker rather than Qt's GUI thread.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    kind = getattr(mesh.visual, "kind", "")
    if kind == "vertex":
        rgba = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        if len(rgba) == len(vertices):
            rows = np.empty(len(vertices), dtype=VERTEX_DTYPE)
            rows["position"] = vertices
            rows["color"] = rgba[:, :4]
            return rows.tobytes(), faces.ravel().tobytes(), len(rows), faces.size
    visual = mesh.visual
    try:
        if kind == "texture":
            visual = visual.to_color()
        colors = np.asarray(visual.face_colors, dtype=np.uint8)
        if colors.shape[0] != len(faces):
            raise ValueError("Face-color count mismatch.")
    except (AttributeError, ValueError, TypeError):
        colors = np.broadcast_to(
            np.array([188, 194, 206, 255], dtype=np.uint8),
            (len(faces), 4))
    if np.all(colors == colors[0]):
        rows = np.empty(len(vertices), dtype=VERTEX_DTYPE)
        rows["position"] = vertices
        rows["color"] = colors[0, :4]
        return rows.tobytes(), faces.ravel().tobytes(), len(rows), faces.size
    rows = np.empty(len(faces) * 3, dtype=VERTEX_DTYPE)
    rows["position"] = vertices[faces.ravel()]
    rows["color"] = np.repeat(colors[:, :4], 3, axis=0)
    return rows.tobytes(), None, len(rows), 0


class GpuModelViewport(QOpenGLWidget):
    FULL_RESOLUTION = True
    PREVIEW_TRIANGLES = 0
    cap_ready = Signal(int, object)
    opengl_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.setMinimumSize(360, 320)
        self.setMouseTracking(True)
        self.original_mesh = None
        self.mesh = None
        self._payload = None
        self._gpu_dirty = False
        self._gpu_ready = False
        self._index_count = 0
        self._vertex_count = 0
        self._line_data = None
        self._line_dirty = True
        self._line_count = 0
        self._plate_dirty = True
        self._plate_count = 0
        self._support_lines = None
        self._support_dirty = True
        self._support_count = 0
        self._support_visible = False
        self.yaw, self.pitch = camera_preset("Isometric")
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=float)
        self.pan_world = np.zeros(3, dtype=float)
        self._last = self._drag = None
        self.slice_mode = False
        self.slice_z = 0.0
        self._base_min_z = 0.0
        self._scale = 1.0
        self._cap_revision = 0
        self._cap_running = False
        self._cap_pending = False
        self.cap_ready.connect(self._finish_cap)
        self._cap_timer = QTimer(self)
        self._cap_timer.setSingleShot(True)
        self._cap_timer.setInterval(220)
        self._cap_timer.timeout.connect(self._start_cap)

    def set_scene(self, source_mesh, gpu_geometry, target_height_mm, reset_camera=False):
        self.set_support_preview_lines(None)
        self.original_mesh = source_mesh
        self._payload = gpu_geometry
        self._gpu_dirty = True
        self._base_min_z = float(source_mesh.bounds[0, 2])
        self._cap_revision += 1
        self._line_data = None
        self._line_dirty = True
        self.set_model_height(target_height_mm)
        self._plate_dirty = True
        if reset_camera:
            self.reset_camera()
        self.update()

    def set_model_height(self, height):
        if self.original_mesh is None:
            return
        if not 0.05 <= height <= 180:
            raise ValueError("Model height must be 0.05–180 mm.")
        mesh = self.original_mesh
        scale = float(height / mesh.extents[2])
        bounds = np.asarray(mesh.bounds, dtype=float) * scale
        bounds[:, 2] -= self._base_min_z * scale
        self.mesh = SimpleNamespace(bounds=bounds, extents=bounds[1] - bounds[0])
        self._scale = scale
        self._plate_dirty = True
        self._cap_revision += 1
        self._line_data = None
        self._line_dirty = True
        self.slice_z = min(self.slice_z, float(height))
        if self.slice_mode:
            self._cap_timer.start()
        self.update()

    def set_support_preview_lines(self, lines):
        if lines is None:
            self._support_lines = None
        else:
            prepared = np.asarray(lines, dtype=np.float32).reshape(-1, 3)
            if len(prepared) % 2:
                raise ValueError("Support preview must contain pairs of GL line vertices.")
            self._support_lines = prepared
        self._support_dirty = True
        self.update()

    def set_show_support_preview(self, active):
        self._support_visible = bool(active)
        self.update()

    def reset_camera(self):
        self.yaw, self.pitch = camera_preset("Isometric")
        self.zoom = 1.0
        self.pan[:] = 0
        self.pan_world[:] = 0
        self.update()

    def set_view_preset(self, name):
        if name == "Fit model":
            self.reset_camera()
            return
        self.yaw, self.pitch = camera_preset(name)
        self.pan[:] = 0
        self.pan_world[:] = 0
        self.update()

    def set_slice_mode(self, active):
        active = bool(active)
        if self.slice_mode == active:
            return
        self.slice_mode = active
        self._cap_revision += 1
        self._line_data = None
        self._line_dirty = True
        if active:
            self._cap_timer.start()
        else:
            self._cap_timer.stop()
        self.update()

    def set_layer_z(self, z_mm):
        if self.mesh is None:
            return
        z = float(np.clip(z_mm, 0, self.mesh.extents[2]))
        if abs(z - self.slice_z) < 1e-10:
            return
        self.slice_z = z
        self._cap_revision += 1
        self._line_data = None
        self._line_dirty = True
        if self.slice_mode:
            self._cap_timer.start()
        self.update()

    def _start_cap(self):
        if not self.slice_mode or self.original_mesh is None:
            return
        if self._cap_running:
            self._cap_pending = True
            return
        self._cap_running = True
        self._cap_pending = False
        revision = self._cap_revision
        mesh = self.original_mesh
        raw_z = float(self._base_min_z + self.slice_z / self._scale)
        raw_z = float(np.clip(raw_z, mesh.bounds[0, 2] + 1e-7,
                              mesh.bounds[1, 2] - 1e-7))

        def calculate():
            try:
                lines = trimesh.intersections.mesh_plane(
                    mesh, [0, 0, 1], [0, 0, raw_z])
                data = np.asarray(lines, dtype=np.float32).reshape(-1, 3)
            except Exception:
                data = np.empty((0, 3), dtype=np.float32)
            self.cap_ready.emit(revision, data)

        threading.Thread(target=calculate, name="UV3D-GPU-cut", daemon=True).start()

    def _finish_cap(self, revision, data):
        self._cap_running = False
        if revision == self._cap_revision and self.slice_mode:
            self._line_data = data
            self._line_dirty = True
            self.update()
        if (self._cap_pending or revision != self._cap_revision) and self.slice_mode:
            self._cap_pending = False
            self._cap_timer.start(15)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._last, self._drag = event.position(), event.button()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._last is None:
            return
        delta = event.position() - self._last
        self._last = event.position()
        if self._drag == Qt.RightButton:
            self.yaw, self.pitch = supp_orbit(self.yaw, self.pitch, delta.x(), delta.y())
        elif self._drag == Qt.MiddleButton:
            dims = self.mesh.extents if self.mesh is not None else [10., 10., 10.]
            magnitude = max(float(max(dims[:2])), float(dims[2]) * 1.45, .0001)
            pixels = .63 * min(self.width(), self.height()) / magnitude * self.zoom
            self.pan_world += np.asarray(pan_focal_delta(
                self.yaw, self.pitch, delta.x(), delta.y(), pixels))
        else:
            return  # Left mouse remains available for selection and future gizmos.
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._last = self._drag = None

    def wheelEvent(self, event):
        self.zoom = supp_zoom(self.zoom, event.angleDelta().y() / 120.)
        self.update()
        event.accept()

    def initializeGL(self):
        try:
            self._program = shaders.compileProgram(
                shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER),
                shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER))
            self._vao = GL.glGenVertexArrays(1)
            self._vbo = GL.glGenBuffers(1)
            self._ebo = GL.glGenBuffers(1)
            self._line_vao = GL.glGenVertexArrays(1)
            self._line_vbo = GL.glGenBuffers(1)
            self._plate_vao = GL.glGenVertexArrays(1)
            self._plate_vbo = GL.glGenBuffers(1)
            self._support_vao = GL.glGenVertexArrays(1)
            self._support_vbo = GL.glGenBuffers(1)
            self._gpu_ready = True
        except Exception as exc:
            self._gpu_ready = False
            self.opengl_failed.emit(str(exc))

    def _upload_geometry(self):
        if not self._gpu_ready or self._payload is None or not self._gpu_dirty:
            return
        vertices, indices, vcount, icount = self._payload
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vertices), vertices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, 16, ctypes.c_void_p(12))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        if indices is not None:
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER, len(indices), indices, GL.GL_STATIC_DRAW)
        self._index_count, self._vertex_count = icount, vcount
        self._payload = None
        self._gpu_dirty = False

    def _set1f(self, name, value):
        GL.glUniform1f(GL.glGetUniformLocation(self._program, name), float(value))

    def _set_uniforms(self, width, height):
        dims, bounds = self.mesh.extents, self.mesh.bounds
        center = (bounds[0] + bounds[1]) * .5
        center[2] = float(dims[2]) * .48
        center += self.pan_world
        magnitude = max(float(max(dims[:2])), float(dims[2]) * 1.45, .0001)
        ratio = self.devicePixelRatioF()
        self._set1f("uScale", self._scale)
        self._set1f("uBaseZ", self._base_min_z)
        GL.glUniform3f(GL.glGetUniformLocation(self._program, "uCenter"),
                       *map(float, center))
        GL.glUniform2f(GL.glGetUniformLocation(self._program, "uViewport"),
                       float(width), float(height))
        GL.glUniform2f(GL.glGetUniformLocation(self._program, "uPan"),
                       float(self.pan[0] * ratio), float(self.pan[1] * ratio))
        self._set1f("uYaw", self.yaw)
        self._set1f("uPitch", self.pitch)
        self._set1f("uPixelsPerUnit", .63 * min(width, height) / magnitude * self.zoom)
        self._set1f("uMaxDim", magnitude)
        self._set1f("uCutZ", self.slice_z)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uSlice"),
                       int(self.slice_mode))

    def _draw_model(self, mode):
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uMode"), mode)
        GL.glBindVertexArray(self._vao)
        if self._index_count:
            GL.glDrawElements(
                GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(0))
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._vertex_count)

    def _draw_build_plate(self):
        """Supp-style Z-up build grid and colored XYZ axes, independent of model detail."""
        if self._plate_dirty:
            bounds = np.asarray(self.mesh.bounds, dtype=float)
            center = bounds.mean(axis=0)
            span = max(10.0, float(max(self.mesh.extents[:2])) * 1.50)
            # Local grid around imported object; avoids shrinking small models
            # to invisibility on a complete 600x900 mm plate.
            x0, x1 = center[0] - span / 2, center[0] + span / 2
            y0, y1 = center[1] - span / 2, center[1] + span / 2
            z = -.015 * span
            points = []
            def line(a, b, color):
                points.extend([(a, color), (b, color)])
            muted = [116, 122, 128, 255]
            border = [179, 183, 187, 255]
            for step in range(11):
                t = step / 10
                x = x0 + t * span
                y = y0 + t * span
                tone = border if step in (0, 10) else muted
                line((x, y0, z), (x, y1, z), tone)
                line((x0, y, z), (x1, y, z), tone)
            origin = (x0 + .07 * span, y0 + .07 * span, z)
            axis_length = .15 * span
            line(origin, (origin[0] + axis_length, origin[1], z),
                 [231, 106, 112, 255])
            line(origin, (origin[0], origin[1] + axis_length, z),
                 [89, 216, 154, 255])
            line(origin, (origin[0], origin[1], z + axis_length),
                 [110, 157, 251, 255])
            packed = np.empty(len(points), dtype=VERTEX_DTYPE)
            packed["position"] = [x[0] for x in points]
            packed["color"] = [x[1] for x in points]
            GL.glBindVertexArray(self._plate_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._plate_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, packed.nbytes,
                            packed.tobytes(), GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                     16, ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE,
                                     16, ctypes.c_void_p(12))
            self._plate_count = len(points)
            self._plate_dirty = False
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uMode"), 4)
        self._set1f("uScale", 1.0)
        self._set1f("uBaseZ", 0.0)
        GL.glBindVertexArray(self._plate_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._plate_count)
        self._set1f("uScale", self._scale)
        self._set1f("uBaseZ", self._base_min_z)

    def _draw_support_wireframe(self):
        """Draw a decimated purple preview, never the full support voxel grid."""
        if self._support_dirty:
            data = (self._support_lines if self._support_lines is not None
                    else np.empty((0, 3), dtype=np.float32))
            packed = np.empty(len(data), dtype=VERTEX_DTYPE)
            packed["position"] = data
            packed["color"] = [169, 116, 230, 165]
            GL.glBindVertexArray(self._support_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._support_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, packed.nbytes,
                            packed.tobytes(), GL.GL_DYNAMIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(
                1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, 16, ctypes.c_void_p(12))
            self._support_count = len(data)
            self._support_dirty = False
        if not self._support_count:
            return
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uMode"), 4)
        GL.glBindVertexArray(self._support_vao)
        # Preview lines are already in scaled model millimeters.
        self._set1f("uScale", 1)
        self._set1f("uBaseZ", 0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_DEPTH_TEST)
        try:
            GL.glDrawArrays(GL.GL_LINES, 0, self._support_count)
        finally:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            self._set1f("uScale", self._scale)
            self._set1f("uBaseZ", self._base_min_z)

    def _draw_cut_lines(self):
        if self._line_dirty:
            data = np.asarray(self._line_data, dtype=np.float32)
            packed = np.empty(len(data), dtype=VERTEX_DTYPE)
            packed["position"] = data
            packed["color"] = [255, 170, 52, 255]
            GL.glBindVertexArray(self._line_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, packed.nbytes,
                            packed.tobytes(), GL.GL_STREAM_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(
                1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, 16, ctypes.c_void_p(12))
            self._line_count = len(data)
            self._line_dirty = False
        GL.glBindVertexArray(self._line_vao)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uMode"), 3)
        # OpenGL 3.3 core-profile drivers (notably on Windows) may advertise
        # only one-pixel line rasterization. glLineWidth(2.0) then triggers
        # GL_INVALID_VALUE (1281), crashing Qt's paintGL callback.
        # Use the universally supported default line width instead.
        GL.glDisable(GL.GL_DEPTH_TEST)
        try:
            GL.glDrawArrays(GL.GL_LINES, 0, self._line_count)
        finally:
            GL.glEnable(GL.GL_DEPTH_TEST)

    def paintGL(self):
        ratio = self.devicePixelRatioF()
        width, height = max(1, int(self.width()*ratio)), max(1, int(self.height()*ratio))
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(54 / 255, 57 / 255, 64 / 255, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self._gpu_ready or self.mesh is None:
            return
        self._upload_geometry()
        GL.glUseProgram(self._program)
        self._set_uniforms(width, height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        self._draw_build_plate()
        if self.slice_mode:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self._draw_model(1)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            GL.glDisable(GL.GL_BLEND)
        self._draw_model(0)
        if self._support_visible and self._support_lines is not None:
            self._draw_support_wireframe()
        if self.slice_mode and self._line_data is not None and len(self._line_data):
            self._draw_cut_lines()
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
