"""GPU print simulation on the original-resolution UV3D model.

A gray cuboid follows an *illustrative* serpentine path above the current layer.
Completed model geometry is shown at native triangle resolution. The in-progress
layer is a texture assembled from actual exported solid/color/support PNGs.
A display-only support surface is translucent and user-tintable. Nothing here
generates real printer movement or hardware commands.
"""
from __future__ import annotations

import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

from .gpu_viewport import GpuModelViewport, VERTEX_DTYPE, VERTEX_SHADER
from .simulation_engine import SimulationFrame
from .gantry import GantryConfig, HeadConfig


SIM_LAYER_VERTEX = (
    VERTEX_SHADER
    .replace("layout(location=1) in vec4 aColor;",
             "layout(location=2) in vec2 aUV;")
    .replace("out vec4 vColor;", "out vec2 vUV;")
    .replace("    vColor = aColor;", "    vUV = aUV;")
)
SIM_LAYER_FRAGMENT = r"""
#version 330 core
in vec2 vUV;
out vec4 outColor;
uniform sampler2D uBody;
uniform sampler2D uColor;
uniform sampler2D uSupport;
uniform vec4 uSupportTint;
uniform vec3 uCurrentModelTint;
uniform vec3 uCurrentSupportTint;
uniform int uRows;
uniform int uRowIndex;
uniform int uPasses;
uniform int uPassesCompleted;
uniform int uScanDirection;
uniform bool uPrinting;
uniform float uRowProgress;
uniform bool uFullLayer;

void main() {
    // All raster rows follow the actual head's Y swath bands. Previous
    // bands are complete; future bands have not been printed yet.
    int row = min(uRows - 1,
                  int(floor(clamp(vUV.y, 0.0, 0.9999999) * float(uRows))));
    float coverage = 0.0;
    if (uFullLayer || row < uRowIndex) {
        coverage = 1.0;
    } else if (row == uRowIndex) {
        // The nozzle traverses the image coordinate matching the physical
        // printhead. A reverse pass MUST paint right-to-left (not based on
        // row parity). Non-printing return moves deposit nothing.
        float already = float(uPassesCompleted);
        float current = 0.0;
        if (uPrinting) {
            float p = clamp(uRowProgress, 0.0, 1.0);
            bool reached = (uScanDirection > 0) ? (vUV.x <= p)
                                                   : (vUV.x >= 1.0 - p);
            current = reached ? 1.0 : 0.0;
        }
        coverage = clamp((already + current) / float(max(1,uPasses)), 0.0, 1.0);
    }
    if (coverage <= 0.0001) discard;
    float body = texture(uBody, vUV).r;
    float support = texture(uSupport, vUV).r;
    if (body > 0.03) {
        // Highlight only the ACTIVE printing layer, not the finished model.
        // Retain a small amount of source color for detail recognition.
        vec4 source = texture(uColor, vUV);
        vec3 tint = mix(uCurrentModelTint, source.rgb, 0.12 * source.a);
        outColor = vec4(tint, coverage * body);
    } else if (support > 0.03) {
        // Different color from the current model and finished support shell.
        outColor = vec4(uCurrentSupportTint, coverage * support * uSupportTint.a);
    } else {
        discard;
    }
}
"""


def make_head_cuboid(center_x: float, center_y: float, bottom_z: float,
                     model_width: float = 0.0, model_depth: float = 0.0,
                     head_spec=None):
    """Render the PHYSICAL Epson i3200-U1 body, not a model-scaled cube.

    Width 69.1, depth 59.5, height 35.6 mm. The nozzle swath is a
    separate 33.8-mm effective print width, not the body depth.
    """
    from .simulation_engine import PrintHeadSpec
    spec = head_spec or PrintHeadSpec()
    width = spec.body_x_mm
    depth = spec.body_y_mm
    tall = spec.body_z_mm
    x0, x1 = center_x - width/2, center_x + width/2
    y0, y1 = center_y - depth/2, center_y + depth/2
    z0, z1 = bottom_z, bottom_z + tall
    corners = np.array([
        [x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0],
        [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1]
    ], dtype=np.float32)
    face_corners = [
        (0,3,2,1), (4,5,6,7), (0,1,5,4),
        (1,2,6,5), (2,3,7,6), (3,0,4,7)
    ]
    shades = [(40,48,59,255), (227,235,242,255), (156,170,182,255),
              (199,212,223,255), (126,143,157,255), (181,193,204,255)]
    packed = np.empty(36, dtype=VERTEX_DTYPE)
    for n, (a,b,c,d) in enumerate(face_corners):
        start = n*6
        packed["position"][start:start+6] = corners[[a,b,c,a,c,d]]
        packed["color"][start:start+6] = shades[n]
    return packed


def make_component_cuboid(cx, cy, bottom_z, sx, sy, sz, rgba):
    """Packed 12-triangle world-space component; dimensions in mm."""
    if min(sx, sy, sz) <= 0:
        raise ValueError("Gantry component dimensions must be positive.")
    x0,x1 = cx-sx/2, cx+sx/2
    y0,y1 = cy-sy/2, cy+sy/2
    z0,z1 = bottom_z, bottom_z+sz
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]
    ], dtype=np.float32)
    indices = [
        (0,3,2,1),(4,5,6,7),(0,1,5,4),
        (1,2,6,5),(2,3,7,6),(3,0,4,7)
    ]
    packed=np.empty(36,dtype=VERTEX_DTYPE)
    color=np.asarray(rgba,dtype=np.uint8)
    for i,(a,b,c,d) in enumerate(indices):
        n=i*6
        packed["position"][n:n+6] = corners[[a,b,c,a,c,d]]
        packed["color"][n:n+6] = color
    return packed


def uv_footprints(frame, gantry, raster_min_x, raster_min_y,
                  raster_max_x, raster_max_y):
    """Visible XY overlap of lit lamp footprints with the printed layer.

    Non-print travel has lamps OFF. Print segments illuminate using all enabled
    lamps; a dedicated curing sweep lights only the lamp assigned to it.
    """
    if frame is None or not frame.lamp_active:
        return []
    gx,gy,_ = frame.head_position_mm
    regions=[]
    for index,lamp in enumerate(gantry.lamps):
        if not lamp.enabled:
            continue
        if frame.curing and frame.active_lamp_index != index:
            continue
        cx,cy = gx+lamp.offset_x_mm, gy+lamp.offset_y_mm
        x0=max(raster_min_x,cx-lamp.width_mm/2)
        x1=min(raster_max_x,cx+lamp.width_mm/2)
        y0=max(raster_min_y,cy-lamp.depth_mm/2)
        y1=min(raster_max_y,cy+lamp.depth_mm/2)
        if x1>x0 and y1>y0:
            regions.append((index,x0,y0,x1,y1))
    return regions


class SimulationViewport(GpuModelViewport):
    """A separate GPU view so slicer camera, meshes and progress stay independent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: SimulationFrame | None = None
        self._support_mesh = None
        self._support_mesh_dirty = False
        self._support_faces = 0
        self._support_rgb = (0.65, 0.42, 0.88)
        self._support_alpha = .32
        self._head_alpha = .65
        self._current_model_rgb = (1.0, 0.66, 0.18)
        self._current_support_rgb = (0.12, 0.95, 0.83)
        self._model_tint = (0.81, 0.82, 0.85)
        self._fallback_model_tint = False
        self._show_future = False
        self._bitmaps = None
        self._bitmap_layer = -1
        self._bitmap_dirty = False
        self._raster = None
        self._gantry_config = GantryConfig(
            heads=[HeadConfig("i3200-U1","color")], lamps=[])

    def initializeGL(self):
        super().initializeGL()
        if not self._gpu_ready:
            return
        try:
            self._layer_program = shaders.compileProgram(
                shaders.compileShader(SIM_LAYER_VERTEX, GL.GL_VERTEX_SHADER),
                shaders.compileShader(SIM_LAYER_FRAGMENT, GL.GL_FRAGMENT_SHADER))
            self._surface_vao = GL.glGenVertexArrays(1)
            self._surface_vbo = GL.glGenBuffers(1)
            self._head_vao = GL.glGenVertexArrays(1)
            self._head_vbo = GL.glGenBuffers(1)
            self._quad_vao = GL.glGenVertexArrays(1)
            self._quad_vbo = GL.glGenBuffers(1)
            self._textures = GL.glGenTextures(3)
            for tex in self._textures:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        except Exception as exc:
            self._gpu_ready = False
            self.opengl_failed.emit(f"Simulation renderer: {exc}")

    def set_simulation_scene(self, source_mesh, gpu_geometry, manifest: dict):
        target_height = float(manifest["scaled_dimensions_mm"][2])
        self.set_scene(source_mesh, gpu_geometry, target_height, reset_camera=True)
        # The real-sized 69.1×59.5×35.6 mm head can be MUCH larger than a
        # tiny imported model. Fit both the model and the head into the
        # initial camera framing, without changing their physical dimensions.
        from .simulation_engine import PrintHeadSpec
        spec = PrintHeadSpec()
        dims = self.mesh.extents
        magnitude = max(float(max(dims[:2])), float(dims[2]) * 1.45, .0001)
        span_with_head = max(
            float(dims[0]) + spec.body_x_mm,
            float(dims[1]) + spec.body_y_mm,
            float(dims[2]) + spec.body_z_mm)
        self.zoom = min(1.0, magnitude / max(magnitude, span_with_head) * 0.92)
        self._raster = manifest
        self._sim_passes = 6
        self.slice_mode = True  # Important: do NOT trigger expensive CPU plane intersections.
        self._cap_timer.stop()
        self._line_data = None
        self._support_mesh = None
        self._support_mesh_dirty = True
        self._bitmaps = None
        self._bitmap_layer = -1
        self.frame = None
        self.update()

    def set_simulation_passes(self, passes: int):
        self._sim_passes = max(1, int(passes))
        self.update()

    def set_gantry_config(self, config: GantryConfig):
        self._gantry_config = GantryConfig.from_dict(config.as_dict())
        self.update()

    def set_frame(self, frame: SimulationFrame):
        self.frame = frame
        self.slice_z = frame.completed_height_mm
        self.update()

    def set_support_surface(self, vertices):
        self._support_mesh = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        self._support_mesh_dirty = True
        self.update()

    def set_support_tint(self, rgb: tuple[int,int,int], opacity: float):
        self._support_rgb = tuple(float(np.clip(x, 0, 255)) / 255 for x in rgb)
        self._support_alpha = float(np.clip(opacity, 0.0, 1.0))
        self.update()

    def set_head_opacity(self, opacity: float):
        self._head_alpha = float(np.clip(opacity, 0.0, 1.0))
        self.update()

    def set_current_layer_colors(self, model_rgb: tuple[int, int, int],
                                 support_rgb: tuple[int, int, int]):
        """Set independent highlight colors for actively deposited materials."""
        self._current_model_rgb = tuple(float(np.clip(c, 0, 255)) / 255
                                        for c in model_rgb)
        self._current_support_rgb = tuple(float(np.clip(c, 0, 255)) / 255
                                          for c in support_rgb)
        self.update()

    def set_model_tint(self, rgb: tuple[int,int,int], fallback_only: bool = True):
        self._model_tint = tuple(float(np.clip(x, 0, 255)) / 255 for x in rgb)
        self._fallback_model_tint = bool(fallback_only)
        self.update()

    def set_show_future(self, enabled: bool):
        self._show_future = bool(enabled)
        self.update()

    def set_layer_bitmaps(self, layer_index: int, body: np.ndarray,
                          color: np.ndarray, support: np.ndarray):
        if body.ndim != 2 or support.shape != body.shape:
            raise ValueError("Invalid simulation body/support raster dimensions")
        if color.shape != (*body.shape,4):
            raise ValueError("Simulation color mask must be RGBA")
        self._bitmaps = (np.ascontiguousarray(body, dtype=np.uint8),
                         np.ascontiguousarray(color, dtype=np.uint8),
                         np.ascontiguousarray(support, dtype=np.uint8))
        self._bitmap_layer = int(layer_index)
        self._bitmap_dirty = True
        self.update()

    def _setup_simple_vao(self, vao, vbo):
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE,
                                 16, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE,
                                 16, ctypes.c_void_p(12))

    def _draw_support_mesh(self):
        if self._support_mesh is None or self._support_alpha <= .001:
            return
        if self._support_mesh_dirty:
            n = len(self._support_mesh)
            packed = np.empty(n, dtype=VERTEX_DTYPE)
            packed["position"] = self._support_mesh
            packed["color"] = [174, 105, 230, 255]
            self._setup_simple_vao(self._surface_vao, self._surface_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, packed.nbytes,
                            packed.tobytes(), GL.GL_STATIC_DRAW)
            self._support_faces = n
            self._support_mesh_dirty = False
        GL.glBindVertexArray(self._surface_vao)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uMode"), 5)
        GL.glUniform4f(GL.glGetUniformLocation(self._program, "uSupportTint"),
                       *self._support_rgb, self._support_alpha)
        self._set1f("uScale", 1)
        self._set1f("uBaseZ", 0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        try:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._support_faces)
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
            self._set1f("uScale", self._scale)
            self._set1f("uBaseZ", self._base_min_z)

    def _upload_layer_bitmaps(self):
        if self._bitmaps is None or not self._bitmap_dirty:
            return
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        try:
            for channel, (tex, bitmap) in enumerate(zip(self._textures, self._bitmaps)):
                height, width = bitmap.shape[:2]
                is_rgba = channel == 1
                GL.glActiveTexture(GL.GL_TEXTURE0 + channel)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8 if is_rgba else GL.GL_R8,
                    width, height, 0, GL.GL_RGBA if is_rgba else GL.GL_RED,
                    GL.GL_UNSIGNED_BYTE, bitmap.tobytes())
        finally:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        self._bitmap_dirty = False

    def _draw_current_layer(self, w, h):
        frame = self.frame
        data = self._raster
        if frame is None or data is None or self._bitmap_layer != frame.layer_index:
            return
        self._upload_layer_bitmaps()
        settings = data["settings"]
        ppm = float(settings["dpi"]) / 25.4
        margin = float(data.get("effective_raster_margin_mm", settings.get("margin_mm", 0)))
        bounds = data["model_bounds_mm"]
        x0 = float(bounds[0][0]) - margin
        y0 = float(bounds[1][1]) + margin
        wpx, hpx = data["raster_size_px"]
        x1 = x0 + float(wpx) / ppm
        y1 = y0 - float(hpx) / ppm
        z = frame.current_top_mm
        # World Y descends from image row 0 to the last row. The 0..1
        # texture coordinate order matches the top-down PNG row data.
        array = np.array([
            (x0,y0,z,0,0), (x1,y0,z,1,0), (x1,y1,z,1,1),
            (x0,y0,z,0,0), (x1,y1,z,1,1), (x0,y1,z,0,1),
        ], dtype=np.float32)
        GL.glBindVertexArray(self._quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, array.nbytes,
                        array.tobytes(), GL.GL_STREAM_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,20,ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2,2,GL.GL_FLOAT,GL.GL_FALSE,20,ctypes.c_void_p(12))
        previous = self._program
        try:
            self._program = self._layer_program
            GL.glUseProgram(self._program)
            self._set_uniforms(w,h)
            self._set1f("uScale",1)
            self._set1f("uBaseZ",0)
            for channel, name in enumerate(("uBody","uColor","uSupport")):
                GL.glActiveTexture(GL.GL_TEXTURE0 + channel)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures[channel])
                GL.glUniform1i(GL.glGetUniformLocation(self._program, name), channel)
            GL.glUniform4f(GL.glGetUniformLocation(self._program, "uSupportTint"),
                           *self._support_rgb, self._support_alpha)
            GL.glUniform3f(GL.glGetUniformLocation(self._program, "uCurrentModelTint"),
                           *self._current_model_rgb)
            GL.glUniform3f(GL.glGetUniformLocation(self._program, "uCurrentSupportTint"),
                           *self._current_support_rgb)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uRows"), frame.rows_per_layer)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uRowIndex"), frame.row_index)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uPasses"),
                           self._sim_passes)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uPassesCompleted"),
                           frame.passes_completed)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uScanDirection"),
                           frame.scan_direction)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uPrinting"),
                           int(frame.printing))
            self._set1f("uRowProgress", frame.row_fraction)
            GL.glUniform1i(GL.glGetUniformLocation(self._program, "uFullLayer"),
                           int(frame.completed))
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
            self._program = previous
            GL.glUseProgram(previous)
            self._set_uniforms(w,h)

    def _draw_component(self, packed, alpha):
        self._setup_simple_vao(self._head_vao, self._head_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, packed.nbytes,
                        packed.tobytes(), GL.GL_STREAM_DRAW)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uMode"), 6)
        GL.glUniform4f(GL.glGetUniformLocation(self._program, "uSupportTint"),
                       1.0, 1.0, 1.0, float(alpha))
        self._set1f("uScale", 1)
        self._set1f("uBaseZ", 0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        try:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(packed))
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
            self._set1f("uScale", self._scale)
            self._set1f("uBaseZ", self._base_min_z)

    def _draw_uv_illumination(self):
        """A pale golden footprint projected onto the current printed layer.

        Glow is VISUAL ONLY; it does not estimate exposure energy or curing.
        Actual rectangle edges are clipped to the printable raster region.
        """
        if self.frame is None or self._raster is None:
            return
        data=self._raster
        ppm=float(data["settings"]["dpi"])/25.4
        margin=float(data.get("effective_raster_margin_mm",
                              data["settings"].get("margin_mm",0)))
        xmin=float(data["model_bounds_mm"][0][0])-margin
        ymax=float(data["model_bounds_mm"][1][1])+margin
        w,h=data["raster_size_px"]
        xmax=xmin+float(w)/ppm
        ymin=ymax-float(h)/ppm
        regions=uv_footprints(self.frame,self._gantry_config,
                              xmin,ymin,xmax,ymax)
        if not regions:
            return
        z=self.frame.current_top_mm + .025
        for _,x0,y0,x1,y1 in regions:
            packed=np.empty(6,dtype=VERTEX_DTYPE)
            packed["position"] = np.asarray([
                (x0,y0,z),(x1,y0,z),(x1,y1,z),
                (x0,y0,z),(x1,y1,z),(x0,y1,z)
            ],dtype=np.float32)
            packed["color"] = [255,231,80,255]
            self._draw_component(packed, .3)

    def _draw_head(self):
        """Draw entire moving carriage, not one disconnected printhead.

        All relative offsets are defined on the Gantry setup page and applied
        equally during print scans, non-print travel and UV curing sweeps.
        """
        if self.frame is None or self.mesh is None:
            return
        gx,gy,z=self.frame.head_position_mm
        config=self._gantry_config
        heads=[h for h in config.heads if h.enabled]
        lamps=[l for l in config.lamps if l.enabled]
        if not heads:
            return
        if self._head_alpha <= .001:
            return
        palette={
            "color":(115,156,202,255),
            "white":(224,226,230,255),
            "varnish":(171,190,168,255),
            "support":(177,121,221,255)
        }
        components=[]
        x_min=x_max=y_min=y_max=None
        highest=0.0

        def include(x,y,sx,sy):
            nonlocal x_min,x_max,y_min,y_max
            a,b=x-sx/2,x+sx/2
            c,d=y-sy/2,y+sy/2
            x_min = a if x_min is None else min(x_min,a)
            x_max = b if x_max is None else max(x_max,b)
            y_min = c if y_min is None else min(y_min,c)
            y_max = d if y_max is None else max(y_max,d)

        for head in heads:
            x=gx+head.offset_x_mm
            y=gy+head.offset_y_mm
            include(x,y,head.width_mm,head.depth_mm)
            highest=max(highest,head.height_mm)
            components.append((make_component_cuboid(
                x,y,z,head.width_mm,head.depth_mm,head.height_mm,
                palette[head.material]),self._head_alpha))
        for lamp in lamps:
            x=gx+lamp.offset_x_mm
            y=gy+lamp.offset_y_mm
            include(x,y,lamp.width_mm,lamp.depth_mm)
            highest=max(highest,lamp.body_height_mm)
            lit=self.frame.lamp_active and (
                not self.frame.curing or
                config.lamps.index(lamp)==self.frame.active_lamp_index)
            components.append((make_component_cuboid(
                x,y,z+max(0,highest-lamp.body_height_mm),
                lamp.width_mm,lamp.depth_mm,lamp.body_height_mm,
                (250,205,78,255) if lit else (132,128,109,255)),
                self._head_alpha))
        # The rectangular bar makes the shared carriage visually obvious.
        # It moves/returns as one piece with all attached heads and lamps.
        if x_min is not None and x_max-x_min>.1:
            beam=make_component_cuboid(
                (x_min+x_max)*.5,(y_min+y_max)*.5,z+highest+1.0,
                x_max-x_min+4, min(12.0,y_max-y_min+2), 7,
                (94,103,119,255))
            components.append((beam,self._head_alpha))
        for packed,opacity in components:
            self._draw_component(packed,opacity)

    def _paint_simulation_frame(self):
        ratio = self.devicePixelRatioF()
        w, h = max(1,int(self.width()*ratio)),max(1,int(self.height()*ratio))
        GL.glViewport(0,0,w,h)
        GL.glClearColor(.18,.19,.22,1.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self._gpu_ready or self.mesh is None:
            return
        self._upload_geometry()
        GL.glUseProgram(self._program)
        self._set_uniforms(w,h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        self._draw_build_plate()
        GL.glUniform1i(GL.glGetUniformLocation(self._program,"uUseFallbackModelTint"),
                       int(self._fallback_model_tint))
        GL.glUniform3f(GL.glGetUniformLocation(self._program,"uFallbackModelTint"),
                       *self._model_tint)
        if self._show_future:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self._draw_model(1)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            GL.glDisable(GL.GL_BLEND)
        if self.frame is not None:
            self._draw_model(0)
            self._draw_support_mesh()
            self._draw_current_layer(w,h)
            self._draw_uv_illumination()
            self._draw_head()
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def paintGL(self):
        """An OpenGL driver error must not propagate from Qt's paint callback.

        PyOpenGL exceptions raised here previously reached Qt's native paint
        machinery, which can terminate the whole application on Windows.
        Disable only this simulation viewport and preserve the application.
        """
        if getattr(self, "_gl_render_failed", False):
            return
        try:
            self._paint_simulation_frame()
        except Exception as exc:
            import traceback
            if not getattr(self, "_gl_render_failed", False):
                self._gl_render_failed = True
                self._gpu_ready = False
                try:
                    self.opengl_failed.emit(
                        f"{type(exc).__name__}: {exc}\\n"
                        f"{traceback.format_exc()}")
                except Exception:
                    pass
            # Avoid further GL calls in a potentially invalid native context.
