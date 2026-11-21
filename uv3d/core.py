"""Offline UV slice generation. NEVER sends commands to any printer.

Output masks are geometric proposals, NOT calibrated ink/channel instructions.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageChops
import trimesh
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union


@dataclass
class SliceSettings:
    layer_mm: float = 0.05
    target_height_mm: float | None = None
    dpi: int = 300
    margin_mm: float = 3.0
    shell_mm: float = 0.3
    body_material: str = "white"
    placement_x_mm: float = 10.0
    placement_y_mm: float = 10.0

    def validate(self) -> None:
        if self.target_height_mm is not None and not (0.05 <= self.target_height_mm <= 180.0):
            raise ValueError("Target height must be between 0.05 and 180 mm.")
        if not (0.01 <= self.layer_mm <= 1.0):
            raise ValueError("Layer thickness must be 0.01–1.0 mm (not a printer calibration).")
        if not (72 <= self.dpi <= 1200):
            raise ValueError("DPI must be 72–1200.")
        if not (0 <= self.margin_mm <= 30 and 0.05 <= self.shell_mm <= 5):
            raise ValueError("Invalid margins or shell width.")
        if self.body_material not in ("white", "varnish"):
            raise ValueError("Body material must be white or varnish.")
        if self.placement_x_mm < 0 or self.placement_y_mm < 0:
            raise ValueError("Placement must be non-negative.")


@dataclass(frozen=True)
class MachineProfile:
    name: str = "Gongzheng H6090EI (unverified integration)"
    bed_x_mm: float = 600.0
    bed_y_mm: float = 900.0
    z_pulses_per_mm: float = 4841.4550469
    z_manual_increment_mm: float = 0.1
    heads: int = 3
    connection: str = "DISABLED - simulation only"


def load_mesh(filename: str | Path) -> tuple[trimesh.Trimesh, list[str]]:
    """Flatten scene transforms; textures are NOT faithfully mapped in v0.1."""
    path = Path(filename)
    if path.suffix.lower() not in (".stl", ".obj", ".glb", ".gltf", ".3mf", ".ply"):
        raise ValueError("Supported formats: STL, OBJ, GLB, GLTF, 3MF, PLY")
    scene = trimesh.load_scene(str(path))
    meshes = [m for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if not meshes:
        raise ValueError("File contains no printable triangle meshes.")
    has_texture = any(getattr(m.visual, "kind", "") == "texture" for m in meshes)
    mesh = trimesh.util.concatenate(meshes)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Model contains non-finite coordinates.")
    warnings = []
    if not mesh.is_watertight:
        warnings.append("Non-watertight mesh: some cross-sections may be missing or open.")
    if has_texture:
        warnings.append("UV texture mapping is NOT implemented. Boundary colors may be approximated; verify against original model.")
    if np.any(mesh.extents <= 0):
        raise ValueError("Model has zero size on at least one axis.")
    return mesh, warnings


def prepare_mesh(mesh: trimesh.Trimesh, target_height_mm: float | None = None) -> trimesh.Trimesh:
    """Uniformly resize XY and Z to a requested model height; rest on z=0.

    The input mesh is NEVER changed. The same transform powers 3D preview
    and offline slice generation, so layer count/geometry cannot disagree.
    """
    height = float(mesh.extents[2])
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Model has no usable Z height.")
    if target_height_mm is not None:
        if not np.isfinite(target_height_mm) or not (0.05 <= target_height_mm <= 180.0):
            raise ValueError("Target height must be between 0.05 and 180 mm.")
        factor = float(target_height_mm) / height
    else:
        factor = 1.0
    result = mesh.copy()
    result.apply_scale(factor)
    result.apply_translation([0.0, 0.0, -float(result.bounds[0][2])])
    return result


def _face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Approximate exterior colors per face; texture handling deliberately conservative."""
    vis = mesh.visual
    try:
        if getattr(vis, "kind", None) == "texture":
            vis = vis.to_color()
        rgb = np.asarray(vis.face_colors, dtype=np.uint8)
        if rgb.shape[0] == len(mesh.faces) and rgb.shape[1] >= 3:
            return rgb[:, :3]
    except (AttributeError, ValueError, TypeError):
        pass
    return np.tile(np.array([190, 190, 190], dtype=np.uint8), (len(mesh.faces), 1))


def _polygons_from_segments(segments: np.ndarray):
    """Polygonize a planar cross-section and retain odd-parity faces (supports holes)."""
    if len(segments) == 0:
        return []
    edges = []
    unique = set()
    for s in segments:
        a = tuple(np.round(s[0, :2], 5))
        b = tuple(np.round(s[1, :2], 5))
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key not in unique:
            unique.add(key)
            edges.append((a, b))
    if not edges:
        return []
    shapes = polygonize(unary_union([LineString(e) for e in edges]))
    a = np.asarray([e[0] for e in edges], dtype=float)
    b = np.asarray([e[1] for e in edges], dtype=float)
    chosen = []
    for polygon in shapes:
        p = polygon.representative_point()
        x, y = p.x, p.y
        crossing = (a[:, 1] > y) != (b[:, 1] > y)
        valid = crossing & (np.abs(b[:, 1] - a[:, 1]) > 1e-12)
        xr = np.empty(len(edges), dtype=float)
        xr.fill(-np.inf)
        xr[valid] = a[valid, 0] + (y - a[valid, 1]) * (b[valid, 0] - a[valid, 0]) / (b[valid, 1] - a[valid, 1])
        if (np.count_nonzero(valid & (x < xr)) % 2) == 1:
            chosen.append(polygon)
    return chosen


def _paint_polygon_mask(polys, size: tuple[int, int], pxy) -> Image.Image:
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for poly in polys:
        d.polygon([pxy(x, y) for x, y in poly.exterior.coords], fill=255)
        for inner in poly.interiors:
            d.polygon([pxy(x, y) for x, y in inner.coords], fill=0)
    return mask


def slice_model(filename: str | Path, output_dir: str | Path, settings: SliceSettings,
                on_progress=None, profile: MachineProfile | None = None) -> Path:
    """Generate numbered PNG masks and manifest; all operations are offline.

    Images share size/origin. Outputs are NOT directly printer-ready: RIP white and
    varnish spot channel mapping, ink deposition and placement must be calibrated.
    """
    settings.validate()
    machine = profile or MachineProfile()
    mesh, warnings = load_mesh(filename)
    original_dimensions = mesh.extents.copy()
    mesh = prepare_mesh(mesh, settings.target_height_mm)
    minimum, maximum = mesh.bounds
    model_width, model_depth, model_height = mesh.extents
    if model_height > 180:
        raise ValueError("Model exceeds 180 mm nominal media clearance; no verified 3D build envelope exists.")
    pixel_per_mm = settings.dpi / 25.4
    w = math.ceil((model_width + 2 * settings.margin_mm) * pixel_per_mm)
    h = math.ceil((model_depth + 2 * settings.margin_mm) * pixel_per_mm)
    if w <= 0 or h <= 0 or w * h > 16_000_000:
        raise ValueError("Raster exceeds 16 million pixels. Reduce model size or DPI.")
    if settings.placement_x_mm + (w / pixel_per_mm) > machine.bed_x_mm or settings.placement_y_mm + (h / pixel_per_mm) > machine.bed_y_mm:
        raise ValueError("Raster placement exceeds the configured 600×900 mm bed.")
    n = math.ceil(model_height / settings.layer_mm)
    if n > 2000:
        raise ValueError("More than 2000 layers; increase layer thickness for this prototype.")
    if n * w * h > 350_000_000:
        raise ValueError("Job too large for this prototype (over 350M layer-pixels).")

    def pxy(x: float, y: float) -> tuple[int, int]:
        return (round((x - minimum[0] + settings.margin_mm) * pixel_per_mm),
                round((maximum[1] - y + settings.margin_mm) * pixel_per_mm))

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export folder must be empty, to avoid mixing jobs.")
    out.mkdir(parents=True, exist_ok=True)
    colors = _face_colors(mesh)
    manifest = {
        "format": "UV3D_OFFLINE_JOB_V1",
        "safety": "PREVIEW ONLY. Never send raster masks directly to printheads or axes.",
        "source_name": Path(filename).name,
        "settings": asdict(settings),
        "original_dimensions_mm": original_dimensions.tolist(),
        "scaled_dimensions_mm": mesh.extents.tolist(),
        "uniform_scale_factor": float(mesh.extents[2] / original_dimensions[2]),
        "machine_profile": asdict(machine),
        "model_bounds_mm": [minimum.tolist(), maximum.tolist()],
        "raster_size_px": [w, h],
        "raster_footprint_mm": [w / pixel_per_mm, h / pixel_per_mm],
        "warnings": warnings + [
            "PrintExp .prt generation and printer API are NOT integrated.",
            "Supports, overhang analysis, true texture mapping and color management are NOT implemented.",
            "White and varnish are grayscale mask proposals; correct RIP spot channels must be validated.",
            "Layer thickness is a target, not validated cured material thickness."
        ],
        "layers": []
    }
    stroke_width = max(1, round(settings.shell_mm * pixel_per_mm))
    for index in range(n):
        z = min(minimum[2] + (index + 0.5) * settings.layer_mm, maximum[2] - 1e-6)
        segs, faces = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z], return_faces=True)
        polys = _polygons_from_segments(segs)
        mask = _paint_polygon_mask(polys, (w, h), pxy)
        white = mask if settings.body_material == "white" else Image.new("L", (w, h), 0)
        varnish = mask if settings.body_material == "varnish" else Image.new("L", (w, h), 0)
        strokes = Image.new("RGB", (w, h), (0, 0, 0))
        alpha = Image.new("L", (w, h), 0)
        cd = ImageDraw.Draw(strokes)
        ad = ImageDraw.Draw(alpha)
        for s, face_id in zip(segs, faces):
            xy = [pxy(*s[0, :2]), pxy(*s[1, :2])]
            rgb = tuple(int(v) for v in colors[face_id])
            cd.line(xy, fill=rgb, width=stroke_width)
            ad.line(xy, fill=255, width=stroke_width)
        alpha = ImageChops.multiply(alpha, mask)
        color = strokes.convert("RGBA")
        color.putalpha(alpha)
        name = f"{index + 1:05d}"
        white.save(out / f"{name}_white.png")
        varnish.save(out / f"{name}_varnish.png")
        color.save(out / f"{name}_color.png")
        manifest["layers"].append({
            "index": index + 1, "slice_center_z_mm": round(float(z), 6),
            "target_stack_height_mm": round(min((index + 1) * settings.layer_mm, model_height), 6),
            "white": f"{name}_white.png", "varnish": f"{name}_varnish.png", "color": f"{name}_color.png",
            "contains_geometry": bool(polys)
        })
        if on_progress:
            on_progress(index + 1, n)
    path = out / "job.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
