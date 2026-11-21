"""Fast offline slicer for UV3D: no printer or motor commands.

Uses sorted Z bounds to exclude irrelevant faces, C-level triangle intersection,
linear contour stitching, and OpenCV for raster/PNG output. Keeps the existing
white / varnish / color PNG naming and JSON format.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .core import (
    MachineProfile, SliceSettings, load_mesh, prepare_mesh,
    _face_colors, _polygons_from_segments,
)
from .job_storage import calculate_job_plan, ensure_disk_space
from .support import effective_raster_margin, generate_support_masks


def _contours_from_segments(segments: np.ndarray, tolerance: float = 1e-5):
    """Stitch ordinary manifold slice edges into loops in O(edges).

    None means the graph is ambiguous; then we use the slower GEOS fallback
    rather than silently inventing a closed region.
    """
    if len(segments) == 0:
        return []
    xy = np.asarray(segments, dtype=np.float64)[:, :, :2]
    if not np.isfinite(xy).all():
        return None
    quantized = np.rint(xy.reshape(-1, 2) / tolerance).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    edges = inverse.reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) < 3:
        return None
    if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
        return None
    coords = np.zeros((int(inverse.max()) + 1, 2), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(coords))
    np.add.at(coords, inverse, xy.reshape(-1, 2))
    coords /= counts[:, None]
    graph = {}
    for a, b in edges:
        a, b = int(a), int(b)
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    if any(len(nbrs) != 2 for nbrs in graph.values()):
        return None
    visited = set()
    loops = []
    for start in graph:
        if start in visited:
            continue
        contour = []
        previous, current = -1, start
        while True:
            if current in visited:
                if current != start:
                    return None
                break
            visited.add(current)
            contour.append(coords[current])
            nbrs = graph[current]
            following = nbrs[1] if nbrs[0] == previous else nbrs[0]
            previous, current = current, following
        if len(contour) < 3:
            return None
        loops.append(np.asarray(contour))
    return loops


def _contour_pixels(segments, minimum, maximum, pixels_per_mm, margin_mm):
    loops = _contours_from_segments(segments)
    if loops is None:
        # For non-manifold / degenerate slices use the older GEOS polygonizer.
        polygons = _polygons_from_segments(segments)
        loops = []
        for polygon in polygons:
            loops.extend([np.asarray(polygon.exterior.coords),
                          *[np.asarray(r.coords) for r in polygon.interiors]])
    output = []
    for loop in loops:
        points = np.asarray(loop)
        xy = np.column_stack((
            (points[:, 0] - minimum[0] + margin_mm) * pixels_per_mm,
            (maximum[1] - points[:, 1] + margin_mm) * pixels_per_mm,
        ))
        output.append(np.rint(xy).astype(np.int32).reshape(-1, 1, 2))
    return output


def slice_fast(filename, output_dir, settings: SliceSettings,
               on_progress=None, profile=None, on_plan=None, on_stage=None):
    """Generate the same offline masks and manifest as the legacy slicer."""
    settings.validate()
    machine = profile or MachineProfile()
    mesh, warnings = load_mesh(filename)
    original_dimensions = mesh.extents.copy()
    mesh = prepare_mesh(mesh, settings.target_height_mm)
    minimum, maximum = mesh.bounds
    width_mm, depth_mm, height_mm = (float(x) for x in mesh.extents)
    if height_mm + (settings.support_top_cap_mm if settings.support_enabled else 0) > 180:
        raise ValueError("Model plus support top cover exceeds 180 mm nominal media clearance; "
                         "no verified 3D build envelope exists.")
    ppm = settings.dpi / 25.4
    raster_margin = effective_raster_margin(settings)
    w = math.ceil((width_mm + 2 * raster_margin) * ppm)
    h = math.ceil((depth_mm + 2 * raster_margin) * ppm)
    if w <= 0 or h <= 0 or w * h > 16_000_000:
        raise ValueError("Raster exceeds 16 million pixels. Reduce model size or DPI.")
    if (settings.placement_x_mm + w / ppm > machine.bed_x_mm
            or settings.placement_y_mm + h / ppm > machine.bed_y_mm):
        raise ValueError("Raster placement exceeds the configured 600×900 mm bed.")
    n = math.ceil(height_mm / settings.layer_mm)
    # Total pixel count affects runtime and disk use, not peak raster RAM:
    # only the active layer is rendered, encoded, then written to disk.
    cap_layers = (math.ceil(settings.support_top_cap_mm / settings.layer_mm - 1e-9)
                  if settings.support_enabled and settings.support_top_cap_mm else 0)
    plan = calculate_job_plan(w, h, n + cap_layers, support_enabled=settings.support_enabled)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export folder must be empty, to avoid mixing jobs.")
    ensure_disk_space(out, plan.pixels_per_layer)
    if on_plan:
        on_plan(plan)
    out.mkdir(parents=True, exist_ok=True)
    colors = _face_colors(mesh)
    manifest = {
        "format": "UV3D_OFFLINE_JOB_V1",
        "safety": "PREVIEW ONLY. Never send raster masks directly to printheads or axes.",
        "source_name": Path(filename).name,
        "source_path": str(Path(filename).resolve()),  # Local-only offline job metadata.
        "settings": asdict(settings),
        "original_dimensions_mm": original_dimensions.tolist(),
        "scaled_dimensions_mm": mesh.extents.tolist(),
        "uniform_scale_factor": float(mesh.extents[2] / original_dimensions[2]),
        "machine_profile": asdict(machine),
        "model_bounds_mm": [minimum.tolist(), maximum.tolist()],
        "raster_size_px": [w, h],
        "raster_footprint_mm": [w / ppm, h / ppm],
        "effective_raster_margin_mm": raster_margin,
        "solid_volume": True,
        "support": {"enabled": settings.support_enabled, "casing_mm": settings.support_casing_mm,
                    "top_cap_mm": settings.support_top_cap_mm,
                    "material": "UNASSIGNED - requires separately validated removable support ink"},
        "storage_plan": plan.as_dict(),
        "warnings": warnings + [
            "PrintExp .prt generation and printer API are NOT integrated.",
            "Experimental conformal support geometry is not validated for real material deposition or dissolving.",
            "Support trapped in sealed cavities may not be removable.",
            "Overhang stability, curing and color management are NOT verified.",
            "White and varnish are grayscale mask proposals; correct RIP spot channels must be validated.",
            "Layer thickness is a target, not validated cured material thickness."
        ],
        "layers": [],
    }

    # Sorted triangle Z bounds: only triangles that can cross the selected
    # plane are passed to the intersection routine. Vertex dots are reused.
    vertex_z = np.asarray(mesh.vertices[:, 2])
    face_z = vertex_z[mesh.faces]
    face_min = np.min(face_z, axis=1)
    face_max = np.max(face_z, axis=1)
    low_order = np.argsort(face_min)
    high_order = np.argsort(face_max)
    low_z = face_min[low_order]
    high_z = face_max[high_order]
    active = np.zeros(len(mesh.faces), dtype=bool)
    low_cursor = high_cursor = 0
    outline_width = max(1, round(settings.shell_mm * ppm))
    png_options = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    def encode(image):
        success, data = cv2.imencode(".png", image, png_options)
        if not success:
            raise RuntimeError("PNG encoding failed.")
        return data.tobytes()

    blank_gray = encode(np.zeros((h, w), dtype=np.uint8))
    blank_color = encode(np.zeros((h, w, 4), dtype=np.uint8))
    prev_mask = None
    prev_encoded = None

    progress_total = n * (2 if settings.support_enabled else 1) + cap_layers
    if on_stage:
        on_stage("Generating solid model volume masks and exterior color boundaries")
    for idx in range(n):
        ensure_disk_space(out, plan.pixels_per_layer)
        z = min(minimum[2] + (idx + .5) * settings.layer_mm, maximum[2] - 1e-6)
        next_low = np.searchsorted(low_z, z, side="right")
        if next_low > low_cursor:
            active[low_order[low_cursor:next_low]] = True
            low_cursor = next_low
        next_high = np.searchsorted(high_z, z, side="left")
        if next_high > high_cursor:
            active[high_order[high_cursor:next_high]] = False
            high_cursor = next_high
        face_ids = np.flatnonzero(active)
        if len(face_ids):
            segments, hit_faces = trimesh.intersections.mesh_plane(
                mesh, [0, 0, 1], [0, 0, z],
                return_faces=True, local_faces=face_ids,
                cached_dots=vertex_z - z)
        else:
            segments = np.empty((0, 2, 3), dtype=float)
            hit_faces = np.empty((0,), dtype=np.int64)

        mask = np.zeros((h, w), dtype=np.uint8)
        if len(segments):
            contours = _contour_pixels(
                segments, minimum, maximum, ppm, raster_margin)
            if contours:
                # One call uses even/odd filling; inner loops remain hollow.
                cv2.fillPoly(mask, contours, 255, lineType=cv2.LINE_8)
        filled = bool(np.any(mask))
        if not filled:
            mask_bytes = blank_gray
        elif prev_mask is not None and np.array_equal(prev_mask, mask):
            mask_bytes = prev_encoded
        else:
            mask_bytes = encode(mask)
            prev_mask = mask.copy()
            prev_encoded = mask_bytes

        if filled:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            xy = np.rint(np.stack((
                (segments[:, :, 0] - minimum[0] + raster_margin) * ppm,
                (maximum[1] - segments[:, :, 1] + raster_margin) * ppm,
            ), axis=-1)).astype(np.int32)
            for segment, face_id in zip(xy, hit_faces):
                r, g, b = (int(value) for value in colors[face_id])
                # OpenCV expects BGRA; output PNG is standard RGBA.
                cv2.line(rgba, tuple(segment[0]), tuple(segment[1]),
                         (b, g, r, 255), thickness=outline_width,
                         lineType=cv2.LINE_8)
            rgba[:, :, 3] = cv2.bitwise_and(rgba[:, :, 3], mask)
            color_bytes = encode(rgba)
        else:
            color_bytes = blank_color

        name = f"{idx + 1:05d}"
        (out / f"{name}_white.png").write_bytes(
            mask_bytes if settings.body_material == "white" else blank_gray)
        (out / f"{name}_varnish.png").write_bytes(
            mask_bytes if settings.body_material == "varnish" else blank_gray)
        (out / f"{name}_color.png").write_bytes(color_bytes)
        manifest["layers"].append({
            "index": idx + 1,
            "slice_center_z_mm": round(float(z), 6),
            "target_stack_height_mm": round(
                min((idx + 1) * settings.layer_mm, height_mm), 6),
            "white": f"{name}_white.png",
            "varnish": f"{name}_varnish.png",
            "color": f"{name}_color.png",
            "contains_geometry": filled,
        })
        if on_progress:
            on_progress(idx + 1, progress_total)
    if settings.support_enabled:
        if on_stage:
            on_stage("Generating conformal support masks from top down to build plate")
        def support_progress(completed, _total):
            if on_progress:
                on_progress(n + completed, progress_total)
        manifest["support"]["statistics"] = generate_support_masks(
            out, manifest["layers"], settings.body_material,
            settings.support_casing_mm, settings.dpi, settings.layer_mm,
            on_progress=support_progress, top_cap_mm=settings.support_top_cap_mm)
    job = out / "job.json"
    job.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return job
