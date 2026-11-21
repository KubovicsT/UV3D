"""3D support wireframe preview from conformal support PNG masks.

Only a sparse set of Z levels is read; this does not alter the actual
full-resolution support masks. All geometry stays in the coordinate system of
the uniformly scaled model used by the GPU viewer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


def build_support_wireframe(job_json: str | Path, max_levels: int = 40,
                            max_vertices: int = 150_000) -> np.ndarray:
    """Return GL_LINES vertices [N,3] for sampled support outlines.

    Contours are displayed as horizontal, translucent purple rings. This is
    a *wireframe*, NOT a watertight mesh or a replacement for the saved masks.
    Sampling avoids loading an entire 1200+ layer support voxel grid into RAM.
    """
    if max_levels < 2 or max_vertices < 4:
        raise ValueError("Invalid preview sampling limits")
    path = Path(job_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data.get("layers", [])
    if not data.get("support", {}).get("enabled") or not layers:
        return np.empty((0, 3), dtype=np.float32)
    ppm = float(data["settings"]["dpi"]) / 25.4
    margin = float(data["effective_raster_margin_mm"])
    minimum, maximum = np.asarray(data["model_bounds_mm"], dtype=float)
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Invalid job raster DPI")
    sample_ids = np.unique(np.rint(np.linspace(
        0, len(layers) - 1, min(len(layers), max_levels))).astype(int))
    sets = []
    remaining = max_vertices
    for index in sample_ids:
        layer = layers[index]
        filename = layer.get("support")
        if not filename:
            continue
        filepath = path.parent / filename
        # cv2.imdecode + numpy.fromfile handle Unicode Windows paths.
        image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8),
                             cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read support mask for 3D preview: {filepath}")
        contours, _ = cv2.findContours(image, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        z = float(layer["slice_center_z_mm"])
        for contour in contours:
            if remaining < 4:
                break
            simplified = cv2.approxPolyDP(contour, epsilon=1.4, closed=True)
            points = simplified.reshape(-1, 2)
            if len(points) < 2:
                continue
            if len(points) * 2 > remaining:
                # Drop tiny details rather than running out of preview memory.
                continue
            xyz = np.empty((len(points), 3), dtype=np.float32)
            xyz[:, 0] = minimum[0] + points[:, 0] / ppm - margin
            xyz[:, 1] = maximum[1] - points[:, 1] / ppm + margin
            xyz[:, 2] = z
            edges = np.empty((len(points) * 2, 3), dtype=np.float32)
            edges[0::2] = xyz
            edges[1::2] = np.roll(xyz, -1, axis=0)
            sets.append(edges)
            remaining -= len(edges)
    if not sets:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(sets, axis=0)
