"""Build a *display-only* 3D support surface from exported grayscale masks.

The full-resolution support PNGs remain authoritative. This module samples
them to a bounded 3D voxel grid and merges adjacent visible voxel faces into
rectangles. The resulting triangle surface can be tinted and alpha-blended to
let the original-color model remain visible behind support.

Runs in a background worker, not the Qt event loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SupportSurface:
    triangles: np.ndarray  # [N,3], GL_TRIANGLES, XYZ in model-millimeter coordinates
    sampled_levels: int
    grid_shape: tuple[int, int, int]


def merge_rectangles(mask: np.ndarray):
    """Yield maximal rectangles for runs shared across adjacent rows.

    Rectangle coordinates are half-open: (row0, row1, col0, col1).
    """
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D boolean mask.")
    active: dict[tuple[int, int], int] = {}
    for row_no, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        spans = {(int(a), int(b)) for a, b in zip(starts, ends)}
        for span in list(active):
            if span not in spans:
                yield (active.pop(span), row_no, span[0], span[1])
        for span in spans:
            active.setdefault(span, row_no)
    for span, start_row in active.items():
        yield (start_row, mask.shape[0], span[0], span[1])


def voxel_surface(volume: np.ndarray, x_edges: np.ndarray,
                  y_edges: np.ndarray, z_edges: np.ndarray,
                  max_vertices: int = 1_200_000) -> np.ndarray:
    """Generate a two-sided triangle surface for exposed solid cells.

    volume is [z, y, x]; y=0 is the TOP raster row (largest world Y).
    No full voxel array is sent to OpenGL; only visible faces are emitted.
    """
    solid = np.asarray(volume, dtype=bool)
    if solid.ndim != 3:
        raise ValueError("Expected support voxels of shape [z,y,x].")
    nz, ny, nx = solid.shape
    if len(x_edges) != nx + 1 or len(y_edges) != ny + 1 or len(z_edges) != nz + 1:
        raise ValueError("Support voxel grid and world-coordinate edges disagree.")
    coords = []
    count = 0

    def add_quad(p0, p1, p2, p3):
        nonlocal count
        if count + 6 > max_vertices:
            raise ValueError("Support preview exceeds geometry budget; reduce preview resolution.")
        coords.extend((p0, p1, p2, p0, p2, p3))
        count += 6

    blank_zy = np.zeros((nz, ny), dtype=bool)
    blank_zx = np.zeros((nz, nx), dtype=bool)
    blank_yx = np.zeros((ny, nx), dtype=bool)

    for ix in range(nx):
        current = solid[:, :, ix]
        for xface, neighbor in (
            (float(x_edges[ix]), solid[:, :, ix - 1] if ix else blank_zy),
            (float(x_edges[ix + 1]), solid[:, :, ix + 1] if ix + 1 < nx else blank_zy),
        ):
            exposed = current & ~neighbor
            if not np.any(exposed):
                continue
            for za, zb, ya, yb in merge_rectangles(exposed):
                add_quad((xface, y_edges[ya], z_edges[za]),
                         (xface, y_edges[yb], z_edges[za]),
                         (xface, y_edges[yb], z_edges[zb]),
                         (xface, y_edges[ya], z_edges[zb]))

    for iy in range(ny):
        current = solid[:, iy, :]
        for yface, neighbor in (
            (float(y_edges[iy]), solid[:, iy - 1, :] if iy else blank_zx),
            (float(y_edges[iy + 1]), solid[:, iy + 1, :] if iy + 1 < ny else blank_zx),
        ):
            exposed = current & ~neighbor
            if not np.any(exposed):
                continue
            for za, zb, xa, xb in merge_rectangles(exposed):
                add_quad((x_edges[xa], yface, z_edges[za]),
                         (x_edges[xb], yface, z_edges[za]),
                         (x_edges[xb], yface, z_edges[zb]),
                         (x_edges[xa], yface, z_edges[zb]))

    for iz in range(nz):
        current = solid[iz]
        for zface, neighbor in (
            (float(z_edges[iz]), solid[iz - 1] if iz else blank_yx),
            (float(z_edges[iz + 1]), solid[iz + 1] if iz + 1 < nz else blank_yx),
        ):
            exposed = current & ~neighbor
            if not np.any(exposed):
                continue
            for ya, yb, xa, xb in merge_rectangles(exposed):
                add_quad((x_edges[xa], y_edges[ya], zface),
                         (x_edges[xb], y_edges[ya], zface),
                         (x_edges[xb], y_edges[yb], zface),
                         (x_edges[xa], y_edges[yb], zface))

    if not coords:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def build_support_surface(job_path: str | Path, max_xy: int = 96,
                          max_levels: int = 96, max_vertices: int = 1_200_000
                          ) -> SupportSurface:
    """Read only sampled support layers (at most ~96) and build a preview mesh."""
    if not 12 <= max_xy <= 256 or not 2 <= max_levels <= 256:
        raise ValueError("Simulation preview resolutions are out of range.")
    job_path = Path(job_path)
    data = json.loads(job_path.read_text(encoding="utf-8"))
    layers = data.get("layers", [])
    if not layers or not data.get("support", {}).get("enabled"):
        return SupportSurface(np.empty((0, 3), dtype=np.float32), 0, (0, 0, 0))
    width_px, height_px = map(int, data["raster_size_px"])
    scale = min(1.0, max_xy / max(width_px, height_px))
    nx = max(2, round(width_px * scale))
    ny = max(2, round(height_px * scale))
    nz = min(max_levels, len(layers))
    # Every source layer contributes to the downsampled preview. Previously
    # one layer out of each 12+ layer band was sampled, so short support
    # features vanished and left false-looking exposed patches on the model.
    # Streaming max-union per Z band keeps peak memory small (~96³ booleans)
    # while preserving every support voxel at display resolution.
    edges = np.linspace(0, len(layers), nz + 1).astype(int)
    sampled = np.zeros((nz, ny, nx), dtype=bool)
    z_edges = np.zeros(nz + 1, dtype=np.float32)
    for k in range(nz):
        first, last = int(edges[k]), int(edges[k + 1])
        last = max(first + 1, last)
        for idx in range(first, last):
            record = layers[idx]
            support_filename = record.get("support")
            if not support_filename:
                raise ValueError(
                    "Support material channel is missing from this completed job.")
            filename = job_path.parent / support_filename
            png = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            if png is None or png.shape != (height_px, width_px):
                raise ValueError(f"Corrupt/inconsistent support PNG: {filename}")
            resized = cv2.resize(
                png, (nx, ny), interpolation=(
                    cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST))
            # Nonzero coverage survives resizing; union every physical Z
            # slice instead of sampling a single possibly empty slice.
            sampled[k] |= resized > 0
        z_edges[k + 1] = float(layers[last - 1]["target_stack_height_mm"])

    settings = data["settings"]
    ppm = float(settings["dpi"]) / 25.4
    margin = float(data.get("effective_raster_margin_mm", settings.get("margin_mm", 0)))
    world_bounds = data["model_bounds_mm"]
    x0 = float(world_bounds[0][0]) - margin
    y0 = float(world_bounds[1][1]) + margin
    x_edges = np.linspace(x0, x0 + width_px / ppm, nx + 1, dtype=np.float32)
    y_edges = np.linspace(y0, y0 - height_px / ppm, ny + 1, dtype=np.float32)
    triangles = voxel_surface(sampled, x_edges, y_edges, z_edges,
                              max_vertices=max_vertices)
    return SupportSurface(triangles, nz, sampled.shape)
