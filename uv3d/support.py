"""Conformal, self-supporting sacrificial material masks for offline UV3D jobs.

We generate solid model masks first, then traverse the stack from TOP to BED:
    envelope[z] = envelope[z+1] OR dilate(model[z], casing_radius)
    support[z] = envelope[z] AND NOT model[z]

This strictly enforces a material column underneath every future occupied
pixel, while creating a shape-following staircase rather than a full XYZ cube.

The output is a *fourth, unmapped material channel*, not white/varnish ink.
No assumption is made about water solubility, curing, accessibility or actual
printhead compatibility. Support sealed into an inaccessible cavity may become
impossible to remove. It is never sent directly to the printer.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .job_storage import ensure_disk_space


def effective_raster_margin(settings) -> float:
    """Give the chosen external casing room in the image without clipping it.

    Keep old behavior when support is disabled. The new margin is reported in
    the manifest; it is shared by all exported channels.
    """
    margin = float(settings.margin_mm)
    if settings.support_enabled:
        # Two output pixels between nominal casing and image border.
        margin = max(margin, float(settings.support_casing_mm) + 2 * 25.4 / settings.dpi)
    return margin


def radius_pixels(casing_mm: float, dpi: int) -> int:
    return max(0, int(round(float(casing_mm) * dpi / 25.4)))


def make_casing_kernel(radius: int):
    if radius <= 0:
        return None
    diameter = radius * 2 + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (diameter, diameter))


def support_layer(part: np.ndarray, envelope_above: np.ndarray, kernel=None):
    """One upper-to-lower step. Return (support, cumulative_envelope).

    Both inputs are uint8 masks, with zero=empty and nonzero=occupied.
    Part/support masks have NO overlapping pixels, including shell boundaries.
    The envelope is monotonically larger going toward the build plate.
    """
    if part.ndim != 2 or part.dtype != np.uint8:
        raise ValueError("Part raster must be a grayscale uint8 image.")
    if part.shape != envelope_above.shape or envelope_above.dtype != np.uint8:
        raise ValueError("Model and support envelope shapes must match.")
    if kernel is None:
        buffered = part
    else:
        buffered = cv2.dilate(part, kernel, iterations=1)
    envelope = cv2.bitwise_or(envelope_above, buffered)
    support = cv2.bitwise_and(envelope, cv2.bitwise_not(part))
    return support, envelope


def generate_support_masks(output_dir: str | Path, layer_entries: list[dict],
                           body_material: str, casing_mm: float, dpi: int,
                           layer_mm: float, on_progress=None,
                           top_cap_mm: float = 0.0) -> dict:
    """Emit numbered 8-bit support PNGs using O(one-layer) raster memory.

    Reads the solid-model channel saved by the slicer and writes support PNGs in
    descending Z order. The manifest is written only after this returns.
    Completed part/support PNGs remain available if the disk runs out.
    """
    if body_material not in ("white", "varnish"):
        raise ValueError("Unsupported solid-model source material.")
    if casing_mm < 0 or top_cap_mm < 0 or dpi <= 0 or layer_mm <= 0:
        raise ValueError("Invalid support construction parameters.")
    out = Path(output_dir)
    if not layer_entries:
        return {
            "support_pixels": 0, "solid_model_pixels": 0,
            "support_volume_mm3": 0.0, "solid_model_volume_mm3": 0.0,
            "cube_support_volume_mm3": 0.0, "saving_vs_cube_percent": 0.0,
        }
    first_path = out / layer_entries[-1][body_material]
    first = cv2.imdecode(np.fromfile(first_path, dtype=np.uint8),
                          cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Unable to read solid-model mask: {first_path}")
    height, width = first.shape
    envelope = np.zeros((height, width), dtype=np.uint8)
    kernel = make_casing_kernel(radius_pixels(casing_mm, dpi))
    mask_options = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    support_pixels = 0
    model_pixels = 0
    original_layers = len(layer_entries)
    cap_layers = int(np.ceil(top_cap_mm / layer_mm - 1e-9)) if top_cap_mm else 0
    total = original_layers + cap_layers

    for completed, layer in enumerate(reversed(layer_entries), start=1):
        ensure_disk_space(out, width * height)
        src = out / layer[body_material]
        part = first if completed == 1 else cv2.imdecode(
            np.fromfile(src, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if part is None or part.shape != (height, width):
            raise RuntimeError(f"Missing or inconsistent solid-model mask: {src}")
        support, envelope = support_layer(part, envelope, kernel)
        layer["support"] = f"{int(layer['index']):05d}_support.png"
        dest = out / layer["support"]
        success, png = cv2.imencode(".png", support, mask_options)
        if not success:
            raise OSError(f"Failed to encode support image: {dest}")
        dest.write_bytes(png.tobytes())
        support_pixels += cv2.countNonZero(support)
        model_pixels += cv2.countNonZero(part)
        layer["contains_support"] = bool(np.any(support))
        if on_progress:
            on_progress(completed, total)

    # A conventional sacrificial encasement needs a roof over the model too,
    # not only columns under overhangs. The roof uses the XY projection of
    # the entire model/support envelope, so even lower/side protrusions are
    # covered above the model. The cap is purely support material.
    if cap_layers:
        top_z = float(layer_entries[-1]["target_stack_height_mm"])
        blank_gray = cv2.imencode(".png", np.zeros_like(envelope), mask_options)[1].tobytes()
        blank_color = cv2.imencode(".png", np.zeros(
            (height, width, 4), dtype=np.uint8), mask_options)[1].tobytes()
        roof_bytes = cv2.imencode(".png", envelope, mask_options)[1].tobytes()
        roof_pixels = cv2.countNonZero(envelope)
        for number in range(1, cap_layers + 1):
            ensure_disk_space(out, width * height)
            index = original_layers + number
            name = f"{index:05d}"
            white = f"{name}_white.png"
            varnish = f"{name}_varnish.png"
            color = f"{name}_color.png"
            support_name = f"{name}_support.png"
            (out / white).write_bytes(blank_gray)
            (out / varnish).write_bytes(blank_gray)
            (out / color).write_bytes(blank_color)
            (out / support_name).write_bytes(roof_bytes)
            layer_entries.append({
                "index": index,
                "slice_center_z_mm": round(top_z + min((number - .5) * layer_mm,
                                                       top_cap_mm), 6),
                "target_stack_height_mm": round(top_z + min(number * layer_mm, top_cap_mm), 6),
                "white": white, "varnish": varnish, "color": color,
                "support": support_name,
                "contains_geometry": False,
                "contains_support": True,
                "support_only_top_cap": True,
            })
            support_pixels += roof_pixels
            if on_progress:
                on_progress(original_layers + number, total)

    # The bottom envelope is the XY footprint of *all* future occupied columns.
    # A traditional fully filled rectangular support block would occupy the
    # rectangular bounding box of this footprint, at every build height.
    ys, xs = np.nonzero(envelope)
    bbox_area_px = ((int(xs.max()) - int(xs.min()) + 1) *
                    (int(ys.max()) - int(ys.min()) + 1)) if len(xs) else 0
    cube_support_px = max(0, bbox_area_px * total - model_pixels)
    voxel_volume = (25.4 / dpi) ** 2 * layer_mm
    saving = (100 * (1 - support_pixels / cube_support_px)
              if cube_support_px > 0 else 0.0)
    return {
        "top_cap_layers": cap_layers,
        "top_cap_mm": round(float(top_cap_mm), 4),
        "support_pixels": int(support_pixels),
        "solid_model_pixels": int(model_pixels),
        "support_volume_mm3": round(support_pixels * voxel_volume, 3),
        "solid_model_volume_mm3": round(model_pixels * voxel_volume, 3),
        "cube_support_volume_mm3": round(cube_support_px * voxel_volume, 3),
        "saving_vs_cube_percent": round(max(0.0, saving), 2),
        "comparison": (
            "Comparison uses a solid rectangular XY bounding box around the "
            "support footprint for the full stack height; model pixels are "
            "excluded from both support totals. Pixel-volume approximation only."
        ),
    }
