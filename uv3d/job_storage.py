"""Streaming job sizing and disk safety for UV3D.

Total pixels describe output work, not peak memory: the slicer holds the
original mesh plus *one* layer's rasters at a time. Never reject a valid
multi-layer job simply because total pixels exceed a static threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

MIB = 1024 ** 2
GIB = 1024 ** 3
MIN_FREE_RESERVE_BYTES = 512 * MIB


@dataclass(frozen=True)
class JobStoragePlan:
    width_px: int
    height_px: int
    layers: int
    png_files: int
    pixels_per_layer: int
    total_layer_pixels: int
    raw_channels_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


def calculate_job_plan(width_px: int, height_px: int, layers: int) -> JobStoragePlan:
    """Calculate descriptive totals without allocating any raster images.

    Two grayscale masks (white/varnish) and one 4-channel RGBA color PNG
    have six bytes per pixel *before* PNG compression. This is an uncompressed
    pixel-data equivalent, NOT a predicted on-disk file size.
    """
    values = (width_px, height_px, layers)
    if any(not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("Raster dimensions and layer count must be positive integers.")
    pixels = width_px * height_px
    return JobStoragePlan(
        width_px=width_px,
        height_px=height_px,
        layers=layers,
        png_files=layers * 3,
        pixels_per_layer=pixels,
        total_layer_pixels=pixels * layers,
        raw_channels_bytes=pixels * layers * 6,
    )


def output_drive_root(output_dir: str | Path) -> Path:
    """Get an existing ancestor, so disk_usage works before a job dir exists."""
    result = Path(output_dir).resolve(strict=False)
    while not result.exists():
        parent = result.parent
        if parent == result:
            raise OSError(f"No existing parent for output path: {output_dir}")
        result = parent
    return result


def ensure_disk_space(output_dir: str | Path, pixels_per_layer: int,
                      reserve_bytes: int = MIN_FREE_RESERVE_BYTES) -> int:
    """Check that one worst-case uncompressed layer fits, plus a reserve.

    The check is repeated before *every* layer. PNGs can compress very
    differently between models; no fixed total-job estimate can guarantee
    available disk. When space becomes critically low, stop cleanly with a
    descriptive error rather than filling the user's volume.
    """
    if pixels_per_layer <= 0:
        raise ValueError("Invalid per-layer raster size.")
    disk = shutil.disk_usage(output_drive_root(output_dir))
    # Generous margin for one layer of white, varnish, and color data.
    # PNG encoding overhead is small compared to these uncompressed buffers.
    safety_required = reserve_bytes + pixels_per_layer * 6 + 2 * MIB
    if disk.free < safety_required:
        need = safety_required / GIB
        available = disk.free / GIB
        raise OSError(
            f"Not enough free disk space for mask generation: "
            f"{available:.2f} GiB free; at least {need:.2f} GiB "
            f"required for the next layer and safety reserve. "
            f"Free some space on the output drive or move UV3D to a larger drive. "
            f"Already generated layers were kept for diagnosis."
        )
    return disk.free
