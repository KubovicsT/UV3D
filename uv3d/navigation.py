"""Supp-compatible camera controls for UV3D (independent of graphics backend).

The view is Z-up and never rolls. Match Supp's established mouse mapping:
right-drag orbit, middle-drag pan, wheel zoom, left click reserved for tools.
"""
from __future__ import annotations
import math

PITCH_LIMIT = math.radians(88.5)
ORBIT_PER_PIXEL = math.radians(0.38)
ZOOM_FACTOR = 1.18
PRESETS = {
    "Isometric": (math.radians(135.0), math.radians(35.264)),
    "Top": (math.radians(180), math.radians(88.5)),
    "Front": (math.radians(180), 0.0),
    "Right": (math.radians(90), 0.0),
}


def orbit(yaw: float, pitch: float, dx: float, dy_down: float) -> tuple[float, float]:
    """Screen coords are x-right, y-down (Qt); drag up raises the camera."""
    return (yaw - dx * ORBIT_PER_PIXEL,
            max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch - dy_down * ORBIT_PER_PIXEL)))


def zoom(current: float, steps: float) -> float:
    """In this viewport zoom is a magnification, not camera distance."""
    return max(.10, min(30., current * (ZOOM_FACTOR ** steps)))


def preset(name: str) -> tuple[float, float]:
    return PRESETS[name]
