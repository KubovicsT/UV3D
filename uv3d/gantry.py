"""Offline multi-head / UV lamp gantry configuration.

Coordinates are millimeters relative to the first head's center (x,y).
Printhead width is the 69.1-mm physical body across carriage X, depth 59.5
along Y, height 35.6; the effective printing swath is 33.8 mm.

These are *visual/planning* coordinates, not calibrated printer offsets or
Hoson commands. Defaults are illustrative, not measured on the user's machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

MATERIALS = ("color", "white", "varnish", "support")
CONFIG_VERSION = 1


def _check_number(value: float, label: str, low: float, high: float):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number.")
    if not low <= float(value) <= high:
        raise ValueError(f"{label} must be in {low}–{high} mm.")


@dataclass
class HeadConfig:
    name: str
    material: str
    offset_x_mm: float = 0
    offset_y_mm: float = 0
    width_mm: float = 69.1
    depth_mm: float = 59.5
    height_mm: float = 35.6
    swath_mm: float = 33.8
    enabled: bool = True

    def validate(self):
        if not self.name.strip():
            raise ValueError("Printhead name cannot be empty.")
        if self.material not in MATERIALS:
            raise ValueError(f"Invalid material for {self.name}.")
        for axis in ("x", "y"):
            _check_number(getattr(self, f"offset_{axis}_mm"),
                          f"{self.name} {axis} offset", -1000, 1000)
        for key in ("width_mm", "depth_mm", "height_mm", "swath_mm"):
            _check_number(getattr(self, key), f"{self.name} {key}", .1, 300)
        if not isinstance(self.enabled, bool):
            raise ValueError(f"{self.name}: enabled must be a checkbox value.")


@dataclass
class LampConfig:
    name: str
    offset_x_mm: float = 0
    offset_y_mm: float = 0
    width_mm: float = 40
    depth_mm: float = 45
    body_height_mm: float = 15
    enabled: bool = True

    def validate(self):
        if not self.name.strip():
            raise ValueError("UV lamp name cannot be empty.")
        for axis in ("x", "y"):
            _check_number(getattr(self, f"offset_{axis}_mm"),
                          f"{self.name} {axis} offset", -1000, 1000)
        for key in ("width_mm", "depth_mm", "body_height_mm"):
            _check_number(getattr(self, key), f"{self.name} {key}", .1, 300)
        if not isinstance(self.enabled, bool):
            raise ValueError(f"{self.name}: enabled must be a checkbox value.")


@dataclass
class GantryConfig:
    heads: list[HeadConfig] = field(default_factory=list)
    lamps: list[LampConfig] = field(default_factory=list)
    cure_passes: int = 1

    def validate(self):
        if len(self.heads) > 12 or len(self.lamps) > 12:
            raise ValueError("Maximum 12 heads and 12 lamps in offline preview.")
        if not any(h.enabled for h in self.heads):
            raise ValueError("At least one enabled printhead is required.")
        if isinstance(self.cure_passes, bool) or not 1 <= self.cure_passes <= 5:
            raise ValueError("UV cure sweeps per lamp must be 1–5.")
        for head in self.heads:
            head.validate()
        for lamp in self.lamps:
            lamp.validate()

    def as_dict(self):
        return {"version": CONFIG_VERSION,
                "heads": [asdict(h) for h in self.heads],
                "lamps": [asdict(l) for l in self.lamps],
                "cure_passes": self.cure_passes}

    @classmethod
    def from_dict(cls, raw: dict):
        if raw.get("version", CONFIG_VERSION) != CONFIG_VERSION:
            raise ValueError("Unsupported gantry configuration version.")
        result = cls(
            heads=[HeadConfig(**h) for h in raw.get("heads", [])],
            lamps=[LampConfig(**l) for l in raw.get("lamps", [])],
            cure_passes=int(raw.get("cure_passes", 1)))
        result.validate()
        return result


def default_gantry():
    """Three i3200-U1 heads and two NOT MEASURED lamp placeholders."""
    return GantryConfig(
        heads=[
            HeadConfig("Head 1 / CMYK", "color", 0, 0),
            HeadConfig("Head 2 / white", "white", 70, 0),
            HeadConfig("Head 3 / varnish", "varnish", 140, 0),
        ],
        lamps=[
            LampConfig("UV lamp L (example)", -80, 0, 40, 45),
            LampConfig("UV lamp R (example)", 220, 0, 40, 45),
        ],
        cure_passes=1)


def gantry_file(root: str | Path) -> Path:
    return Path(root) / "workspace" / "machine" / "gantry.json"


def load_gantry(root: str | Path) -> GantryConfig:
    path = gantry_file(root)
    if not path.is_file():
        result = default_gantry()
        result.validate()
        return result
    return GantryConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_gantry(root: str | Path, config: GantryConfig) -> Path:
    config.validate()
    path = gantry_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config.as_dict(), indent=2),
                         encoding="utf-8")
    temporary.replace(path)
    return path
