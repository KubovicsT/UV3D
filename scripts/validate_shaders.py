"""Compile-check GLSL source strings without opening a GPU context.

Requires the open-source glslangValidator command (CI only, not UV3D runtime).
"""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

# Run directly from scripts/ in CI without requiring a system-wide install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uv3d.gpu_viewport import VERTEX_SHADER, FRAGMENT_SHADER
from uv3d.simulation_viewport import SIM_LAYER_VERTEX, SIM_LAYER_FRAGMENT


def main():
    shaders = {
        "model.vert": (VERTEX_SHADER, "vert"),
        "model.frag": (FRAGMENT_SHADER, "frag"),
        "print_layer.vert": (SIM_LAYER_VERTEX, "vert"),
        "print_layer.frag": (SIM_LAYER_FRAGMENT, "frag"),
    }
    with TemporaryDirectory() as directory:
        for name, (source, stage) in shaders.items():
            filename = Path(directory) / name
            filename.write_text(source, encoding="utf-8")
            result = subprocess.run(
                ["glslangValidator", "-S", stage, str(filename)],
                capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"{name} failed GLSL shader validation:\n"
                    f"{result.stdout}\n{result.stderr}")
            print(f"{name}: GLSL validation passed")


if __name__ == "__main__":
    main()
