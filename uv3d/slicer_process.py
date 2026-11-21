"""Isolated UV3D mask generator.

Run with:
    python -m uv3d.slicer_process --request X:/UV3D/workspace/tasks/job.json

Structured JSON progress is printed to stdout, flushed per event.
Python exceptions and native crash tracebacks go to a log in workspace/logs.
This process never connects to a printer.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import traceback


def report(kind: str, **kwargs) -> None:
    print(json.dumps({"type": kind, **kwargs}), flush=True)


def run_request(request_path: str | Path) -> int:
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(request["source"])
    output = Path(request["output"])
    logs = Path(request["log_dir"])
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (request_path.stem + ".log")
    # Keep the fault log open throughout native C/C++ calls. A fatal signal
    # dumps the Python thread stacks here even if Python cannot catch it.
    with log_path.open("a", encoding="utf-8", buffering=1) as logfile:
        faulthandler.enable(file=logfile, all_threads=True)
        logfile.write(f"UV3D slicing started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logfile.write(f"Source: {source}\nOutput: {output}\n")
        start = time.perf_counter()
        last_logged = 0
        # Import native dependencies after fault handler is active, so even
        # startup faults have a diagnostic trace and an existing log file.
        from .core import SliceSettings, slice_model, _slice_model_legacy

        def on_progress(current: int, total: int) -> None:
            nonlocal last_logged
            # Avoid thousands of expensive stdout wake-ups for big jobs.
            if current == total or current == 1 or current - last_logged >= max(1, total // 200):
                last_logged = current
                report("progress", current=current, total=total,
                       elapsed_seconds=round(time.perf_counter() - start, 2))
                logfile.write(f"layer {current}/{total}\n")

        def on_stage(message: str) -> None:
            logfile.write("STAGE: " + message + "\n")
            report("stage", message=message)

        def on_plan(plan) -> None:
            message = (
                f"Preparing {plan.layers:,} layers at "
                f"{plan.width_px:,} × {plan.height_px:,} px; "
                f"{plan.png_files:,} output PNGs; "
                f"{plan.total_layer_pixels:,} total layer-pixels. "
                f"Raw channel data equivalent: "
                f"{plan.raw_channels_bytes / (1024 ** 3):.2f} GiB "
                f"(actual PNG sizes depend on compression)."
            )
            report("stage", message=message)
            logfile.write(message + "\n")

        try:
            options = SliceSettings(**request["settings"])
            report("stage", message="Loading source and preparing layers")
            compatibility_mode = bool(request.get("compatibility_mode", False))
            logfile.write("Mode: " + ("compatibility" if compatibility_mode else "accelerated") + "\n")
            report("stage", message="Using " + ("compatibility" if compatibility_mode else "accelerated") + " slicing")
            fn = _slice_model_legacy if compatibility_mode else slice_model
            result = fn(source, output, options, on_progress=on_progress,
                        on_plan=on_plan, on_stage=on_stage)
            if not result.exists():
                raise RuntimeError("Slicer did not produce a job manifest")
            logfile.write("SUCCESS\n")
            report("finished", job=str(result),
                   elapsed_seconds=round(time.perf_counter() - start, 3))
            return 0
        except BaseException as exc:
            # Python-level errors are fully shown. Native crashes cannot be
            # caught, but leave a nonzero process status and faulthandler log.
            error = "".join(traceback.format_exception(exc))
            logfile.write("FAILED:\n" + error + "\n")
            report("error", message=f"{type(exc).__name__}: {exc}",
                   log=str(log_path))
            return 1
        finally:
            faulthandler.disable()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    return run_request(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
