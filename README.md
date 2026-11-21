# UV3D

**Prototype v0.4 — offline experimental UV slicing. No printhead, motor, UV lamp, or printer control is implemented.**

UV3D targets the Gongzheng H6090EI (three Epson i3200 heads, CMYK/white/varnish), with FlexiPRINT MiniTX GongZheng Edition and Gongzheng PrintExp. It is independent of both vendor applications.

## Features

- Windows desktop GUI to import STL, OBJ, GLB, GLTF, 3MF and PLY triangle meshes.
- **Responsive background imports:** load meshes and simplify preview geometry without blocking the Windows UI. A status message and indeterminate progress bar show which stage is running, and the log records timings. On Windows with OpenGL, the import worker prepares **all original triangles** for the graphics card rather than creating a reduced mesh. The MIT-licensed `fast-simplification` library is retained only for the software fallback. All libraries, including PyOpenGL and OpenCV, are installed into the existing UV3D `runtime/` directory, not system-wide.
- Interactive orbitable **3D build viewport**: left-drag orbit, right/middle-drag pan, wheel zoom, Reset view.
- **Set model height in millimeters**: the mesh is scaled uniformly in X/Y/Z and the exact same transformed geometry is used for both preview and exported slices.
- Slicer-style **vertical layer slider**, synchronized numeric layer selector, translucent full-model silhouette, visible solid portion below the slice and amber highlighted cross-section **before export**.
- **Supp-style view controls and appearance:** right-drag orbit, middle-drag pan, wheel zoom, Z-up camera with no roll, isometric/top/front/right/fit presets, charcoal-gray background, build grid and XYZ axis markers. Left click is reserved for future selection and transform tools.
- **Windows 3D viewport: full original mesh resolution.** The OpenGL 3.3 viewer uploads the original mesh triangles and uses the graphics card for orbit, pan, zoom and live layer clipping, with **no simplification**. Slice boundaries are calculated off the main thread.
- Headless test machines and unsupported display environments retain the lower-detail software renderer as a fallback. GPU viewer requires functioning OpenGL 3.3 drivers and enough available graphics memory; Windows graphics driver support must be verified on the actual machine.
- Exported slices always use the complete original mesh geometry, regardless of viewer.
- **Crash-resistant layer generation:** masks are generated in a **separate Python process**, so a native OpenCV/geometry failure can terminate the worker without closing UV3D. The main window shows progress, a Cancel button and an error dialog with a log path. Every job gets a diagnostic log under `workspace/logs`; incomplete output remains separated under `workspace/jobs`.
- **Compatibility slicing:** optional checkbox switches to the older, slower PIL/GEOS workflow if a particular model fails in the accelerated worker.
- **Faster layer generation:** Z-range face filtering, loop stitching, OpenCV filling/PNG encoding, and reuse of unchanged/blank layer masks. Still produces same-size white, varnish and approximate colored boundary PNGs for every layer, plus a JSON job manifest.
- Support hollow regions and warn about open geometry.
- **Large jobs are supported without a fixed total-layer-pixel cap.** The slicer streams one raster layer at a time and saves it before moving to the next, rather than holding all layers in memory. A 60.021 mm tall, 0.05 mm, 300 DPI model can therefore generate 1,201 layers without the former 350-million-layer-pixel rejection.
- Storage safety: before starting and before every layer, UV3D checks that its output volume has enough free space for one uncompressed layer plus a 512 MiB safety reserve. If space runs low it stops with a clear error and preserves the partial job. The slicer log shows layer dimensions, expected PNG count and uncompressed pixel-data equivalent; actual PNG file sizes vary with compression.
- Preview exported 2D masks by channel, inspect layers inside the 3D view and simulate job progress with **no printer commands**.
- Read-only machine profile: 600 x 900 mm nominal bed, 3 heads, 4,841.455 Z motor pulses/mm (observed factory configuration).
- **Update** button: portable Git fetches origin/main, fast-forwards only, refreshes folder-local dependencies if necessary, then restarts. No UV3D releases.

**Not implemented:** PrintExp .prt export, RIP white/varnish channel mapping, hardware API, Z movement, completion feedback, UV curing and real ink thickness, accurate UV textures, actual full-color printing, supports, or automatic printer calibration. PNGs are geometric **proposals**, not printer-ready files. Never send these files directly to a head controller.

## Windows install — single application folder

Default suggested location: X:\UV3D; choose any other writable folder if preferred. Windows 10/11 64-bit required.

1. Download the GitHub **main branch ZIP** using Code → Download ZIP. This is not a release.
2. Extract the *contents* of the ZIP's UV3D-main folder into X:\UV3D (or your chosen folder).
3. Open PowerShell in that folder and run:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-UV3D.ps1

4. The script downloads portable Git and official Python, puts both under runtime\, checks out main and installs the Python wheels into runtime\site-packages. No separate Python, Git, or Qt installation is needed.
5. Launch from Start-UV3D.cmd. Inside the app, click **Update from GitHub main** for future changes. This now opens a visible PowerShell window with step-by-step progress and any errors (the window remains open so you can read it).

### Updating manually

Close UV3D, then double-click `Update-UV3D.cmd` in the application folder. It shows Git fetch progress, dependency checks, the installed commit and any errors. It remains open at the end.

Or use PowerShell directly:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "X:\UV3D\Update-UV3D.ps1"

Run from your actual installation folder if different. Pass `-NoRestart` if you only want to update without relaunching UV3D. Logs are saved to `workspace\logs\update.log` inside the app. Existing `workspace\` and `runtime\` are preserved; updates do not force-reset or create GitHub releases.

The installer does not register system-wide Python or Git, modify the global PATH, or install packages into Program Files or AppData. UV3D uses runtime\tmp and runtime\pip-cache for its managed working files. Jobs and logs go in workspace\. The OS and third-party components may still create their own caches elsewhere; we cannot promise to override Windows itself.

### Folder layout

    X:\UV3D\
      .git\                 Repository main checkout
      uv3d\                 Python application
      Install-UV3D.ps1      Portable initial setup / repair
      Start-UV3D.cmd        Launcher
      Update-UV3D.ps1       Fast-forward updater with visible progress
      Update-UV3D.cmd       Double-click update launcher
      requirements.txt      Local Python dependencies
      runtime\              Portable Git/Python, packages, downloads and cache (git-ignored)
      workspace\            User jobs, logs and working files (git-ignored)

**Important first-time ZIP installation:** The initial setup converts a fresh extracted ZIP into a Git checkout using a hard reset to origin/main. Do not place other files in this folder before initial setup. Later Update operations use a non-destructive fast-forward and refuse modified tracked source files. Runtime and workspace survive updates.

## Development and tests

For developer machines that already have Python, install requirements and pytest in a dev environment and run:

    python -m pytest -q

The production app doesn't rely on system Python. MIT licensed: see LICENSE.

## Known limitations

- Non-manifold/open meshes may produce incomplete polygons.
- GPU display is un-decimated, but requires OpenGL 3.3 and enough video memory for the original mesh. Very large or per-face-colored meshes can use substantial video memory. Headless/software fallback previews remain simplified.
- Height resizing is **uniform** rather than Z-only stretching; X and Y dimensions change proportionally. Changing model height now reuses the original preview mesh instead of reprocessing the entire file. Existing job previews are invalidated when height or slice settings change.
- Textured mesh colors are not reproduced accurately. Surface colors are only approximated along visible layer boundaries.
- White/varnish masks need verified mapping to real RIP spot channels.
- Registration in actual printer coordinates, ink deposition, mechanical Z positioning, cured thickness, software travel limits, and collision protection need validation.
- Nominal media thickness is not a verified free-standing 3D build volume.

## Diagnosing slow imports

Open a model, then check the import messages in the slicer log. It records **geometry load seconds** and **preview preparation seconds** separately so you can see which stage needs additional optimization. Large files may still take time to read or upload to GPU memory, but the import worker keeps most of the processing off the UI thread. Only one import/export task runs at a time; the program intentionally prevents shutdown while the worker is active.

## Large-job slicing and disk space

Layer-pixels = raster width × raster height × number of slices. This estimates how much *work* a job contains; it is not the amount of RAM the slicer needs. Previously UV3D refused jobs exceeding 350 million layer-pixels, despite generating one slice at a time. That arbitrary rejection has been removed in both slicing modes.

**Example:** a 44.52 × 44.87 × 60.021 mm model at 300 DPI, 3 mm raster margins and 0.05 mm layers produces about 597 × 601 pixels × 1,201 layers: approximately 431 million layer-pixels and 3,603 mask PNGs. UV3D now accepts that job (subject to output drive capacity, source geometry, and the still-enforced 16-million-pixels-per-layer memory guard).

UV3D checks the available space on the drive containing `workspace/jobs` at the start and again before each layer. It reserves at least 512 MiB plus one layer's uncompressed buffers for safety. Large exports may take significant time and generate thousands of files. The program reports how many PNGs it expects and estimates the raw equivalent of all channels, **not** the final compressed disk usage.

## Layer-generator crashes and logs

For any mask-generation failure, UV3D remains open and displays a dialog instead of closing silently. The log path appears in the dialog and left-hand activity log, normally:

    X:\UV3D\workspace\logs\<model>_<timestamp>.log

The file records the job, processing mode and most recent completed layer, plus Python tracebacks or a fatal-error thread dump when available. A native Windows termination or out-of-memory kill may leave only the last successful layer and an exit code; no error-reporting mechanism can guarantee a stack trace for every system crash.

Unexpected exceptions or native faults in the main **3D viewer/Qt process** are logged separately (when possible) to `workspace/logs/uv3d-gui-crash.log`. If UV3D itself closes, share that file along with the latest layer-job log.

If a particular model fails with accelerated slicing, check **Compatibility slicing**, retry, and share the diagnostic log if the failure persists. Jobs are never sent to printer hardware. The separate worker is intended to isolate slicer faults, not fix every possible source of bad geometry or memory exhaustion.

## Next milestones

1. Improve 3D manipulation: rotation/orientation, placement and accurate UV texture colors (uniform height resizing and layer preview are implemented).
2. Validate a FlexiPRINT MiniTX test-layer job and stable XY registration.
3. Obtain supported Hoson/PrintExp SDK or documented status API.
4. Measure physical ink builds and investigate guarded supervised Z stepping.
