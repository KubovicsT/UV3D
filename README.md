# UV3D

**Prototype v0.2 — offline experimental UV slicing. No printhead, motor, UV lamp, or printer control is implemented.**

UV3D targets the Gongzheng H6090EI (three Epson i3200 heads, CMYK/white/varnish), with FlexiPRINT MiniTX GongZheng Edition and Gongzheng PrintExp. It is independent of both vendor applications.

## Features

- Windows desktop GUI to import STL, OBJ, GLB, GLTF, 3MF and PLY triangle meshes.
- Interactive orbitable **3D build viewport**: left-drag orbit, right/middle-drag pan, wheel zoom, Reset view.
- **Set model height in millimeters**: the mesh is scaled uniformly in X/Y/Z and the exact same transformed geometry is used for both preview and exported slices.
- Slicer-style **vertical layer slider**, synchronized numeric layer selector, translucent full-model silhouette, visible solid portion below the slice and amber highlighted cross-section **before export**.
- The 3D view uses Qt software painting (no extra 3D graphics libraries or separate drivers). To stay responsive, it samples high-triangle-count meshes for surface rendering; the cross-section and exported masks use the complete geometry.
- Generate same-size white, varnish and approximate colored boundary PNG masks for each horizontal slice, plus a JSON job manifest.
- Support hollow regions, warn about open geometry and reject oversized jobs.
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
- The viewport is an approximate software-rendered 3D preview; complex meshes can render more slowly, and displayed exterior faces are sampled for performance. Exported slices are calculated from full-resolution geometry.
- Height resizing is **uniform** rather than Z-only stretching; X and Y dimensions change proportionally. Existing job previews are invalidated when height or slice settings change.
- Textured mesh colors are not reproduced accurately. Surface colors are only approximated along visible layer boundaries.
- White/varnish masks need verified mapping to real RIP spot channels.
- Registration in actual printer coordinates, ink deposition, mechanical Z positioning, cured thickness, software travel limits, and collision protection need validation.
- Nominal media thickness is not a verified free-standing 3D build volume.

## Next milestones

1. Improve 3D manipulation: rotation/orientation, placement and accurate UV texture colors (uniform height resizing and layer preview are implemented).
2. Validate a FlexiPRINT MiniTX test-layer job and stable XY registration.
3. Obtain supported Hoson/PrintExp SDK or documented status API.
4. Measure physical ink builds and investigate guarded supervised Z stepping.
