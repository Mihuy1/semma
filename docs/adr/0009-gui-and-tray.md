# ADR 0009: Desktop GUI, Onboarding, Theming, and System Tray

## Status

Accepted

## Context

Semma ships a PySide6 (Qt Widgets) settings/status window, first-run onboarding wizard, and system tray icon targeting KDE Plasma and GNOME on Wayland and X11.

## Decisions

### 1. Wayland Activation and Packaging Compatibility
- The GUI sets the Wayland `app_id` to `io.github.Semma.desktop` and forwards activation tokens for Qt Wayland window activation.
- Tray launches account for the packaged Python running with `-I` (isolated mode).
- The D-Bus GUI client recognizes unknown-method/interface/object errors so it remains compatible with an older running daemon that lacks `ReloadConfig`.

### 2. Onboarding and Settings UX
- An empty SQLite database is created on daemon startup; onboarding state checks whether an index actually exists rather than file presence alone.
- While setup is open, the daemon avoids loading the model so setup can change the compute device without requiring a restart.
- Unlike the settings page (which only warns when nothing would be indexed), the onboarding wizard refuses to continue until something will be indexed.
- Closing the onboarding window schedules finish logic outside `closeEvent` because Qt ignores a `close()` call made while a close event is already being handled.
- Re-running setup prefills saved folders and pauses indexing; cancelling leaves indexing paused, whereas choosing "Keep current settings" discards edits and resumes indexing.
- If `libtesseract` is missing while saved config has OCR enabled, the checkbox remains checked and enabled so the user can turn it off.

### 3. Qt Widgets, Progress Bars, Theming, and Icon Tinting
- `QProgressBar` accepts signed 32-bit integers, whereas model file byte counts exceed `2**31`; download progress is scaled before updating the bar.
- Theme portal D-Bus queries do not use a `with ThreadPoolExecutor` context manager because `__exit__` blocks until the worker finishes, which would defeat the portal query timeout.
- Icon tinting converts images to `RGBA8888` (byte-ordered R, G, B, A across architectures, unlike endian-dependent `ARGB32`).
- Monochrome detection runs per pixmap rather than per icon name: themes such as Breeze ship monochrome 16–24 px icons and full-color 32 px+ icons under the same name (e.g., `folder`), and tinting full-color pixmaps would flatten them into silhouettes.
- Stylesheets preserve child `QSpinBox` palette `Base` backgrounds and exact `Window` background corners on welcome icons to avoid dark-square artifacts.
- Tray status polling serializes in-flight polls so a new poll fires only after the previous one resolves.
