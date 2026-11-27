# Semma

Semantic file search for KDE KRunner and GNOME Shell search, powered by EmbeddingGemma 2. A background service indexes your files locally, and you search them by meaning from the desktop's search.

## What can it find?

Describe what you remember, rather than the filename. These are illustrative queries, not guaranteed matches:

| Content                      | Example query                   | What is searched                                                              |
| ---------------------------- | ------------------------------- | ----------------------------------------------------------------------------- |
| Documents and notes          | “notes about renewing my lease” | Text, Markdown, ODT, DOCX, RTF, and supported presentations                   |
| PDFs                         | “paper about battery recycling” | Extracted page text; scanned or mostly pictorial pages are embedded as images |
| Images                       | “a dog running on a beach”      | Visual content, without needing a caption or OCR                              |
| Screenshots with OCR enabled | “connection refused”            | Recognized screenshot text, in addition to the image's visual content         |
| Audio                        | “recording of birds singing”    | Audio content embedded directly, not a generated transcript                   |
| Video                        | “someone riding a bicycle”      | Sampled frames and the audio track, when present                              |

File types and size limits are configurable. Prefix a query with `image:`, `video:`, or `screenshot:` (for example `image: a dog on a beach` or `screenshot: connection refused`) to restrict results to that type. By default, only the first hour of each audio/video file is indexed; adjust this in **Settings → Media**. Source-code files are not indexed.

HEIC/HEIF photos are decoded with ffmpeg, which needs an HEVC decoder; Fedora's `ffmpeg-free` has none (install RPM Fusion's `ffmpeg` or `libavcodec-freeworld`). Multi-tile photos, as phones take them, need FFmpeg 7.1 or newer.

## Requirements

- Linux with KDE Plasma/KRunner or GNOME Shell, a session D-Bus, and a systemd user service for desktop integration. Release packages target x86-64 Fedora 41+, Ubuntu 24.04+, and Debian 13+.
- The packages bundle Python, Python dependencies, and llama.cpp with CPU and Vulkan backends. A GPU is optional; Vulkan acceleration needs a working system Vulkan loader and driver. CUDA is not included or required by the supplied build.
- About **865 MB** for the initial model download, plus space for the package and an index that grows with your files. RAM use and indexing time depend on the content and hardware; the download size is not a RAM requirement.

## Install

Download the package for x86-64 Fedora 41+, Ubuntu 24.04+, or Debian 13+ from [Releases](https://github.com/Mihuy1/semma/releases/latest), then install it:

```sh
sudo dnf install ./semma-0.6.2-1.x86_64.rpm     # Fedora
sudo apt install ./semma_0.6.2-1_amd64.deb      # Ubuntu, Debian
```

Then open **Semma** from the application menu and go through the setup wizard. Semma does not index anything until you finish the wizard. Review the indexed folders, exclusions, file types, and device settings before starting. The model (865 MB) downloads in the background the first time.

On GNOME, log out and back in for the search provider to appear. Once files are indexed, search from KRunner or the GNOME overview; the Semma window shows indexing progress and settings.

Normal package installs also pull in recommended media, OCR, and GUI libraries. If you disable recommended/weak dependencies, see the source runtime requirements below for features that need system libraries.

## Privacy and indexed folders

- **Fully local:** everything runs locally on your machine, and nothing leaves your system. After the initial model download from [Hugging Face](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) on first use, indexing and search work completely offline.
- **Default scope:** your home folder plus readable data mounts detected under `/mnt`, `/media`, and `/run/media/<user>`. Hidden files/folders, common dependency caches, game libraries, and project build folders are skipped by default.
- Change the scope and excluded paths in the setup wizard or **Settings → Folders**. File types and excluded extensions are under **Settings → File types**; the corresponding configuration keys include `[scan]` `roots`, `include_mounts`, `exclude_paths`, and `exclude_extensions`.

Default locations (respecting `XDG_CONFIG_HOME` and `XDG_DATA_HOME`):

| Data              | Default path                    |
| ----------------- | ------------------------------- |
| Configuration     | `~/.config/semma/config.toml`   |
| SQLite index      | `~/.local/share/semma/index.db` |
| Downloaded models | `~/.local/share/semma/models/`  |

## Text in screenshots

Semma can also read the text in your screenshots with Tesseract, so you can find a screenshot by the words in it. This is off by default; turn it on in Semma's settings (or with `enabled = true` under `[ocr]` in `~/.config/semma/config.toml`). It reads your system's language and English when their language data is installed. To add a language, install its Tesseract language pack, for example `sudo dnf install tesseract-langpack-deu` (Fedora) or `sudo apt install tesseract-ocr-deu` (Ubuntu, Debian).

## Tray icon

Semma includes a system-tray icon (`StatusNotifierItem`) that displays the current indexing state, lets you pause or resume indexing, and has an **Open Semma** action to open the window.

It starts automatically at login after installing a package (`/etc/xdg/autostart/io.github.Semma.Tray.desktop`) or running `semma install` (`~/.config/autostart/io.github.Semma.Tray.desktop`, respecting `XDG_CONFIG_HOME`). You can also launch it manually with `semma tray` (or `uv run semma tray` from a source checkout).

On GNOME, tray icons require the **AppIndicator and KStatusNotifierItem Support** GNOME Shell extension. It is preinstalled on Ubuntu; on Fedora, install it with:

```sh
sudo dnf install gnome-shell-extension-appindicator
```

To disable autostart, turn off **Semma tray icon** in your desktop's Autostart settings, or remove/override the autostart file:

- Source install: delete `~/.config/autostart/io.github.Semma.Tray.desktop`.
- Package install: copy `/etc/xdg/autostart/io.github.Semma.Tray.desktop` to `~/.config/autostart/` and add `Hidden=true` (or `X-GNOME-Autostart-enabled=false`).

## Uninstall

```sh
sudo dnf remove semma     # Fedora
sudo apt remove semma     # Ubuntu, Debian
```

Your index and the model stay in `~/.local/share/semma` (or your configured data location). After stopping/uninstalling Semma, delete that folder to remove them too. Configuration remains separately in `~/.config/semma` by default.

## From source

Needs Git, uv, and Python **3.13** (uv can provision it). The default llama.cpp build also needs **Podman or Docker**; it downloads the pinned source and builds the shared libraries inside an Ubuntu container.

```sh
git clone https://github.com/Mihuy1/semma.git
cd semma
uv sync --locked --python 3.13
scripts/build-llama.sh
uv run semma install      # add desktop integration and start the user service
uv run semma gui          # complete the setup wizard before indexing starts
```

For a native build, use `scripts/build-llama.sh --native` instead of the container build. Install CMake, Ninja, a C++ compiler, Git, `glslc`, and Vulkan development libraries plus SPIR-V headers first. The built libraries are discovered automatically in `build/llama/lib`.

**System runtime dependencies are not installed by `uv sync`:**

- Audio/video and HEIC: `ffmpeg` and `ffprobe` on `PATH`, with codecs for your files.
- Optional screenshot OCR: the Tesseract shared library and language data (`tesseract-libs` / `tesseract-langpack-eng` on Fedora; `libtesseract5` / `tesseract-ocr-eng` on Ubuntu/Debian).
- GUI: PySide6 provides Qt, but its platform plugins still need system D-Bus, EGL/OpenGL, font, GLib, Wayland or X11/XCB, and xkbcommon libraries. See [the package dependency lists](packaging/nfpm.yaml) for distribution-specific names, including XCB helper libraries.
- Inference/downloads: the C/C++ and OpenMP runtimes and system CA certificates; a Vulkan loader and GPU driver for optional acceleration.

`install` detects KDE or GNOME; pass `--desktop kde|gnome` to choose. For a source install on GNOME, run the printed `sudo install` command if prompted to register the provider, then log out and back in. Keep the checkout and its virtual environment in place: the service runs from them.

- Search from the terminal: `uv run semma search "notes about renewing my lease"`
- Open the window: `uv run semma gui`
- Remove source desktop integration and stop its service: `uv run semma uninstall`

## How it works

1. A background user service scans the configured folders, extracts text or media chunks, and watches for file changes.
2. EmbeddingGemma 2 runs locally through llama.cpp to turn content into vectors stored in SQLite with sqlite-vec, alongside metadata and full-text search indexes.
3. A query is embedded locally and ranked against indexed content, combining semantic similarity with word and filename matches. Results are exposed through KRunner and GNOME Shell search.

## Development

> [!NOTE]
> Semma is primarily AI-written (mostly with Claude and a little Gemini). I use it daily on Fedora KDE.

- Run the tests: `uv run pytest -q`
- Build the packages: `scripts/package.sh` (needs `uv`, `curl`, and Podman or Docker); they land in `dist/`.

## License

Semma's code is **MIT** licensed; see [LICENSE](LICENSE).

Semma is built on [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) by Google DeepMind and its [GGUF conversion](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) (**Apache-2.0**), [llama.cpp](https://github.com/ggml-org/llama.cpp) (**MIT**), and [sqlite-vec](https://github.com/asg017/sqlite-vec) (**MIT or Apache-2.0**). The window uses [Qt/PySide6](https://doc.qt.io/qtforpython-6/) under **LGPL-3.0**. Packages also include [Python](https://www.python.org/) (**PSF license**) and Python dependencies under their own licenses; optional [Tesseract](https://github.com/tesseract-ocr/tesseract) OCR is **Apache-2.0**.

The packages carry the license of everything they include in `/usr/share/doc/semma/THIRD_PARTY_NOTICES.txt`, generated from the shipped files by [scripts/notices.py](scripts/notices.py) at build time. These dependencies keep their own terms; Semma's MIT license does not replace them.
