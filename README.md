# Semma

Semantic file search for KDE KRunner and GNOME Shell overview search, powered by EmbeddingGemma 2 and `sqlite-vec`, with a background daemon on the session D-Bus (`io.github.Semma`).

## Install

### From a package (x86-64: Fedora 41+, Ubuntu 24.04+, Debian 13+)

```sh
sudo dnf install ./semma-0.2.0-1.x86_64.rpm     # Fedora
sudo apt install ./semma_0.2.0-1_amd64.deb      # Ubuntu, Debian
```

Then log out and back in. The package adds Semma to the application menu, KRunner, and GNOME Shell search for every user and starts its service with the session; nothing else needs to be set up. Opening **Semma** from the application menu (or `semma gui` in a graphical session) shows the status of the model download and indexing, and the settings.

- Everything Semma runs on is inside the package, under `/usr/lib/semma`: its own Python 3.13 with its libraries (including a trimmed Qt Widgets runtime for `semma gui`), and llama.cpp with the Vulkan and CPU backends. The system's Python is not used.
- The model (865 MB) is not inside. The service downloads it to `~/.local/share/semma/models` the first time it runs, checks it, and continues an interrupted download where it stopped.
- Vulkan (`vulkan-loader` / `libvulkan1`), `ffmpeg`, and the Wayland/X11 system libraries for the window are recommended, not required: without Vulkan the model runs on the processor, without ffmpeg audio and video files are skipped, and a minimal or headless install (`--no-install-recommends` / `install_weak_deps=False`) skips the desktop libraries while keeping the daemon and CLI.
- The first search waits for that download; `journalctl --user -u semma -f` shows it. Offline, searches find nothing until the model is there.
- After an update or a removal, log out and back in: a service that is already running keeps the old version until then. Removing the package leaves each user's index and model in `~/.local/share/semma`.
- To turn it off for one user: `systemctl --user mask --now semma.service`.
- If you ran `semma install` from a source checkout before, run `semma uninstall` there first: the files it wrote in your home folder take precedence over the package's.

Build the packages with `scripts/package.sh` (needs `uv`, `curl`, and `podman` or `docker`); they land in `dist/`. `scripts/check-package.sh dist/<package> [image]` installs one in a clean container, checks the window and runs an index and a search with the real model.

### From a source checkout

`uv run semma install` auto-detects the desktop from `XDG_CURRENT_DESKTOP` (falling back to `XDG_SESSION_DESKTOP` and `DESKTOP_SESSION`); pass `--desktop kde|gnome` to override. On other desktops it exits without making changes.

- Auto-detected: `uv run semma install [--desktop kde|gnome] [--python PATH] [--no-enable] [--no-restart]` (`--no-restart` is KDE only; `--python PATH` overrides existing service files on GNOME), `uv run semma uninstall [--desktop kde|gnome]`
- Explicit KDE: `uv run semma install-kde [--no-enable] [--no-restart]`, `uv run semma uninstall-kde`
- Explicit GNOME: `uv run semma install-gnome [--python PATH] [--no-enable]`, `uv run semma uninstall-gnome`
- Status: `uv run semma kde-status`, `uv run semma gnome-status`

Both `install-kde` and `install-gnome` also add **Semma** to the application menu (`~/.local/share/applications/io.github.Semma.desktop`) to open the window (status of the model download and indexing; settings for folders and file types).

## CLI

- Build and inspect the index: `uv run semma scan`, `uv run semma index`, `uv run semma status [--failed]`
- Query from the terminal: `uv run semma search <query> [-n N] [--kind KIND] [--json]`
- Run or control the daemon: `uv run semma daemon`, `uv run semma ctl status|pause|resume|rescan|retry-failed`
- Open the window with the daemon's status and the settings: `uv run semma gui` (needs PySide6; closing it does not stop indexing)

## GNOME notes

- GNOME Shell only reads search-provider files from system data directories, so `install-gnome` writes `io.github.Semma.search-provider.ini` to `/usr/local/share/gnome-shell/search-providers/` or, without permission, stages it in `~/.local/share/semma/` (`$XDG_DATA_HOME/semma/`) and prints a one-line `sudo install` command to run (`uninstall-gnome` likewise prints a `sudo rm` command if needed).
- The generated `.desktop` file (`~/.local/share/applications/io.github.Semma.desktop`) serves as both the application menu entry (on KDE and GNOME) and GNOME Shell's `DesktopId`.
- Log out and back in (or restart GNOME Shell) after installing for the new search provider to appear.
- Result activation opens files with `gio open` (falling back to `xdg-open`); clicking the provider icon (`LaunchSearch`) opens the Semma window.

## Both desktops

The daemon serves KRunner (`org.kde.krunner1`) at `/runner` and GNOME (`org.gnome.Shell.SearchProvider2`) at `/io/github/Semma/SearchProvider` on `io.github.Semma` at the same time. Uninstalling one desktop's integration keeps the shared application entry, D-Bus service, and `semma.service` systemd user unit active while the other desktop is still installed.

## Testing GNOME from KDE

Run `./scripts/gnome_devkit.sh` (requires `sudo dnf install mutter-devkit` along with `gnome-shell`, and the `semma` daemon running on your session bus):

- Starts `gnome-shell --devkit --wayland` in a private `dbus-run-session` so it cannot take over Plasma's notifications, status tray, or `xdg-desktop-portal` backends.
- Isolates the nested shell with `AutoStart=false` (preventing D-Bus activation of a second daemon) and `GSETTINGS_BACKEND=memory` (preventing changes to your real dconf settings).
- Runs `scripts/gnome_bridge.py` to forward only the search provider (`org.gnome.Shell.SearchProvider2` at `/io/github/Semma/SearchProvider` on `io.github.Semma`) back to the real host session bus.
- Close the devkit window (or press `Ctrl+C`) to stop both GNOME Shell and the bridge.

## Development

- `scripts/build-llama.sh` builds the pinned llama.cpp into `build/llama/lib`; a source checkout uses that build when it exists (`SEMMA_LLAMA_LIB_DIR` or `model.llama_lib_dir` override it).
- Run tests: `uv run pytest -q`
- Evaluate search ranking against a local JSON query set (`evals/local/search_queries.json` by default): `uv run python scripts/eval_search.py`

## License

MIT, see [LICENSE](LICENSE). The packages also carry the licences of Python and llama.cpp in /usr/share/doc/semma.
