# GemmaSearch

Semantic file search for KDE KRunner and GNOME Shell overview search, powered by EmbeddingGemma 2 and `sqlite-vec`, with a background daemon on the session D-Bus (`io.github.GemmaSearch`).

## Install

`uv run gemmasearch install` auto-detects the desktop from `XDG_CURRENT_DESKTOP` (falling back to `XDG_SESSION_DESKTOP` and `DESKTOP_SESSION`); pass `--desktop kde|gnome` to override. On other desktops it exits without making changes.

- Auto-detected: `uv run gemmasearch install [--desktop kde|gnome] [--python PATH] [--no-enable] [--no-restart]` (`--no-restart` is KDE only; `--python PATH` overrides existing service files on GNOME), `uv run gemmasearch uninstall [--desktop kde|gnome]`
- Explicit KDE: `uv run gemmasearch install-kde [--no-enable] [--no-restart]`, `uv run gemmasearch uninstall-kde`
- Explicit GNOME: `uv run gemmasearch install-gnome [--python PATH] [--no-enable]`, `uv run gemmasearch uninstall-gnome`
- Status: `uv run gemmasearch kde-status`, `uv run gemmasearch gnome-status`

## CLI

- Build and inspect the index: `uv run gemmasearch scan`, `uv run gemmasearch index`, `uv run gemmasearch status [--failed]`
- Query from the terminal: `uv run gemmasearch search <query> [-n N] [--kind KIND] [--json]`
- Run or control the daemon: `uv run gemmasearch daemon`, `uv run gemmasearch ctl status|pause|resume|rescan|retry-failed`

## GNOME notes

- GNOME Shell only reads search-provider files from system data directories, so `install-gnome` writes `io.github.GemmaSearch.search-provider.ini` to `/usr/local/share/gnome-shell/search-providers/` or, without permission, stages it in `~/.local/share/gemmasearch/` (`$XDG_DATA_HOME/gemmasearch/`) and prints a one-line `sudo install` command to run (`uninstall-gnome` likewise prints a `sudo rm` command if needed).
- The generated `.desktop` file (`~/.local/share/applications/io.github.GemmaSearch.desktop`) includes `OnlyShowIn=GNOME;` so it never appears in KDE Plasma.
- Log out and back in (or restart GNOME Shell) after installing for the new search provider to appear.
- Result activation opens files with `gio open` (falling back to `xdg-open`); `LaunchSearch` (clicking the provider icon) currently does nothing.

## Both desktops

The daemon serves KRunner (`org.kde.krunner1`) at `/runner` and GNOME (`org.gnome.Shell.SearchProvider2`) at `/io/github/GemmaSearch/SearchProvider` on `io.github.GemmaSearch` at the same time. Uninstalling one desktop's integration keeps the shared D-Bus service and `gemmasearch.service` systemd user unit active while the other desktop is still installed.

## Testing GNOME from KDE

Run `./scripts/gnome_devkit.sh` (requires `sudo dnf install mutter-devkit` along with `gnome-shell`, and the `gemmasearch` daemon running on your session bus):

- Starts `gnome-shell --devkit --wayland` in a private `dbus-run-session` so it cannot take over Plasma's notifications, status tray, or `xdg-desktop-portal` backends.
- Isolates the nested shell with `AutoStart=false` (preventing D-Bus activation of a second daemon) and `GSETTINGS_BACKEND=memory` (preventing changes to your real dconf settings).
- Runs `scripts/gnome_bridge.py` to forward only the search provider (`org.gnome.Shell.SearchProvider2` at `/io/github/GemmaSearch/SearchProvider` on `io.github.GemmaSearch`) back to the real host session bus.
- Close the devkit window (or press `Ctrl+C`) to stop both GNOME Shell and the bridge.

## Development

- Run tests: `uv run pytest -q`
- Evaluate search ranking against a local JSON query set (`evals/local/search_queries.json` by default): `uv run python scripts/eval_search.py`
