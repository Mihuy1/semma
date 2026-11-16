"""Open a PDF or audio/video search hit at its page or timestamp in the default viewer.

``gio open`` / ``kde-open`` / ``xdg-open`` take only a file, so for a hit with a spot the
default application for its MIME type is looked up (``xdg-mime query default <mime>``), its
desktop entry's ``Exec`` line read, and the viewer's own seek option added in front of the file:

  okular     --page N
  evince     --page-index=N
  papers     --page-index=N
  zathura    --page=N
  mpv        --start=S
  vlc/cvlc   --start-time=S
  celluloid  --mpv-start=S          (also gnome-mpv)
  browsers   file:///x#page=N       Firefox (pdf.js) and Chromium-based browsers (Chrome,
             file:///x#t=S          Chromium, Brave, Edge, Vivaldi, Opera: one shared viewer)

Flatpak entries (``flatpak run ... --command=okular org.kde.okular @@u %U @@``) and entries with
a ``--`` end-of-options marker (``mpv ... -- %U``) work the same: the options go in front of
those markers. Any other viewer, a missing entry or an ``Exec`` line we cannot parse returns
None and the caller opens the file plainly. The page is the 1-based page index stored in
``Chunk.position``; audio/video seek times are start seconds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
import mimetypes
import os
from pathlib import Path
import re
import shlex
import shutil

from . import filetypes as ft
from .search import SearchResult

log = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
MIME_QUERY_TIMEOUT_SECONDS = 2.0

FILE_FIELD_CODES = frozenset({"%f", "%F", "%u", "%U"})
DROPPED_FIELD_CODES = frozenset({"%i", "%c", "%k"})
FLATPAK_FORWARD_MARKERS = frozenset({"@@", "@@u"})
_PREFIX_MARKERS = FLATPAK_FORWARD_MARKERS | {"--"}

_EXTRA_MEDIA_MIMES: dict[str, str] = {
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".opus": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


@dataclass(frozen=True, slots=True)
class Spot:
    """Where in a file to open: a 1-based page index, or a seek time in seconds."""

    page: int | None = None
    seconds: float | None = None

    @property
    def is_time(self) -> bool:
        return self.seconds is not None


def spot_for(result: SearchResult) -> Spot | None:
    """Where to open a hit (PDF page or audio/video start second), or None to open plainly."""
    if result.position is None:
        return None
    if result.kind == ft.PDF:
        page = int(result.position)
        return Spot(page) if page >= 1 else None
    if result.kind in (ft.AUDIO, ft.VIDEO) and result.position > 0:
        return Spot(seconds=float(result.position))
    return None


def media_mime(path: str) -> str | None:
    """MIME type for an audio/video file, or None when unknown or not audio/video."""
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith(("audio/", "video/")):
        return mime
    return _EXTRA_MEDIA_MIMES.get(Path(path).suffix.lower())


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def _page_uri(path: str, spot: Spot) -> str:
    return f"{Path(path).as_uri()}#page={spot.page}"


def _time_uri(path: str, spot: Spot) -> str:
    assert spot.seconds is not None
    return f"{Path(path).as_uri()}#t={_format_seconds(spot.seconds)}"


def _time_flag(prefix: str) -> Callable[[Spot], list[str]]:
    def build(spot: Spot) -> list[str]:
        assert spot.seconds is not None
        return [f"{prefix}{_format_seconds(spot.seconds)}"]

    return build


# Program name -> options placed before the file.
OPTION_VIEWERS: dict[str, Callable[[Spot], list[str]]] = {
    "okular": lambda spot: ["--page", str(spot.page)],
    "evince": lambda spot: [f"--page-index={spot.page}"],
    "papers": lambda spot: [f"--page-index={spot.page}"],
    "zathura": lambda spot: [f"--page={spot.page}"],
}
TIME_OPTION_VIEWERS: dict[str, Callable[[Spot], list[str]]] = {
    "mpv": _time_flag("--start="),
    "vlc": _time_flag("--start-time="),
    "cvlc": _time_flag("--start-time="),
    "celluloid": _time_flag("--mpv-start="),
    "gnome-mpv": _time_flag("--mpv-start="),
}
# Program name -> the file argument replaced by a URI.
BROWSERS = (
    "firefox", "firefox-esr", "firefox-bin",
    "chromium", "chromium-browser", "chrome", "google-chrome", "google-chrome-stable",
    "google-chrome-beta", "google-chrome-unstable", "brave", "brave-browser", "brave-browser-stable",
    "brave-browser-beta", "brave-browser-nightly", "microsoft-edge", "microsoft-edge-stable",
    "microsoft-edge-beta", "microsoft-edge-dev", "edge", "vivaldi", "vivaldi-stable", "vivaldi-snapshot",
    "opera", "opera-beta", "opera-developer",
)
URI_VIEWERS: dict[str, Callable[[str, Spot], str]] = dict.fromkeys(BROWSERS, _page_uri)
TIME_URI_VIEWERS: dict[str, Callable[[str, Spot], str]] = dict.fromkeys(BROWSERS, _time_uri)
FLATPAK_APPS: dict[str, str] = {
    "org.kde.okular": "okular",
    "org.gnome.Evince": "evince",
    "org.gnome.Papers": "papers",
    "org.pwmt.zathura": "zathura",
    "io.mpv.Mpv": "mpv",
    "org.videolan.VLC": "vlc",
    "io.github.celluloid_player.Celluloid": "celluloid",
    "org.mozilla.firefox": "firefox",
    "org.chromium.Chromium": "chromium",
    "io.github.ungoogled_software.ungoogled_chromium": "chromium",
    "com.google.Chrome": "chrome",
    "com.brave.Browser": "brave",
    "com.microsoft.Edge": "edge",
    "com.vivaldi.Vivaldi": "vivaldi",
    "com.opera.Opera": "opera",
}

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DESKTOP_ESCAPE_RE = re.compile(r"\\(.)")
_DESKTOP_ESCAPES = {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}


def viewer_name(exec_argv: list[str]) -> str | None:
    """The viewer an ``Exec`` argv runs: its program name, or a known Flatpak app's."""
    i = 0
    if i < len(exec_argv) and os.path.basename(exec_argv[i]) == "env":
        i += 1
        while i < len(exec_argv) and _ENV_ASSIGNMENT_RE.match(exec_argv[i]):
            i += 1
    if i >= len(exec_argv):
        return None
    program = os.path.basename(exec_argv[i])
    if program != "flatpak":
        return program
    rest = exec_argv[i + 1 :]
    if "run" not in rest:
        return None
    for arg in rest:
        if arg.startswith("--command="):
            return os.path.basename(arg.removeprefix("--command="))
    for arg in rest[rest.index("run") + 1 :]:
        if not arg.startswith("-"):
            return FLATPAK_APPS.get(arg)
    return None


def _viewer_action(name: str | None, path: str, spot: Spot) -> tuple[list[str], str] | None:
    if spot.is_time:
        option_table, uri_table = TIME_OPTION_VIEWERS, TIME_URI_VIEWERS
    elif spot.page is not None:
        option_table, uri_table = OPTION_VIEWERS, URI_VIEWERS
    else:
        return None
    if name in option_table:
        return option_table[name](spot), path
    if name in uri_table:
        return [], uri_table[name](path, spot)
    return None


def _option_insert_index(out: list[str]) -> int:
    at = len(out)
    while at > 0 and out[at - 1] in _PREFIX_MARKERS:
        at -= 1
    return at


def build_argv(exec_argv: list[str], path: str, spot: Spot) -> list[str] | None:
    """The command opening ``path`` at ``spot`` with this ``Exec`` argv, or None if the viewer
    is not one we know how to point at that spot."""
    action = _viewer_action(viewer_name(exec_argv), path, spot)
    if action is None:
        return None
    options, target = action

    out: list[str] = []
    placed = False
    for arg in exec_argv:
        if arg in FILE_FIELD_CODES:
            if placed:
                continue
            at = _option_insert_index(out)
            out[at:at] = options
            out.append(target)
            placed = True
        elif arg in DROPPED_FIELD_CODES:
            continue
        else:
            stripped = arg.replace("%%", "")
            if "%" in stripped:  # a field code inside an argument ("--file=%f"): not handled
                return None
            out.append(arg.replace("%%", "%"))
    if not placed:
        at = _option_insert_index(out)
        out[at:at] = options
        out.append(target)
    return out


def parse_exec(value: str) -> list[str] | None:
    """Split a desktop entry ``Exec`` value into argv (string escapes, then quoting)."""
    unescaped = _DESKTOP_ESCAPE_RE.sub(lambda m: _DESKTOP_ESCAPES.get(m[1], "\\" + m[1]), value)
    try:
        argv = shlex.split(unescaped)
    except ValueError:
        return None
    return argv or None


def read_exec(desktop_file: Path) -> list[str] | None:
    """The ``Exec`` argv of a desktop file's ``[Desktop Entry]`` group."""
    try:
        lines = desktop_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    in_entry = False
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            in_entry = line == "[Desktop Entry]"
        elif in_entry and line.startswith("Exec"):
            key, sep, value = line.partition("=")
            if sep and key.strip() == "Exec":
                return parse_exec(value.strip())
    return None


def applications_dirs() -> list[Path]:
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [Path(d) / "applications" for d in [data_home, *data_dirs.split(":")] if d]


def find_desktop_file(desktop_id: str) -> Path | None:
    if "/" in desktop_id or not desktop_id.endswith(".desktop"):
        return None
    for directory in applications_dirs():
        candidate = directory / desktop_id
        if candidate.is_file():
            return candidate
    return None


async def default_desktop_id(mime: str = PDF_MIME) -> str | None:
    """The default application's desktop file id for ``mime``, via ``xdg-mime``."""
    xdg_mime = shutil.which("xdg-mime")
    if not xdg_mime:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            xdg_mime, "query", "default", mime,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as e:
        log.debug("xdg-mime failed to start: %s", e)
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), MIME_QUERY_TIMEOUT_SECONDS)
    except (TimeoutError, asyncio.TimeoutError):
        proc.kill()
        await proc.wait()
        log.debug("xdg-mime query default %s timed out", mime)
        return None
    lines = out.decode(errors="replace").split()
    return lines[0] if proc.returncode == 0 and lines else None


async def viewer_argv(path: str, spot: Spot) -> list[str] | None:
    """The command opening ``path`` at ``spot`` with the default viewer, or None when the file
    type or viewer is unknown or cannot be found (open the file plainly then)."""
    if spot.is_time:
        mime = media_mime(path)
        if not mime:
            return None
    else:
        mime = PDF_MIME
    desktop_id = await default_desktop_id(mime)
    if not desktop_id:
        return None
    desktop_file = find_desktop_file(desktop_id)
    if desktop_file is None:
        log.debug("desktop file %s not found", desktop_id)
        return None
    exec_argv = read_exec(desktop_file)
    if not exec_argv:
        return None
    argv = build_argv(exec_argv, path, spot)
    if argv is None:
        log.debug("default viewer %s: no seek support known", desktop_id)
    return argv
