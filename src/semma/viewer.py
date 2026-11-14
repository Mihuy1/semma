"""Open a PDF search hit at its page, and at the matching text where the viewer can search.

``gio open`` / ``kde-open`` / ``xdg-open`` take only a file, so for a PDF hit with a page the
default PDF application is looked up (``xdg-mime query default application/pdf``), its desktop
entry's ``Exec`` line read, and the viewer's own options added in front of the file:

  viewer    page                 text (the hit's opening words)
  okular    --page N             --find TEXT    searches forward from page N, across lines;
                                                no match leaves it on page N
  evince    --page-index=N       --find=TEXT
  zathura   --page=N             --find=TEXT
  papers    --page-index=N       (no search option since the Evince fork)
  firefox   file:///x.pdf#page=N&search=TEXT&phrase=true   (pdf.js)

Flatpak entries (``flatpak run ... --command=okular org.kde.okular @@u %U @@``) work the same:
the options go in front of the ``@@u`` file-forwarding block. Any other viewer, a missing
entry or an ``Exec`` line we cannot parse returns None and the caller opens the file plainly.
The page is the 1-based page index the extractor stored (``Chunk.position``), not a page label.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
from urllib.parse import quote

from . import filetypes as ft
from .search import SearchResult

log = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
MIME_QUERY_TIMEOUT_SECONDS = 2.0

# The search text is the start of the hit's chunk. Chunks start at a paragraph or sentence
# where they can, so a few words usually sit on one line of the page; a long phrase is more
# likely to trip over a line break, hyphenation or ligature the viewer extracts differently.
FIND_MAX_WORDS = 5
FIND_MAX_CHARS = 48
FIND_MIN_ALNUM = 6

FILE_FIELD_CODES = frozenset({"%f", "%F", "%u", "%U"})
DROPPED_FIELD_CODES = frozenset({"%i", "%c", "%k"})
FLATPAK_FORWARD_MARKERS = frozenset({"@@", "@@u"})


@dataclass(frozen=True, slots=True)
class Spot:
    """Where in a document to open: a 1-based page index and optionally text to find on it."""

    page: int
    find: str | None = None


def find_phrase(snippet: str) -> str | None:
    """The first few words of a snippet, trimmed of punctuation at both ends, or None."""
    words = snippet.removesuffix("…").split()
    picked: list[str] = []
    for word in words[:FIND_MAX_WORDS]:
        if picked and len(" ".join([*picked, word])) > FIND_MAX_CHARS:
            break
        picked.append(word)
    phrase = " ".join(picked)
    start, end = 0, len(phrase)
    while start < end and not phrase[start].isalnum():
        start += 1
    while end > start and not phrase[end - 1].isalnum():
        end -= 1
    phrase = phrase[start:end]
    if sum(c.isalnum() for c in phrase) < FIND_MIN_ALNUM:
        return None
    return phrase


def spot_for(result: SearchResult) -> Spot | None:
    """The page (and text) to open a PDF hit at; None for other kinds or without a page."""
    if result.kind != ft.PDF or result.position is None:
        return None
    page = int(result.position)
    if page < 1:
        return None
    # Scanned pages are image chunks: their snippet is not text on the page.
    find = find_phrase(result.snippet) if result.modality == "text" and result.snippet else None
    return Spot(page, find)


def _okular(spot: Spot) -> list[str]:
    return ["--page", str(spot.page)] + (["--find", spot.find] if spot.find else [])


def _evince(spot: Spot) -> list[str]:
    return [f"--page-index={spot.page}"] + ([f"--find={spot.find}"] if spot.find else [])


def _zathura(spot: Spot) -> list[str]:
    return [f"--page={spot.page}"] + ([f"--find={spot.find}"] if spot.find else [])


def _papers(spot: Spot) -> list[str]:
    return [f"--page-index={spot.page}"]


def _pdfjs_uri(path: str, spot: Spot) -> str:
    fragment = f"page={spot.page}"
    if spot.find:
        fragment += f"&search={quote(spot.find, safe='')}&phrase=true"
    return f"{Path(path).as_uri()}#{fragment}"


# Program name -> options placed before the file.
OPTION_VIEWERS: dict[str, Callable[[Spot], list[str]]] = {
    "okular": _okular,
    "evince": _evince,
    "zathura": _zathura,
    "papers": _papers,
}
# Program name -> the file argument replaced by a URI.
URI_VIEWERS: dict[str, Callable[[str, Spot], str]] = {
    "firefox": _pdfjs_uri,
    "firefox-esr": _pdfjs_uri,
    "firefox-bin": _pdfjs_uri,
}
FLATPAK_APPS: dict[str, str] = {
    "org.kde.okular": "okular",
    "org.gnome.Evince": "evince",
    "org.gnome.Papers": "papers",
    "org.pwmt.zathura": "zathura",
    "org.mozilla.firefox": "firefox",
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


def build_argv(exec_argv: list[str], path: str, spot: Spot) -> list[str] | None:
    """The command opening ``path`` at ``spot`` with this ``Exec`` argv, or None if the viewer
    is not one we know how to point at a page."""
    name = viewer_name(exec_argv)
    if name in OPTION_VIEWERS:
        options, target = OPTION_VIEWERS[name](spot), path
    elif name in URI_VIEWERS:
        options, target = [], URI_VIEWERS[name](path, spot)
    else:
        return None

    out: list[str] = []
    placed = False
    for arg in exec_argv:
        if arg in FILE_FIELD_CODES:
            if placed:
                continue
            at = len(out) - 1 if out and out[-1] in FLATPAK_FORWARD_MARKERS else len(out)
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
        out += [*options, target]
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
    """The command opening the PDF ``path`` at ``spot`` with the default PDF viewer, or None
    when that viewer is unknown or cannot be found (open the file plainly then)."""
    desktop_id = await default_desktop_id(PDF_MIME)
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
        log.debug("default PDF viewer %s: no page support known", desktop_id)
    return argv
