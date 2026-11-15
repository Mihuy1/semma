"""What the Status page shows, computed from the daemon's Status() dict. No Qt here.

The daemon may be an older build that lacks some keys, so every key is read with a default.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import filetypes as ft

KIND_NAMES: dict[str, str] = {
    ft.TEXT: "Text",
    ft.PDF: "PDF",
    ft.OFFICE: "Office documents",
    ft.IMAGE: "Images",
    ft.AUDIO: "Audio",
    ft.VIDEO: "Video",
}

# What the daemon reports while it pauses for the user (`Daemon.pause`).
USER_PAUSE_REASON = "paused by request"


# -- formatting -----------------------------------------------------------------------------

def format_count(n: float) -> str:
    """1204 -> "1,204"."""
    return f"{int(n):,}"


def plural(n: float, one: str, many: str | None = None) -> str:
    """plural(1, "file") -> "1 file", plural(3, "file") -> "3 files"."""
    word = one if int(n) == 1 else (many or one + "s")
    return f"{format_count(n)} {word}"


_UNITS = ("B", "kB", "MB", "GB", "TB")


def _unit_for(n: float) -> int:
    i = 0
    while n >= 1000 and i < len(_UNITS) - 1:
        n /= 1000
        i += 1
    return i


def _scaled(n: float, i: int) -> str:
    v = n / 1000**i
    if i == 0:
        return f"{v:.0f}"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def format_bytes(n: float) -> str:
    """Decimal units, as file managers show sizes: 865_000_000 -> "865 MB"."""
    i = _unit_for(n)
    return f"{_scaled(n, i)} {_UNITS[i]}"


def format_bytes_of(done: float, total: float) -> str:
    """Both numbers in the unit of the total: "412 of 865 MB"."""
    i = _unit_for(total)
    return f"{_scaled(min(done, total), i)} of {_scaled(total, i)} {_UNITS[i]}"


def format_ago(when: float, now: float | None = None) -> str:
    """A wall-clock timestamp relative to now: "just now", "12 minutes ago", "3 days ago"."""
    if not when:
        return "Not yet"
    now = time.time() if now is None else now
    s = max(0.0, now - when)
    if s < 60:
        return "Just now"
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if s >= size:
            return f"{plural(s // size, unit)} ago"
    return "Just now"  # unreachable


def format_duration(seconds: float) -> str:
    """A rough duration: "less than a minute", "about 15 minutes", "about 2 hours"."""
    if seconds < 60:
        return "less than a minute"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {plural(minutes, 'minute')}"
    hours = seconds / 3600
    if hours < 48:
        h = round(hours)
        return f"about {plural(h, 'hour')}"
    return f"about {plural(round(hours / 24), 'day')}"


def short_path(path: str, home: Path | str) -> str:
    """A path under the home folder with "~": /home/u/Documents -> ~/Documents."""
    h = str(home).rstrip("/")
    if not path or not h:
        return path
    if path == h:
        return "~"
    if path.startswith(h + "/"):
        return "~" + path[len(h):]
    return path


def elide_middle(text: str, limit: int) -> str:
    """Shorten `text` to `limit` characters with "…" in the middle (for a line that should not
    wrap and jump while it changes every second)."""
    if len(text) <= limit:
        return text
    keep = limit - 1
    return text[: keep // 2] + "…" + text[len(text) - (keep - keep // 2):]


SCAN_DIR_LIMIT = 48

# -- the view -------------------------------------------------------------------------------

@dataclass
class Headline:
    """The sentence at the top of the Status page and its progress bar.

    `progress` is (done, total) for a bar that fills up, None for no bar; `busy` shows a bar
    that moves without a known end (total unknown)."""
    text: str
    detail: str = ""        # a second line: the file or folder being worked on
    secondary: str = ""     # speed and time left
    progress: tuple[int, int] | None = None
    busy: bool = False


@dataclass
class KindRow:
    kind: str
    name: str
    indexed: int
    pending: int
    failed: int


@dataclass
class StatusView:
    headline: Headline
    kinds: list[KindRow] = field(default_factory=list)
    indexed: int = 0
    pending: int = 0
    failed: int = 0
    paused: bool = False
    last_scan: str = "Not yet"
    model_loaded: bool = False
    watched_dirs: int = 0


def _num(st: dict[str, Any], key: str) -> float:
    try:
        v = float(st.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _int(st: dict[str, Any], key: str) -> int:
    return int(_num(st, key))


def _by_kind(st: dict[str, Any], key: str) -> dict[str, int]:
    d = st.get(key)
    return {str(k): int(v or 0) for k, v in d.items()} if isinstance(d, dict) else {}


def kind_rows(st: dict[str, Any]) -> list[KindRow]:
    """One row per known kind, in the order of filetypes.KINDS, plus any unknown kind the daemon
    reports with files; kinds with no files at all are left out. A daemon too old to count per
    kind gets one "All files" row."""
    if not any(isinstance(st.get(k), dict) for k in ("indexed_by_kind", "pending_by_kind", "failed_by_kind")):
        row = KindRow("all", "All files", _int(st, "indexed"), _int(st, "pending"), _int(st, "failed"))
        return [row] if row.indexed or row.pending or row.failed else []
    idx, pen, fail = (_by_kind(st, k) for k in ("indexed_by_kind", "pending_by_kind", "failed_by_kind"))
    kinds = list(ft.KINDS) + sorted(({*idx, *pen, *fail}) - set(ft.KINDS))
    rows = []
    for kind in kinds:
        row = KindRow(kind, KIND_NAMES.get(kind, kind.capitalize()), idx.get(kind, 0), pen.get(kind, 0),
                      fail.get(kind, 0))
        if row.indexed or row.pending or row.failed:
            rows.append(row)
    return rows


def pause_text(reason: str) -> str:
    if not reason or reason == USER_PAUSE_REASON:
        return "Paused by you"
    return f"Paused: {reason}"


def headline(st: dict[str, Any], home: Path | str) -> Headline:
    state = str(st.get("state") or "")
    indexed, pending = _int(st, "indexed"), _int(st, "pending")
    total = indexed + pending

    if state == "downloading" or st.get("download_file"):
        done, size = _num(st, "download_all_done"), _num(st, "download_all_total")
        if size <= 0:
            done, size = _num(st, "download_done"), _num(st, "download_total")
        if size > 0:
            return Headline(f"Downloading the model: {format_bytes_of(done, size)}",
                            detail=str(st.get("download_file") or ""), progress=(int(done), int(size)))
        return Headline("Downloading the model", detail=str(st.get("download_file") or ""), busy=True)

    if state == "scanning":
        seen = _int(st, "scan_files")
        where = elide_middle(short_path(str(st.get("scan_dir") or ""), home), SCAN_DIR_LIMIT)
        text = f"Scanning {where}" if where else "Scanning folders"
        if seen:
            text += f" ({plural(seen, 'file')} seen)"
        detail = ""
        if _int(st, "scan_dirs"):
            detail = f"{plural(_int(st, 'scan_dirs'), 'folder')} listed, {format_count(_int(st, 'scan_matched'))} to index"
        return Headline(text, detail=detail, busy=True)

    if state == "loading":
        return Headline("Loading the model", busy=True)

    paused = bool(st.get("paused")) or state == "paused"
    if paused and state in ("paused", "idle", "indexing", ""):
        bar = (indexed, total) if pending else None
        detail = f"{plural(pending, 'file')} waiting to be indexed" if pending else ""
        return Headline(pause_text(str(st.get("pause_reason") or "")), detail=detail, progress=bar)

    if state == "indexing":
        current = str(st.get("current_file") or "")
        rate = _num(st, "run_rate")
        secondary = ""
        if rate > 0:
            secondary = f"{rate:.1f} files per second" if rate < 10 else f"{rate:.0f} files per second"
            if pending:
                secondary += f", {format_duration(pending / rate)} left"
        return Headline(f"Indexing: {format_count(indexed)} of {plural(total, 'file')}",
                        detail=short_path(current, home), secondary=secondary, progress=(indexed, total))

    if state == "idle" or not state:
        if pending:
            return Headline(f"Waiting to index {plural(pending, 'file')}", progress=(indexed, total))
        if indexed:
            return Headline(f"Up to date: {plural(indexed, 'file')} indexed")
        return Headline("Nothing indexed yet")

    return Headline(state.capitalize())


def status_view(st: dict[str, Any], home: Path | str, now: float | None = None) -> StatusView:
    return StatusView(
        headline=headline(st, home),
        kinds=kind_rows(st),
        indexed=_int(st, "indexed"),
        pending=_int(st, "pending"),
        failed=_int(st, "failed"),
        paused=bool(st.get("paused")) or st.get("state") == "paused",
        last_scan=format_ago(_num(st, "last_scan_at"), now),
        model_loaded=bool(st.get("model_loaded")),
        watched_dirs=_int(st, "watched_dirs"),
    )
