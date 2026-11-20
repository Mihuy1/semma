"""KRunner D-Bus runner implementation (``org.kde.krunner1``).

Exports ``Match``, ``Actions``, ``Run``, ``Teardown`` and ``SetActivationToken`` using ``dbus-fast``.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
import logging
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from dbus_fast import Message, MessageType, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

from . import filetypes as ft, mpris
from .dbus_names import RUNNER_INTERFACE
from .query_filters import parse_query
from .search import SearchResult
from .thumbnails import IconData, Thumbnailer
from .viewer import Spot, spot_for, viewer_argv

log = logging.getLogger(__name__)

CATEGORY_LABEL = "Semantic file search"
ACTION_OPEN_FOLDER = "open-folder"
ACTION_COPY_PATH = "copy-path"
MATCH_ACTIONS = [ACTION_OPEN_FOLDER, ACTION_COPY_PATH]

THUMBNAIL_BUDGET_SECONDS = 0.150
RECENT_RESULTS_MAX = 512
SNIPPET_MAX_CHARS = 100

_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

KIND_FALLBACK_ICONS: dict[str, str] = {
    ft.TEXT: "text-plain",
    ft.PDF: "application-pdf",
    ft.OFFICE: "x-office-document",
    ft.IMAGE: "image-x-generic",
    ft.AUDIO: "audio-x-generic",
    ft.VIDEO: "video-x-generic",
}

FILEMANAGER_BUS = "org.freedesktop.FileManager1"
FILEMANAGER_PATH = "/org/freedesktop/FileManager1"
FILEMANAGER_IFACE = "org.freedesktop.FileManager1"

KLIPPER_BUS = "org.kde.klipper"
KLIPPER_PATH = "/klipper"
KLIPPER_IFACE = "org.kde.klipper.klipper"


def count_alnum(text: str) -> int:
    return len(_ALNUM_RE.findall(text))


def icon_name_for(path: str, kind: str) -> str:
    """Freedesktop icon name from MIME type, falling back to generic icon per kind."""
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime.replace("/", "-")
    return KIND_FALLBACK_ICONS.get(kind, "text-plain")


def shorten_home(path: Path | str, home: Path | None = None) -> str:
    h = str(home or Path.home())
    s = str(path)
    if s == h:
        return "~"
    prefix = h.rstrip("/") + "/"
    if s.startswith(prefix):
        return "~/" + s[len(prefix) :]
    return s


def trim_snippet(snippet: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    collapsed = _WS_RE.sub(" ", snippet).strip()
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[: limit - 1].rstrip()
    last_space = cut.rfind(" ")
    if last_space >= limit // 2:
        cut = cut[:last_space].rstrip()
    return cut + "…"


def format_subtext(result: SearchResult, home: Path | None = None) -> str:
    """Format the one-line subtext for a search result."""
    snippet = trim_snippet(result.snippet) if result.snippet else ""
    locator = _WS_RE.sub(" ", result.locator).strip() if result.locator else ""

    if snippet:
        base = f"{locator} · {snippet}" if locator else snippet
    else:
        parent = shorten_home(Path(result.path).parent, home)
        base = f"{locator} · {parent}" if locator else parent

    if result.duplicates > 0:
        base = f"{base}  (+{result.duplicates} similar)"
    return base


STRONG_SCORE = 4.0
MODERATE_Z = 5.0
RELEVANCE_STEP = 1e-4
WEAK_SCORE_RATIO = 0.5


def trim_weak_results(results: list[SearchResult], max_results: int | None = None) -> list[SearchResult]:
    """Drop hits scoring below ``WEAK_SCORE_RATIO`` of the top hit, then cap at ``max_results``."""
    if not results:
        return []
    cutoff = results[0].score * WEAK_SCORE_RATIO
    kept = [r for r in results if r.score >= cutoff]
    return kept[:max_results] if max_results is not None else kept


def is_strong(r: SearchResult) -> bool:
    lexical = "name" in r.matched or "text" in r.matched
    return r.score >= STRONG_SCORE and (lexical or (r.z is not None and r.z >= MODERATE_Z))


def category_relevances(results: list[SearchResult]) -> list[int]:
    """50 (Moderate) for the leading results that are strong matches (is_strong), 30 (Low) from
    the first weaker one on. KRunner sorts by category relevance before relevance, so this must
    never increase down our ranking."""
    out: list[int] = []
    strong = True
    for r in results:
        strong = strong and is_strong(r)
        out.append(50 if strong else 30)
    return out


def raw_relevance(score: float) -> float:
    """Map the search score linearly from [0, 8] to [0.3, 0.9] (clamped)."""
    t = max(0.0, min(1.0, score / 8.0))
    return round(0.3 + t * (0.9 - 0.3), 6)


def compute_monotonic_relevances(results: list[SearchResult]) -> list[float]:
    """Relevance in (0, 1] per result, strictly decreasing in rank (results come best first), so
    KRunner keeps our order."""
    relevances: list[float] = []
    for r in results:
        rel = raw_relevance(r.score)
        if relevances:
            rel = min(rel, relevances[-1] - RELEVANCE_STEP)
        relevances.append(round(max(rel, RELEVANCE_STEP), 6))
    return relevances


def file_uri(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = p.resolve()
    return p.as_uri()


class Launcher:
    """Process spawning and session D-Bus calls for ``KRunnerService.Run``.

    Injectable so tests can inspect calls without opening windows or touching the clipboard.
    """

    def __init__(self, bus: MessageBus | None = None) -> None:
        self._bus = bus

    def _spawn_detached(self, argv: list[str], activation_token: str | None = None) -> bool:
        env = os.environ.copy()
        if activation_token:
            env["XDG_ACTIVATION_TOKEN"] = activation_token
        try:
            subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
            return True
        except OSError as e:
            log.warning("failed to spawn %s: %s", argv, e)
            return False

    def _open_with_desktop_opener(self, target: str, activation_token: str | None = None) -> None:
        opener = shutil.which("kde-open") or shutil.which("xdg-open") or "kde-open"
        if not self._spawn_detached([opener, target], activation_token=activation_token) and opener != "xdg-open":
            fallback = shutil.which("xdg-open")
            if fallback:
                self._spawn_detached([fallback, target], activation_token=activation_token)

    async def _call_session_dbus(
        self,
        destination: str,
        path: str,
        interface: str,
        member: str,
        signature: str,
        body: list[Any],
    ) -> bool:
        bus = self._bus
        own_bus = False
        try:
            if bus is None:
                bus = await MessageBus().connect()
                own_bus = True
            reply = await bus.call(
                Message(
                    destination=destination,
                    path=path,
                    interface=interface,
                    member=member,
                    signature=signature,
                    body=body,
                )
            )
            return reply is not None and reply.message_type == MessageType.METHOD_RETURN
        except Exception as e:
            log.debug("D-Bus call %s.%s failed: %s", interface, member, e)
            return False
        finally:
            if own_bus and bus is not None:
                try:
                    bus.disconnect()
                except Exception:
                    pass

    async def open_file(self, path: str, activation_token: str | None = None, spot: Spot | None = None) -> None:
        """Open ``path`` with the default application (``kde-open``, fallback ``xdg-open``).

        With a ``spot`` (a PDF hit's page, an audio/video hit's start second), a default viewer
        that can be pointed there is started directly with its option instead. Other media
        players are sought to the start second over MPRIS once they have loaded the file."""
        if spot is not None:
            argv = await viewer_argv(path, spot)
            if argv and self._spawn_detached(argv, activation_token=activation_token):
                return
        if spot is not None and spot.seconds is not None:
            before = await mpris.snapshot()
            self._open_with_desktop_opener(path, activation_token=activation_token)
            mpris.seek_in_background(path, spot.seconds, before)
            return
        self._open_with_desktop_opener(path, activation_token=activation_token)

    async def open_folder(self, path: str, activation_token: str | None = None) -> None:
        """Select ``path`` in the file manager via ``org.freedesktop.FileManager1.ShowItems``,
        falling back to opening the parent directory with ``kde-open`` / ``xdg-open``."""
        uri = file_uri(path)
        startup_id = activation_token or ""
        ok = await self._call_session_dbus(
            FILEMANAGER_BUS,
            FILEMANAGER_PATH,
            FILEMANAGER_IFACE,
            "ShowItems",
            "ass",
            [[uri], startup_id],
        )
        if not ok:
            parent = str(Path(path).parent)
            self._open_with_desktop_opener(parent, activation_token=activation_token)

    async def copy_path(self, path: str) -> None:
        """Copy ``path`` to the clipboard via Klipper D-Bus, falling back to ``wl-copy``."""
        ok = await self._call_session_dbus(
            KLIPPER_BUS,
            KLIPPER_PATH,
            KLIPPER_IFACE,
            "setClipboardContents",
            "s",
            [path],
        )
        if not ok:
            wl_copy = shutil.which("wl-copy")
            if wl_copy:
                self._spawn_detached([wl_copy, "--", path])


class KRunnerService(ServiceInterface):
    """D-Bus service implementing ``org.kde.krunner1``."""

    def __init__(
        self,
        search: Callable[[str], Awaitable[list[SearchResult]]],
        *,
        max_results: int = 10,
        on_activity: Callable[[], None] | None = None,
        launcher: Launcher | None = None,
        thumbnailer: Thumbnailer | None = None,
        thumbnail_budget: float = THUMBNAIL_BUDGET_SECONDS,
        home: Path | None = None,
    ) -> None:
        super().__init__(RUNNER_INTERFACE)
        self._search = search
        self._max_results = max(1, int(max_results))
        self._on_activity = on_activity
        self._launcher = launcher if launcher is not None else Launcher()
        self._thumbnailer = thumbnailer if thumbnailer is not None else Thumbnailer()
        self._thumbnail_budget = thumbnail_budget
        self._home = home
        self._recent: OrderedDict[str, SearchResult] = OrderedDict()
        self._activation_token: str | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _remember(self, result: SearchResult) -> str:
        match_id = result.path
        self._recent[match_id] = result
        self._recent.move_to_end(match_id)
        while len(self._recent) > RECENT_RESULTS_MAX:
            self._recent.popitem(last=False)
        return match_id

    async def _gather_thumbnails(self, results: list[SearchResult]) -> dict[str, IconData]:
        media = [r for r in results if r.kind in (ft.IMAGE, ft.VIDEO)]
        if not media:
            return {}

        ready: dict[str, IconData] = {}
        pending_results: list[SearchResult] = []
        for r in media:
            cached = self._thumbnailer.get_cached(r.path, r.mtime)
            if cached is not None:
                ready[r.path] = cached
            else:
                pending_results.append(r)

        if not pending_results or self._thumbnail_budget <= 0:
            for r in pending_results:
                t = asyncio.create_task(self._safe_thumb(r))
                self._background_tasks.add(t)
                t.add_done_callback(self._background_tasks.discard)
            return ready

        task_to_path: dict[asyncio.Task[IconData | None], str] = {}
        for r in pending_results:
            task = asyncio.create_task(self._safe_thumb(r))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task_to_path[task] = r.path

        done, _ = await asyncio.wait(task_to_path.keys(), timeout=self._thumbnail_budget)
        for task in done:
            if not task.cancelled() and task.exception() is None:
                icon = task.result()
                if icon is not None:
                    ready[task_to_path[task]] = icon
        return ready

    async def _safe_thumb(self, result: SearchResult) -> IconData | None:
        try:
            return await self._thumbnailer.icon_data(result.path, result.kind, result.mtime)
        except Exception as e:
            log.debug("thumbnail error for %s: %s", result.path, e)
            return None

    @method()
    async def Match(self, query: "s") -> "a(sssida{sv})":  # noqa: F821, UP037
        q = query.strip()
        if count_alnum(parse_query(q).text) < 3:
            return []

        if self._on_activity is not None:
            try:
                self._on_activity()
            except Exception:
                log.exception("on_activity callback failed")

        try:
            raw_results = await self._search(q)
        except asyncio.CancelledError:
            return []
        except Exception:
            log.exception("search failed for query %r", q)
            return []

        results = trim_weak_results(raw_results, self._max_results)
        if not results:
            return []
        relevances = compute_monotonic_relevances(results)
        cat_rels = category_relevances(results)
        try:
            thumbs = await self._gather_thumbnails(results)
        except asyncio.CancelledError:
            return []
        except Exception:
            log.exception("thumbnail gathering failed")
            thumbs = {}

        matches: list[list[Any]] = []
        for result, relevance, cat_rel in zip(results, relevances, cat_rels, strict=True):
            match_id = self._remember(result)
            text = Path(result.path).name or result.path
            icon_name = icon_name_for(result.path, result.kind)
            subtext = format_subtext(result, self._home)

            props: dict[str, Variant] = {
                "subtext": Variant("s", subtext),
                "urls": Variant("as", [file_uri(result.path)]),
                "category": Variant("s", CATEGORY_LABEL),
                "actions": Variant("as", list(MATCH_ACTIONS)),
            }
            icon = thumbs.get(result.path)
            if icon is not None:
                w, h, rowstride, has_alpha, bps, channels, raw_bytes = icon
                props["icon-data"] = Variant(
                    "(iiibiiay)",
                    [int(w), int(h), int(rowstride), bool(has_alpha), int(bps), int(channels), bytes(raw_bytes)],
                )

            matches.append([match_id, text, icon_name, int(cat_rel), float(relevance), props])

        return matches

    @method()
    def Actions(self) -> "a(sss)":  # noqa: F821, UP037
        return [
            [ACTION_OPEN_FOLDER, "Open Containing Folder", "document-open-folder"],
            [ACTION_COPY_PATH, "Copy Path", "edit-copy"],
        ]

    @method()
    def SetActivationToken(self, token: "s"):  # noqa: F821, UP037
        self._activation_token = token or None

    @method()
    async def Run(self, matchId: "s", actionId: "s"):  # noqa: F821, N803, UP037
        token, self._activation_token = self._activation_token, None
        result = self._recent.get(matchId)
        if result is None:
            return
        path = result.path
        try:
            if actionId == "":
                await self._launcher.open_file(path, activation_token=token, spot=spot_for(result))
            elif actionId == ACTION_OPEN_FOLDER:
                await self._launcher.open_folder(path, activation_token=token)
            elif actionId == ACTION_COPY_PATH:
                await self._launcher.copy_path(path)
            else:
                log.debug("unknown KRunner action %r for %s", actionId, path)
        except Exception:
            log.exception("Run(%r, %r) failed", matchId, actionId)

    @method()
    def Teardown(self):
        self._activation_token = None
        while len(self._recent) > self._max_results * 2:
            self._recent.popitem(last=False)
