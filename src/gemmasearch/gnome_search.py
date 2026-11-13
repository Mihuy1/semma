"""GNOME Shell overview search provider (``org.gnome.Shell.SearchProvider2``).

Exports ``GetInitialResultSet``, ``GetSubsearchResultSet``, ``GetResultMetas``, ``ActivateResult``
and ``LaunchSearch`` using ``dbus-fast``. Result ids are file paths.

Icons: GNOME Shell 51's ``ui/remoteSearch.js`` ``createIcon(size, meta)`` checks the metas in
this order and uses the first one present::

    if (meta['icon'])            gicon = Gio.icon_deserialize(meta['icon']);
    else if (meta['gicon'])      gicon = Gio.icon_new_for_string(meta['gicon']);
    else if (meta['icon-data'])  { const [width, height, rowStride, hasAlpha,
                                          bitsPerSample, nChannels_, data] = meta['icon-data'];
                                   gicon = Shell.util_create_pixbuf_from_data(data, ...); }

So a thumbnail goes in ``icon-data`` (``(iiibiiay)``, the same raw-pixel tuple KRunner takes:
no PNG encoding, no temp files), and ``gicon`` (a themed MIME icon name) must then be left
out, because it would win over ``icon-data``. ``gicon`` uses the ``. GThemedIcon a b`` string
form so a theme missing the MIME icon falls back to the generic one.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from dbus_fast import Variant
from dbus_fast.service import ServiceInterface, method

from . import filetypes as ft
from .dbus_names import GNOME_INTERFACE
from .krunner import (
    _WS_RE,
    KIND_FALLBACK_ICONS,
    RECENT_RESULTS_MAX,
    count_alnum,
    icon_name_for,
    shorten_home,
    trim_snippet,
)
from .search import SearchResult
from .thumbnails import IconData, Thumbnailer

log = logging.getLogger(__name__)

FOLDER_MAX_CHARS = 48

# GNOME Shell asks GetResultMetas once per id and caches the result actor until the search is
# cleared (search.js _ensureResultActors), so a thumbnail that misses this budget never shows.
# It is much longer than KRunner's budget, which can update a match later.
GNOME_THUMBNAIL_BUDGET_SECONDS = 0.75
# GNOME's list view shows this many results per provider; only those get thumbnails prewarmed.
GNOME_VISIBLE_RESULTS = 5

# sd_booted(3): systemd is PID 1 when this directory exists.
SYSTEMD_RUNTIME_DIR = "/run/systemd/system"
# How long to watch a ``systemd-run --scope`` launch for an early failure before trusting it.
SCOPE_START_TIMEOUT_SECONDS = 1.0


def shorten_folder(path: str, limit: int = FOLDER_MAX_CHARS) -> str:
    """Shorten a home-shortened folder path in the middle to fit within ``limit`` chars."""
    if len(path) <= limit:
        return path

    if path.startswith("~/"):
        head = "~"
        rest = [p for p in path[2:].split("/") if p]
    elif path.startswith("/"):
        parts = [p for p in path[1:].split("/") if p]
        if len(parts) <= 1:
            return path
        head = "/" + parts[0]
        rest = parts[1:]
    else:
        parts = [p for p in path.split("/") if p]
        if len(parts) <= 2:
            return path
        head = parts[0]
        rest = parts[1:]

    if len(rest) <= 1:
        return path

    prefix = f"{head}/…/"
    kept = [rest[-1]]
    total = len(prefix) + len(rest[-1])
    for part in reversed(rest[:-1]):
        added = 1 + len(part)
        if total + added > limit:
            break
        kept.append(part)
        total += added

    return prefix + "/".join(reversed(kept))


def gnome_description(result: SearchResult, home: Path | None = None) -> str:
    """Format the one-line description for a GNOME Shell search result."""
    snippet = trim_snippet(result.snippet) if result.snippet else ""
    locator = _WS_RE.sub(" ", result.locator).strip() if result.locator else ""

    if snippet:
        base = f"{locator} · {snippet}" if locator else snippet
    else:
        parent = shorten_folder(shorten_home(Path(result.path).parent, home))
        base = f"{locator} · {parent}" if locator else parent

    if result.duplicates > 0:
        base = f"{base}  (+{result.duplicates} similar)"
    return base


def gicon_string(path: str, kind: str) -> str:
    """``Gio.icon_new_for_string`` form of a themed icon: the MIME icon, then the generic icon
    for the kind as a fallback for themes that lack the specific one."""
    names = [icon_name_for(path, kind)]
    fallback = KIND_FALLBACK_ICONS.get(kind)
    if fallback and fallback not in names:
        names.append(fallback)
    if len(names) == 1:
        return names[0]
    return ". GThemedIcon " + " ".join(names)


class GnomeLauncher:
    """Opens files with the default application (``gio open``, fallback ``xdg-open``).

    The daemon runs as ``gemmasearch.service`` with ``KillMode=control-group``: anything it
    spawns directly stays in the service's cgroup and is killed by ``systemctl --user
    restart/stop gemmasearch``, taking the user's open documents with it. Under systemd the
    opener is therefore started in its own transient scope (``systemd-run --user --scope``),
    falling back to a plain spawn if ``systemd-run`` is missing or fails to start.

    Injectable so tests can inspect calls without opening windows.
    """

    def _scope_prefix(self) -> list[str] | None:
        """``systemd-run`` argv prefix that moves the child into its own scope, if usable."""
        if not (os.environ.get("INVOCATION_ID") or os.path.isdir(SYSTEMD_RUNTIME_DIR)):
            return None
        systemd_run = shutil.which("systemd-run")
        if not systemd_run:
            return None
        return [systemd_run, "--user", "--scope", "--collect", "--quiet", "--"]

    def _popen(self, argv: list[str]) -> subprocess.Popen[bytes] | None:
        try:
            return subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=os.environ.copy(),
            )
        except OSError as e:
            log.warning("failed to spawn %s: %s", argv, e)
            return None

    def _spawn_detached(self, argv: list[str]) -> bool:
        return self._popen(argv) is not None

    async def _spawn_scoped(self, argv: list[str]) -> bool:
        """Run ``argv`` in a transient systemd scope. False if that is unavailable or failed.

        ``systemd-run --scope`` registers the scope and then execs ``argv`` in place, so an
        early non-zero exit means either the scope could not be created or the opener itself
        failed; the plain retry is harmless in the latter case (it fails the same way).
        """
        prefix = self._scope_prefix()
        if prefix is None:
            return False
        proc = self._popen(prefix + argv)
        if proc is None:
            return False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SCOPE_START_TIMEOUT_SECONDS
        while proc.poll() is None and loop.time() < deadline:
            await asyncio.sleep(0.02)
        if proc.returncode not in (None, 0):
            log.info("systemd-run scope launch of %s exited %s; spawning directly", argv, proc.returncode)
            return False
        return True

    async def spawn(self, argv: list[str]) -> bool:
        """Start ``argv`` detached, outside the service's cgroup when possible."""
        if await self._spawn_scoped(argv):
            return True
        return self._spawn_detached(argv)

    async def open_file(self, path: str) -> None:
        gio = shutil.which("gio")
        if gio and await self.spawn([gio, "open", path]):
            return
        xdg_open = shutil.which("xdg-open")
        if xdg_open:
            await self.spawn([xdg_open, path])
        else:
            log.warning("neither gio nor xdg-open found: cannot open %s", path)


class GnomeSearchService(ServiceInterface):
    """D-Bus service implementing ``org.gnome.Shell.SearchProvider2``."""

    def __init__(
        self,
        search: Callable[[str], Awaitable[list[SearchResult]]],
        *,
        max_results: int = 10,
        on_activity: Callable[[], None] | None = None,
        launcher: GnomeLauncher | None = None,
        thumbnailer: Thumbnailer | None = None,
        thumbnail_budget: float = GNOME_THUMBNAIL_BUDGET_SECONDS,
        home: Path | None = None,
    ) -> None:
        super().__init__(GNOME_INTERFACE)
        self._search = search
        self._max_results = max(1, int(max_results))
        self._on_activity = on_activity
        self._launcher = launcher if launcher is not None else GnomeLauncher()
        self._thumbnailer = thumbnailer if thumbnailer is not None else Thumbnailer()
        self._thumbnail_budget = thumbnail_budget
        self._home = home
        self._recent: OrderedDict[str, SearchResult] = OrderedDict()
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _remember(self, result: SearchResult) -> str:
        result_id = result.path
        self._recent[result_id] = result
        self._recent.move_to_end(result_id)
        while len(self._recent) > RECENT_RESULTS_MAX:
            self._recent.popitem(last=False)
        return result_id

    async def _run_search(self, terms: list[str]) -> list[str]:
        q = " ".join(terms).strip()
        if count_alnum(q) < 3:
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

        results = list((raw_results or [])[: self._max_results])
        ids = [self._remember(r) for r in results]
        # Start thumbnails now so GetResultMetas (which follows right away) finds them cached.
        # Only for the results GNOME will actually display; the rest get one on demand.
        for r in results[:GNOME_VISIBLE_RESULTS]:
            if r.kind in (ft.IMAGE, ft.VIDEO) and self._thumbnailer.get_cached(r.path, r.mtime) is None:
                self._spawn_thumb(r)
        return ids

    def _spawn_thumb(self, result: SearchResult) -> asyncio.Task[IconData | None]:
        task = asyncio.create_task(self._safe_thumb(result))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _safe_thumb(self, result: SearchResult) -> IconData | None:
        try:
            return await self._thumbnailer.icon_data(result.path, result.kind, result.mtime)
        except Exception as e:
            log.debug("thumbnail error for %s: %s", result.path, e)
            return None

    async def _gather_thumbnails(self, results: list[SearchResult]) -> dict[str, IconData]:
        """Thumbnails ready within the budget; the rest keep generating in the background."""
        ready: dict[str, IconData] = {}
        pending: dict[asyncio.Task[IconData | None], str] = {}
        for r in results:
            if r.kind not in (ft.IMAGE, ft.VIDEO):
                continue
            cached = self._thumbnailer.get_cached(r.path, r.mtime)
            if cached is not None:
                ready[r.path] = cached
            else:
                # The thumbnailer dedupes in-flight work, so this joins any prewarm task.
                pending[self._spawn_thumb(r)] = r.path

        if pending and self._thumbnail_budget > 0:
            done, _ = await asyncio.wait(pending.keys(), timeout=self._thumbnail_budget)
            for task in done:
                if not task.cancelled() and task.exception() is None:
                    icon = task.result()
                    if icon is not None:
                        ready[pending[task]] = icon
        return ready

    @method()
    async def GetInitialResultSet(self, terms: "as") -> "as":  # noqa: F821, N802, UP037
        return await self._run_search(list(terms))

    @method()
    async def GetSubsearchResultSet(self, previous_results: "as", terms: "as") -> "as":  # noqa: F821, N802, UP037
        # Semantic results don't narrow monotonically as the query grows: search again.
        return await self._run_search(list(terms))

    def _synthetic_meta(self, path: str) -> dict[str, Variant]:
        """Minimal meta for an id we no longer remember (daemon restart, LRU eviction).

        GNOME Shell's remoteSearch.js rejects a reply whose length differs from the request
        ("Wrong number of result metas"), blanking the whole provider, so every id needs one.
        Ids are file paths, which is enough for a name, folder and themed icon.
        """
        p = Path(path)
        kind = ft.classify(p.name) or ft.TEXT
        return {
            "id": Variant("s", path),
            "name": Variant("s", p.name or path),
            "description": Variant("s", shorten_folder(shorten_home(p.parent, self._home))),
            "clipboardText": Variant("s", path),
            "gicon": Variant("s", gicon_string(path, kind)),
        }

    @method()
    async def GetResultMetas(self, identifiers: "as") -> "aa{sv}":  # noqa: F821, N802, UP037
        results = [self._recent[i] for i in identifiers if i in self._recent]
        try:
            thumbs = await self._gather_thumbnails(results)
        except asyncio.CancelledError:
            thumbs = {}
        except Exception:
            log.exception("thumbnail gathering failed")
            thumbs = {}

        metas: list[dict[str, Variant]] = []
        for identifier in identifiers:
            r = self._recent.get(identifier)
            if r is None:
                metas.append(self._synthetic_meta(identifier))
                continue
            meta: dict[str, Variant] = {
                "id": Variant("s", r.path),
                "name": Variant("s", Path(r.path).name or r.path),
                "description": Variant("s", gnome_description(r, self._home)),
                "clipboardText": Variant("s", r.path),
            }
            icon = thumbs.get(r.path)
            if icon is not None:
                w, h, rowstride, has_alpha, bps, channels, raw_bytes = icon
                meta["icon-data"] = Variant(
                    "(iiibiiay)",
                    [int(w), int(h), int(rowstride), bool(has_alpha), int(bps), int(channels), bytes(raw_bytes)],
                )
            else:
                meta["gicon"] = Variant("s", gicon_string(r.path, r.kind))
            metas.append(meta)
        return metas

    @staticmethod
    def _openable_unknown_id(identifier: str) -> bool:
        """Whether an id we don't remember may still be opened.

        Ids are paths we handed to GNOME Shell earlier (before a restart or LRU eviction).
        Opening one is no more than the caller could do itself: only same-user processes can
        call us on the session bus, and they can run ``gio open`` directly. Still, only accept
        what we could have returned: an absolute path to an existing regular file of an
        indexable kind, which keeps ``.desktop`` files, scripts and other launchables out.
        """
        return (
            os.path.isabs(identifier)
            and ft.classify(os.path.basename(identifier)) is not None
            and os.path.isfile(identifier)
        )

    @method()
    async def ActivateResult(self, identifier: "s", terms: "as", timestamp: "u"):  # noqa: F821, N802, UP037
        result = self._recent.get(identifier)
        if result is not None:
            path = result.path
        elif self._openable_unknown_id(identifier):
            path = identifier
        else:
            log.debug("ActivateResult(%r): unknown id, ignored", identifier)
            return
        try:
            await self._launcher.open_file(path)
        except Exception:
            log.exception("ActivateResult(%r) failed", identifier)

    @method()
    def LaunchSearch(self, terms: "as", timestamp: "u"):  # noqa: F821, N802, UP037
        log.debug("LaunchSearch(%r) ignored", terms)
