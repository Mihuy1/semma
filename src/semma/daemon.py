"""semmad: the long-running service behind KRunner.

Runs as a systemd user service (Type=dbus) that owns `dbus_names.BUS_NAME` on the session bus,
answers KRunner queries (`krunner.KRunnerService` at RUNNER_PATH) and GNOME Shell overview
searches (`gnome_search.GnomeSearchService` at GNOME_PATH) and keeps the index current in the
background. `Control` at CONTROL_PATH reports status, takes pause/resume/rescan and
ReloadConfig (read config.toml again: the worker swaps the new Config in at a file boundary,
then rescans if what is indexed or its priority changed).

Threads and what runs where:

  event loop (main thread)   D-Bus, query debounce/stale-cancellation, idle-unload timer.
                             Never touches SQLite or the model directly.
  semma-search (1)     every query: embed_query + Searcher.search_vector on this one
                             thread (its read-only connection and the in-memory vector cache);
                             also RetryFailed.
  semma-status (1)     Status counts and Failed() (its own read-only connection), so a status
                             window polling every second never waits behind a query, and a
                             query never waits behind it. A query can sit on the search thread
                             for minutes while the model downloads. Also reads the config file
                             for ReloadConfig.
  semma-model (1)      model load (shared by all queries waiting for it) and unload.
  semma-worker         scans (startup, inotify, daily, Rescan()) and indexing slices.
                             nice 10 + idle IO class; the indexer's extraction thread and the
                             ffmpeg processes it starts inherit both.
  semma-gpumon         while a slice runs: samples other processes' GPU use every
                             gpu_check_seconds and stops the slice when the GPU is busy.
  semma-watcher        (watcher.DirWatcher) inotify; changes become partial rescans.

The model is shared: queries take it through the embedder's GpuLock with priority, so a query
waits for at most the index batch already on the GPU. Indexing runs in slices
(`Indexer.run` with a stop event): pause, GPU busy and shutdown stop a slice at once (files are
either committed or left pending), scan requests stop it at the next file boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import ctypes
import logging
import os
import platform
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from dbus_fast import DBusError, Message, MessageFlag, MessageType, NameFlag, RequestNameReply, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, dbus_method

from . import dbus_names
from . import filetypes as ft
from .config import Config, ConfigError, canonical, load_config
from .indexer import Indexer, IndexStats, ModelMismatch
from .onboarding import onboarding_done
from .scanner import Scanner, _is_app_install
from .search import Searcher, SearchResult
from .store import Store

log = logging.getLogger("semma.daemon")

MIN_QUERY_CHARS = 3
PROGRESS_LOG_EVERY = 100  # files
GPU_BUSY_RECHECK_S = 60.0
GPU_BUSY_UNLOAD_S = 60.0  # unload after 1 min without queries while another app uses the GPU
# A slice that only skipped/deferred files (changed since the scan, database busy) leaves them
# pending; don't spin on them, wait for the rescan their change triggers (or this long).
NO_PROGRESS_RETRY_S = 300.0
WORKER_BACKOFF_S = (5.0, 600.0)  # after an unexpected error: first wait, maximum wait
DOWNLOAD_BACKOFF_S = (5.0, 300.0)  # model download failed: first wait, maximum wait
DOWNLOAD_JOIN_S = 2.0  # don't let a stuck transfer hold up shutdown; the part file resumes
SHUTDOWN_JOIN_S = 60.0
MAX_PARTIAL_SCAN_DIRS = 64  # more changed directories than this: one full scan instead

_DEFAULT: Any = object()  # "construct the real collaborator"; None means "none"

# How `Daemon.reload_config` applies each Config field, with the config.toml name it reports.
# Taken over only by a restart (the embedder, the store and its vector cache are built from them):
RESTART_FIELDS: dict[str, str] = {
    "db_path": "db_path", "model_name": "model.name", "device": "model.device",
    "query_device": "model.query_device", "llama_lib_dir": "model.llama_lib_dir",
    "model_dir": "model.model_dir", "batch_sizes": "index.batch_sizes",
    "vector_cache_max_chunks": "search.vector_cache_max_chunks",
}
# Decide what is indexed: a change runs a full rescan.
SCAN_FIELDS: dict[str, str] = {
    "roots": "scan.roots", "include_mounts": "scan.include_mounts", "skip_hidden": "scan.skip_hidden",
    "skip_app_installs": "scan.skip_app_installs", "kinds": "scan.kinds",
    "exclude_extensions": "scan.exclude_extensions", "exclude_dir_names": "scan.exclude_dir_names",
    "root_system_dir_names": "scan.root_system_dir_names",
    "project_build_dir_names": "scan.project_build_dir_names", "project_markers": "scan.project_markers",
    "exclude_paths": "scan.exclude_paths", "max_bytes": "scan.max_file_mb",
}
# Priority scores: a full rescan stores the new score of every file, changed or not.
PRIORITY_FIELDS: dict[str, str] = {
    "locations": "priority.locations", "other_location": "priority.other_location",
    "type_weights": "priority.type_weights", "project_penalty": "priority.project_penalty",
    "depth_free_levels": "priority.depth_free_levels",
    "depth_penalty_per_level": "priority.depth_penalty_per_level",
    "depth_penalty_max": "priority.depth_penalty_max", "recency": "priority.recency",
    "tiny_bytes": "priority.tiny_bytes", "tiny_penalty": "priority.tiny_penalty",
    "huge_video_bytes": "priority.huge_video_bytes", "huge_video_penalty": "priority.huge_video_penalty",
}
# Read when a file is indexed (or a slice starts): files indexed from then on; nothing re-indexed.
NEXT_FILE_FIELDS: dict[str, str] = {
    "max_media_seconds": "media.max_minutes", "media_segment_seconds": "media.segment_seconds",
    "prefetch_chunks": "index.prefetch_chunks",
}
# Read at use time (or pushed to the component that copied them).
LIVE_FIELDS: dict[str, str] = {
    "z_gate": "search.z_gate", "z_floor": "search.z_floor", "knn_k": "search.knn_k",
    "location_prior": "search.location_prior", "duplicate_similarity": "search.duplicate_similarity",
    "model_idle_unload_minutes": "daemon.model_idle_unload_minutes",
    "index_model_unload_seconds": "daemon.index_model_unload_seconds",
    "rescan_hours": "daemon.rescan_hours", "watch_debounce_seconds": "daemon.watch_debounce_seconds",
    "pause_when_gpu_busy": "daemon.pause_when_gpu_busy",
    "gpu_busy_threshold": "daemon.gpu_busy_threshold", "gpu_check_seconds": "daemon.gpu_check_seconds",
    "krunner_max_results": "daemon.krunner_max_results", "query_debounce_ms": "daemon.query_debounce_ms",
}
# First-run setup. false stops scanning and indexing; true starts a full scan. The index is kept.
ONBOARDING_FIELDS: dict[str, str] = {"onboarding_complete": "onboarding_complete"}
# Not settings: always those the daemon started with.
FIXED_FIELDS: tuple[str, ...] = ("home", "source_path", "source_home")

# Parents of the drives `include_mounts` adds (config.detect_mounts). With it on, files below
# them are "configured": a drive that is unplugged keeps its index.
MOUNT_PARENTS: tuple[Path, ...] = (Path("/mnt"), Path("/media"), Path("/run/media"))


def configured_areas(cfg: Config, scanned_roots: Iterable[Path] = ()) -> list[Path]:
    """Where indexed files may be under these settings, available right now or not: the
    configured roots (even when missing or unmounted), the roots just scanned and, with
    include_mounts, everything below the mount parents."""
    areas = [canonical(r) for r in cfg.roots] + list(scanned_roots)
    if cfg.include_mounts:
        for parent in MOUNT_PARENTS:
            areas += [parent, canonical(parent)]
    return areas


class GpuMonitorLike(Protocol):
    def busy(self, threshold: float | None = None) -> bool: ...


class WatcherLike(Protocol):
    def set_dirs(self, dirs: Iterable[str]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


WatcherFactory = Callable[[Callable[[set[str]], None], float], WatcherLike]
RunnerFactory = Callable[..., ServiceInterface]


# -- thread priority ------------------------------------------------------------------------

# ioprio_set(2) syscall numbers; other architectures just keep the normal IO class.
_IOPRIO_SET_NR = {"x86_64": 251, "aarch64": 30, "riscv64": 30, "ppc64le": 273, "ppc64": 273,
                  "i386": 289, "i686": 289, "armv7l": 314}
IOPRIO_WHO_PROCESS = 1
IOPRIO_CLASS_IDLE = 3
IOPRIO_CLASS_SHIFT = 13


def lower_thread_priority(nice: int = 10) -> None:
    """Lower the calling thread's CPU priority (Linux nice is per thread) and put its IO in the
    idle class. Threads and processes it starts inherit both. Best effort: never raises."""
    tid = threading.get_native_id()
    try:
        if os.getpriority(os.PRIO_PROCESS, tid) < nice:
            os.setpriority(os.PRIO_PROCESS, tid, nice)
    except (OSError, AttributeError) as exc:
        log.debug("could not lower CPU priority: %s", exc)
    nr = _IOPRIO_SET_NR.get(platform.machine())
    if nr is None:
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        res = libc.syscall(ctypes.c_long(nr), ctypes.c_int(IOPRIO_WHO_PROCESS), ctypes.c_int(tid),
                           ctypes.c_int(IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT))
        if res != 0:
            log.debug("ioprio_set failed: %s", os.strerror(ctypes.get_errno()))
    except Exception as exc:  # no libc, no syscall(): keep the default IO class
        log.debug("ioprio_set unavailable: %s", exc)


# -- partial rescans ------------------------------------------------------------------------

def partial_scan_roots(cfg: Config, changed: Iterable[str], roots: list[Path] | None = None) -> list[Path] | None:
    """Directories to rescan for a set of changed directories, or None for a full scan.

    The scanner walks each given directory as if it were a root: it does not know whether the
    directory lies inside a software project, and it prunes "root system" names (Windows,
    Program Files...) directly below it. Where that would differ from a full scan, this returns
    None. Directories the full scan prunes (hidden, excluded, venvs...) are dropped; deleted
    directories are replaced by their nearest existing parent.
    """
    roots = cfg.effective_roots() if roots is None else roots
    root_strs = sorted((str(r) for r in roots), key=len, reverse=True)  # most specific first
    anchors = {str(p) for p in cfg.locations}
    excluded = {str(p) for p in cfg.exclude_paths}
    markers = cfg.project_markers
    out: set[str] = set()
    for d in changed:
        d = os.path.normpath(d)
        while not os.path.isdir(d):
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        root = next((r for r in root_strs if d == r or d.startswith(r.rstrip("/") + "/")), None)
        if root is None:
            continue  # outside every root (an unmounted drive, an excluded path's parent...)
        if d == root:
            out.add(d)
            continue
        path, in_project, pruned = root, False, False
        names: list[str] = []
        for name in d[len(root):].strip("/").split("/"):
            parent, path = path, os.path.join(path, name)
            folded = name.casefold()
            if ((cfg.skip_hidden and name.startswith(".")) or folded in cfg.exclude_dir_names
                    or (parent == root and folded in cfg.root_system_dir_names)
                    or (in_project and folded in cfg.project_build_dir_names)
                    or path in excluded or path in root_strs or os.path.ismount(path)):
                pruned = True
                break
            try:
                names = os.listdir(path)
            except OSError:
                return None
            if path not in anchors:
                if "pyvenv.cfg" in names or (cfg.skip_app_installs and _is_app_install(names)):
                    pruned = True
                    break
                if not in_project and any(n in markers for n in names):
                    in_project = True
        if pruned:
            continue
        if in_project or any(n.casefold() in cfg.root_system_dir_names for n in names):
            return None
        out.add(d)
    # Keep only the topmost of nested directories (the scanner walks subdirectories itself).
    top: list[str] = []
    for d in sorted(out):
        if not any(d == t or d.startswith(t.rstrip("/") + "/") for t in top):
            top.append(d)
    if len(top) > MAX_PARTIAL_SCAN_DIRS:
        return None
    return [Path(d) for d in top]


class _ScanAborted(Exception):
    pass


# -- the daemon -----------------------------------------------------------------------------

class Daemon:
    """Search service plus background indexing; every collaborator can be injected (tests).

    gpu_monitor / watcher_factory / runner_factory / gnome_factory: `_DEFAULT` builds the real one (if its
    module is available), None disables it.
    """

    gpu_busy_recheck_s = GPU_BUSY_RECHECK_S

    def __init__(self, cfg: Config, *, store: Store | None = None, embedder: Any = None,
                 extract_fn: Any = None, gpu_monitor: GpuMonitorLike | None = _DEFAULT,
                 watcher_factory: WatcherFactory | None = _DEFAULT,
                 runner_factory: RunnerFactory | None = _DEFAULT,
                 gnome_factory: RunnerFactory | None = _DEFAULT,
                 model_download: Callable[[Config], None] | None = _DEFAULT,
                 bus_name: str = dbus_names.BUS_NAME,
                 config_loader: Callable[[], Config] | None = None) -> None:
        self.cfg = cfg
        # ReloadConfig: the settings the daemon started with (restart-only settings compare
        # against these), the newest settings accepted, and the loader of the config file.
        self._start_cfg = cfg
        self._accepted_cfg = cfg
        self._reload_lock = threading.Lock()
        self._config_loader = config_loader or self._load_config_file
        self.bus_name = bus_name
        self.store = store if store is not None else Store(
            cfg.db_path, vector_cache=True, cache_max_chunks=cfg.vector_cache_max_chunks)
        # None: the caller disabled the download (tests). `_DEFAULT`: download the GGUF files.
        self._model_download = download_model_files if model_download is _DEFAULT else model_download
        self._download_error = ""
        self._download_thread: threading.Thread | None = None
        self._setup_done = onboarding_done(cfg, self.store)
        self._logged_wait_setup = False
        self._logged_wait_model = False
        if embedder is None:
            from .embedder import Embedder

            embedder = Embedder(cfg)
        self.embedder = embedder
        self.searcher = Searcher(self.store, cfg, embedder)
        self.indexer = Indexer(self.store, cfg, embedder, extract_fn=extract_fn)
        self.gpu_monitor = _default_gpu_monitor(cfg, embedder=embedder) if gpu_monitor is _DEFAULT else gpu_monitor
        self._watcher_factory = _default_watcher_factory() if watcher_factory is _DEFAULT else watcher_factory
        self._runner_factory = _default_runner_factory() if runner_factory is _DEFAULT else runner_factory
        self._gnome_factory = _default_gnome_factory() if gnome_factory is _DEFAULT else gnome_factory
        self.watcher: WatcherLike | None = None
        self._services: list[Any] = []  # the KRunner / GNOME interfaces (serve sets them)

        # Worker state, guarded by _cv (simple reads of single attributes are fine without it).
        self._cv = threading.Condition()
        self._wakeup = False
        self._stopping = threading.Event()
        self._slice_stop = threading.Event()
        self._user_paused = False
        self._scan_full = False
        self._scan_dirs: set[str] = set()
        self._pending_cfg: Config | None = None  # accepted by a reload, applied by the worker
        self._phase = "idle"  # idle | scanning | indexing | paused
        self._pause_reason = ""
        self._current_file = ""
        self._last_scan_at = 0.0  # wall clock
        self._last_full_scan = float("-inf")  # monotonic
        self._no_progress_until = 0.0
        self._last_work = 0.0  # monotonic: last file indexed
        self._watched: set[str] = set()
        self._gpu_busy = False
        self._gpu_checked_at = float("-inf")
        self._gpu_sample_lock = threading.Lock()
        self._run_logged = 0
        # Progress for Status, each replaced whole by the worker (one attribute write) and read
        # by the event loop: (done, total, failed, files per second) of the running slice, and
        # the scanner of the running scan (its stats are only read).
        self._run_progress: tuple[int, int, int, float] = (0, 0, 0, 0.0)
        self._scanner: Scanner | None = None
        self._worker: threading.Thread | None = None
        self._gpu_thread: threading.Thread | None = None

        # Query state (event loop only).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._query_seq = 0
        self._newest: asyncio.Future[None] | None = None
        self._queries_in_flight = 0
        self._last_query = 0.0  # monotonic
        self._load_task: asyncio.Future[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._search_exec = ThreadPoolExecutor(1, thread_name_prefix="semma-search")
        self._model_exec = ThreadPoolExecutor(1, thread_name_prefix="semma-model")
        self._status_exec = ThreadPoolExecutor(1, thread_name_prefix="semma-status")
        self._started = time.monotonic()
        self._running = False
        self._closed = False

    # -- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the background threads and timers (call from the running event loop).

        The model download starts immediately. A full scan starts only once setup is finished;
        until then the worker waits and searches return nothing.
        """
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._started = time.monotonic()
        self._setup_done = onboarding_done(self.cfg, self.store)
        if self._watcher_factory is not None:
            try:
                self.watcher = self._watcher_factory(self._on_watch_change, self.cfg.watch_debounce_seconds)
                self.watcher.start()
            except Exception:
                log.exception("file watcher unavailable; relying on the daily rescan")
                self.watcher = None
        if self._setup_done:
            with self._cv:
                self._scan_full = True
        self._worker = threading.Thread(target=self._worker_main, name="semma-worker", daemon=True)
        self._worker.start()
        self._start_download()
        if self.gpu_monitor is not None:
            self._gpu_thread = threading.Thread(target=self._gpu_main, name="semma-gpumon", daemon=True)
            self._gpu_thread.start()
        self._idle_task = asyncio.ensure_future(self._idle_loop())

    async def stop(self) -> None:
        """Graceful shutdown: the running index batch finishes (files are committed whole or
        left pending), the watcher stops, the database is closed."""
        if self._closed:
            return
        self._closed = True
        log.info("shutting down")
        self._stopping.set()
        self._slice_stop.set()
        self._wake()
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._idle_task
        if self.watcher is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.watcher.stop)
        clean = True
        for thread, timeout in (
            (self._download_thread, DOWNLOAD_JOIN_S),
            (self._worker, SHUTDOWN_JOIN_S),
            (self._gpu_thread, SHUTDOWN_JOIN_S),
        ):
            if thread is not None:
                await asyncio.to_thread(thread.join, timeout)
                if thread.is_alive():
                    log.warning("%s did not stop within %.0fs", thread.name, timeout)
                    if thread is not self._download_thread:
                        clean = False
        await asyncio.to_thread(self._search_exec.shutdown, True, cancel_futures=True)
        await asyncio.to_thread(self._model_exec.shutdown, True, cancel_futures=True)
        await asyncio.to_thread(self._status_exec.shutdown, True, cancel_futures=True)
        if clean:
            self.store.close()
        self._running = False
        log.info("stopped")

    # -- queries --------------------------------------------------------------------------
    def record_activity(self) -> None:
        self._last_query = time.monotonic()

    async def search(self, query: str) -> list[SearchResult]:
        """Search for KRunner: best first, at most `krunner_max_results`.

        Only the newest query is worked on: a query superseded by a newer one (during the
        debounce, while the model loads or while waiting for the search thread / GPU) returns
        [] at once. Never raises.
        """
        loop = asyncio.get_running_loop()
        self._query_seq += 1
        seq = self._query_seq
        if self._newest is not None and not self._newest.done():
            self._newest.set_result(None)
        newer = self._newest = loop.create_future()
        self._queries_in_flight += 1
        self.record_activity()
        t0 = time.perf_counter()
        try:
            q = query.strip()
            if len(q) < MIN_QUERY_CHARS or self._closed:
                return []
            # Setup is still open: don't load the model (that would pin the device). Run setup
            # again keeps the index, so a model that is loaded already still answers.
            if not self._setup_done and not self._can_search_during_setup():
                return []
            debounce = self.cfg.query_debounce_ms / 1000
            if debounce > 0:
                await asyncio.wait({newer}, timeout=debounce)
                if newer.done():
                    log.debug("query %r superseded during debounce", q)
                    return []
            t1 = time.perf_counter()
            if not self._model_ready():
                ok, _ = await self._unless_superseded(self.ensure_model(), newer)
                if not ok:
                    log.debug("query %r superseded while the model loaded", q)
                    return []
            ok, results = await self._unless_superseded(
                loop.run_in_executor(self._search_exec, self._search_sync, q, seq), newer)
            if not ok or results is None:
                log.debug("query %r superseded", q)
                return []
            log.debug("query %r: %d results in %.0f ms (+%.0f ms debounce)", q, len(results),
                      (time.perf_counter() - t1) * 1000, (t1 - t0) * 1000)
            return results
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("query failed")
            return []
        finally:
            self._queries_in_flight -= 1
            self.record_activity()

    @staticmethod
    async def _unless_superseded(aw: Any, newer: asyncio.Future[None]) -> tuple[bool, Any]:
        """(True, result) if `aw` finishes first, (False, None) if a newer query arrives first.
        `aw` is not cancelled (a shared model load must go on)."""
        fut = asyncio.ensure_future(aw)
        await asyncio.wait({fut, newer}, return_when=asyncio.FIRST_COMPLETED)
        if fut.done():
            return True, fut.result()
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # no "never retrieved"
        return False, None

    def _search_sync(self, query: str, seq: int) -> list[SearchResult] | None:
        """On the search thread. None if the query went stale while it waited."""
        if seq != self._query_seq:
            return None
        qvec = self.embedder.embed_query(query)  # GpuLock: ahead of waiting index batches
        if seq != self._query_seq:
            return None
        return self.searcher.search_vector(qvec, query, self.cfg.krunner_max_results)

    def _can_search_during_setup(self) -> bool:
        """Only after Run setup again (the flag is false, not absent), and only with a model
        that is in memory already: the query model, or the index model, which answers too."""
        return self.cfg.onboarding_complete is False and self._model_ready()

    def _model_ready(self) -> bool:
        return bool(getattr(self.embedder, "query_ready", getattr(self.embedder, "loaded", True)))

    def _query_model_loaded(self) -> bool:
        if hasattr(self.embedder, "query_model_loaded"):
            return bool(self.embedder.query_model_loaded)
        return bool(getattr(self.embedder, "loaded", True))

    def _index_model_loaded(self) -> bool:
        if hasattr(self.embedder, "index_model_loaded"):
            return bool(self.embedder.index_model_loaded)
        return bool(getattr(self.embedder, "loaded", True))

    async def ensure_model(self) -> None:
        """Load the query model if needed (reusing the full index model if loaded);
        concurrent callers share one load."""
        if self._model_ready():
            return
        if self._load_task is None or self._load_task.done():
            self._load_task = asyncio.ensure_future(self._load_model())
        await asyncio.shield(self._load_task)

    async def _load_model(self) -> None:
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        log.info("loading the model")
        load_fn = getattr(self.embedder, "load_query", self.embedder.load)
        await loop.run_in_executor(self._model_exec, load_fn)
        log.info("model ready in %.1fs", time.monotonic() - started)

    # -- idle unload ----------------------------------------------------------------------
    def _idle_seconds(self) -> float:
        return self.cfg.model_idle_unload_minutes * 60

    def _index_unload_seconds(self) -> float:
        return self.cfg.index_model_unload_seconds

    async def _idle_loop(self) -> None:
        while True:
            # Re-read every round: a reload may change the times (0: never unload).
            candidates = [x for x in (self._idle_seconds(), self._index_unload_seconds()) if x > 0]
            interval = min(60.0, max(0.02, min(candidates) / 4)) if candidates else 60.0
            await asyncio.sleep(interval)
            try:
                await self.check_idle()
            except Exception:
                log.exception("idle check failed")

    def _unload_index_if_no_work(self) -> bool:
        """On the model thread (keeps SQLite off the event loop)."""
        if self._phase == "indexing" or (self._phase != "paused" and self.indexer.count_pending() > 0):
            return False
        return self.embedder.unload_index()

    async def check_idle(self) -> bool:
        """Unload the full index model after its grace period once indexing finishes, and unload
        the query model (or legacy single model) after `model_idle_unload_minutes` without queries."""
        if self._load_task is not None and not self._load_task.done():
            return False
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        any_unloaded = False

        has_split = hasattr(self.embedder, "unload_index") and hasattr(self.embedder, "unload_query")
        if has_split:
            # 1. Full index model: unload after index_model_unload_seconds once not indexing and no
            #    pending work. Paused indexing (GPU busy, user pause) gives it back even with
            #    files pending; the next slice loads it again.
            idx_grace = self._index_unload_seconds()
            if (idx_grace > 0 and self._index_model_loaded() and self._phase != "indexing"
                    and not self._queries_in_flight):
                last_idx = max(self._last_work, float(getattr(self.embedder, "index_last_used", 0.0)))
                idx_dev = str(getattr(self.embedder, "device", "") or "").strip().lower()
                idx_on_gpu = idx_dev != "cpu"
                needed_idx = min(idx_grace, GPU_BUSY_UNLOAD_S) if (self._gpu_busy and idx_on_gpu) else idx_grace
                if now - last_idx >= needed_idx:
                    unloaded = await loop.run_in_executor(self._model_exec, self._unload_index_if_no_work)
                    if unloaded is not False:
                        log.info("index model unloaded after %.0fs without indexing%s", needed_idx,
                                 " (GPU busy)" if (self._gpu_busy and idx_on_gpu) else "")
                        any_unloaded = True

            # 2. Text-only query model: unload after model_idle_unload_minutes without queries.
            q_idle = self._idle_seconds()
            if q_idle > 0 and self._query_model_loaded() and not self._queries_in_flight:
                last_q = max(self._last_query, float(getattr(self.embedder, "query_last_used", 0.0)))
                q_on_gpu = str(getattr(self.embedder, "query_device", "") or "").strip().lower().startswith(
                    ("cuda", "vulkan", "gpu")
                )
                needed_q = min(q_idle, GPU_BUSY_UNLOAD_S) if (self._gpu_busy and q_on_gpu) else q_idle
                if now - last_q >= needed_q:
                    unloaded = await loop.run_in_executor(self._model_exec, self.embedder.unload_query)
                    if unloaded is not False:
                        log.info("query model unloaded after %.0f min without queries%s", needed_q / 60,
                                 " (GPU busy)" if (self._gpu_busy and q_on_gpu) else "")
                        any_unloaded = True
            return any_unloaded

        # Legacy single-model embedder fallback (e.g. simple test doubles without unload_index/unload_query).
        idle = self._idle_seconds()
        if (idle <= 0 or not self._model_ready() or not hasattr(self.embedder, "unload")
                or self._queries_in_flight or self._phase == "indexing"):
            return False
        last = max(self._last_query, self._last_work, float(getattr(self.embedder, "last_used", 0.0)))
        # While another application (a game) keeps the GPU busy, give its VRAM back quickly.
        needed = min(idle, GPU_BUSY_UNLOAD_S) if self._gpu_busy else idle
        if now - last < needed:
            return False
        unloaded = await loop.run_in_executor(self._model_exec, self.embedder.unload)
        if unloaded is not False:
            log.info("model unloaded after %.0f min without queries or indexing%s", needed / 60,
                     " (GPU busy)" if self._gpu_busy else "")
        return True

    # -- control --------------------------------------------------------------------------
    def pause(self) -> None:
        with self._cv:
            self._user_paused = True
        self._wake(stop_slice=True)
        log.info("indexing paused by request")

    def resume(self) -> None:
        with self._cv:
            was = self._user_paused
            self._user_paused = False
            self._no_progress_until = 0.0
        self._wake()
        if was:
            log.info("indexing resumed by request")

    def rescan(self) -> None:
        if not self._setup_done:
            log.info("full rescan requested, but setup is not finished; it runs when setup is done")
            return
        with self._cv:
            self._scan_full = True
        self._wake()
        log.info("full rescan requested")

    async def retry_failed(self) -> int:
        """Queue every failed file again now; returns how many."""
        loop = asyncio.get_running_loop()
        n = await loop.run_in_executor(
            self._search_exec, lambda: self.store.requeue(("failed",), reset_attempts=False))
        with self._cv:
            self._no_progress_until = 0.0
        self._wake()
        log.info("retrying %d failed files", n)
        return n

    def state(self, downloading: bool = False) -> str:
        if downloading:
            return "downloading"
        if self._phase == "scanning":
            return "scanning"
        if (self._load_task is not None and not self._load_task.done()) or getattr(self.embedder, "loading", False):
            return "loading"
        return self._phase

    async def status(self) -> dict[str, Any]:
        """Everything `Control.Status` reports; see STATUS_SIGNATURES for the types. The keys are
        always all present (zero or empty when that part is not running)."""
        loop = asyncio.get_running_loop()
        by_kind, chunks, dl, files = await loop.run_in_executor(self._status_exec, self._status_sync)
        counts = {status: sum(kinds.values()) for status, kinds in by_kind.items()}
        q_loaded = self._query_model_loaded()
        i_loaded = self._index_model_loaded()
        run_done, run_total, run_failed, run_rate = self._run_progress
        scanner = self._scanner
        scan = scanner.stats if scanner is not None else None
        dl = dl or {}
        return {
            "state": self.state(downloading=bool(dl)),
            "model_loaded": q_loaded or i_loaded,
            "query_model_loaded": q_loaded,
            "index_model_loaded": i_loaded,
            "paused": self._user_paused or self._phase == "paused",
            "pause_reason": self._pause_reason if (self._user_paused or self._phase == "paused") else "",
            "pending": counts.get("pending", 0),
            "indexed": counts.get("indexed", 0),
            "failed": counts.get("failed", 0),
            "chunks": chunks,
            "current_file": self._current_file,
            "last_scan_at": self._last_scan_at,
            "uptime": time.monotonic() - self._started,
            "watched_dirs": len(self._watched),
            "pending_by_kind": by_kind["pending"],
            "indexed_by_kind": by_kind["indexed"],
            "failed_by_kind": by_kind["failed"],
            "download_file": dl.get("file", ""),
            "download_done": dl.get("done", 0),
            "download_total": dl.get("total", 0),
            "download_all_done": dl.get("all_done", 0),
            "download_all_total": dl.get("all_total", 0),
            "setup_pending": not self._setup_done,
            "download_error": self._download_error,
            "model_files_ready": bool(files.get("ready")),
            "model_bytes_done": int(files.get("done") or 0),
            "model_bytes_total": int(files.get("total") or 0),
            "run_done": run_done,
            "run_total": run_total,
            "run_failed": run_failed,
            "run_rate": run_rate,
            "scan_dir": scan.current_dir if scan is not None else "",
            "scan_dirs": scan.dirs if scan is not None else 0,
            "scan_files": scan.files_seen if scan is not None else 0,
            "scan_matched": scan.matched if scan is not None else 0,
        }

    def _status_sync(self) -> tuple[dict[str, dict[str, int]], int, dict[str, Any] | None, dict[str, Any]]:
        """On the status thread: file counts per status and kind, chunks, the model download."""
        db = self.store.reader()
        by_kind: dict[str, dict[str, int]] = {s: dict.fromkeys(ft.KINDS, 0) for s in ("pending", "indexed", "failed")}
        for status, kind, n in db.execute("SELECT status, kind, count(*) FROM files GROUP BY status, kind"):
            by_kind.setdefault(status, dict.fromkeys(ft.KINDS, 0))[kind] = n
        chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return by_kind, chunks, self._download_sync(), self._model_files_status()

    def _model_files_status(self) -> dict[str, Any]:
        """Bytes on disk. A test double with no probe is treated as ready so indexing is not blocked."""
        probe = getattr(self.embedder, "model_files_status", None)
        if probe is None:
            return {"ready": True, "done": 0, "total": 0}
        try:
            status = probe()
        except Exception:
            log.debug("model file status unavailable", exc_info=True)
            return {"ready": True, "done": 0, "total": 0}
        return {
            "ready": bool(status.get("ready")),
            "done": int(status.get("done") or 0),
            "total": int(status.get("total") or 0),
        }

    def _model_files_ready(self) -> bool:
        return bool(self._model_files_status().get("ready"))

    def _download_sync(self) -> dict[str, Any] | None:
        probe = getattr(self.embedder, "download_progress", None)  # test doubles have none
        if probe is None:
            return None
        try:
            return probe()
        except Exception:
            log.debug("download progress unavailable", exc_info=True)
            return None

    async def failed(self, limit: int = 0) -> list[tuple[str, str]]:
        """(path, error) of failed files, the most important first (as `semma status --failed`);
        limit 0: all of them."""
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(
            self._status_exec, lambda: self.store.failed_files(limit or None, db=self.store.reader()))
        return [(r["path"], r["error"] or "") for r in rows]

    # -- model download -------------------------------------------------------------------
    def _start_download(self) -> None:
        """Start the download thread unless one is running (or the download is disabled)."""
        thread = self._download_thread
        if self._model_download is None or (thread is not None and thread.is_alive()):
            return
        self._download_thread = threading.Thread(
            target=self._download_main, name="semma-download", daemon=True)
        self._download_thread.start()

    def _download_main(self) -> None:
        """Fetch the GGUF files without loading them. Retries until they verify or we stop."""
        backoff = DOWNLOAD_BACKOFF_S[0]
        while not self._stopping.is_set():
            try:
                assert self._model_download is not None
                self._model_download(self.cfg)
            except Exception as exc:
                self._download_error = str(exc)
                log.warning("model download failed: %s; retrying in %.0fs", exc, backoff)
                if self._stopping.wait(backoff):
                    return
                backoff = min(backoff * 2, DOWNLOAD_BACKOFF_S[1])
                continue
            self._download_error = ""
            self._logged_wait_model = False
            self._wake()
            log.info("model files are ready")
            return

    # -- worker ---------------------------------------------------------------------------
    def _wake(self, stop_slice: bool = False) -> None:
        with self._cv:
            self._wakeup = True
            self._cv.notify_all()
        if stop_slice:
            self._slice_stop.set()

    def _wait(self, timeout: float) -> None:
        with self._cv:
            if not self._wakeup and not self._stopping.is_set():
                self._cv.wait(max(0.0, timeout))
            self._wakeup = False

    def _set_phase(self, phase: str, reason: str = "") -> None:
        with self._cv:
            if phase != self._phase or reason != self._pause_reason:
                if phase == "paused" and reason != self._pause_reason:
                    log.info("indexing paused: %s", reason)
                elif self._phase == "paused" and phase == "indexing":
                    log.info("indexing resumed")
            self._phase, self._pause_reason = phase, reason
            self._cv.notify_all()

    def _on_watch_change(self, changed: set[str]) -> None:
        """Watcher thread: remember what changed; the worker rescans it at the next file boundary."""
        with self._cv:
            if "*" in changed:
                self._scan_full = True
            else:
                self._scan_dirs |= changed
            self._wakeup = True
            self._cv.notify_all()
        log.debug("changes in %d directories", len(changed))

    def _worker_main(self) -> None:
        lower_thread_priority()
        backoff = WORKER_BACKOFF_S[0]
        while not self._stopping.is_set():
            try:
                self._worker_step()
                backoff = WORKER_BACKOFF_S[0]
            except Exception:
                log.exception("background worker error; retrying in %.0fs", backoff)
                self._current_file = ""
                self._set_phase("idle")
                self._wait(backoff)
                backoff = min(backoff * 2, WORKER_BACKOFF_S[1])
        self._set_phase("idle")

    def _worker_step(self) -> None:
        with self._cv:
            new_cfg, self._pending_cfg = self._pending_cfg, None
            full, dirs = self._scan_full, self._scan_dirs
            self._scan_full, self._scan_dirs = False, set()
        if new_cfg is not None:
            self._apply_config(new_cfg)
        if not self._setup_done:
            if not self._logged_wait_setup:
                log.info("indexing waits until setup is finished")
                self._logged_wait_setup = True
            self._current_file = ""
            self._set_phase("idle")
            self._wait(5)
            return
        self._logged_wait_setup = False
        if full or dirs:
            self._scan(None if full else dirs)
            return
        now = time.monotonic()
        rescan_s = self.cfg.rescan_hours * 3600
        if rescan_s > 0 and now - self._last_full_scan >= rescan_s:
            log.info("daily rescan")
            self._scan(None)
            return
        sleep = min(3600.0, self._last_full_scan + rescan_s - now) if rescan_s > 0 else 3600.0
        if self._user_paused:
            self._set_phase("paused", "paused by request")
            self._wait(sleep)
            return
        if now < self._no_progress_until:
            self._set_phase("idle")
            self._wait(min(sleep, self._no_progress_until - now))
            return
        pending = self.indexer.count_pending()
        if pending and not self._model_files_ready():
            if not self._logged_wait_model:
                log.info("indexing waits until the model has been downloaded")
                self._logged_wait_model = True
            self._start_download()  # the files were there and are gone again: fetch them
            self._set_phase("idle")
            self._wait(min(sleep, 2.0))
            return
        self._logged_wait_model = False
        if pending == 0:
            if self._phase != "idle":
                self._went_idle()
            self._wait(sleep)
            return
        if self._gpu_is_busy():
            self._set_phase("paused", f"the GPU is in use by another application "
                                      f"(over {self.cfg.gpu_busy_threshold:g}% SM)")
            self._wait(self.gpu_busy_recheck_s)
            return
        self._index_slice(pending)

    def _went_idle(self) -> None:
        self._set_phase("idle")
        self._current_file = ""
        release = getattr(self.embedder, "release_cache", None)
        if release is not None and self._model_ready():
            try:
                release()  # cached video/batch activations go back to the driver
            except Exception:
                log.debug("release_cache failed", exc_info=True)

    def _index_slice(self, pending: int) -> None:
        self._slice_stop.clear()
        # Requests that set _slice_stop before the clear above are seen here.
        if self._stopping.is_set() or self._user_paused:
            return
        with self._cv:
            if self._scan_full or self._scan_dirs or self._pending_cfg is not None:
                return
        if self._phase != "indexing":
            log.info("indexing %d pending files", pending)
        self._run_progress = (0, pending, 0, 0.0)  # before the phase: Status sees both together
        self._set_phase("indexing")
        self._run_logged = 0
        try:
            st = self.indexer.run(stop=self._slice_stop, progress=self._on_progress)
        except ModelMismatch as exc:
            self._set_phase("paused", f"{exc}; run `semma index --rebuild`")
            self._wait(600)
            return
        finally:
            self._current_file = ""
            self._run_progress = (0, 0, 0, 0.0)
            self._last_work = time.monotonic()
        if st.done:
            log.info("indexing %s: %d indexed, %d failed, %d skipped, %d deferred, %d chunks in %.0fs",
                     "paused" if st.stopped else "run done", st.indexed, st.failed, st.skipped,
                     st.deferred, st.chunks, st.seconds)
        if not st.stopped and st.indexed + st.failed == 0:
            self._no_progress_until = time.monotonic() + NO_PROGRESS_RETRY_S

    def _on_progress(self, st: IndexStats) -> None:
        """Indexer callback, after every file (a file boundary)."""
        self._current_file = st.current
        self._last_work = time.monotonic()
        rate = st.done / st.seconds if st.seconds > 0 else 0.0
        self._run_progress = (st.done, st.total, st.failed, rate)
        if st.done - self._run_logged >= PROGRESS_LOG_EVERY:
            self._run_logged = st.done
            log.info("indexed %d/%d files (%d failed), %.1f files/s", st.indexed, st.total, st.failed, rate)
        with self._cv:
            scan_wanted = self._scan_full or bool(self._scan_dirs) or self._pending_cfg is not None
        if scan_wanted or self._user_paused or self._stopping.is_set():
            self._slice_stop.set()

    # -- GPU ------------------------------------------------------------------------------
    def _sample_gpu(self) -> bool:
        if not self.cfg.pause_when_gpu_busy:
            self._gpu_busy = False
            self._gpu_checked_at = float("-inf")
            return False
        busy = False
        mon = self.gpu_monitor
        if mon is not None:
            with self._gpu_sample_lock:
                try:
                    busy = bool(mon.busy(self.cfg.gpu_busy_threshold))
                except Exception:
                    log.warning("GPU monitor failed; assuming the GPU is free", exc_info=True)
        self._gpu_checked_at = time.monotonic()
        if busy != self._gpu_busy:
            log.info("GPU %s", "busy with another application" if busy else "free again")
        self._gpu_busy = busy
        return busy

    def _gpu_is_busy(self) -> bool:
        if self.gpu_monitor is None or not self.cfg.pause_when_gpu_busy:
            self._gpu_busy = False
            self._gpu_checked_at = float("-inf")
            return False
        if self._gpu_busy or time.monotonic() - self._gpu_checked_at >= self.cfg.gpu_check_seconds:
            return self._sample_gpu()
        return False

    def _gpu_main(self) -> None:
        """While a slice runs, sample every gpu_check_seconds; stop the slice when busy."""
        while not self._stopping.is_set():
            interval = max(0.01, self.cfg.gpu_check_seconds)  # a reload may change it
            with self._cv:
                self._cv.wait_for(
                    lambda: (self._phase == "indexing" and self.cfg.pause_when_gpu_busy)
                    or self._stopping.is_set(),
                    timeout=5,
                )
            if self._stopping.wait(interval):
                return
            if (
                self._phase != "indexing"
                or not self.cfg.pause_when_gpu_busy
                or time.monotonic() - self._gpu_checked_at < interval * 0.9
            ):
                continue
            if self._sample_gpu():
                self._slice_stop.set()

    # -- scanning -------------------------------------------------------------------------
    def _scan(self, dirs: set[str] | None) -> None:
        roots: list[Path] | None = None
        if dirs is not None:
            roots = partial_scan_roots(self.cfg, dirs)
            if roots is None:
                log.debug("changes need a full scan")
            elif not roots:
                return  # only pruned or vanished places changed
        full = roots is None
        self._set_phase("scanning")
        scanner = Scanner(self.cfg)
        self._scanner = scanner  # Status reads its stats until the scan is done
        try:
            self._scan_roots(scanner, roots, full)
        finally:
            self._scanner = None

    def _scan_roots(self, scanner: Scanner, roots: list[Path] | None, full: bool) -> None:
        gen = self.store.begin_scan()

        def files() -> Any:
            for f in scanner.scan(roots):
                if self._stopping.is_set() or self._pending_cfg is not None:
                    raise _ScanAborted
                yield f

        try:
            upserted = self.store.upsert_scanned(files(), gen)
        except _ScanAborted:
            # Files not seen keep their old generation: nothing lost.
            if self._stopping.is_set():
                log.info("scan aborted")
                return
            log.info("scan stopped: the settings changed; it starts again with the new ones")
            with self._cv:
                if full or roots is None:
                    self._scan_full = True
                else:
                    self._scan_dirs |= {str(r) for r in roots}
            return
        st = scanner.stats
        cfg = scanner.cfg
        removed = self.store.finish_scan(gen, st.roots, st.failed_dirs)
        if full:
            # finish_scan only cleans below the roots just walked: forget what the settings no
            # longer cover anywhere else (a removed root, mounts no longer included, a kind
            # turned off on an unplugged drive). Configured but unavailable roots are kept.
            removed += self.store.forget_unconfigured(configured_areas(cfg, st.roots), cfg.exclude_paths,
                                                      cfg.kinds, cfg.exclude_extensions, cfg.max_bytes)
        self._last_scan_at = time.time()
        if full:
            self._last_full_scan = time.monotonic()
        self._no_progress_until = 0.0
        self._update_watches(None if full else st.roots, getattr(st, "walked_dirs", None))
        log.log(logging.INFO if full else logging.DEBUG,
                "%s scan of %s in %.1fs: %d dirs, %d indexable files, %d removed, %d unreadable",
                "full" if full else "partial", ", ".join(map(str, st.roots)) if not full or len(st.roots) < 4
                else f"{len(st.roots)} roots", st.seconds, st.dirs, upserted, removed, st.errors)

    # -- settings reload ------------------------------------------------------------------
    def _load_config_file(self) -> Config:
        """Read the config file the daemon was started with again (raises ConfigError)."""
        path = self._start_cfg.source_path
        if path is None:
            raise ConfigError("the service was not started from a config file; restart it instead")
        return load_config(path, self._start_cfg.source_home, strict=True)

    async def reload_config(self) -> list[str]:
        """`Control.ReloadConfig`: read the config file again and apply what can be applied
        while running. Returns the config.toml names of settings that differ from those the
        service started with but only take effect after a restart ([]: everything applied).
        Raises ConfigError (and keeps the current settings) when the file cannot be read or
        is not valid."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._status_exec, self._reload_sync)

    def _reload_sync(self) -> list[str]:
        """Off the event loop. The new Config is handed to the worker, which swaps it in at a
        file boundary (`_apply_config`): nothing reads a half-updated Config."""
        try:
            new = self._config_loader()
        except ConfigError as exc:
            log.warning("settings not reloaded: %s", exc)
            raise
        start = self._start_cfg
        with self._reload_lock:
            base = self._accepted_cfg
            # The device is read when the model is loaded. Until then a new choice applies now,
            # so first-run setup does not restart the service (and does not interrupt the download).
            chosen_device = {attr: getattr(new, attr) for attr in ("device", "query_device")}
            restart = [name for attr, name in RESTART_FIELDS.items() if getattr(new, attr) != getattr(start, attr)]
            # A load that has started has read the device already.
            loading = ((self._load_task is not None and not self._load_task.done())
                       or bool(getattr(self.embedder, "loading", False)))
            models_loaded = self._index_model_loaded() or self._query_model_loaded() or loading
            hot: list[str] = []
            if not models_loaded:
                restart = [name for name in restart if name not in ("model.device", "model.query_device")]
                hot = [RESTART_FIELDS[attr] for attr, value in chosen_device.items()
                       if value != getattr(base, attr)]
            for attr in (*RESTART_FIELDS, *FIXED_FIELDS):
                setattr(new, attr, getattr(start, attr))  # keep running with what we started with
            if not models_loaded:
                for attr, value in chosen_device.items():
                    setattr(new, attr, value)
                if hot:
                    # `_start_cfg` may be the live Config. Copy before changing it so a device
                    # choice does not show up on the running settings before the worker swaps them.
                    if self._start_cfg is self.cfg or self._start_cfg is base:
                        self._start_cfg = copy.copy(self._start_cfg)
                    for attr, value in chosen_device.items():
                        setattr(self._start_cfg, attr, value)

            def diff(fields: dict[str, str]) -> list[str]:
                return [name for attr, name in fields.items() if getattr(new, attr) != getattr(base, attr)]

            rescan = diff(SCAN_FIELDS) + diff(PRIORITY_FIELDS)
            setup = diff(ONBOARDING_FIELDS)
            changed = rescan + diff(NEXT_FILE_FIELDS) + diff(LIVE_FIELDS) + setup + hot
            setup_done = onboarding_done(new, self.store) if setup else None
            if changed:
                self._accepted_cfg = new
                with self._cv:
                    self._pending_cfg = new
                    if rescan or setup_done:
                        self._scan_full = True
                    if setup_done is False:
                        self._scan_full = False
                        self._scan_dirs = set()
                    self._wakeup = True
                    self._cv.notify_all()
            if setup_done is False:
                self._slice_stop.set()
        if changed or restart:
            log.info("settings reloaded: %s%s%s", ", ".join(changed) or "nothing changed",
                     "; rescanning" if rescan or setup_done else "",
                     f"; need a restart: {', '.join(restart)}" if restart else "")
        else:
            log.info("settings reloaded: nothing changed")
        return restart

    def _apply_config(self, cfg: Config) -> None:
        """On the worker, between scans and slices: every holder of the Config gets the new one
        (the next scan builds its Scanner and Scorer from it)."""
        self.cfg = cfg
        self.indexer.cfg = cfg
        self._setup_done = onboarding_done(cfg, self.store)
        if self._setup_done:
            self._logged_wait_setup = False
        embedder = self.embedder
        if hasattr(embedder, "requested_device"):
            embedder.requested_device = cfg.device
        if hasattr(embedder, "requested_query_device"):
            embedder.requested_query_device = cfg.query_device
        self._no_progress_until = 0.0
        if not cfg.pause_when_gpu_busy:
            self._gpu_busy = False
        self._gpu_checked_at = float("-inf")
        with contextlib.suppress(RuntimeError):  # shutting down
            # On the search thread, between queries: a query sees one Config throughout.
            self._search_exec.submit(setattr, self.searcher, "cfg", cfg)
        watcher = self.watcher
        if watcher is not None and hasattr(watcher, "debounce"):
            watcher.debounce = float(cfg.watch_debounce_seconds)
        loop = self._loop
        if loop is not None:
            n = max(1, int(cfg.krunner_max_results))
            for svc in self._services:
                if hasattr(svc, "_max_results"):
                    with contextlib.suppress(RuntimeError):  # loop closed
                        loop.call_soon_threadsafe(setattr, svc, "_max_results", n)
        log.debug("new settings applied")

    def _update_watches(self, partial_roots: list[Path] | None, walked: list[str] | None) -> None:
        if walked is None:
            return  # scanner without walked_dirs: keep watching what we had
        db_dir = str(Path(self.cfg.db_path).parent)
        new = {d for d in walked if d != db_dir}  # our own writes must not trigger rescans
        if partial_roots is None:
            self._watched = new
        else:
            tops = {str(r) for r in partial_roots}
            prefixes = tuple(t.rstrip("/") + "/" for t in tops)
            self._watched = {d for d in self._watched if d not in tops and not d.startswith(prefixes)} | new
        if self.watcher is not None:
            try:
                self.watcher.set_dirs(set(self._watched))
            except Exception:
                log.exception("updating the watched directories failed")


def _default_gpu_monitor(cfg: Config, embedder: Any = None) -> GpuMonitorLike | None:
    try:
        from .gpumon import GpuMonitor
    except ImportError:
        log.warning("gpumon unavailable: indexing will not pause for other GPU users")
        return None

    def own_pids() -> set[int]:
        pids = {os.getpid()}
        if embedder is not None and hasattr(embedder, "worker_pids"):
            pids |= embedder.worker_pids()
        return pids

    return GpuMonitor(own_pids=own_pids, threshold=cfg.gpu_busy_threshold)


def _default_watcher_factory() -> WatcherFactory | None:
    try:
        from .watcher import DirWatcher
    except ImportError:
        log.warning("watcher unavailable: changes are only picked up by the daily rescan")
        return None
    return lambda on_change, debounce: DirWatcher(on_change, debounce=debounce)


def download_model_files(cfg: Config) -> None:
    """Download both GGUF files if they are missing. Does not load the model into memory."""
    from .embedder import ensure_model_file, resolve_model_dir

    model_dir = resolve_model_dir(cfg)
    ensure_model_file("text", model_dir)
    ensure_model_file("mmproj", model_dir)


def _default_runner_factory() -> RunnerFactory | None:
    try:
        from .krunner import KRunnerService
    except ImportError:
        log.exception("krunner module unavailable: not serving KRunner")
        return None
    return KRunnerService


def _default_gnome_factory() -> RunnerFactory | None:
    try:
        from .gnome_search import GnomeSearchService
    except ImportError:
        log.exception("gnome_search module unavailable: not serving GNOME Shell search")
        return None
    return GnomeSearchService


# -- D-Bus ----------------------------------------------------------------------------------

STATUS_SIGNATURES = {
    "state": "s", "model_loaded": "b", "query_model_loaded": "b", "index_model_loaded": "b",
    "paused": "b", "pause_reason": "s", "pending": "x", "indexed": "x", "failed": "x",
    "chunks": "x", "current_file": "s", "last_scan_at": "d", "uptime": "d", "watched_dirs": "x",
    # file counts per kind (every kind in filetypes.KINDS, 0 if none)
    "pending_by_kind": "a{sx}", "indexed_by_kind": "a{sx}", "failed_by_kind": "a{sx}",
    # the model download in progress ("" / 0 when none): the file, its bytes on disk and
    # expected, and the same over every model file
    "download_file": "s", "download_done": "x", "download_total": "x",
    "download_all_done": "x", "download_all_total": "x",
    # first-run setup is not finished: nothing is scanned or indexed and searches return nothing
    "setup_pending": "b",
    "download_error": "s", "model_files_ready": "b", "model_bytes_done": "x", "model_bytes_total": "x",
    # the running indexing slice (0 when not indexing): files done (indexed, failed, skipped or
    # deferred) of the files queued when it started, failed among them, files per second
    "run_done": "x", "run_total": "x", "run_failed": "x", "run_rate": "d",
    # the running scan ("" / 0 when not scanning): the directory being listed, directories
    # listed, regular files seen and indexable files among them
    "scan_dir": "s", "scan_dirs": "x", "scan_files": "x", "scan_matched": "x",
}


class Control(ServiceInterface):
    """`io.github.Semma.Control1`: status and indexing control for the CLI."""

    def __init__(self, daemon: Daemon) -> None:
        super().__init__(dbus_names.CONTROL_INTERFACE)
        self._daemon = daemon

    @dbus_method()
    async def Status(self) -> "a{sv}":  # noqa: F821, N802
        status = await self._daemon.status()
        return {k: Variant(STATUS_SIGNATURES[k], v) for k, v in status.items()}

    @dbus_method()
    def Pause(self):  # noqa: N802
        self._daemon.pause()

    @dbus_method()
    def Resume(self):  # noqa: N802
        self._daemon.resume()

    @dbus_method()
    def Rescan(self):  # noqa: N802
        self._daemon.rescan()

    @dbus_method()
    async def RetryFailed(self) -> "u":  # noqa: F821, N802
        return await self._daemon.retry_failed()

    @dbus_method()
    async def Failed(self, limit: "u") -> "a(ss)":  # noqa: F821, N802
        """(path, error) of failed files, the most important first; limit 0: all."""
        return [[path, error] for path, error in await self._daemon.failed(limit)]

    @dbus_method()
    async def ReloadConfig(self) -> "as":  # noqa: F821, N802
        """Read the config file again and apply it; returns the changed settings that need a
        restart. An unreadable or invalid file: CONFIG_ERROR, the settings stay as they were."""
        try:
            return await self._daemon.reload_config()
        except ConfigError as exc:
            raise DBusError(dbus_names.CONFIG_ERROR, str(exc)) from exc


async def serve(daemon: Daemon, *, bus_address: str | None = None, stop_event: asyncio.Event | None = None) -> int:
    """Export the interfaces, take the bus name and run until `stop_event` or the bus goes
    away. Returns the exit status (1 if the name is owned by another process)."""
    stop_event = stop_event or asyncio.Event()
    try:
        bus = await MessageBus(bus_address=bus_address).connect()
    except Exception as exc:
        log.error("cannot connect to the session bus: %s", exc)
        daemon.store.close()
        return 1
    runner = None
    if daemon._runner_factory is not None:
        try:
            runner = daemon._runner_factory(daemon.search, max_results=daemon.cfg.krunner_max_results,
                                            on_activity=daemon.record_activity)
            bus.export(dbus_names.RUNNER_PATH, runner)
        except Exception:
            log.exception("could not export the KRunner interface")
            runner = None
    gnome = None
    if daemon._gnome_factory is not None:
        try:
            gnome = daemon._gnome_factory(daemon.search, max_results=daemon.cfg.krunner_max_results,
                                          on_activity=daemon.record_activity)
            bus.export(dbus_names.GNOME_PATH, gnome)
        except Exception:
            log.exception("could not export the GNOME Shell search provider interface")
            gnome = None
    daemon._services = [svc for svc in (runner, gnome) if svc is not None]
    bus.export(dbus_names.CONTROL_PATH, Control(daemon))
    try:
        reply = await bus.request_name(daemon.bus_name, NameFlag.DO_NOT_QUEUE)
    except Exception as exc:
        reply = None
        log.error("requesting %s failed: %s", daemon.bus_name, exc)
    if reply not in (RequestNameReply.PRIMARY_OWNER, RequestNameReply.ALREADY_OWNER):
        if reply is not None:
            log.error("%s is already owned by another process; exiting", daemon.bus_name)
        bus.disconnect()
        daemon.store.close()
        return 1
    log.info("serving %s (KRunner %s, GNOME %s)", daemon.bus_name, "on" if runner else "off",
             "on" if gnome else "off")
    daemon.start()
    stop_wait = asyncio.ensure_future(stop_event.wait())
    disconnect = asyncio.ensure_future(bus.wait_for_disconnect())
    await asyncio.wait({stop_wait, disconnect}, return_when=asyncio.FIRST_COMPLETED)
    bus_lost = disconnect.done()
    if bus_lost:
        log.warning("lost the session bus connection")
    stop_wait.cancel()
    if runner is not None and not bus_lost:
        with contextlib.suppress(Exception):
            bus.unexport(dbus_names.RUNNER_PATH)
    if gnome is not None and not bus_lost:
        with contextlib.suppress(Exception):
            bus.unexport(dbus_names.GNOME_PATH)
    await daemon.stop()
    if not bus_lost:
        with contextlib.suppress(Exception):
            await bus.release_name(daemon.bus_name)
        bus.disconnect()
    disconnect.cancel()
    return 0 if not bus_lost else 1


def setup_logging(level: str | int | None = None) -> None:
    """stderr logging (journald adds timestamps under systemd). Leaves existing handlers alone."""
    root = logging.getLogger()
    if root.handlers:
        return
    level = level or os.environ.get("SEMMA_LOG_LEVEL", "INFO")
    fmt = "%(levelname)s %(name)s: %(message)s"
    if "JOURNAL_STREAM" not in os.environ:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # one INFO line per Hub request otherwise
    # The unit sets HF_HUB_OFFLINE=1, so the Hub's "offline mode, using cached commit" warning is
    # expected on every model load (and huggingface_hub prints it twice via its own handler).
    # huggingface_hub resets its logger level when imported, so set its own verbosity variable.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def run_daemon(cfg: Config, *, bus_address: str | None = None, bus_name: str = dbus_names.BUS_NAME) -> int:
    """Entry point of `semma daemon`: runs until SIGTERM/SIGINT, returns the exit status."""
    setup_logging()

    async def main() -> int:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        try:
            daemon = Daemon(cfg, bus_name=bus_name)
        except Exception:
            log.exception("cannot start")
            return 1
        return await serve(daemon, bus_address=bus_address, stop_event=stop)

    return asyncio.run(main())


# -- client side (CLI) ----------------------------------------------------------------------

class DaemonNotRunning(RuntimeError):
    pass


_NOT_RUNNING_ERRORS = {"org.freedesktop.DBus.Error.ServiceUnknown", "org.freedesktop.DBus.Error.NameHasNoOwner"}


def _unwrap(value: Any) -> Any:
    if isinstance(value, Variant):
        return _unwrap(value.value)
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


async def control_call(method: str, *args: Any, signature: str = "", bus_address: str | None = None,
                       bus_name: str = dbus_names.BUS_NAME, timeout: float = 10.0,
                       auto_start: bool = False) -> Any:
    """Call a Control method of a running daemon and return its (unwrapped) result.

    By default the call does not D-Bus-activate the daemon. Raises DaemonNotRunning if nobody
    owns the name (or there is no session bus), DBusError for other errors.
    """
    try:
        bus = await MessageBus(bus_address=bus_address).connect()
    except Exception as exc:
        raise DaemonNotRunning(f"no session bus: {exc}") from exc
    try:
        msg = Message(destination=bus_name, path=dbus_names.CONTROL_PATH,
                      interface=dbus_names.CONTROL_INTERFACE, member=method, signature=signature,
                      body=list(args), flags=MessageFlag.NONE if auto_start else MessageFlag.NO_AUTOSTART)
        reply = await asyncio.wait_for(bus.call(msg), timeout)
        assert reply is not None
        if reply.message_type == MessageType.ERROR:
            text = reply.body[0] if reply.body else ""
            if reply.error_name in _NOT_RUNNING_ERRORS:
                raise DaemonNotRunning(text)
            raise DBusError(reply.error_name or "org.freedesktop.DBus.Error.Failed", text)
        body = _unwrap(reply.body)
        if not body:
            return None
        return body[0] if len(body) == 1 else tuple(body)
    finally:
        bus.disconnect()


def daemon_call(method: str, *args: Any, **kwargs: Any) -> Any:
    """Synchronous `control_call` (for the CLI; not from inside a running event loop)."""
    return asyncio.run(control_call(method, *args, **kwargs))


def daemon_status(*, bus_address: str | None = None, bus_name: str = dbus_names.BUS_NAME,
                  timeout: float = 10.0) -> dict[str, Any] | None:
    """The daemon's Status() as a plain dict, or None if it is not running."""
    try:
        return daemon_call("Status", bus_address=bus_address, bus_name=bus_name, timeout=timeout)
    except (DaemonNotRunning, TimeoutError):
        return None


def daemon_reload_config(*, bus_address: str | None = None, bus_name: str = dbus_names.BUS_NAME,
                         timeout: float = 10.0) -> list[str] | None:
    """Make the daemon read its config file again (ReloadConfig): the changed settings that need
    a restart ([] when all were applied), or None if it is not running. An invalid config file
    raises DBusError (type dbus_names.CONFIG_ERROR, the reason as its text)."""
    try:
        names = daemon_call("ReloadConfig", bus_address=bus_address, bus_name=bus_name, timeout=timeout)
    except (DaemonNotRunning, TimeoutError):
        return None
    return [str(n) for n in names or []]


def daemon_failed(limit: int = 0, *, bus_address: str | None = None, bus_name: str = dbus_names.BUS_NAME,
                  timeout: float = 10.0) -> list[tuple[str, str]] | None:
    """The daemon's Failed(limit) as (path, error) pairs, or None if it is not running."""
    try:
        rows = daemon_call("Failed", limit, signature="u", bus_address=bus_address, bus_name=bus_name,
                           timeout=timeout)
    except (DaemonNotRunning, TimeoutError):
        return None
    return [(path, error) for path, error in rows or []]
