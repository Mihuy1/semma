"""semmad: the long-running service behind KRunner.

Runs as a systemd user service (Type=dbus) that owns `dbus_names.BUS_NAME` on the session bus,
answers KRunner queries (`krunner.KRunnerService` at RUNNER_PATH) and GNOME Shell overview
searches (`gnome_search.GnomeSearchService` at GNOME_PATH) and keeps the index current in the
background. `Control` at CONTROL_PATH reports status and takes pause/resume/rescan.

Threads and what runs where:

  event loop (main thread)   D-Bus, query debounce/stale-cancellation, idle-unload timer.
                             Never touches SQLite or the model directly.
  semma-search (1)     every query: embed_query + Searcher.search_vector on this one
                             thread (its read-only connection and the in-memory vector cache);
                             also Status counts and RetryFailed.
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
from .config import Config
from .indexer import Indexer, IndexStats, ModelMismatch
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
SHUTDOWN_JOIN_S = 60.0
MAX_PARTIAL_SCAN_DIRS = 64  # more changed directories than this: one full scan instead

_DEFAULT: Any = object()  # "construct the real collaborator"; None means "none"


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
                 bus_name: str = dbus_names.BUS_NAME) -> None:
        self.cfg = cfg
        self.bus_name = bus_name
        self.store = store if store is not None else Store(
            cfg.db_path, vector_cache=True, cache_max_chunks=cfg.vector_cache_max_chunks)
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

        # Worker state, guarded by _cv (simple reads of single attributes are fine without it).
        self._cv = threading.Condition()
        self._wakeup = False
        self._stopping = threading.Event()
        self._slice_stop = threading.Event()
        self._user_paused = False
        self._scan_full = False
        self._scan_dirs: set[str] = set()
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
        self._run_stats: IndexStats | None = None
        self._run_logged = 0
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
        self._started = time.monotonic()
        self._running = False
        self._closed = False

    # -- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        """Start the background threads and timers (call from the running event loop).
        The first thing the worker does is a full scan."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._started = time.monotonic()
        if self._watcher_factory is not None:
            try:
                self.watcher = self._watcher_factory(self._on_watch_change, self.cfg.watch_debounce_seconds)
                self.watcher.start()
            except Exception:
                log.exception("file watcher unavailable; relying on the daily rescan")
                self.watcher = None
        with self._cv:
            self._scan_full = True
        self._worker = threading.Thread(target=self._worker_main, name="semma-worker", daemon=True)
        self._worker.start()
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
        for thread in (self._worker, self._gpu_thread):
            if thread is not None:
                await asyncio.to_thread(thread.join, SHUTDOWN_JOIN_S)
                if thread.is_alive():
                    log.warning("%s did not stop within %.0fs", thread.name, SHUTDOWN_JOIN_S)
                    clean = False
        await asyncio.to_thread(self._search_exec.shutdown, True, cancel_futures=True)
        await asyncio.to_thread(self._model_exec.shutdown, True, cancel_futures=True)
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
        q_idle = self._idle_seconds()
        i_idle = self._index_unload_seconds()
        candidates = [x for x in (q_idle, i_idle) if x > 0]
        if not candidates:
            return  # 0: never unload
        interval = min(60.0, max(0.02, min(candidates) / 4))
        while True:
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

    def state(self) -> str:
        if self._phase == "scanning":
            return "scanning"
        if (self._load_task is not None and not self._load_task.done()) or getattr(self.embedder, "loading", False):
            return "loading"
        return self._phase

    async def status(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        counts, chunks = await loop.run_in_executor(self._search_exec, self._counts_sync)
        q_loaded = self._query_model_loaded()
        i_loaded = self._index_model_loaded()
        return {
            "state": self.state(),
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
        }

    def _counts_sync(self) -> tuple[dict[str, int], int]:
        db = self.store.reader()
        counts = {r[0]: r[1] for r in db.execute("SELECT status, count(*) FROM files GROUP BY status")}
        chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return counts, chunks

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
            full, dirs = self._scan_full, self._scan_dirs
            self._scan_full, self._scan_dirs = False, set()
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
            if self._scan_full or self._scan_dirs:
                return
        if self._phase != "indexing":
            log.info("indexing %d pending files", pending)
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
        if st.done - self._run_logged >= PROGRESS_LOG_EVERY:
            self._run_logged = st.done
            rate = st.done / st.seconds if st.seconds > 0 else 0.0
            log.info("indexed %d/%d files (%d failed), %.1f files/s", st.indexed, st.total, st.failed, rate)
        with self._cv:
            scan_wanted = self._scan_full or bool(self._scan_dirs)
        if scan_wanted or self._user_paused or self._stopping.is_set():
            self._slice_stop.set()

    # -- GPU ------------------------------------------------------------------------------
    def _sample_gpu(self) -> bool:
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
        if self.gpu_monitor is None:
            return False
        if self._gpu_busy or time.monotonic() - self._gpu_checked_at >= self.cfg.gpu_check_seconds:
            return self._sample_gpu()
        return False

    def _gpu_main(self) -> None:
        """While a slice runs, sample every gpu_check_seconds; stop the slice when busy."""
        interval = max(0.01, self.cfg.gpu_check_seconds)
        while not self._stopping.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._phase == "indexing" or self._stopping.is_set(), timeout=5)
            if self._stopping.wait(interval):
                return
            if self._phase != "indexing" or time.monotonic() - self._gpu_checked_at < interval * 0.9:
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
        gen = self.store.begin_scan()

        def files() -> Any:
            for f in scanner.scan(roots):
                if self._stopping.is_set():
                    raise _ScanAborted
                yield f

        try:
            upserted = self.store.upsert_scanned(files(), gen)
        except _ScanAborted:
            log.info("scan aborted")  # files not seen keep their old generation: nothing lost
            return
        st = scanner.stats
        removed = self.store.finish_scan(gen, st.roots, st.failed_dirs)
        self._last_scan_at = time.time()
        if full:
            self._last_full_scan = time.monotonic()
        self._no_progress_until = 0.0
        self._update_watches(None if full else st.roots, getattr(st, "walked_dirs", None))
        log.log(logging.INFO if full else logging.DEBUG,
                "%s scan of %s in %.1fs: %d dirs, %d indexable files, %d removed, %d unreadable",
                "full" if full else "partial", ", ".join(map(str, st.roots)) if not full or len(st.roots) < 4
                else f"{len(st.roots)} roots", st.seconds, st.dirs, upserted, removed, st.errors)

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
