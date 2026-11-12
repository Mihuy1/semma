"""Non-recursive inotify directory watcher with debouncing and max-delay cap."""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable

from inotify_simple import Event, INotify, flags

log = logging.getLogger(__name__)

WATCH_MASK: int = (
    flags.CREATE
    | flags.DELETE
    | flags.CLOSE_WRITE
    | flags.MOVED_FROM
    | flags.MOVED_TO
    | flags.DELETE_SELF
    | flags.MOVE_SELF
    | flags.ATTRIB
)

TEMP_SUFFIXES: tuple[str, ...] = ("~", ".part", ".crdownload", ".tmp", ".swp")

_POLL_INTERVAL_MS = 50


def is_temp_name(name: str) -> bool:
    """Return True if *name* looks like a temporary or editor swap file."""
    if not name:
        return False
    if name.startswith("."):
        return True
    low = name.lower()
    return low.endswith(TEMP_SUFFIXES)


class DirWatcher:
    """Watch a set of directories via inotify and batch change notifications."""

    def __init__(
        self,
        on_change: Callable[[set[str]], None],
        debounce: float = 5.0,
        max_delay: float = 30.0,
    ) -> None:
        self.on_change = on_change
        self.debounce = float(debounce)
        self.max_delay = float(max_delay)

        self._lock = threading.RLock()
        self._inotify: INotify | None = INotify()
        self._path_to_wd: dict[str, int] = {}
        self._wd_to_path: dict[int, str] = {}
        self._limit_warned = False

        self._pending: set[str] = set()
        self._first_event_at: float | None = None
        self._last_event_at: float | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def watched_dirs(self) -> set[str]:
        with self._lock:
            return set(self._path_to_wd)

    def set_dirs(self, dirs: Iterable[str | os.PathLike[str]]) -> None:
        """Update the set of watched directories incrementally (thread-safe)."""
        target = {os.fspath(d) for d in dirs}
        with self._lock:
            if self._inotify is None:
                return
            to_remove = set(self._path_to_wd) - target
            to_add = [d for d in target if d not in self._path_to_wd]

            for path in to_remove:
                wd = self._path_to_wd.pop(path, None)
                if wd is not None:
                    self._wd_to_path.pop(wd, None)
                    try:
                        self._inotify.rm_watch(wd)
                    except OSError:
                        pass

            for path in to_add:
                try:
                    wd = self._inotify.add_watch(path, WATCH_MASK | flags.ONLYDIR)
                except OSError as exc:
                    if exc.errno == errno.ENOSPC:
                        if not self._limit_warned:
                            self._limit_warned = True
                            log.warning(
                                "Kernel inotify watch limit reached (ENOSPC); "
                                "stopping addition of further directory watches."
                            )
                        break
                    if exc.errno in (errno.ENOENT, errno.EACCES, errno.EPERM, errno.ENOTDIR):
                        continue
                    log.debug("Skipping inotify watch for %s: %s", path, exc)
                    continue
                self._path_to_wd[path] = wd
                self._wd_to_path[wd] = path

    def start(self) -> None:
        """Start the background daemon thread reading inotify events."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._inotify is None:
                self._inotify = INotify()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="gemmasearch-watcher",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close the inotify file descriptor."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            self._thread = None
            if self._inotify is not None:
                try:
                    self._inotify.close()
                except OSError:
                    pass
                self._inotify = None
            self._path_to_wd.clear()
            self._wd_to_path.clear()
            self._pending.clear()
            self._first_event_at = None
            self._last_event_at = None

    def _invoke_on_change(self, changed: set[str]) -> None:
        try:
            self.on_change(changed)
        except Exception:
            log.exception("Unhandled exception in DirWatcher on_change callback")

    def _handle_event(self, event: Event, now: float | None = None) -> None:
        """Process a single inotify event."""
        mask = event.mask
        if mask & flags.Q_OVERFLOW:
            with self._lock:
                self._pending.clear()
                self._first_event_at = None
                self._last_event_at = None
            self._invoke_on_change({"*"})
            return

        if mask & flags.IGNORED:
            with self._lock:
                path = self._wd_to_path.pop(event.wd, None)
                if path is not None and self._path_to_wd.get(path) == event.wd:
                    self._path_to_wd.pop(path, None)
            return

        is_dir = bool(mask & flags.ISDIR)
        if not is_dir and is_temp_name(event.name):
            return

        with self._lock:
            dirpath = self._wd_to_path.get(event.wd)
            if dirpath is None:
                return
            ts = time.monotonic() if now is None else now
            self._pending.add(dirpath)
            if is_dir and (mask & (flags.CREATE | flags.MOVED_TO)) and event.name:
                self._pending.add(os.path.join(dirpath, event.name))
            if self._first_event_at is None:
                self._first_event_at = ts
            self._last_event_at = ts

    def _handle_events(self, events: Iterable[Event], now: float | None = None) -> None:
        for ev in events:
            self._handle_event(ev, now=now)

    def _next_wait_ms(self, now: float) -> int:
        with self._lock:
            if not self._pending or self._first_event_at is None or self._last_event_at is None:
                return _POLL_INTERVAL_MS
            due = min(
                self._last_event_at + self.debounce,
                self._first_event_at + self.max_delay,
            )
            if now >= due:
                return 0
            rem_ms = max(1, int((due - now) * 1000.0 + 0.999))
            return min(_POLL_INTERVAL_MS, rem_ms)

    def _pop_due_changes(self, now: float | None = None) -> set[str] | None:
        with self._lock:
            if not self._pending or self._first_event_at is None or self._last_event_at is None:
                return None
            ts = time.monotonic() if now is None else now
            due = min(
                self._last_event_at + self.debounce,
                self._first_event_at + self.max_delay,
            )
            if ts < due:
                return None
            changed = set(self._pending)
            self._pending.clear()
            self._first_event_at = None
            self._last_event_at = None
            return changed

    def _run(self) -> None:
        while not self._stop_event.is_set():
            inotify = self._inotify
            if inotify is None:
                break
            timeout_ms = self._next_wait_ms(time.monotonic())
            try:
                events = inotify.read(timeout=timeout_ms)
            except (OSError, ValueError):
                if self._stop_event.is_set():
                    break
                log.exception("Error reading inotify events")
                break

            if self._stop_event.is_set():
                break

            if events:
                now = time.monotonic()
                for ev in events:
                    self._handle_event(ev, now=now)

            changed = self._pop_due_changes()
            if changed:
                self._invoke_on_change(changed)
