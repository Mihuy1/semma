"""A stand-in for gui.client.DaemonClient: answers at once from canned data, records calls."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from semma.daemon import DaemonNotRunning


def _done(value: Any = None, exc: BaseException | None = None) -> Future[Any]:
    f: Future[Any] = Future()
    if exc is not None:
        f.set_exception(exc)
    else:
        f.set_result(value)
    return f


class FakeClient:
    def __init__(self, status: dict[str, Any] | None = None) -> None:
        self.running = status is not None
        self.status_value = status
        self.failed_rows: list[tuple[str, str]] = []
        self.reload_result: list[str] | None = []  # None: an older daemon without ReloadConfig
        self.systemctl_result: tuple[int, str] = (0, "")
        self.status_error: BaseException | None = None
        self.calls: list[str] = []

    def status(self) -> Future[Any]:
        self.calls.append("Status")
        if self.status_error is not None:
            return _done(exc=self.status_error)
        return _done(self.status_value if self.running else None)

    def failed(self, limit: int = 0) -> Future[Any]:
        self.calls.append("Failed")
        return _done(list(self.failed_rows) if self.running else None)

    def call(self, method: str) -> Future[Any]:
        self.calls.append(method)
        if not self.running:
            return _done(exc=DaemonNotRunning("not running"))
        return _done(len(self.failed_rows) if method == "RetryFailed" else None)

    def reload_config(self) -> Future[Any]:
        self.calls.append("ReloadConfig")
        if not self.running:
            return _done(exc=DaemonNotRunning("not running"))
        return _done(self.reload_result)

    def systemctl(self, action: str) -> Future[Any]:
        self.calls.append(f"systemctl {action}")
        return _done(self.systemctl_result)

    def close(self) -> None:
        pass


def status(**overrides: Any) -> dict[str, Any]:
    """A complete Status() dict of an idle daemon, with `overrides` applied."""
    kinds = ("text", "pdf", "office", "image", "audio", "video")
    st: dict[str, Any] = {
        "state": "idle", "model_loaded": False, "query_model_loaded": False, "index_model_loaded": False,
        "paused": False, "pause_reason": "", "pending": 0, "indexed": 0, "failed": 0, "chunks": 0,
        "current_file": "", "last_scan_at": 0.0, "uptime": 60.0, "watched_dirs": 0,
        "pending_by_kind": dict.fromkeys(kinds, 0), "indexed_by_kind": dict.fromkeys(kinds, 0),
        "failed_by_kind": dict.fromkeys(kinds, 0),
        "download_file": "", "download_done": 0, "download_total": 0, "download_all_done": 0,
        "download_all_total": 0, "run_done": 0, "run_total": 0, "run_failed": 0, "run_rate": 0.0,
        "scan_dir": "", "scan_dirs": 0, "scan_files": 0, "scan_matched": 0,
    }
    st.update(overrides)
    return st
