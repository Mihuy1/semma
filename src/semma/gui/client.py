"""The window's connection to the daemon. No Qt here.

Every call runs on one worker thread with its own asyncio loop (dbus-fast is asyncio-based)
and returns a `concurrent.futures.Future` at once, so the Qt event loop never waits for D-Bus:
a call can take up to its 10 s timeout. `gui.qtasync.when_done` delivers the result back on
the Qt thread. Each call opens its own bus connection through `daemon.control_call`, the
helper the CLI uses; that costs a few milliseconds, which is nothing at one poll a second.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any

from .. import dbus_names
from ..daemon import DaemonNotRunning, control_call

# Error names of a call to a method, interface or object the daemon does not have (an older
# daemon without `ReloadConfig`).
UNKNOWN_METHOD_ERRORS = frozenset({
    "org.freedesktop.DBus.Error.UnknownMethod",
    "org.freedesktop.DBus.Error.UnknownInterface",
    "org.freedesktop.DBus.Error.UnknownObject",
})


class NotSupported(Exception):
    """The running daemon is an older build without this method."""

    def __init__(self, method: str) -> None:
        super().__init__(f"The running service is an older version that has no {method}; "
                         f"restart it to use this.")
        self.method = method


async def _known(coro: Coroutine[Any, Any, Any], method: str) -> Any:
    from dbus_fast.errors import DBusError

    try:
        return await coro
    except DBusError as exc:
        if exc.type in UNKNOWN_METHOD_ERRORS:
            raise NotSupported(method) from exc
        raise


class DaemonClient:
    """Calls the daemon's Control interface from a background thread.

    `status()`, `state()` and `failed()` resolve to None when the daemon is not running; the
    other calls raise `DaemonNotRunning` through the future. A method an older daemon lacks
    raises `NotSupported` (except `state()`, which falls back to `Status`). Other errors
    (DBusError, TimeoutError) are raised through the future too, so the window can show them."""

    def __init__(self, *, bus_address: str | None = None, bus_name: str = dbus_names.BUS_NAME,
                 timeout: float = 10.0) -> None:
        self.bus_address = bus_address
        self.bus_name = bus_name
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="semma-gui-dbus", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _call(self, method: str, *args: Any, signature: str = "") -> Any:
        return await control_call(method, *args, signature=signature, bus_address=self.bus_address,
                                  bus_name=self.bus_name, timeout=self.timeout)

    async def _or_none(self, method: str, *args: Any, signature: str = "") -> Any:
        try:
            return await self._call(method, *args, signature=signature)
        except DaemonNotRunning:
            return None

    # -- the calls ------------------------------------------------------------------------
    def status(self) -> Future[dict[str, Any] | None]:
        return self.submit(self._or_none("Status"))

    def state(self) -> Future[dict[str, Any] | None]:
        """Cheap `State()` for frequent polling; falls back to `Status()` on older daemons."""
        async def run() -> dict[str, Any] | None:
            try:
                return await _known(self._or_none("State"), "State")
            except NotSupported:
                return await self._or_none("Status")
        return self.submit(run())

    def failed(self, limit: int = 0) -> Future[list[tuple[str, str]] | None]:
        async def run() -> list[tuple[str, str]] | None:
            rows = await _known(self._or_none("Failed", limit, signature="u"), "Failed")
            return None if rows is None else [(str(p), str(e)) for p, e in rows]
        return self.submit(run())

    def call(self, method: str) -> Future[Any]:
        """A Control method without arguments: Pause, Resume, Rescan, RetryFailed."""
        return self.submit(self._call(method))

    def reload_config(self) -> Future[list[str] | None]:
        """`ReloadConfig() -> as`: the changed settings that need a service restart ([] when all
        were applied). None if this daemon has no ReloadConfig (an older build)."""
        async def run() -> list[str] | None:
            try:
                names = await _known(self._call("ReloadConfig"), "ReloadConfig")
            except NotSupported:
                return None
            return [str(n) for n in names or []]
        return self.submit(run())

    def systemctl(self, action: str) -> Future[tuple[int, str]]:
        """`systemctl --user <action> semma.service`: (exit status, what it printed)."""
        async def run() -> tuple[int, str]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "systemctl", "--user", action, dbus_names.SYSTEMD_UNIT,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            except OSError as exc:
                return 127, str(exc)
            out, _ = await proc.communicate()
            return proc.returncode or 0, out.decode(errors="replace").strip()
        return self.submit(run())


def describe_error(exc: BaseException) -> str:
    """A one-line message for the window about a failed daemon call."""
    from dbus_fast.errors import DBusError

    if isinstance(exc, DaemonNotRunning):
        return "The Semma service is not running."
    if isinstance(exc, TimeoutError):
        return "The Semma service did not answer in time."
    if isinstance(exc, DBusError):
        return exc.text or str(exc.type)
    return str(exc) or type(exc).__name__
