"""One window per session: the first `semma gui` owns a small D-Bus name, a second one asks it
to come forward and exits. No Qt here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

from dbus_fast import Message, MessageType, NameFlag, RequestNameReply
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

WINDOW_BUS_NAME = "io.github.Semma.Window"
WINDOW_PATH = "/io/github/Semma/Window"
WINDOW_INTERFACE = "io.github.Semma.Window1"

log = logging.getLogger(__name__)


class _Window(ServiceInterface):
    def __init__(self, on_activate: Callable[[str], None]) -> None:
        super().__init__(WINDOW_INTERFACE)
        self._on_activate = on_activate

    @method()
    def Activate(self, token: "s"):  # noqa: F821, N802
        """Bring the window forward; `token` is the caller's XDG activation token (Wayland) or ""."""
        self._on_activate(token)


async def claim(on_activate: Callable[[str], None], *, bus_address: str | None = None,
                bus_name: str = WINDOW_BUS_NAME, timeout: float = 3.0) -> MessageBus | None | bool:
    """Take `bus_name` and serve Activate (called on the bus's thread) while the returned bus
    stays connected. If another window owns the name, ask it to come forward and return False.
    None: no session bus (the window still opens, just without this)."""
    try:
        bus = await asyncio.wait_for(MessageBus(bus_address=bus_address).connect(), timeout)
    except Exception as exc:
        log.debug("no session bus for single-instance: %s", exc)
        return None
    try:
        bus.export(WINDOW_PATH, _Window(on_activate))
        reply = await asyncio.wait_for(bus.request_name(bus_name, NameFlag.DO_NOT_QUEUE), timeout)
        if reply in (RequestNameReply.PRIMARY_OWNER, RequestNameReply.ALREADY_OWNER):
            return bus
        token = os.environ.get("XDG_ACTIVATION_TOKEN", "")
        msg = Message(destination=bus_name, path=WINDOW_PATH, interface=WINDOW_INTERFACE, member="Activate",
                      signature="s", body=[token])
        answer = await asyncio.wait_for(bus.call(msg), timeout)
        if answer is None or answer.message_type == MessageType.ERROR:
            log.warning("the open Semma window did not answer; opening another one")
            bus.disconnect()
            return None
        bus.disconnect()
        return False
    except Exception as exc:
        log.warning("single-instance check failed: %s", exc)
        bus.disconnect()
        return None
