"""D-Bus bridge forwarding org.gnome.Shell.SearchProvider2 to the real session bus.

Runs on the nested private session bus (default DBUS_SESSION_BUS_ADDRESS), owns
``io.github.Semma``, exports ``org.gnome.Shell.SearchProvider2`` at
``/io/github/Semma/SearchProvider``, and forwards all 5 SearchProvider2 calls
to the upstream session bus daemon.

Usage:
    SEMMA_UPSTREAM_BUS="$DBUS_SESSION_BUS_ADDRESS" \\
        dbus-run-session -- uv run --project . python scripts/gnome_bridge.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Any

from dbus_fast import Message, MessageType, RequestNameReply, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method

from semma.dbus_names import BUS_NAME, GNOME_INTERFACE, GNOME_PATH

log = logging.getLogger(__name__)

UPSTREAM_ENV_VAR = "SEMMA_UPSTREAM_BUS"
DEFAULT_TIMEOUT_SECONDS = 10.0


def _ensure_metas_variants(metas: Any) -> list[dict[str, Variant]]:
    """Ensure GetResultMetas payload is ``list[dict[str, Variant]]``."""
    if not isinstance(metas, list):
        return []
    out: list[dict[str, Variant]] = []
    for item in metas:
        if not isinstance(item, dict):
            continue
        out.append({str(k): v for k, v in item.items() if isinstance(v, Variant)})
    return out


class SearchProviderBridge(ServiceInterface):
    """Forwards ``org.gnome.Shell.SearchProvider2`` calls to an upstream bus."""

    def __init__(
        self,
        upstream: MessageBus,
        *,
        upstream_address: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(GNOME_INTERFACE)
        self.upstream = upstream
        self._upstream_address = upstream_address
        self._timeout = timeout
        self._reconnect_lock = asyncio.Lock()

    async def _get_upstream(self) -> MessageBus:
        if self.upstream.connected:
            return self.upstream
        async with self._reconnect_lock:
            if not self.upstream.connected:
                log.info("Reconnecting to upstream D-Bus at %s", self._upstream_address)
                self.upstream = await MessageBus(bus_address=self._upstream_address).connect()
            return self.upstream

    async def _forward(
        self,
        member: str,
        signature: str,
        body: list[Any],
        default: Any,
    ) -> Any:
        try:
            bus = await self._get_upstream()
            msg = Message(
                destination=BUS_NAME,
                path=GNOME_PATH,
                interface=GNOME_INTERFACE,
                member=member,
                signature=signature,
                body=body,
            )
            reply = await asyncio.wait_for(bus.call(msg), timeout=self._timeout)
        except TimeoutError:
            log.warning("Upstream %s timed out after %.1fs", member, self._timeout)
            return default
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Upstream %s call failed", member)
            return default

        if reply.message_type != MessageType.METHOD_RETURN:
            log.warning(
                "Upstream %s returned error %s: %s",
                member,
                reply.error_name,
                reply.body,
            )
            return default

        if default is None:
            return None
        return reply.body[0] if reply.body else default

    @method()
    async def GetInitialResultSet(self, terms: "as") -> "as":  # noqa: F821, N802, UP037
        res = await self._forward("GetInitialResultSet", "as", [list(terms)], [])
        return list(res) if isinstance(res, list) else []

    @method()
    async def GetSubsearchResultSet(  # noqa: N802
        self,
        previous_results: "as",  # noqa: F821, UP037
        terms: "as",  # noqa: F821, UP037
    ) -> "as":  # noqa: F821, UP037
        res = await self._forward(
            "GetSubsearchResultSet",
            "asas",
            [list(previous_results), list(terms)],
            [],
        )
        return list(res) if isinstance(res, list) else []

    @method()
    async def GetResultMetas(self, identifiers: "as") -> "aa{sv}":  # noqa: F821, N802, UP037
        res = await self._forward("GetResultMetas", "as", [list(identifiers)], [])
        return _ensure_metas_variants(res)

    @method()
    async def ActivateResult(  # noqa: N802
        self,
        identifier: "s",  # noqa: F821, UP037
        terms: "as",  # noqa: F821, UP037
        timestamp: "u",  # noqa: F821, UP037
    ):
        await self._forward(
            "ActivateResult",
            "sasu",
            [identifier, list(terms), int(timestamp)],
            None,
        )

    @method()
    async def LaunchSearch(  # noqa: N802
        self,
        terms: "as",  # noqa: F821, UP037
        timestamp: "u",  # noqa: F821, UP037
    ):
        await self._forward(
            "LaunchSearch",
            "asu",
            [list(terms), int(timestamp)],
            None,
        )


def _safe_disconnect(bus: MessageBus | None) -> None:
    if bus is None:
        return
    try:
        bus.disconnect()
    except Exception:
        pass


async def _run_bridge(upstream_address: str, timeout: float) -> int:
    upstream: MessageBus | None = None
    local: MessageBus | None = None
    bridge: SearchProviderBridge | None = None
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(sig_name: str) -> None:
        log.info("Received %s, shutting down", sig_name)
        stop_event.set()

    installed_signals: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
            installed_signals.append(sig)
        except NotImplementedError:
            pass

    try:
        upstream = await MessageBus(bus_address=upstream_address).connect()
        local = await MessageBus().connect()

        bridge = SearchProviderBridge(
            upstream,
            upstream_address=upstream_address,
            timeout=timeout,
        )
        local.export(GNOME_PATH, bridge)

        reply = await local.request_name(BUS_NAME)
        if reply != RequestNameReply.PRIMARY_OWNER:
            log.error("Failed to own %s on local session bus: %s", BUS_NAME, reply)
            return 1

        log.info("Bridging %s %s (%s) -> %s", BUS_NAME, GNOME_PATH, GNOME_INTERFACE, upstream_address)

        stop_task = asyncio.create_task(stop_event.wait(), name="bridge-stop")
        disconnect_task = asyncio.create_task(local.wait_for_disconnect(), name="local-disconnect")
        done, pending = await asyncio.wait(
            {stop_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.exception() if not task.cancelled() else None
        return 0
    finally:
        for sig in installed_signals:
            loop.remove_signal_handler(sig)
        _safe_disconnect(local)
        _safe_disconnect(bridge.upstream if bridge is not None else upstream)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bridge org.gnome.Shell.SearchProvider2 from a nested session bus to the host session bus.",
    )
    p.add_argument(
        "--upstream-address",
        default=os.environ.get(UPSTREAM_ENV_VAR),
        help=f"Upstream D-Bus session address (default: ${UPSTREAM_ENV_VAR})",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-call timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.upstream_address:
        print(
            f"Missing upstream D-Bus address. Pass --upstream-address or set {UPSTREAM_ENV_VAR}.",
            file=sys.stderr,
        )
        return 1
    if args.timeout <= 0:
        print("--timeout must be positive.", file=sys.stderr)
        return 1
    try:
        return asyncio.run(_run_bridge(args.upstream_address, args.timeout))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
