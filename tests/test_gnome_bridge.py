from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Any

from dbus_fast import DBusError, Message, MessageType, RequestNameReply, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, method
import pytest

from gemmasearch.dbus_names import BUS_NAME, GNOME_INTERFACE, GNOME_PATH

BRIDGE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gnome_bridge.py"


def _load_gnome_bridge():
    spec = importlib.util.spec_from_file_location("gnome_bridge", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gnome_bridge = _load_gnome_bridge()


def _start_private_dbus() -> tuple[str, int]:
    proc = subprocess.run(
        ["dbus-daemon", "--session", "--fork", "--print-address=1", "--print-pid=1"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2, f"Unexpected dbus-daemon output: {proc.stdout!r}"
    return lines[0], int(lines[1])


@pytest.fixture()
def two_private_buses():
    if shutil.which("dbus-daemon") is None:
        pytest.skip("dbus-daemon is not installed")
    pids: list[int] = []
    try:
        upstream_addr, upstream_pid = _start_private_dbus()
        pids.append(upstream_pid)
        local_addr, local_pid = _start_private_dbus()
        pids.append(local_pid)
        yield upstream_addr, local_addr
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


class FakeUpstreamSearchProvider(ServiceInterface):
    def __init__(self) -> None:
        super().__init__(GNOME_INTERFACE)
        self.mode: str = "ok"
        self.delay: float = 0.0
        self.icon_pixels = bytes(range(16))

    @method()
    async def GetInitialResultSet(self, terms: "as") -> "as":  # noqa: F821, N802, UP037
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.mode == "error":
            raise DBusError("io.github.GemmaSearch.TestError", "upstream failure")
        return [f"doc:{'_'.join(terms)}"]

    @method()
    async def GetSubsearchResultSet(  # noqa: N802
        self,
        previous_results: "as",  # noqa: F821, UP037
        terms: "as",  # noqa: F821, UP037
    ) -> "as":  # noqa: F821, UP037
        return [r for r in previous_results if all(t in r for t in terms)]

    @method()
    async def GetResultMetas(self, identifiers: "as") -> "aa{sv}":  # noqa: F821, N802, UP037
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.mode == "error":
            raise DBusError("io.github.GemmaSearch.TestError", "metas failure")
        return [
            {
                "id": Variant("s", ident),
                "name": Variant("s", f"Title for {ident}"),
                "description": Variant("s", "Semantic match"),
                "icon-data": Variant(
                    "(iiibiiay)",
                    [2, 2, 8, True, 8, 4, self.icon_pixels],
                ),
            }
            for ident in identifiers
        ]

    @method()
    async def ActivateResult(  # noqa: N802
        self,
        identifier: "s",  # noqa: F821, UP037
        terms: "as",  # noqa: F821, UP037
        timestamp: "u",  # noqa: F821, UP037
    ):
        return None

    @method()
    async def LaunchSearch(  # noqa: N802
        self,
        terms: "as",  # noqa: F821, UP037
        timestamp: "u",  # noqa: F821, UP037
    ):
        return None


async def _call_gnome(
    bus: MessageBus,
    member: str,
    signature: str,
    body: list[Any],
) -> Message:
    return await bus.call(
        Message(
            destination=BUS_NAME,
            path=GNOME_PATH,
            interface=GNOME_INTERFACE,
            member=member,
            signature=signature,
            body=body,
        )
    )


def test_bridge_forwards_results_metas_errors_and_timeouts(two_private_buses) -> None:
    upstream_addr, local_addr = two_private_buses

    async def main() -> None:
        upstream_server = await MessageBus(bus_address=upstream_addr).connect()
        bridge_upstream = await MessageBus(bus_address=upstream_addr).connect()
        bridge_local = await MessageBus(bus_address=local_addr).connect()
        client = await MessageBus(bus_address=local_addr).connect()

        try:
            fake = FakeUpstreamSearchProvider()
            upstream_server.export(GNOME_PATH, fake)
            assert await upstream_server.request_name(BUS_NAME) == RequestNameReply.PRIMARY_OWNER

            bridge = gnome_bridge.SearchProviderBridge(
                bridge_upstream,
                upstream_address=upstream_addr,
                timeout=0.25,
            )
            bridge_local.export(GNOME_PATH, bridge)
            assert await bridge_local.request_name(BUS_NAME) == RequestNameReply.PRIMARY_OWNER

            # Confirm dbus-fast unpacks aa{sv} values directly as Variant objects.
            direct_metas = await bridge_upstream.call(
                Message(
                    destination=BUS_NAME,
                    path=GNOME_PATH,
                    interface=GNOME_INTERFACE,
                    member="GetResultMetas",
                    signature="as",
                    body=[["doc:alpha"]],
                )
            )
            assert direct_metas.message_type == MessageType.METHOD_RETURN
            assert all(isinstance(v, Variant) for v in direct_metas.body[0][0].values())

            # 1. GetInitialResultSet round-trip through the bridge.
            res = await _call_gnome(client, "GetInitialResultSet", "as", [["rocket", "launch"]])
            assert res.message_type == MessageType.METHOD_RETURN
            assert res.body[0] == ["doc:rocket_launch"]

            # 2. GetResultMetas with icon-data (iiibiiay) round-trip through the bridge.
            metas_msg = await _call_gnome(client, "GetResultMetas", "as", [["doc:rocket_launch"]])
            assert metas_msg.message_type == MessageType.METHOD_RETURN
            metas = metas_msg.body[0]
            assert len(metas) == 1
            meta = metas[0]
            assert meta["id"].value == "doc:rocket_launch"
            assert meta["name"].value == "Title for doc:rocket_launch"
            assert meta["description"].value == "Semantic match"
            icon_data = meta["icon-data"]
            assert icon_data.signature == "(iiibiiay)"
            assert icon_data.value == (2, 2, 8, True, 8, 4, fake.icon_pixels)

            # 3. Upstream error -> empty default.
            fake.mode = "error"
            err_res = await _call_gnome(client, "GetInitialResultSet", "as", [["boom"]])
            assert err_res.message_type == MessageType.METHOD_RETURN
            assert err_res.body[0] == []
            err_metas = await _call_gnome(client, "GetResultMetas", "as", [["doc:boom"]])
            assert err_metas.message_type == MessageType.METHOD_RETURN
            assert err_metas.body[0] == []

            # 4. Upstream timeout -> empty default.
            fake.mode = "ok"
            fake.delay = 0.6
            to_res = await _call_gnome(client, "GetInitialResultSet", "as", [["slow"]])
            assert to_res.message_type == MessageType.METHOD_RETURN
            assert to_res.body[0] == []
            to_metas = await _call_gnome(client, "GetResultMetas", "as", [["doc:slow"]])
            assert to_metas.message_type == MessageType.METHOD_RETURN
            assert to_metas.body[0] == []
        finally:
            client.disconnect()
            bridge_local.disconnect()
            bridge_upstream.disconnect()
            upstream_server.disconnect()

    asyncio.run(main())
