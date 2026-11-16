from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import itertools
import os
from pathlib import Path

import pytest
from dbus_fast import Variant
from dbus_fast.aio import MessageBus
from dbus_fast.constants import PropertyAccess
from dbus_fast.service import ServiceInterface, dbus_property, method

from semma import mpris
from semma.mpris import track_matches, url_path

requires_session_bus = pytest.mark.skipif(
    not os.environ.get("DBUS_SESSION_BUS_ADDRESS"),
    reason="DBUS_SESSION_BUS_ADDRESS is not set",
)
_COUNTER = itertools.count(1)


def test_url_path() -> None:
    assert url_path("file:///home/u/My%20Pod/ep%201.mp3") == "/home/u/My Pod/ep 1.mp3"
    assert url_path("/home/u/a.mp3") == "/home/u/a.mp3"
    assert url_path("https://example.com/a.mp3") is None


def test_track_matches(tmp_path: Path) -> None:
    f = tmp_path / "ep 1.mp3"
    f.touch()
    link = tmp_path / "link.mp3"
    link.symlink_to(f)
    assert track_matches(f.as_uri(), str(f), new_player=False)
    assert track_matches(link.as_uri(), str(f), new_player=False)  # same file through a symlink
    portal = "file:///run/user/1000/doc/abc123/ep%201.mp3"  # a Flatpak player's view of the file
    assert track_matches(portal, str(f), new_player=True)
    assert not track_matches(portal, str(f), new_player=False)
    assert not track_matches(None, str(f), new_player=True)
    assert not track_matches("file:///x/other.mp3", str(f), new_player=True)


class FakePlayer(ServiceInterface):
    """MPRIS Player that ignores the first ``ignore`` SetPosition calls (still loading)."""

    def __init__(self, url: str, ignore: int = 0, trackid: str | None = "/org/semma/track/1") -> None:
        super().__init__(mpris.PLAYER_IFACE)
        self.url, self.ignore, self.trackid = url, ignore, trackid
        self.position = 0
        self.set_calls: list[tuple[str, int]] = []
        self.seek_calls: list[int] = []

    @dbus_property(access=PropertyAccess.READ)
    def Metadata(self) -> "a{sv}":  # noqa: F821, N802, UP037
        md = {"xesam:url": Variant("s", self.url), "mpris:length": Variant("x", 600_000_000)}
        if self.trackid:
            md["mpris:trackid"] = Variant("o", self.trackid)
        return md

    @dbus_property(access=PropertyAccess.READ)
    def Position(self) -> "x":  # noqa: F821, N802, UP037
        return self.position

    @method()
    def SetPosition(self, trackid: "o", position: "x"):  # noqa: F821, N802, UP037
        self.set_calls.append((trackid, position))
        if self.ignore > 0:
            self.ignore -= 1
        elif trackid == self.trackid:
            self.position = position

    @method()
    def Seek(self, offset: "x"):  # noqa: F821, N802, UP037
        self.seek_calls.append(offset)
        self.position += offset


@asynccontextmanager
async def exported(player: FakePlayer) -> AsyncIterator[str]:
    name = f"{mpris.MPRIS_PREFIX}semmatest{os.getpid()}_{next(_COUNTER)}"
    bus = await MessageBus().connect()
    try:
        bus.export(mpris.MPRIS_PATH, player)
        await bus.request_name(name)
        yield name
    finally:
        bus.disconnect()


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mpris, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(mpris, "SETTLE_SECONDS", 0.02)


@requires_session_bus
def test_seeks_the_player_that_loaded_the_file(tmp_path: Path) -> None:
    f = tmp_path / "a b.mp4"

    async def go() -> None:
        player = FakePlayer(f.as_uri(), ignore=2)
        other = FakePlayer((tmp_path / "other.mp4").as_uri())
        async with exported(other), exported(player) as name:
            assert name in await mpris.snapshot()
            assert await mpris.seek_when_loaded(str(f), 125.5, timeout=5)
        assert player.position == 125_500_000
        assert len(player.set_calls) == 3  # retried while the player ignored it
        assert other.set_calls == [] and other.position == 0

    asyncio.run(go())


@requires_session_bus
def test_prefers_a_new_player_over_an_older_one_on_the_same_file(tmp_path: Path) -> None:
    f = tmp_path / "a.mp3"

    async def go() -> None:
        old, new = FakePlayer(f.as_uri()), FakePlayer(f.as_uri())
        async with exported(old):
            before = await mpris.snapshot()
            async with exported(new):
                assert await mpris.seek_when_loaded(str(f), 60, before, timeout=5)
        assert new.position == 60_000_000 and old.set_calls == []

    asyncio.run(go())


@requires_session_bus
def test_relative_seek_without_a_track_id(tmp_path: Path) -> None:
    f = tmp_path / "a.mp3"

    async def go() -> None:
        player = FakePlayer(f.as_uri(), trackid=None)
        player.position = 2_000_000
        async with exported(player):
            assert await mpris.seek_when_loaded(str(f), 60, timeout=5)
        assert player.seek_calls == [58_000_000] and player.position == 60_000_000

    asyncio.run(go())


@requires_session_bus
def test_gives_up_when_no_player_has_the_file(tmp_path: Path) -> None:
    async def go() -> None:
        other = FakePlayer((tmp_path / "other.mp3").as_uri())
        async with exported(other):
            assert not await mpris.seek_when_loaded(str(tmp_path / "a.mp3"), 60, timeout=0.3)
        assert other.set_calls == []

    asyncio.run(go())
