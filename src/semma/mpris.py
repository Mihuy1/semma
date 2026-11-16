"""Seek a media player to a search hit's start second over MPRIS, after it opened the file.

Most players have no "start at" command-line option (Dragon Player, Elisa, Haruna, Totem,
Showtime, Clapper, Amberol, ...), but nearly all implement MPRIS2 on the session bus. The
file is opened normally, then ``seek_when_loaded`` polls for a player
(``org.mpris.MediaPlayer2.*``) whose current track is that file and calls
``Player.SetPosition(trackid, µs)``, retrying until ``Position`` is near the target (players
ignore a seek while the track is still loading) or the timeout passes.

Which player: one whose ``xesam:url`` is the file. Players that appeared after the snapshot
taken before the open are preferred, so a second, older window on the same file is left
alone; a single-instance player (Elisa) that reused its existing name still matches by URL.
A sandboxed (Flatpak) player sees the file through the document portal under another path,
so a newly appeared player whose track has the same file name also matches.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import logging
import os
from typing import Any
from urllib.parse import unquote, urlparse

from dbus_fast import Message, MessageType, Variant
from dbus_fast.aio import MessageBus

log = logging.getLogger(__name__)

MPRIS_PREFIX = "org.mpris.MediaPlayer2."
MPRIS_PATH = "/org/mpris/MediaPlayer2"
PLAYER_IFACE = "org.mpris.MediaPlayer2.Player"
PROPS_IFACE = "org.freedesktop.DBus.Properties"

SEEK_TIMEOUT_SECONDS = 15.0
POLL_SECONDS = 0.25
SETTLE_SECONDS = 0.4  # wait after SetPosition before reading Position back
TOLERANCE_SECONDS = 3.0
CALL_TIMEOUT_SECONDS = 1.0

_tasks: set[asyncio.Task[bool]] = set()


def url_path(url: str) -> str | None:
    """The local path of a ``file://`` URL (or a bare absolute path), else None."""
    if url.startswith("/"):
        return url
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return unquote(parsed.path)


def _same_file(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (OSError, ValueError):
        return False


def track_matches(track_url: str | None, path: str, *, new_player: bool) -> bool:
    if not track_url:
        return False
    track_path = url_path(track_url)
    if track_path is None:
        return False
    if _same_file(track_path, path):
        return True
    return new_player and os.path.basename(track_path) == os.path.basename(path)


def _unwrap(value: Any) -> Any:
    return value.value if isinstance(value, Variant) else value


async def _call(bus: MessageBus, destination: str, interface: str, member: str,
                signature: str = "", body: list[Any] | None = None) -> list[Any] | None:
    try:
        reply = await asyncio.wait_for(
            bus.call(Message(destination=destination, path=MPRIS_PATH if destination != "org.freedesktop.DBus"
                             else "/org/freedesktop/DBus", interface=interface, member=member,
                             signature=signature, body=body or [])),
            CALL_TIMEOUT_SECONDS,
        )
    except Exception as e:  # timeouts, a player vanishing mid-call
        log.debug("%s %s.%s failed: %s", destination, interface, member, e)
        return None
    if reply is None or reply.message_type != MessageType.METHOD_RETURN:
        return None
    return reply.body


async def player_names(bus: MessageBus) -> set[str]:
    body = await _call(bus, "org.freedesktop.DBus", "org.freedesktop.DBus", "ListNames")
    if not body:
        return set()
    return {n for n in body[0] if isinstance(n, str) and n.startswith(MPRIS_PREFIX)}


async def _get(bus: MessageBus, name: str, prop: str) -> Any:
    body = await _call(bus, name, PROPS_IFACE, "Get", "ss", [PLAYER_IFACE, prop])
    return _unwrap(body[0]) if body else None


async def _try_seek(bus: MessageBus, name: str, metadata: dict[str, Any], target_us: int) -> bool:
    """One seek attempt; True once the player reports a position near the target."""
    trackid = _unwrap(metadata.get("mpris:trackid"))
    length = _unwrap(metadata.get("mpris:length"))
    if isinstance(length, int) and 0 < length < target_us:
        log.debug("%s: track shorter (%d µs) than the target %d µs", name, length, target_us)
        return True  # nothing sensible to do; stop trying
    if isinstance(trackid, str) and trackid:
        await _call(bus, name, PLAYER_IFACE, "SetPosition", "ox", [trackid, target_us])
    else:  # no track id: relative seek from wherever it is
        position = await _get(bus, name, "Position")
        if not isinstance(position, int):
            return False
        await _call(bus, name, PLAYER_IFACE, "Seek", "x", [target_us - position])
    await asyncio.sleep(SETTLE_SECONDS)
    position = await _get(bus, name, "Position")
    return isinstance(position, int) and abs(position - target_us) <= TOLERANCE_SECONDS * 1_000_000


async def _candidates(bus: MessageBus, path: str, before: set[str]) -> list[tuple[str, dict[str, Any]]]:
    names = await player_names(bus)
    found = []
    for name in sorted(names, key=lambda n: n in before):  # newly appeared players first
        metadata = await _get(bus, name, "Metadata")
        if not isinstance(metadata, dict):
            continue
        url = _unwrap(metadata.get("xesam:url"))
        if track_matches(url if isinstance(url, str) else None, path, new_player=name not in before):
            found.append((name, metadata))
    new = [c for c in found if c[0] not in before]
    return new or found


async def seek_when_loaded(path: str, seconds: float, before: Iterable[str] = (),
                           *, bus: MessageBus | None = None, timeout: float = SEEK_TIMEOUT_SECONDS) -> bool:
    """Wait for a player that has ``path`` loaded and move it to ``seconds``. True on success."""
    before = set(before)
    target_us = int(round(seconds * 1_000_000))
    own_bus = bus is None
    try:
        if bus is None:
            bus = await MessageBus().connect()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            for name, metadata in (await _candidates(bus, path, before))[:1]:
                if await _try_seek(bus, name, metadata, target_us):
                    log.debug("seeked %s to %.1f s in %s", path, seconds, name)
                    return True
            await asyncio.sleep(POLL_SECONDS)
        log.info("no MPRIS player seeked %s to %.0f s within %.0f s", path, seconds, timeout)
        return False
    except Exception as e:
        log.debug("MPRIS seek of %s failed: %s", path, e)
        return False
    finally:
        if own_bus and bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass


async def snapshot() -> set[str]:
    """The MPRIS players on the session bus right now (empty when the bus is unreachable)."""
    try:
        bus = await MessageBus().connect()
    except Exception as e:
        log.debug("session bus unavailable: %s", e)
        return set()
    try:
        return await player_names(bus)
    finally:
        bus.disconnect()


def seek_in_background(path: str, seconds: float, before: Iterable[str]) -> asyncio.Task[bool]:
    """Run ``seek_when_loaded`` without blocking the caller (a D-Bus method reply)."""
    task = asyncio.get_running_loop().create_task(seek_when_loaded(path, seconds, before))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
