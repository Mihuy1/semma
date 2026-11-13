"""Tests for ``gemmasearch.gnome_search`` (D-Bus ``org.gnome.Shell.SearchProvider2`` service)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import itertools
import os
from pathlib import Path

from dbus_fast.aio import MessageBus
import pytest

from gemmasearch import filetypes as ft
from gemmasearch.dbus_names import GNOME_INTERFACE, GNOME_PATH
from gemmasearch.gnome_search import (
    GnomeLauncher,
    GnomeSearchService,
    gicon_string,
    gnome_description,
    shorten_folder,
)
from gemmasearch.krunner import format_subtext
from gemmasearch.search import SearchResult
from gemmasearch.thumbnails import IconData, Thumbnailer

_NAME_COUNTER = itertools.count(1)

requires_session_bus = pytest.mark.skipif(
    not os.environ.get("DBUS_SESSION_BUS_ADDRESS"),
    reason="DBUS_SESSION_BUS_ADDRESS is not set",
)

ICON: IconData = (2, 1, 8, True, 8, 4, bytes(range(8)))


class FakeLauncher(GnomeLauncher):
    def __init__(self) -> None:
        self.opened: list[str] = []

    async def open_file(self, path: str) -> None:
        self.opened.append(path)


class FakeThumbnailer(Thumbnailer):
    """Returns ICON after `delay` seconds for images/videos and caches it."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__(lru_capacity=32)
        self.delay = delay
        self.calls = 0

    async def icon_data(self, path: str, kind: str, mtime: float, size: int = 64) -> IconData | None:
        if kind not in (ft.IMAGE, ft.VIDEO):
            return None
        self.calls += 1
        await asyncio.sleep(self.delay)
        self._put_cached(self._lru_key(path, mtime, size), ICON)
        return ICON


def _result(
    path: str,
    kind: str = ft.TEXT,
    snippet: str = "some text",
    locator: str = "",
    duplicates: int = 0,
) -> SearchResult:
    return SearchResult(
        path=path, kind=kind, score=3.0, similarity=0.7, modality="text", locator=locator,
        snippet=snippet, position=None, mtime=1700000000.0, matched=["vector"], duplicates=duplicates,
        duplicate_paths=[], z=None,
    )


def _service(results: list[SearchResult], **kw) -> tuple[GnomeSearchService, list[str]]:
    queries: list[str] = []

    async def fake_search(q: str) -> list[SearchResult]:
        queries.append(q)
        return list(results)

    kw.setdefault("launcher", FakeLauncher())
    kw.setdefault("thumbnailer", FakeThumbnailer())
    return GnomeSearchService(fake_search, **kw), queries


@asynccontextmanager
async def exported_gnome(service: GnomeSearchService) -> AsyncIterator[tuple[object, MessageBus, str]]:
    bus_name = f"io.github.GemmaSearch.Test{os.getpid()}_g{next(_NAME_COUNTER)}"
    server_bus = await MessageBus().connect()
    client_bus = await MessageBus().connect()
    try:
        server_bus.export(GNOME_PATH, service)
        await server_bus.request_name(bus_name)
        introspection = await client_bus.introspect(bus_name, GNOME_PATH)
        proxy = client_bus.get_proxy_object(bus_name, GNOME_PATH, introspection)
        yield proxy.get_interface(GNOME_INTERFACE), client_bus, bus_name
    finally:
        try:
            await server_bus.release_name(bus_name)
        except Exception:
            pass
        server_bus.disconnect()
        client_bus.disconnect()


def test_gicon_string_has_generic_fallback() -> None:
    assert gicon_string("/a/b.png", ft.IMAGE) == ". GThemedIcon image-png image-x-generic"
    assert gicon_string("/a/b.pdf", ft.PDF) == "application-pdf"
    assert gicon_string("/a/noext", ft.TEXT) == "text-plain"


def test_short_query_returns_empty_without_searching() -> None:
    async def check() -> None:
        activity = 0

        def on_act() -> None:
            nonlocal activity
            activity += 1

        svc, queries = _service([_result("/x/a.txt")], on_activity=on_act)
        for terms in ([], [""], ["a"], ["a", "b"], ["--", "?"], ["1", "2"]):
            assert await svc._run_search(terms) == []
        assert queries == [] and activity == 0
        assert await svc._run_search(["a", "b1"]) == ["/x/a.txt"]
        assert queries == ["a b1"] and activity == 1

    asyncio.run(check())


def test_order_cap_and_subsearch_reruns() -> None:
    async def check() -> None:
        results = [_result(f"/x/{i}.txt") for i in (5, 1, 9, 3, 7)]
        svc, queries = _service(results, max_results=3)
        assert await svc._run_search(["rocket", "launch"]) == ["/x/5.txt", "/x/1.txt", "/x/9.txt"]
        # Subsearch ignores the previous ids and searches again with the new terms.
        assert await svc._run_search(["rocket", "launches"]) == ["/x/5.txt", "/x/1.txt", "/x/9.txt"]
        assert queries == ["rocket launch", "rocket launches"]

    asyncio.run(check())


def test_search_exception_returns_empty() -> None:
    async def check() -> None:
        async def boom(q: str) -> list[SearchResult]:
            raise RuntimeError("model died")

        async def cancelled(q: str) -> list[SearchResult]:
            raise asyncio.CancelledError

        for fn in (boom, cancelled):
            svc = GnomeSearchService(fn, launcher=FakeLauncher(), thumbnailer=FakeThumbnailer())
            assert await svc._run_search(["rocket"]) == []

    asyncio.run(check())


@requires_session_bus
def test_dbus_round_trip_metas_activate_and_signatures(tmp_path: Path) -> None:
    async def check() -> None:
        home = Path("/home/user")
        results = [
            _result("/home/user/notes/rockets.md", snippet="liquid  oxygen\nand kerosene", locator="§2"),
            _result("/home/user/Pictures/launch.png", kind=ft.IMAGE, snippet=""),
            _result("/home/user/Videos/slow.mp4", kind=ft.VIDEO, snippet=""),
        ]
        thumb = FakeThumbnailer(delay=0.0)
        launcher = FakeLauncher()
        svc, queries = _service(results, launcher=launcher, thumbnailer=thumb, home=home)
        async with exported_gnome(svc) as (iface, client_bus, bus_name):
            node = await client_bus.introspect(bus_name, GNOME_PATH)
            ifaces = [i for i in node.interfaces if i.name == GNOME_INTERFACE]
            methods = {m.name: m for m in ifaces[0].methods}
            sigs = {n: ([a.signature for a in m.in_args], [a.signature for a in m.out_args])
                    for n, m in methods.items()}
            assert sigs == {
                "GetInitialResultSet": (["as"], ["as"]),
                "GetSubsearchResultSet": (["as", "as"], ["as"]),
                "GetResultMetas": (["as"], ["aa{sv}"]),
                "ActivateResult": (["s", "as", "u"], []),
                "LaunchSearch": (["as", "u"], []),
            }

            assert await iface.call_get_initial_result_set(["ro"]) == []
            ids = await iface.call_get_initial_result_set(["rocket", "launch"])
            assert ids == [r.path for r in results]
            ids2 = await iface.call_get_subsearch_result_set(ids, ["rocket", "launch", "pad"])
            assert ids2 == ids
            assert queries == ["rocket launch", "rocket launch pad"]

            metas = await iface.call_get_result_metas(["/unknown/file.txt", *ids])
            assert len(metas) == 3
            text, image, video = ({k: v.value for k, v in m.items()} for m in metas)
            assert text == {
                "id": "/home/user/notes/rockets.md",
                "name": "rockets.md",
                "description": "§2 · liquid oxygen and kerosene",
                "clipboardText": "/home/user/notes/rockets.md",
                "gicon": text["gicon"],
            }
            assert "icon-data" not in text and text["gicon"]
            # Thumbnails replace the themed icon (gicon would win over icon-data in the shell).
            for m in (image, video):
                assert "gicon" not in m
                assert list(m["icon-data"]) == [2, 1, 8, True, 8, 4, bytes(range(8))]
            assert image["description"] == "~/Pictures"

            await iface.call_activate_result("/unknown/file.txt", ["x"], 0)
            await iface.call_activate_result(ids[1], ["rocket"], 123)
            assert launcher.opened == ["/home/user/Pictures/launch.png"]
            await iface.call_launch_search(["rocket"], 0)

    asyncio.run(check())


def test_metas_fall_back_to_gicon_past_thumbnail_budget() -> None:
    async def check() -> None:
        thumb = FakeThumbnailer(delay=0.3)
        svc, _ = _service([_result("/p/a.jpg", kind=ft.IMAGE)], thumbnailer=thumb, thumbnail_budget=0.02)
        ids = await svc._run_search(["sunset"])
        metas = await _call_metas(svc, ids)
        assert "gicon" in metas[0] and "icon-data" not in metas[0]
        # Generation continues in the background; the next lookup gets the thumbnail.
        await asyncio.sleep(0.4)
        metas = await _call_metas(svc, ids)
        assert "icon-data" in metas[0]
        assert thumb.calls >= 1

    asyncio.run(check())


async def _call_metas(svc: GnomeSearchService, ids: list[str]):
    # dbus-fast's @method keeps the undecorated coroutine function as __wrapped__.
    metas = await type(svc).__dict__["GetResultMetas"].__wrapped__(svc, ids)
    return [{k: v.value for k, v in m.items()} for m in metas]


def test_activate_failure_never_raises() -> None:
    class Broken(FakeLauncher):
        async def open_file(self, path: str) -> None:
            raise OSError("no opener")

    async def check() -> None:
        svc, _ = _service([_result("/x/a.txt")], launcher=Broken())
        ids = await svc._run_search(["abc"])
        fn = type(svc).__dict__["ActivateResult"].__wrapped__
        await fn(svc, ids[0], ["abc"], 0)

    asyncio.run(check())


def test_launcher_prefers_gio_then_xdg_open(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    available = {"gio": "/usr/bin/gio", "xdg-open": "/usr/bin/xdg-open"}
    monkeypatch.setattr("gemmasearch.gnome_search.shutil.which", lambda n: available.get(n))
    launcher = GnomeLauncher()
    monkeypatch.setattr(launcher, "_spawn_detached", lambda argv: spawned.append(argv) or True)
    asyncio.run(launcher.open_file("/x/a b.txt"))
    assert spawned == [["/usr/bin/gio", "open", "/x/a b.txt"]]

    spawned.clear()
    del available["gio"]
    asyncio.run(launcher.open_file("/x/a.txt"))
    assert spawned == [["/usr/bin/xdg-open", "/x/a.txt"]]


def test_shorten_folder_examples_and_edges() -> None:
    assert (
        shorten_folder("/mnt/games/code/thesisProject/backend/datasets/coco2017/evaluation/prepared_standalone")
        == "/mnt/…/coco2017/evaluation/prepared_standalone"
    )
    assert (
        shorten_folder("~/Documents/alpha/beta/gamma/delta/epsilon/very_long_folder_name")
        == "~/…/gamma/delta/epsilon/very_long_folder_name"
    )
    assert (
        shorten_folder("~/Documents/a/b/c/d/e/f/g/very_long_folder_name_here")
        == "~/…/a/b/c/d/e/f/g/very_long_folder_name_here"
    )
    assert shorten_folder("~/Pictures/Trips") == "~/Pictures/Trips"
    assert shorten_folder("/mnt/games") == "/mnt/games"

    # Single-component paths (or no middle components to elide) stay untouched.
    assert shorten_folder("/") == "/"
    assert shorten_folder("~") == "~"
    long_single = "/" + "a" * 60
    assert shorten_folder(long_single) == long_single
    long_home_single = "~/" + "a" * 60
    assert shorten_folder(long_home_single) == long_home_single
    long_root_child = "/mnt/" + "a" * 60
    assert shorten_folder(long_root_child) == long_root_child

    # Always keeps at least the last component even when longer than limit.
    long_last = "x" * 60
    assert shorten_folder(f"/mnt/middle/{long_last}") == f"/mnt/\u2026/{long_last}"
    assert shorten_folder(f"~/middle/{long_last}") == f"~/\u2026/{long_last}"

    # Limit boundary: len == limit untouched, len == limit + 1 shortened; exact fit at limit.
    at_limit = "/mnt/alpha/beta/gamma"
    assert shorten_folder(at_limit, limit=len(at_limit)) == at_limit
    assert shorten_folder(at_limit, limit=len(at_limit) - 1) == "/mnt/…/beta/gamma"
    exact_fit = "/mnt/alpha/bb/cc"
    assert shorten_folder(exact_fit, limit=len("/mnt/…/bb/cc")) == "/mnt/…/bb/cc"
    assert shorten_folder(exact_fit, limit=len("/mnt/…/bb/cc") - 1) == "/mnt/…/cc"

    # Unicode folder names.
    uni = "~/Документы/проекты/фотографии/лето_2025/архив_снимков_выходные"
    short_uni = shorten_folder(uni)
    assert short_uni == "~/…/фотографии/лето_2025/архив_снимков_выходные"
    assert len(short_uni) <= 48
    assert "\u2026" in short_uni


def test_gnome_description_matches_format_subtext_for_short_paths_and_snippets() -> None:
    home = Path("/home/user")
    cases = [
        _result("/home/user/Pictures/Trips/a.jpg", kind=ft.IMAGE, snippet=""),
        _result("/home/user/Pictures/Trips/a.jpg", kind=ft.IMAGE, snippet="", locator="01:00–02:00", duplicates=2),
        _result(
            "/mnt/games/code/thesisProject/backend/datasets/coco2017/evaluation/prepared_standalone/notes.txt",
            snippet="  first   line\n\t" + ("word " * 30),
            locator=" p.\n 12 ",
            duplicates=3,
        ),
        _result("/home/user/notes/a.txt", snippet="plain snippet", duplicates=1),
    ]
    for r in cases:
        assert gnome_description(r, home=home) == format_subtext(r, home=home)


def test_gnome_description_shortens_long_parent_and_keeps_suffix() -> None:
    home = Path("/home/user")
    r_abs = _result(
        "/mnt/games/code/thesisProject/backend/datasets/coco2017/evaluation/prepared_standalone/img.jpg",
        kind=ft.IMAGE,
        snippet="",
        duplicates=3,
    )
    assert gnome_description(r_abs, home=home) == "/mnt/…/coco2017/evaluation/prepared_standalone  (+3 similar)"

    r_home = _result(
        "/home/user/Documents/alpha/beta/gamma/delta/epsilon/very_long_folder_name/doc.pdf",
        kind=ft.PDF,
        snippet="",
        locator="  p.\n 4 ",
        duplicates=2,
    )
    assert gnome_description(r_home, home=home) == "p. 4 · ~/…/gamma/delta/epsilon/very_long_folder_name  (+2 similar)"

    async def check_metas() -> None:
        svc, _ = _service([r_abs], home=home)
        ids = await svc._run_search(["coco"])
        metas = await _call_metas(svc, ids)
        assert metas[0]["description"] == "/mnt/…/coco2017/evaluation/prepared_standalone  (+3 similar)"

    asyncio.run(check_metas())
