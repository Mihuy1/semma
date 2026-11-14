"""Tests for ``semma.krunner`` (D-Bus ``org.kde.krunner1`` service)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import itertools
import os
from pathlib import Path

from dbus_fast.aio import MessageBus
import numpy as np
from PIL import Image
import pytest

from semma import filetypes as ft
from semma.dbus_names import RUNNER_INTERFACE, RUNNER_PATH
from semma.krunner import (
    ACTION_COPY_PATH,
    ACTION_OPEN_FOLDER,
    CATEGORY_LABEL,
    KRunnerService,
    Launcher,
    MODERATE_Z,
    STRONG_SCORE,
    category_relevances,
    compute_monotonic_relevances,
    format_subtext,
    icon_name_for,
    raw_relevance,
)
from semma.search import SearchResult
from semma.thumbnails import IconData, Thumbnailer

_NAME_COUNTER = itertools.count(1)

requires_session_bus = pytest.mark.skipif(
    not os.environ.get("DBUS_SESSION_BUS_ADDRESS"),
    reason="DBUS_SESSION_BUS_ADDRESS is not set",
)


class FakeLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__(bus=None)
        self.opened_files: list[tuple[str, str | None]] = []
        self.opened_folders: list[tuple[str, str | None]] = []
        self.copied_paths: list[str] = []

    async def open_file(self, path: str, activation_token: str | None = None) -> None:
        self.opened_files.append((path, activation_token))

    async def open_folder(self, path: str, activation_token: str | None = None) -> None:
        self.opened_folders.append((path, activation_token))

    async def copy_path(self, path: str) -> None:
        self.copied_paths.append(path)


class SlowThumbnailer(Thumbnailer):
    def __init__(self, delay: float, icon: IconData) -> None:
        super().__init__(lru_capacity=32)
        self.delay = delay
        self.icon = icon
        self.calls = 0

    async def icon_data(self, path: str, kind: str, mtime: float, size: int = 64) -> IconData | None:
        if kind not in (ft.IMAGE, ft.VIDEO):
            return None
        cached = self.get_cached(path, mtime, size)
        if cached is not None:
            return cached
        self.calls += 1

        async def _produce() -> IconData:
            await asyncio.sleep(self.delay)
            self._put_cached(self._lru_key(path, mtime, size), self.icon)
            return self.icon

        key = self._lru_key(path, mtime, size)
        fut = self._inflight.get(key)
        if fut is None:
            fut = asyncio.ensure_future(_produce())
            self._inflight[key] = fut
            fut.add_done_callback(lambda _f: self._inflight.pop(key, None))
        return await asyncio.shield(fut)


def _sample_result(
    path: str = "/home/user/Documents/thesis.pdf",
    kind: str = ft.PDF,
    score: float = 3.0,
    similarity: float | None = 0.78,
    locator: str = "p. 4",
    snippet: str = "Introduction to semantic desktop search.",
    mtime: float = 1700000000.0,
    matched: list[str] | None = None,
    duplicates: int = 0,
    z: float | None = None,
) -> SearchResult:
    return SearchResult(
        path=path,
        kind=kind,
        score=score,
        similarity=similarity,
        modality="text" if kind != ft.IMAGE else "image",
        locator=locator,
        snippet=snippet,
        position=None,
        mtime=mtime,
        matched=matched if matched is not None else ["vector"],
        duplicates=duplicates,
        duplicate_paths=[],
        z=z,
    )


@asynccontextmanager
async def exported_krunner(
    service: KRunnerService,
) -> AsyncIterator[tuple[object, MessageBus, str]]:
    """Export ``service`` on the user's session bus under a unique throwaway name and yield a client proxy."""
    bus_name = f"io.github.Semma.Test{os.getpid()}_{next(_NAME_COUNTER)}"
    server_bus = await MessageBus().connect()
    client_bus = await MessageBus().connect()
    try:
        server_bus.export(RUNNER_PATH, service)
        await server_bus.request_name(bus_name)
        introspection = await client_bus.introspect(bus_name, RUNNER_PATH)
        proxy = client_bus.get_proxy_object(bus_name, RUNNER_PATH, introspection)
        iface = proxy.get_interface(RUNNER_INTERFACE)
        yield iface, client_bus, bus_name
    finally:
        try:
            await server_bus.release_name(bus_name)
        except Exception:
            pass
        server_bus.disconnect()
        client_bus.disconnect()


@requires_session_bus
def test_dbus_introspection_exact_signatures() -> None:
    async def check() -> None:
        async def fake_search(q: str) -> list[SearchResult]:
            return []

        svc = KRunnerService(fake_search)
        async with exported_krunner(svc) as (_iface, client_bus, bus_name):
            node = await client_bus.introspect(bus_name, RUNNER_PATH)
            krunner_ifaces = [i for i in node.interfaces if i.name == "org.kde.krunner1"]
            assert len(krunner_ifaces) == 1
            methods = {m.name: m for m in krunner_ifaces[0].methods}
            assert set(methods) == {"Match", "Actions", "Run", "Teardown", "SetActivationToken"}

            # Match(s query) -> a(sssida{sv})
            m_match = methods["Match"]
            assert [a.signature for a in m_match.in_args] == ["s"]
            assert [a.signature for a in m_match.out_args] == ["a(sssida{sv})"]

            # Actions() -> a(sss)
            m_actions = methods["Actions"]
            assert [a.signature for a in m_actions.in_args] == []
            assert [a.signature for a in m_actions.out_args] == ["a(sss)"]

            # Run(s matchId, s actionId) -> void
            m_run = methods["Run"]
            assert [a.signature for a in m_run.in_args] == ["s", "s"]
            assert [a.signature for a in m_run.out_args] == []

            # SetActivationToken(s token) -> void
            m_token = methods["SetActivationToken"]
            assert [a.signature for a in m_token.in_args] == ["s"]
            assert [a.signature for a in m_token.out_args] == []

            # Teardown() -> void
            m_teardown = methods["Teardown"]
            assert [a.signature for a in m_teardown.in_args] == []
            assert [a.signature for a in m_teardown.out_args] == []

    asyncio.run(check())


@requires_session_bus
def test_short_queries_return_empty_without_calling_search() -> None:
    async def check() -> None:
        called = 0
        activity = 0

        async def fake_search(q: str) -> list[SearchResult]:
            nonlocal called
            called += 1
            return [_sample_result()]

        def on_act() -> None:
            nonlocal activity
            activity += 1

        svc = KRunnerService(fake_search, on_activity=on_act)
        async with exported_krunner(svc) as (iface, _, _):
            for q in ("", "  ", "a", "ab", "a!", "  12  ", "--?"):
                res = await iface.call_match(q)
                assert res == []
            assert called == 0
            assert activity == 0

            # Exactly 3 letters/digits triggers search and on_activity
            res = await iface.call_match("  ab1 ")
            assert len(res) == 1
            assert called == 1
            assert activity == 1

    asyncio.run(check())


def test_mapping_helpers_subtext_icons_and_monotonic_relevance() -> None:
    home = Path("/home/alice")

    # 1. Locator + snippet (whitespace collapsed & trimmed) + duplicates
    long_text = "word " * 30  # 150 chars
    r1 = _sample_result(
        path="/home/alice/Documents/paper.pdf",
        locator="p. 12",
        snippet=f"  first   line\n\t{long_text}",
        duplicates=2,
    )
    sub1 = format_subtext(r1, home=home)
    assert sub1.startswith("p. 12 · first line word ")
    assert sub1.endswith("…  (+2 similar)")
    assert "\n" not in sub1 and "\t" not in sub1

    # 2. No snippet -> shortened parent folder (~) + locator if any + duplicates
    r2 = _sample_result(
        path="/home/alice/Pictures/Trips/photo.jpg",
        kind=ft.IMAGE,
        locator="",
        snippet="",
        duplicates=1,
    )
    assert format_subtext(r2, home=home) == "~/Pictures/Trips  (+1 similar)"

    r3 = _sample_result(
        path="/home/alice/Videos/clip.mp4",
        kind=ft.VIDEO,
        locator="01:00–02:00",
        snippet="",
        duplicates=0,
    )
    assert format_subtext(r3, home=home) == "01:00–02:00 · ~/Videos"

    # 3. Icon names from MIME and fallbacks
    assert icon_name_for("/tmp/a.pdf", ft.PDF) == "application-pdf"
    assert icon_name_for("/tmp/a.png", ft.IMAGE) == "image-png"
    assert icon_name_for("/tmp/a.unknown_ext_xyz", ft.TEXT) == "text-plain"
    assert icon_name_for("/tmp/a.unknown_ext_xyz", ft.OFFICE) == "x-office-document"
    assert icon_name_for("/tmp/a.unknown_ext_xyz", ft.IMAGE) == "image-x-generic"
    assert icon_name_for("/tmp/a.unknown_ext_xyz", ft.AUDIO) == "audio-x-generic"
    assert icon_name_for("/tmp/a.unknown_ext_xyz", ft.VIDEO) == "video-x-generic"

    # 4. Relevance: linear in the search score ([0, 8] -> [0.3, 0.9]), strictly decreasing in rank.
    # Similarity no longer matters: the score already ranks the results.
    results = [
        _sample_result(path="/a", score=4.0, similarity=None, matched=["name"]),
        _sample_result(path="/b", score=4.0, similarity=0.80, matched=["vector"], z=6.0),  # a tie
        _sample_result(path="/c", score=0.0, similarity=0.55, matched=["vector"]),
        _sample_result(path="/d", score=2.0, similarity=0.90, matched=["vector"]),  # out of order
    ]
    rels = compute_monotonic_relevances(results)
    assert all(0.0 < x <= 0.9 for x in rels)
    assert all(a > b for a, b in itertools.pairwise(rels))
    assert rels[0] == pytest.approx(0.6)
    assert rels[1] == pytest.approx(0.6 - 1e-4)  # ties are split in rank order
    assert rels[2] == pytest.approx(0.3)
    assert rels[3] < rels[2]
    assert raw_relevance(100.0) == pytest.approx(0.9) and raw_relevance(-1.0) == pytest.approx(0.3)

    # Category relevance never increases down the list: 50 while results are strong, then 30.
    assert category_relevances(results) == [50, 50, 30, 30]
    assert category_relevances([_sample_result(score=1.0), _sample_result(score=9.0)]) == [30, 30]
    assert category_relevances([_sample_result(score=STRONG_SCORE, matched=["text"])]) == [50]


def test_moderate_needs_words_or_a_meaning_match_that_stands_far_out() -> None:
    """z is relative: every query has a z 3-4.5 tail, so a vector-only result is "Moderate" only
    from MODERATE_Z; one with the query's words in its name or text needs just STRONG_SCORE."""
    assert category_relevances([_sample_result(score=4.7, matched=["vector"], z=4.5)]) == [30]
    assert category_relevances([_sample_result(score=5.5, matched=["vector"], z=MODERATE_Z)]) == [50]
    assert category_relevances([_sample_result(score=4.6, matched=["vector", "text"], z=3.6)]) == [50]
    assert category_relevances([_sample_result(score=5.0, matched=["name"], z=None)]) == [50]
    assert category_relevances([_sample_result(score=3.0, matched=["vector", "text"], z=2.0)]) == [30]
    assert category_relevances([_sample_result(score=4.2, matched=["vector", "folder"], z=3.9)]) == [30]
    # Strong after weak is still Low: category relevance never increases down the ranking.
    assert category_relevances([_sample_result(score=6.0, matched=["vector"], z=4.0),
                                _sample_result(score=5.0, matched=["text"])]) == [30, 30]


def test_relevances_never_reorder_results() -> None:
    """Whatever the scores, KRunner sorting by (category relevance, relevance) keeps our order."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        scores = np.where(rng.random(n) < 0.2, rng.choice([4.0, 0.0], size=n), rng.uniform(-1, 12, size=n))
        results = [_sample_result(path=f"/{i}", score=float(sc)) for i, sc in enumerate(scores)]
        rels = compute_monotonic_relevances(results)
        cats = category_relevances(results)
        order = sorted(range(n), key=lambda i: (-cats[i], -rels[i]))
        assert order == list(range(n))


@requires_session_bus
def test_match_properties_urls_thumbnails_and_actions(tmp_path: Path) -> None:
    img_path = tmp_path / "kesä kuva (1).png"
    Image.new("RGBA", (120, 60), (255, 128, 0, 255)).save(img_path, format="PNG")
    mtime = img_path.stat().st_mtime

    doc_path = tmp_path / "Muistiinpanot & raportti.pdf"
    doc_path.write_bytes(b"%PDF-1.4")

    results = [
        _sample_result(
            path=str(doc_path),
            kind=ft.PDF,
            score=6.4,
            similarity=0.80,
            locator="p. 4",
            snippet="Johdanto kappale.",
            matched=["vector"],
            duplicates=3,
            z=6.0,
        ),
        _sample_result(
            path=str(img_path),
            kind=ft.IMAGE,
            similarity=0.68,
            locator="",
            snippet="",
            mtime=mtime,
            matched=["vector"],
            duplicates=0,
        ),
    ]

    async def check() -> None:
        async def fake_search(q: str) -> list[SearchResult]:
            return results

        thumb = Thumbnailer(cache_dir=tmp_path / "cache")
        svc = KRunnerService(fake_search, thumbnailer=thumb, home=tmp_path)
        async with exported_krunner(svc) as (iface, _, _):
            actions = await iface.call_actions()
            assert [list(a) for a in actions] == [
                [ACTION_OPEN_FOLDER, "Open Containing Folder", "document-open-folder"],
                [ACTION_COPY_PATH, "Copy Path", "edit-copy"],
            ]

            matches = await iface.call_match("johdanto")
            assert len(matches) == 2

            # Match 0: PDF (no icon-data, score 6.4 >= STRONG_SCORE -> cat_rel 50)
            m0_id, m0_text, m0_icon, m0_cat_rel, m0_rel, m0_props = matches[0]
            assert m0_id == str(doc_path)
            assert m0_text == "Muistiinpanot & raportti.pdf"
            assert m0_icon == "application-pdf"
            assert m0_cat_rel == 50
            assert m0_rel == pytest.approx(0.78, abs=1e-4)
            assert m0_props["subtext"].value == "p. 4 · Johdanto kappale.  (+3 similar)"
            assert m0_props["urls"].value == [doc_path.as_uri()]
            assert "%20" in m0_props["urls"].value[0]
            assert m0_props["category"].value == CATEGORY_LABEL
            assert m0_props["actions"].value == [ACTION_OPEN_FOLDER, ACTION_COPY_PATH]
            assert "icon-data" not in m0_props

            # Match 1: Image (has icon-data, score 3.0 < STRONG_SCORE -> cat_rel 30)
            m1_id, m1_text, m1_icon, m1_cat_rel, m1_rel, m1_props = matches[1]
            assert m1_id == str(img_path)
            assert m1_text == "kesä kuva (1).png"
            assert m1_icon == "image-png"
            assert m1_cat_rel == 30
            assert m1_rel <= m0_rel
            assert m1_props["subtext"].value == "~"
            assert m1_props["urls"].value == [img_path.as_uri()]
            assert "icon-data" in m1_props
            icon_struct = m1_props["icon-data"].value
            w, h, rowstride, has_alpha, bps, channels, raw_bytes = icon_struct
            assert (w, h) == (64, 32)
            assert rowstride == 64 * 4
            assert has_alpha is True
            assert bps == 8
            assert channels == 4
            assert len(raw_bytes) == rowstride * h

    asyncio.run(check())


@requires_session_bus
def test_run_actions_and_activation_token() -> None:
    res = _sample_result(path="/home/user/Documents/report.pdf")

    async def check() -> None:
        async def fake_search(q: str) -> list[SearchResult]:
            return [res]

        launcher = FakeLauncher()
        svc = KRunnerService(fake_search, launcher=launcher)
        async with exported_krunner(svc) as (iface, _, _):
            matches = await iface.call_match("report")
            match_id = matches[0][0]

            # 1. SetActivationToken + default Run ("")
            await iface.call_set_activation_token("wayland-token-123")
            await iface.call_run(match_id, "")
            assert launcher.opened_files == [("/home/user/Documents/report.pdf", "wayland-token-123")]

            # Token must be cleared after Run
            await iface.call_run(match_id, "")
            assert launcher.opened_files[-1] == ("/home/user/Documents/report.pdf", None)

            # 2. open-folder action with token
            await iface.call_set_activation_token("token-folder-456")
            await iface.call_run(match_id, ACTION_OPEN_FOLDER)
            assert launcher.opened_folders == [("/home/user/Documents/report.pdf", "token-folder-456")]

            # 3. copy-path action
            await iface.call_run(match_id, ACTION_COPY_PATH)
            assert launcher.copied_paths == ["/home/user/Documents/report.pdf"]

            # 4. Unknown matchId is safely ignored
            await iface.call_run("/nonexistent/id", "")
            assert len(launcher.opened_files) == 2

            # 5. Teardown clears token and keeps recent map bounded
            await iface.call_set_activation_token("token-to-clear")
            await iface.call_teardown()
            await iface.call_run(match_id, "")
            assert launcher.opened_files[-1] == ("/home/user/Documents/report.pdf", None)

    asyncio.run(check())


@requires_session_bus
def test_search_exception_and_concurrent_matches_and_thumbnail_budget() -> None:
    img_res = _sample_result(
        path="/home/user/Pictures/photo.jpg",
        kind=ft.IMAGE,
        similarity=0.77,
        locator="",
        snippet="",
        mtime=1700000100.0,
    )
    fake_icon: IconData = (4, 4, 16, True, 8, 4, b"\xff" * 64)

    async def check() -> None:
        mode = "error"

        async def fake_search(q: str) -> list[SearchResult]:
            if mode == "error":
                raise RuntimeError("boom in search")
            await asyncio.sleep(0.01)
            return [img_res]

        slow_thumb = SlowThumbnailer(delay=0.12, icon=fake_icon)
        svc = KRunnerService(
            fake_search,
            thumbnailer=slow_thumb,
            thumbnail_budget=0.02,  # 20 ms budget, shorter than 120 ms generation
        )
        async with exported_krunner(svc) as (iface, _, _):
            # 1. Search exception returns [] cleanly over D-Bus
            assert await iface.call_match("failing query") == []

            # 2. Slow thumbnail misses the first Match budget -> returned without icon-data,
            #    then cached on the second Match after background task completes.
            mode = "ok"
            m_first = await iface.call_match("photo")
            assert len(m_first) == 1
            assert "icon-data" not in m_first[0][5]

            await asyncio.sleep(0.15)
            m_second = await iface.call_match("photo")
            assert len(m_second) == 1
            assert "icon-data" in m_second[0][5]
            assert tuple(m_second[0][5]["icon-data"].value) == fake_icon
            assert slow_thumb.calls == 1

            # 3. Multiple concurrent Match calls in flight (one per keystroke)
            concurrent = await asyncio.gather(
                iface.call_match("pho"),
                iface.call_match("phot"),
                iface.call_match("photo"),
            )
            assert all(len(r) == 1 for r in concurrent)

    asyncio.run(check())
