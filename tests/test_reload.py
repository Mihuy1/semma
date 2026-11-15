"""ReloadConfig: the daemon reads its config file again and applies it without a restart.

Same doubles as test_daemon (fake embedder and extractor, stub watcher); the config comes from
a real TOML file, written per test.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
import tomlkit
from fakes import FakeExtractor
from test_daemon import (DOCS, StubWatcher, counts, make_daemon, needs_bus, test_bus_name, until_idle, wait_until,
                         write_docs)

from semma import daemon as daemon_mod
from semma import dbus_names
from semma.config import Config, ConfigError, load_config
from semma.daemon import Daemon, DaemonNotRunning, configured_areas, control_call, daemon_reload_config, serve
from semma.store import Store


def base_settings(home: Path) -> dict[str, Any]:
    """The settings of test_daemon.make_cfg, as config.toml."""
    return {
        "onboarding_complete": True,
        "db_path": str(home / "db" / "index.db"),
        "scan": {"roots": [str(home / "Documents")], "include_mounts": False},
        "priority": {"locations": {str(home / "Documents"): 40}},
        "search": {"z_gate": 1.0, "z_floor": 0.0},
        "daemon": {"query_debounce_ms": 0, "gpu_check_seconds": 0.05, "model_idle_unload_minutes": 0,
                   "index_model_unload_seconds": 0},
    }


def write_config(home: Path, settings: dict[str, Any]) -> Path:
    path = home / "config.toml"
    path.write_text(tomlkit.dumps(settings), encoding="utf-8")
    return path


def start_cfg(home: Path, settings: dict[str, Any] | None = None) -> Config:
    path = write_config(home, settings or base_settings(home))
    return load_config(path, home=home, strict=True)


def paths(db_path: Path) -> dict[str, float]:
    """Indexed path -> stored priority."""
    s = Store(db_path)
    try:
        return {r[0]: r[1] for r in s.db.execute("SELECT path, priority FROM files")}
    finally:
        s.close()


async def reload_and_settle(d: Daemon, home: Path, settings: dict[str, Any], rescan: bool = True) -> list[str]:
    """Write the settings, reload, and wait until the worker applied them (and rescanned)."""
    write_config(home, settings)
    scanned_at = d._last_scan_at
    restart = await d.reload_config()
    await wait_until(lambda: d._pending_cfg is None, what="settings applied")
    if rescan:
        await wait_until(lambda: d._last_scan_at > scanned_at, what="rescan after the reload")
        await until_idle(d)
    return restart


@pytest.fixture
def home(tmp_path: Path) -> Path:
    write_docs(tmp_path, DOCS)
    return tmp_path


def test_every_config_field_has_a_reload_rule() -> None:
    groups = [daemon_mod.RESTART_FIELDS, daemon_mod.SCAN_FIELDS, daemon_mod.PRIORITY_FIELDS,
              daemon_mod.NEXT_FILE_FIELDS, daemon_mod.LIVE_FIELDS, daemon_mod.ONBOARDING_FIELDS,
              dict.fromkeys(daemon_mod.FIXED_FIELDS)]
    names = [attr for g in groups for attr in g]
    assert len(names) == len(set(names))
    assert set(names) == {f.name for f in dataclasses.fields(Config)}


def test_reload_without_changes_is_a_noop(home: Path) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            full_scan, scanned_at = d._last_full_scan, d._last_scan_at
            assert await d.reload_config() == []
            assert d._pending_cfg is None and not d._scan_full
            await asyncio.sleep(0.1)
            assert d.cfg is cfg and d._last_full_scan == full_scan and d._last_scan_at == scanned_at
        finally:
            await d.stop()

    asyncio.run(main())


def test_removed_root_is_forgotten_while_unavailable_root_is_kept(home: Path) -> None:
    async def main() -> None:
        write_docs(home, {"comet.md": "a comet tail glows near the sun"}, sub="../Other")
        write_docs(home, {"harbor.md": "fishing boats rest in the quiet harbor"}, sub="../Drive")
        settings = base_settings(home)
        settings["scan"]["roots"] += [str(home / "Other"), str(home / "Drive")]
        cfg = start_cfg(home, settings)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            assert counts(cfg.db_path).get("indexed") == 5
            assert (await d.search("comet tail sun"))[0].path == str(home / "Other" / "comet.md")
            # The drive is unplugged (its mount point is gone) and Other is removed from the roots.
            shutil.move(home / "Drive", home / "Drive.away")
            settings["scan"]["roots"].remove(str(home / "Other"))
            assert await reload_and_settle(d, home, settings) == []
            indexed = paths(cfg.db_path)
            assert str(home / "Other" / "comet.md") not in indexed
            assert str(home / "Drive" / "harbor.md") in indexed  # configured, just not there now
            assert len(indexed) == 4
            res = await d.search("comet tail sun")  # the in-memory vectors went with the rows
            assert all("Other" not in r.path for r in res)
            # The daily scan with the drive still away keeps it too.
            d.rescan()
            scanned_at = d._last_scan_at
            await wait_until(lambda: d._last_scan_at > scanned_at, what="rescan")
            assert str(home / "Drive" / "harbor.md") in paths(cfg.db_path)
            # Plugged in again: found without re-indexing.
            shutil.move(home / "Drive.away", home / "Drive")
            assert (await d.search("fishing boats harbor"))[0].path == str(home / "Drive" / "harbor.md")
        finally:
            await d.stop()

    asyncio.run(main())


def test_disabled_kind_and_excluded_extension_are_removed(home: Path) -> None:
    async def main() -> None:
        write_docs(home, {"sunset.jpg": "orange sunset over the sea"})
        cfg = start_cfg(home)
        d = make_daemon(cfg, extractor=FakeExtractor())
        d.start()
        try:
            await until_idle(d)
            assert len(paths(cfg.db_path)) == 4
            settings = base_settings(home)
            settings["scan"]["kinds"] = ["text", "pdf", "office", "audio", "video"]
            settings["scan"]["exclude_extensions"] = ["txt"]
            assert await reload_and_settle(d, home, settings) == []
            assert {Path(p).name for p in paths(cfg.db_path)} == {"cats.md", "rockets.md"}
            assert d.cfg.kinds == frozenset({"text", "pdf", "office", "audio", "video"})
            for query in ("knead dough bread oven", "orange sunset sea"):
                assert all(Path(r.path).name in {"cats.md", "rockets.md"} for r in await d.search(query))
        finally:
            await d.stop()

    asyncio.run(main())


def test_max_file_mb_change_removes_too_large_and_requeues_when_raised(home: Path) -> None:
    async def main() -> None:
        big = home / "Documents" / "big.md"
        write_docs(home, {"big.md": "giant whale swimming in the deep ocean " * 60})
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            assert str(big) in paths(cfg.db_path)
            assert counts(cfg.db_path) == {"indexed": 4}

            # Lower the text limit below big.md (~2.3 KiB) while keeping the small docs under it.
            settings = base_settings(home)
            settings["scan"]["max_file_mb"] = {"text": 0.001}
            assert await reload_and_settle(d, home, settings) == []
            assert str(big) not in paths(cfg.db_path)
            assert counts(cfg.db_path) == {"indexed": 3}
            assert all(r.path != str(big) for r in await d.search("whale swimming ocean"))

            # Pause indexing so the rescan after raising the limit leaves big.md observable as pending.
            d.pause()
            await wait_until(lambda: d._phase == "paused", what="paused")
            settings["scan"]["max_file_mb"] = {"text": 10}
            write_config(home, settings)
            scanned_at = d._last_scan_at
            assert await d.reload_config() == []
            await wait_until(lambda: d._last_scan_at > scanned_at, what="rescan after raising max_file_mb")
            assert str(big) in paths(cfg.db_path)
            assert counts(cfg.db_path) == {"indexed": 3, "pending": 1}

            # Resume: big.md gets indexed again.
            d.resume()
            await until_idle(d)
            assert counts(cfg.db_path) == {"indexed": 4}
        finally:
            await d.stop()

    asyncio.run(main())


def test_new_root_is_scanned_and_watched(home: Path) -> None:
    async def main() -> None:
        StubWatcher.instances.clear()
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            w = StubWatcher.instances[-1]
            assert str(home / "Other") not in w.dirs
            write_docs(home, {"comet.md": "a comet tail glows near the sun"}, sub="../Other/space")
            settings = base_settings(home)
            settings["scan"]["roots"].append(str(home / "Other"))
            assert await reload_and_settle(d, home, settings) == []
            assert str(home / "Other" / "space" / "comet.md") in paths(cfg.db_path)
            assert {str(home / "Other"), str(home / "Other" / "space"), str(home / "Documents")} <= w.dirs
            assert (await d.search("comet tail sun"))[0].path == str(home / "Other" / "space" / "comet.md")
            # And watches follow a root that is removed again.
            settings["scan"]["roots"].remove(str(home / "Other"))
            await reload_and_settle(d, home, settings)
            assert not any(x.startswith(str(home / "Other")) for x in w.dirs)
            assert str(home / "Documents") in w.dirs
        finally:
            await d.stop()

    asyncio.run(main())


def test_priority_change_rescores_unchanged_files(home: Path) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            cats = str(home / "Documents" / "cats.md")
            before = paths(cfg.db_path)[cats]
            settings = base_settings(home)
            settings["priority"]["locations"][str(home / "Documents")] = 10
            settings["priority"]["type_weights"] = {"text": 5}
            assert await reload_and_settle(d, home, settings) == []
            after = paths(cfg.db_path)[cats]
            assert after == pytest.approx(before - 30 - 20)
            assert counts(cfg.db_path) == {"indexed": 3}  # rescored, not re-indexed
        finally:
            await d.stop()

    asyncio.run(main())


def test_restart_keys_are_reported_and_not_applied(home: Path) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            settings = base_settings(home)
            settings["db_path"] = str(home / "elsewhere.db")
            settings["model"] = {"device": "cpu", "name": "other/model"}
            settings["index"] = {"batch_sizes": {"text": 2}}
            settings["search"]["vector_cache_max_chunks"] = 5
            settings["daemon"]["krunner_max_results"] = 3  # live, applied
            restart = await reload_and_settle(d, home, settings, rescan=False)
            # Indexing has loaded the model, so the device still waits for a restart.
            assert sorted(restart) == ["db_path", "index.batch_sizes", "model.device", "model.name",
                                       "search.vector_cache_max_chunks"]
            assert d.cfg.db_path == cfg.db_path and d.cfg.device is None and d.cfg.model_name == cfg.model_name
            assert d.cfg.batch_sizes == cfg.batch_sizes and d.cfg.krunner_max_results == 3
            # Still pending a restart on the next reload; nothing else changes.
            assert sorted(await d.reload_config()) == sorted(restart)
            # Back to the started values: nothing to restart for.
            assert await reload_and_settle(d, home, base_settings(home), rescan=False) == []
        finally:
            await d.stop()

    asyncio.run(main())


def test_device_applies_without_a_restart_before_the_model_is_loaded(home: Path) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        from test_daemon import LifecycleEmbedder
        emb = LifecycleEmbedder()
        emb.requested_device = None
        emb.requested_query_device = "auto"
        d = make_daemon(cfg, embedder=emb)
        settings = base_settings(home)
        settings["model"] = {"device": "cpu"}
        write_config(home, settings)
        restart = await d.reload_config()
        assert "model.device" not in restart
        assert d.cfg.device is None  # not swapped in until the worker applies it
        assert d._pending_cfg is not None and d._pending_cfg.device == "cpu"
        d._apply_config(d._pending_cfg)
        assert d.cfg.device == "cpu" and emb.requested_device == "cpu"
        # A second reload of the same file does not ask for a restart again.
        assert "model.device" not in await d.reload_config()

    asyncio.run(main())


def test_device_change_needs_a_restart_while_the_model_is_loading(home: Path) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        from test_daemon import LifecycleEmbedder
        emb = LifecycleEmbedder()
        emb.loading = True  # the load has read the device already
        d = make_daemon(cfg, embedder=emb)
        settings = base_settings(home)
        settings["model"] = {"device": "cpu"}
        write_config(home, settings)
        assert await d.reload_config() == ["model.device"]

    asyncio.run(main())


def test_device_change_needs_a_restart_once_the_model_is_loaded(home: Path) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            d.embedder.load()
            settings = base_settings(home)
            settings["model"] = {"device": "cpu"}
            restart = await reload_and_settle(d, home, settings, rescan=False)
            assert restart == ["model.device"]
            assert d.cfg.device is None
        finally:
            await d.stop()

    asyncio.run(main())


def test_live_settings_reach_every_holder_without_a_rescan(home: Path) -> None:
    async def main() -> None:
        StubWatcher.instances.clear()
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        svc = type("Svc", (), {"_max_results": 10})()
        d._services = [svc]
        d.start()
        try:
            await until_idle(d)
            full_scan = d._last_full_scan
            settings = base_settings(home)
            settings["search"]["z_gate"] = 1.5
            settings["daemon"].update({"krunner_max_results": 4, "watch_debounce_seconds": 2,
                                       "query_debounce_ms": 7, "rescan_hours": 12})
            settings["media"] = {"max_minutes": 5}
            assert await reload_and_settle(d, home, settings, rescan=False) == []
            new = d.cfg
            assert new is not cfg and new.z_gate == 1.5 and new.max_media_seconds == 300
            assert d.indexer.cfg is new
            await wait_until(lambda: d.searcher.cfg is new, what="searcher swapped")
            await wait_until(lambda: svc._max_results == 4, what="runner limit")
            assert StubWatcher.instances[-1].debounce == 2
            assert d._last_full_scan == full_scan
            assert counts(cfg.db_path) == {"indexed": 3}  # nothing re-indexed
        finally:
            await d.stop()

    asyncio.run(main())


def test_reload_toggles_pause_when_gpu_busy_live(home: Path) -> None:
    from test_daemon import StubGpu

    async def main() -> None:
        settings = base_settings(home)
        settings["daemon"]["gpu_check_seconds"] = 60.0
        cfg = start_cfg(home, settings)
        gpu = StubGpu(busy=True)
        d = make_daemon(cfg, gpu=gpu)
        d.start()
        try:
            await wait_until(lambda: d._phase == "paused", what="initial GPU pause")
            st = await d.status()
            assert st["paused"] and "GPU" in st["pause_reason"] and st["indexed"] == 0

            # Turning pause_when_gpu_busy off during a GPU pause resumes indexing without a restart.
            settings["daemon"]["pause_when_gpu_busy"] = False
            assert await reload_and_settle(d, home, settings, rescan=False) == []
            await until_idle(d)
            st2 = await d.status()
            assert not st2["paused"] and st2["pause_reason"] == "" and st2["indexed"] == 3

            # Turning it back on makes the next sample count immediately (despite gpu_check_seconds=60).
            calls_before = gpu.calls
            settings["daemon"]["pause_when_gpu_busy"] = True
            assert await reload_and_settle(d, home, settings, rescan=False) == []
            write_docs(home, {"extra.md": "bright nebula in the night sky"})
            d.rescan()
            await wait_until(lambda: d._phase == "paused", what="GPU pause after re-enabling")
            assert gpu.calls > calls_before
            st3 = await d.status()
            assert st3["paused"] and "GPU" in st3["pause_reason"] and st3["pending"] == 1

            gpu.is_busy = False
            await until_idle(d)
            assert counts(cfg.db_path) == {"indexed": 4}
        finally:
            await d.stop()

    asyncio.run(main())


def test_reload_during_a_slice_applies_at_a_file_boundary(home: Path) -> None:
    async def main() -> None:
        write_docs(home, {f"note{i}.md": f"note number {i} about gardens" for i in range(6)})
        gate = asyncio.Event()
        seen: list[float] = []
        loop = asyncio.get_running_loop()

        def slow(path: Path, i: int) -> None:
            time.sleep(0.05)

        ext = FakeExtractor(on_chunk=slow)
        orig = ext.__call__

        def spy(path: Path, kind: str, c: Config):
            seen.append(c.max_media_seconds)
            if len(seen) == 2:
                loop.call_soon_threadsafe(gate.set)
            return orig(path, kind, c)

        cfg = start_cfg(home)
        d = make_daemon(cfg, extractor=spy)  # type: ignore[arg-type]
        d.start()
        try:
            await asyncio.wait_for(gate.wait(), 5)
            settings = base_settings(home)
            settings["media"] = {"max_minutes": 1}
            assert await reload_and_settle(d, home, settings, rescan=False) == []
            await until_idle(d)
            assert counts(cfg.db_path) == {"indexed": 9}
            assert seen[0] == cfg.max_media_seconds and seen[-1] == 60  # later files: new settings
        finally:
            await d.stop()

    asyncio.run(main())


@pytest.mark.parametrize("content", ["this is [not toml", None, "scan = { roots = 3 }",
                                     "priority = { depth_free_levels = 'deep' }"])
def test_invalid_config_keeps_the_settings(home: Path, content: str | None) -> None:
    async def main() -> None:
        cfg = start_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            path = home / "config.toml"
            if content is None:
                path.unlink()
            else:
                path.write_text(content, encoding="utf-8")
            with pytest.raises(ConfigError) as err:
                await d.reload_config()
            assert str(path) in str(err.value)
            assert d.cfg is cfg and d._accepted_cfg is cfg and d._pending_cfg is None and not d._scan_full
            assert counts(cfg.db_path) == {"indexed": 3}
        finally:
            await d.stop()

    asyncio.run(main())


def test_configured_areas_keep_mount_parents_only_with_include_mounts(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "none.toml", home=tmp_path)
    cfg.roots = [tmp_path / "gone"]
    cfg.include_mounts = True
    assert tmp_path / "gone" in configured_areas(cfg) and Path("/run/media") in configured_areas(cfg)
    cfg.include_mounts = False
    assert configured_areas(cfg, [tmp_path / "x"]) == [tmp_path / "gone", tmp_path / "x"]


def test_store_forget_unconfigured(tmp_path: Path) -> None:
    from semma.priority import Score
    from semma.scanner import ScannedFile

    store = Store(tmp_path / "i.db")
    try:
        files = ["/home/u/a.md", "/home/u/junk/b.md", "/home/u/junk/keep/c.md", "/old/d.md",
                 "/mnt/usb/e.md", "/home/u/f.TXT", "/home/u/g.tar.gz", "/home/u/p.jpg", "/home/u/x_y.md"]
        kinds = {"jpg": "image"}
        gen = store.begin_scan()
        store.upsert_scanned((ScannedFile(p, kinds.get(p.rsplit(".", 1)[1], "text"), 1, 1.0, False,
                                          Score(1, {})) for p in files), gen)

        def left() -> list[str]:
            return sorted(r[0] for r in store.db.execute("SELECT path FROM files"))

        # Nothing configured to change: nothing removed.
        assert store.forget_unconfigured([Path("/")]) == 0
        # /old is no longer a root; /mnt is kept (include_mounts, the drive unplugged).
        assert store.forget_unconfigured([Path("/home/u"), Path("/mnt")]) == 1
        assert "/old/d.md" not in left() and "/mnt/usb/e.md" in left()
        # An excluded path below a root goes, unless a root lies inside it (walked on its own).
        assert store.forget_unconfigured([Path("/home/u"), Path("/mnt"), Path("/home/u/junk/keep")],
                                         [Path("/home/u/junk")]) == 0
        assert store.forget_unconfigured([Path("/home/u"), Path("/mnt")], [Path("/home/u/junk")]) == 2
        # Kinds and extensions (after the last dot, any case; "_" is not a wildcard).
        assert store.forget_unconfigured([Path("/")], kinds=["text"], exclude_extensions=["txt", "tar.gz", "y.md"]) == 2
        assert left() == ["/home/u/a.md", "/home/u/g.tar.gz", "/home/u/x_y.md", "/mnt/usb/e.md"]
        assert store.forget_unconfigured([Path("/")], exclude_extensions=["m_", "%"]) == 0
        assert store.forget_unconfigured([Path("/")], exclude_extensions=["md"]) == 3
        assert left() == ["/home/u/g.tar.gz"]
        assert store.forget_unconfigured([Path("/")], max_bytes={"text": 0}) == 1
        assert left() == []
    finally:
        store.close()


@needs_bus
def test_reload_config_over_dbus(home: Path) -> None:
    async def main() -> None:
        name = test_bus_name()
        cfg = start_cfg(home)
        d = make_daemon(cfg, bus_name=name)
        stop = asyncio.Event()
        task = asyncio.ensure_future(serve(d, stop_event=stop))
        try:
            await wait_until(lambda: d._running, what="daemon running")
            await until_idle(d)
            assert await control_call("ReloadConfig", bus_name=name) == []
            settings = base_settings(home)
            settings["model"] = {"query_device": "cpu"}
            settings["daemon"]["krunner_max_results"] = 5
            write_config(home, settings)
            assert await asyncio.to_thread(daemon_reload_config, bus_name=name) == ["model.query_device"]
            await wait_until(lambda: d.cfg.krunner_max_results == 5, what="applied")
            (home / "config.toml").write_text("[scan\n", encoding="utf-8")
            from dbus_fast import DBusError

            with pytest.raises(DBusError) as err:
                await control_call("ReloadConfig", bus_name=name)
            assert err.value.type == dbus_names.CONFIG_ERROR and "not valid TOML" in err.value.text
            assert d.cfg.krunner_max_results == 5
        finally:
            stop.set()
            assert await asyncio.wait_for(task, 10) == 0
        assert await asyncio.to_thread(daemon_reload_config, bus_name=name) is None
        with pytest.raises(DaemonNotRunning):
            await control_call("ReloadConfig", bus_name=name)

    asyncio.run(main())
