"""Daemon tests: no GPU, fake embedder, stub watcher/GPU monitor.

D-Bus tests use the user's session bus with a throwaway bus name per test (never the real
`io.github.Semma`); they are skipped when there is no session bus.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder, FakeExtractor

from semma import daemon as daemon_mod
from semma.config import Config, load_config
from semma.daemon import (Daemon, DaemonNotRunning, control_call, daemon_failed, daemon_status, partial_scan_roots,
                          serve)
from semma.store import Store

DOCS = {
    "cats.md": "the cat sleeps on the warm sofa all afternoon",
    "rockets.md": "rocket engines burn liquid oxygen and kerosene during launch",
    "bread.txt": "knead the dough then bake the sourdough bread in a hot oven",
}

needs_bus = pytest.mark.skipif(not os.environ.get("DBUS_SESSION_BUS_ADDRESS"), reason="no session bus")
_names = itertools.count()


def test_bus_name() -> str:
    return f"io.github.Semma.Test{os.getpid()}_{next(_names)}"


test_bus_name.__test__ = False  # type: ignore[attr-defined]


class LifecycleEmbedder(FakeEmbedder):
    """FakeEmbedder with the real Embedder's lifecycle: separate query (text-only) and index
    (full) models, reuse of the index model for queries, and load/unload counts."""

    def __init__(self, load_seconds: float = 0.0) -> None:
        super().__init__()
        self.load_seconds = load_seconds
        self.load_calls = 0
        self.query_load_calls = 0
        self.index_load_calls = 0
        self.loads = 0
        self.query_loads = 0
        self.index_loads = 0
        self.unloads = 0
        self.query_unloads = 0
        self.index_unloads = 0
        self.queries: list[str] = []
        self.last_used = 0.0
        self.query_last_used = 0.0
        self.index_last_used = 0.0
        self.loading = False
        self.query_device = "cuda"
        self._query_loaded = False
        self._index_loaded = False
        self._lock = threading.Lock()

    @property
    def query_model_loaded(self) -> bool:
        return self._query_loaded

    @property
    def index_model_loaded(self) -> bool:
        return self._index_loaded

    @property
    def loaded(self) -> bool:
        return self._query_loaded or self._index_loaded

    @property
    def query_ready(self) -> bool:
        return self._query_loaded or self._index_loaded

    def load(self) -> None:
        self.load_calls += 1
        self.index_load_calls += 1
        self._load_index()

    def load_query(self) -> None:
        self.load_calls += 1
        self.query_load_calls += 1
        self._load_query()

    def _load_index(self) -> None:
        with self._lock:
            if self._index_loaded:
                return
            self.loading = True
            time.sleep(self.load_seconds)
            self.loads += 1
            self.index_loads += 1
            self._index_loaded = True
            if self._query_loaded and self.query_device.startswith(("cuda", "vulkan", "gpu")):
                self._query_loaded = False
            self.loading = False
            now = time.monotonic()
            self.last_used = now
            self.index_last_used = now

    def _load_query(self) -> None:
        with self._lock:
            if self._index_loaded or self._query_loaded:
                return
            self.loading = True
            time.sleep(self.load_seconds)
            self.loads += 1
            self.query_loads += 1
            self._query_loaded = True
            self.loading = False
            now = time.monotonic()
            self.last_used = now
            self.query_last_used = now

    def unload_index(self) -> bool:
        with self._lock:
            if not self._index_loaded:
                return False
            self._index_loaded = False
            self.unloads += 1
            self.index_unloads += 1
            now = time.monotonic()
            self.last_used = now
            self.index_last_used = now
            return True

    def unload_query(self) -> bool:
        with self._lock:
            if not self._query_loaded:
                return False
            self._query_loaded = False
            self.unloads += 1
            self.query_unloads += 1
            now = time.monotonic()
            self.last_used = now
            self.query_last_used = now
            return True

    def unload(self) -> bool:
        u_i = self.unload_index()
        u_q = self.unload_query()
        return u_i or u_q

    def embed_query(self, query: str):
        self._load_query()
        self.queries.append(query)
        now = time.monotonic()
        self.last_used = now
        self.query_last_used = now
        return super().embed_query(query)

    def embed_batch(self, modality, contents, titles):
        self._load_index()
        now = time.monotonic()
        self.last_used = now
        self.index_last_used = now
        return super().embed_batch(modality, contents, titles)


class StubGpu:
    def __init__(self, busy: bool = False) -> None:
        self.is_busy = busy
        self.calls = 0

    def busy(self, threshold: float | None = None) -> bool:
        self.calls += 1
        return self.is_busy


class StubWatcher:
    instances: list[StubWatcher] = []

    def __init__(self, on_change: Callable[[set[str]], None], debounce: float) -> None:
        self.on_change = on_change
        self.debounce = debounce
        self.dirs: set[str] = set()
        self.started = self.stopped = False
        StubWatcher.instances.append(self)

    def set_dirs(self, dirs) -> None:
        self.dirs = set(dirs)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def make_cfg(home: Path) -> Config:
    cfg = load_config(path=home / "nonexistent.toml", home=home)
    cfg.include_mounts = False
    cfg.db_path = home / "db" / "index.db"
    cfg.roots = [home / "Documents"]
    cfg.locations = {home / "Documents": 40}
    # Tiny indexes: with n vectors no z-score can exceed sqrt(n - 1), so the default gate (2.0)
    # and floor (4.5) are out of reach for 3-5 files.
    cfg.z_gate = 1.0
    cfg.z_floor = 0.0
    cfg.query_debounce_ms = 0
    cfg.gpu_check_seconds = 0.05
    cfg.model_idle_unload_minutes = 0  # tests that need it set it
    cfg.index_model_unload_seconds = 0  # tests that need it set it
    cfg.canonicalize()
    return cfg


def write_docs(home: Path, docs: dict[str, str], sub: str = "") -> None:
    for name, text in docs.items():
        p = home / "Documents" / sub / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def make_daemon(cfg: Config, *, embedder: Any = None, extractor: FakeExtractor | None = None,
                gpu: StubGpu | None = None, watcher: bool = True, **kw: Any) -> Daemon:
    kw.setdefault("runner_factory", None)
    kw.setdefault("gnome_factory", None)
    d = Daemon(cfg, embedder=embedder or LifecycleEmbedder(), extract_fn=extractor or FakeExtractor(),
               gpu_monitor=gpu, watcher_factory=StubWatcher if watcher else None, **kw)
    d.gpu_busy_recheck_s = 0.05
    return d


async def wait_until(pred: Callable[[], bool], timeout: float = 5.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


def counts(store_path: Path) -> dict[str, int]:
    s = Store(store_path)
    try:
        return {r[0]: r[1] for r in s.db.execute("SELECT status, count(*) FROM files GROUP BY status")}
    finally:
        s.close()


async def until_idle(d: Daemon, timeout: float = 5.0) -> None:
    async def done() -> bool:
        st = await d.status()
        return st["state"] == "idle" and st["pending"] == 0 and st["indexed"] > 0

    deadline = time.monotonic() + timeout
    while not await done():
        if time.monotonic() > deadline:
            raise AssertionError(f"daemon not idle: {await d.status()}")
        await asyncio.sleep(0.02)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    write_docs(tmp_path, DOCS)
    return tmp_path


# -- queries -------------------------------------------------------------------------------

def test_search_after_startup_indexing(home: Path) -> None:
    async def main() -> None:
        d = make_daemon(make_cfg(home))
        d.start()
        try:
            await until_idle(d)
            res = await d.search("rocket launch")
            assert res and Path(res[0].path).name == "rockets.md"
            assert await d.search("ro") == []  # too short
        finally:
            await d.stop()

    asyncio.run(main())


def test_stale_query_is_dropped_during_debounce(home: Path) -> None:
    async def main() -> None:
        cfg = make_cfg(home)
        cfg.query_debounce_ms = 80
        emb = LifecycleEmbedder()
        d = make_daemon(cfg, embedder=emb)
        d.start()
        try:
            await until_idle(d)
            emb.queries.clear()
            t0 = time.monotonic()
            first = asyncio.ensure_future(d.search("cats sofa"))
            await asyncio.sleep(0.02)
            second = asyncio.ensure_future(d.search("rocket launch"))
            r1 = await first
            stale_after = time.monotonic() - t0
            r2 = await second
            assert r1 == []
            assert stale_after < 0.07  # returned when the newer query arrived, not after the debounce
            assert r2 and Path(r2[0].path).name == "rockets.md"
            assert emb.queries == ["rocket launch"]  # the stale query never reached the model
        finally:
            await d.stop()

    asyncio.run(main())


def test_stale_query_returns_immediately_while_model_loads(home: Path) -> None:
    async def main() -> None:
        emb = LifecycleEmbedder()
        d = make_daemon(make_cfg(home), embedder=emb)
        d.start()
        try:
            await until_idle(d)
            emb.unload()
            emb.loads = 0
            emb.load_seconds = 0.4
            t0 = time.monotonic()
            first = asyncio.ensure_future(d.search("cats sofa"))
            await asyncio.sleep(0.05)
            second = asyncio.ensure_future(d.search("bread oven"))
            assert await first == []
            assert time.monotonic() - t0 < 0.3
            r2 = await second
            assert r2 and Path(r2[0].path).name == "bread.txt"
            assert emb.loads == 1
        finally:
            await d.stop()

    asyncio.run(main())


def test_concurrent_queries_share_one_model_load(home: Path) -> None:
    async def main() -> None:
        emb = LifecycleEmbedder()
        d = make_daemon(make_cfg(home), embedder=emb)
        d.start()
        try:
            await until_idle(d)
            emb.unload()
            emb.load_calls = 0
            emb.load_seconds = 0.2
            await asyncio.gather(*(d.ensure_model() for _ in range(5)))
            assert emb.load_calls == 1 and emb.loaded
            emb.unload()
            emb.load_calls = 0
            results = await asyncio.gather(*(d.search(q) for q in ("cat sofa", "cats sofa warm", "rocket launch")))
            assert emb.load_calls == 1
            assert results[0] == [] and results[1] == []
            assert Path(results[2][0].path).name == "rockets.md"
        finally:
            await d.stop()

    asyncio.run(main())


def test_query_failure_returns_empty_and_daemon_survives(home: Path) -> None:
    async def main() -> None:
        emb = LifecycleEmbedder()
        d = make_daemon(make_cfg(home), embedder=emb)
        d.start()
        try:
            await until_idle(d)
            original = emb.embed_query
            emb.embed_query = lambda q: (_ for _ in ()).throw(RuntimeError("CUDA error"))  # type: ignore[method-assign]
            assert await d.search("rocket launch") == []
            emb.embed_query = original  # type: ignore[method-assign]
            assert await d.search("rocket launch")
        finally:
            await d.stop()

    asyncio.run(main())


# -- idle unload ---------------------------------------------------------------------------

def test_idle_unload_and_reload(home: Path) -> None:
    async def main() -> None:
        cfg = make_cfg(home)
        cfg.model_idle_unload_minutes = 0.3 / 60  # 0.3 s
        cfg.index_model_unload_seconds = 0.1  # 0.1 s
        emb = LifecycleEmbedder()
        d = make_daemon(cfg, embedder=emb)
        d.start()
        try:
            await until_idle(d)
            assert await d.search("rocket launch")
            assert emb.loaded
            await wait_until(lambda: not emb.loaded, 3, "idle unload")
            assert emb.index_unloads == 1
            st = await d.status()
            assert st["model_loaded"] is False
            assert st["query_model_loaded"] is False
            assert st["index_model_loaded"] is False
            res = await d.search("bread oven")
            assert res and Path(res[0].path).name == "bread.txt"
            # After the full index model was unloaded, the query loaded the text-only query model.
            assert emb.query_model_loaded and not emb.index_model_loaded
            assert emb.query_loads == 1 and emb.index_loads == 1
            st2 = await d.status()
            assert st2["model_loaded"] is True
            assert st2["query_model_loaded"] is True
            assert st2["index_model_loaded"] is False
            await wait_until(lambda: not emb.loaded, 3, "query model idle unload")
            assert emb.query_unloads == 1
        finally:
            await d.stop()

    asyncio.run(main())


def test_index_model_unloaded_after_grace_period_and_query_reuses_while_loaded(home: Path) -> None:
    async def main() -> None:
        cfg = make_cfg(home)
        cfg.index_model_unload_seconds = 0.25
        cfg.model_idle_unload_minutes = 10  # long: only index grace period fires
        emb = LifecycleEmbedder()
        d = make_daemon(cfg, embedder=emb)
        d.start()
        try:
            await until_idle(d)
            # Right after indexing finishes, the full model is still loaded (within grace period).
            assert emb.index_model_loaded and not emb.query_model_loaded
            # A query arriving while the full model is loaded reuses it instead of loading the query model.
            res = await d.search("rocket launch")
            assert res and Path(res[0].path).name == "rockets.md"
            assert emb.query_loads == 0 and emb.index_loads == 1
            assert emb.index_model_loaded and not emb.query_model_loaded

            # After the grace period expires, the full index model is unloaded.
            await wait_until(lambda: not emb.index_model_loaded, 3, "index model grace-period unload")
            assert emb.index_unloads == 1
            assert not emb.loaded

            # Now a query loads only the text-only query model, which stays loaded (10 min idle).
            res2 = await d.search("cats sofa")
            assert res2 and Path(res2[0].path).name == "cats.md"
            assert emb.query_model_loaded and not emb.index_model_loaded
            assert emb.query_loads == 1 and emb.index_loads == 1
        finally:
            await d.stop()

    asyncio.run(main())


def test_no_unload_while_queries_keep_coming(home: Path) -> None:
    async def main() -> None:
        cfg = make_cfg(home)
        cfg.model_idle_unload_minutes = 0.3 / 60
        cfg.index_model_unload_seconds = 0.05
        emb = LifecycleEmbedder()
        d = make_daemon(cfg, embedder=emb)
        d.start()
        try:
            await until_idle(d)
            await wait_until(lambda: not emb.index_model_loaded, 2, "index unload")
            assert await d.search("rocket launch")
            assert emb.query_model_loaded and emb.query_unloads == 0
            for _ in range(8):
                await d.search("rocket launch")
                await asyncio.sleep(0.1)
            assert emb.query_unloads == 0 and emb.query_model_loaded
        finally:
            await d.stop()

    asyncio.run(main())


@pytest.mark.parametrize(("query_device", "expect_unload"), [
    ("vulkan", True),
    ("vulkan:0", True),
    ("gpu", True),
    ("cuda", True),
    ("cpu", False),
])
def test_gpu_busy_unloads_gpu_query_device_only(
    home: Path, monkeypatch: pytest.MonkeyPatch, query_device: str, expect_unload: bool
) -> None:
    async def main() -> None:
        monkeypatch.setattr(daemon_mod, "GPU_BUSY_UNLOAD_S", 0.05)
        cfg = make_cfg(home)
        cfg.model_idle_unload_minutes = 10  # long normal idle; only GPU_BUSY_UNLOAD_S fires
        cfg.index_model_unload_seconds = 0
        emb = LifecycleEmbedder()
        emb.query_device = query_device
        d = make_daemon(cfg, embedder=emb)
        emb._load_query()
        d._gpu_busy = True
        d._last_query = time.monotonic() - 1.0
        emb.query_last_used = d._last_query
        unloaded = await d.check_idle()
        assert unloaded is expect_unload
        assert emb.query_model_loaded is (not expect_unload)

    asyncio.run(main())


def test_config_vulkan_query_device_and_llama_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)

    for dev in ("vulkan", "vulkan:0", "gpu", "gpu:0", "cuda", "cuda:0", "cpu", "auto"):
        conf = tmp_path / "c.toml"
        conf.write_text(f'[model]\nquery_device = "{dev}"\n', encoding="utf-8")
        cfg = load_config(conf, home=tmp_path)
        assert cfg.query_device == dev
        assert capsys.readouterr().err == ""

    conf = tmp_path / "paths.toml"
    conf.write_text(
        '[model]\n'
        'device = "vulkan"\n'
        'llama_lib_dir = "~/llama-libs"\n'
        'model_dir = "$HOME/link/models"\n'
        '[daemon]\n'
        'query_device = "VULKAN:0"\n',
        encoding="utf-8",
    )
    cfg = load_config(conf, home=tmp_path)
    assert capsys.readouterr().err == ""
    assert cfg.device == "vulkan"
    assert cfg.query_device == "vulkan:0"
    assert cfg.llama_lib_dir == (tmp_path / "llama-libs").resolve()
    assert cfg.model_dir == real_dir.resolve() / "models"

    for bad_spec in ("tpu", "vulkan:", "vulkan:abc", "gpu:-1"):
        bad = tmp_path / "bad.toml"
        bad.write_text(f'[model]\ndevice = "{bad_spec}"\n[daemon]\nquery_device = "{bad_spec}"\n', encoding="utf-8")
        cfg_bad = load_config(bad, home=tmp_path)
        assert cfg_bad.device is None
        assert cfg_bad.query_device == "auto"
        err = capsys.readouterr().err
        assert "model.device" in err and "query_device" in err

    auto_conf = tmp_path / "auto.toml"
    auto_conf.write_text('[model]\ndevice = " AUTO "\n', encoding="utf-8")
    cfg_auto = load_config(auto_conf, home=tmp_path)
    assert cfg_auto.device is None
    assert capsys.readouterr().err == ""


# -- GPU pause -----------------------------------------------------------------------------

def test_indexing_waits_while_gpu_busy_then_resumes(home: Path) -> None:
    async def main() -> None:
        gpu = StubGpu(busy=True)
        d = make_daemon(make_cfg(home), gpu=gpu)
        d.start()
        try:
            await wait_until(lambda: d._phase == "paused", what="GPU pause")
            st = await d.status()
            assert st["paused"] and "GPU" in st["pause_reason"] and st["pending"] == 3 and st["indexed"] == 0
            calls = gpu.calls
            await asyncio.sleep(0.3)
            assert gpu.calls > calls  # re-checked while paused
            assert (await d.status())["indexed"] == 0
            gpu.is_busy = False
            await until_idle(d)
        finally:
            await d.stop()

    asyncio.run(main())


def test_gpu_busy_pauses_a_running_slice(tmp_path: Path) -> None:
    async def main() -> None:
        write_docs(tmp_path, {f"doc{i:02d}.md": f"note number {i} about gardening" for i in range(30)})
        gpu = StubGpu()
        ext = FakeExtractor(on_chunk=lambda p, i: time.sleep(0.03))
        d = make_daemon(make_cfg(tmp_path), gpu=gpu, extractor=ext)
        d.start()
        try:
            await wait_until(lambda: d._phase == "indexing", what="indexing")
            await asyncio.sleep(0.15)
            gpu.is_busy = True
            await wait_until(lambda: d._phase == "paused", 3, "pause")
            n = (await d.status())["indexed"]
            assert 0 < n < 30
            await asyncio.sleep(0.3)
            assert (await d.status())["indexed"] == n
            gpu.is_busy = False
            await until_idle(d, 10)
            assert (await d.status())["indexed"] == 30
        finally:
            await d.stop()

    asyncio.run(main())


def test_user_pause_and_resume(tmp_path: Path) -> None:
    async def main() -> None:
        write_docs(tmp_path, {f"doc{i:02d}.md": f"note number {i} about gardening" for i in range(20)})
        ext = FakeExtractor(on_chunk=lambda p, i: time.sleep(0.03))
        d = make_daemon(make_cfg(tmp_path), extractor=ext)
        d.start()
        try:
            await wait_until(lambda: d._phase == "indexing", what="indexing")
            await wait_until(lambda: counts(d.cfg.db_path).get("indexed", 0) > 0, what="first file")
            d.pause()
            await wait_until(lambda: d._phase == "paused", what="pause")
            st = await d.status()
            assert st["paused"] and st["state"] == "paused" and st["pause_reason"] == "paused by request"
            n = st["indexed"]
            await asyncio.sleep(0.2)
            assert (await d.status())["indexed"] == n
            assert await d.search("gardening note")  # search still works while paused
            d.resume()
            await until_idle(d)
        finally:
            await d.stop()

    asyncio.run(main())


@pytest.mark.parametrize("why", ["gpu_busy", "user"])
def test_index_model_unloaded_while_indexing_is_paused_with_pending_files(tmp_path: Path, why: str) -> None:
    async def main() -> None:
        write_docs(tmp_path, {f"doc{i:02d}.md": f"note number {i} about gardening" for i in range(30)})
        cfg = make_cfg(tmp_path)
        cfg.index_model_unload_seconds = 0.2
        gpu = StubGpu()
        emb = LifecycleEmbedder()
        ext = FakeExtractor(on_chunk=lambda p, i: time.sleep(0.03))
        d = make_daemon(cfg, embedder=emb, gpu=gpu, extractor=ext)
        d.start()
        try:
            await wait_until(lambda: emb.index_model_loaded, what="index model loaded")
            if why == "gpu_busy":
                gpu.is_busy = True  # a game started: its VRAM matters more than our pending files
            else:
                d.pause()
            await wait_until(lambda: d._phase == "paused", 3, "pause")
            assert (await d.status())["pending"] > 0
            await wait_until(lambda: not emb.index_model_loaded, 3, "index model unload while paused")
            assert (await d.status())["index_model_loaded"] is False
            if why == "gpu_busy":
                gpu.is_busy = False
            else:
                d.resume()
            await until_idle(d, 10)
            assert (await d.status())["indexed"] == 30
        finally:
            await d.stop()

    asyncio.run(main())


# -- watcher -------------------------------------------------------------------------------

def test_watcher_change_rescans_and_indexes_new_file(home: Path) -> None:
    async def main() -> None:
        StubWatcher.instances.clear()
        cfg = make_cfg(home)
        d = make_daemon(cfg)
        d.start()
        try:
            await until_idle(d)
            w = StubWatcher.instances[-1]
            docs = str(home / "Documents")
            assert w.started and docs in w.dirs
            write_docs(home, {"volcano.md": "lava erupts from the volcano crater"}, sub="geology")
            w.on_change({docs, os.path.join(docs, "geology")})
            await wait_until(lambda: counts(cfg.db_path).get("indexed", 0) == 4, what="new file indexed")
            res = await d.search("volcano lava")
            assert res and Path(res[0].path).name == "volcano.md"
            assert os.path.join(docs, "geology") in w.dirs and docs in w.dirs  # partial scan kept the rest
            # A deleted directory is rescanned from its parent and its files disappear.
            (home / "Documents" / "geology" / "volcano.md").unlink()
            (home / "Documents" / "geology").rmdir()
            w.on_change({os.path.join(docs, "geology")})
            await wait_until(lambda: sum(counts(cfg.db_path).values()) == 3, what="deleted file removed")
            await wait_until(lambda: os.path.join(docs, "geology") not in w.dirs, what="watch removed")
            w.on_change({"*"})  # overflow: full rescan
            await until_idle(d)
        finally:
            await d.stop()
        assert StubWatcher.instances[-1].stopped

    asyncio.run(main())


def test_partial_scan_roots(tmp_path: Path) -> None:
    write_docs(tmp_path, {"a.md": "x"}, sub="notes/deep")
    write_docs(tmp_path, {"b.md": "x"}, sub="proj/src")
    (tmp_path / "Documents" / "proj" / ".git").mkdir()
    (tmp_path / "Documents" / ".hidden").mkdir()
    (tmp_path / "Documents" / "old" / "Windows").mkdir(parents=True)
    cfg = make_cfg(tmp_path)
    docs = tmp_path / "Documents"
    roots = cfg.effective_roots()
    assert partial_scan_roots(cfg, [str(docs / "notes" / "deep")], roots) == [docs / "notes" / "deep"]
    # nested dirs collapse; a vanished dir becomes its parent
    assert partial_scan_roots(cfg, [str(docs / "notes" / "deep"), str(docs / "notes" / "gone")], roots) == [docs / "notes"]
    # hidden or outside every root: nothing to scan
    assert partial_scan_roots(cfg, [str(docs / ".hidden"), "/nonexistent/x"], roots) == []
    # inside a project, or a child the scanner would treat as an OS folder: full scan
    assert partial_scan_roots(cfg, [str(docs / "proj" / "src")], roots) is None
    assert partial_scan_roots(cfg, [str(docs / "old")], roots) is None
    assert partial_scan_roots(cfg, [str(docs)], roots) == [docs]


# -- shutdown ------------------------------------------------------------------------------

def test_graceful_stop_mid_indexing_leaves_db_consistent(tmp_path: Path) -> None:
    async def main() -> Config:
        write_docs(tmp_path, {f"doc{i:02d}.md": f"note {i} about birds" for i in range(40)})
        ext = FakeExtractor({f"doc{i:02d}.md": [f"part {j} of note {i} about birds" for j in range(3)]
                             for i in range(40)}, on_chunk=lambda p, i: time.sleep(0.01))
        cfg = make_cfg(tmp_path)
        d = make_daemon(cfg, extractor=ext)
        d.start()
        await wait_until(lambda: d._phase == "indexing", what="indexing")
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        await d.stop()
        assert time.monotonic() - t0 < 3
        assert not d._worker.is_alive()
        return cfg

    cfg = asyncio.run(main())
    s = Store(cfg.db_path)
    try:
        db = s.db
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        st = dict(db.execute("SELECT status, count(*) FROM files GROUP BY status").fetchall())
        assert 0 < st.get("indexed", 0) < 40 and st.get("pending", 0) > 0
        chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        vecs = db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
        fts = db.execute("SELECT count(*) FROM fts_chunks_docsize").fetchone()[0]
        assert chunks == vecs == fts == 3 * st["indexed"]  # every indexed file complete, nothing partial
        partial = db.execute("SELECT count(*) FROM files f WHERE status = 'indexed' AND "
                             "(SELECT count(*) FROM chunks c WHERE c.file_id = f.id) != 3").fetchone()[0]
        assert partial == 0
    finally:
        s.close()


def test_worker_survives_errors(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def main() -> None:
        monkeypatch.setattr(daemon_mod, "WORKER_BACKOFF_S", (0.05, 0.1))
        d = make_daemon(make_cfg(home))
        real = d.indexer.count_pending
        fails = {"n": 2}

        def flaky(*a: Any, **k: Any) -> int:
            if fails["n"]:
                fails["n"] -= 1
                raise RuntimeError("disk on fire")
            return real(*a, **k)

        d.indexer.count_pending = flaky  # type: ignore[method-assign]
        d.start()
        try:
            await until_idle(d)
        finally:
            await d.stop()

    asyncio.run(main())


def test_indexing_slices_do_not_leak_connections(tmp_path: Path) -> None:
    write_docs(tmp_path, {f"doc{i}.md": f"text {i}" for i in range(5)})
    cfg = make_cfg(tmp_path)
    from semma.indexer import Indexer
    from semma.scanner import Scanner

    store = Store(cfg.db_path)
    try:
        for _ in range(5):
            gen = store.begin_scan()
            sc = Scanner(cfg)
            store.upsert_scanned(sc.scan(), gen)
            store.requeue(("indexed",))
            Indexer(store, cfg, FakeEmbedder(), extract_fn=FakeExtractor()).run()
        assert len(store._conns) <= 2
    finally:
        store.close()


# -- status details (what a status window polls every second) ------------------------------

def test_status_counts_per_kind_and_failed_files(home: Path) -> None:
    async def main() -> None:
        (home / "Documents" / "report.pdf").write_text("quarterly figures", encoding="utf-8")
        ext = FakeExtractor({"report.pdf": RuntimeError("broken pdf"), "bread.txt": RuntimeError("bad text")})
        d = make_daemon(make_cfg(home), extractor=ext)
        d.start()
        try:
            await until_idle(d)
            st = await d.status()
            kinds = set(daemon_mod.ft.KINDS)
            for key in ("pending_by_kind", "indexed_by_kind", "failed_by_kind"):
                assert set(st[key]) == kinds
            assert st["indexed_by_kind"]["text"] == 2 and st["failed_by_kind"] == {
                **dict.fromkeys(kinds, 0), "text": 1, "pdf": 1}
            assert sum(st["failed_by_kind"].values()) == st["failed"] == 2
            assert sum(st["indexed_by_kind"].values()) == st["indexed"]
            failed = dict(await d.failed())
            assert set(failed) == {str(home / "Documents" / "report.pdf"), str(home / "Documents" / "bread.txt")}
            assert "broken pdf" in failed[str(home / "Documents" / "report.pdf")]
            assert len(await d.failed(1)) == 1
            # The per-kind counts read only the (status, kind) index, not the table.
            plan = " ".join(r[3] for r in d.store.reader().execute(
                "EXPLAIN QUERY PLAN SELECT status, kind, count(*) FROM files GROUP BY status, kind"))
            assert "files_status_kind" in plan, plan
        finally:
            await d.stop()

    asyncio.run(main())


def test_status_does_not_wait_for_the_search_thread(home: Path) -> None:
    """A query can sit on the search thread for minutes (the model downloading); Status must not."""
    async def main() -> None:
        d = make_daemon(make_cfg(home))
        release = threading.Event()
        blocked = d._search_exec.submit(release.wait, 10)
        try:
            st = await asyncio.wait_for(d.status(), 2)
            assert st["state"] == "idle"
            assert await asyncio.wait_for(d.failed(), 2) == []
        finally:
            release.set()
            blocked.result()
            await d.stop()

    asyncio.run(main())


def test_status_reports_indexing_run_progress(tmp_path: Path) -> None:
    async def main() -> None:
        write_docs(tmp_path, {f"doc{i:02d}.md": f"note number {i} about gardening" for i in range(20)})
        ext = FakeExtractor({"doc05.md": RuntimeError("unreadable")}, on_chunk=lambda p, i: time.sleep(0.03))
        d = make_daemon(make_cfg(tmp_path), extractor=ext)
        d.start()
        try:
            await wait_until(lambda: d._phase == "indexing", what="indexing")
            st = await d.status()
            assert st["state"] == "indexing" and st["run_total"] == 20
            await wait_until(lambda: d._run_progress[0] >= 8, what="files done")
            st = await d.status()
            assert st["run_total"] == 20 and 8 <= st["run_done"] < 20
            assert st["run_rate"] > 0
            await until_idle(d, 10)
            st = await d.status()
            assert (st["run_done"], st["run_total"], st["run_failed"], st["run_rate"]) == (0, 0, 0, 0.0)
        finally:
            await d.stop()
        assert counts(d.cfg.db_path) == {"indexed": 19, "failed": 1}

    asyncio.run(main())


def test_status_run_progress_counts_failures(tmp_path: Path) -> None:
    async def main() -> None:
        write_docs(tmp_path, {f"doc{i:02d}.md": f"note number {i}" for i in range(6)})
        seen: list[tuple[int, int, int, float]] = []
        ext = FakeExtractor({"doc00.md": RuntimeError("x"), "doc01.md": RuntimeError("y")})
        d = make_daemon(make_cfg(tmp_path), extractor=ext)
        orig = d._on_progress

        def spy(st: Any) -> None:
            orig(st)
            seen.append(d._run_progress)

        d._on_progress = spy  # type: ignore[method-assign]
        d.start()
        try:
            await until_idle(d)
        finally:
            await d.stop()
        assert seen[-1][:3] == (6, 6, 2)

    asyncio.run(main())


class SlowScanner(daemon_mod.Scanner):
    def scan(self, roots=None, now=None):
        for f in super().scan(roots, now):
            time.sleep(0.05)  # the walk is suspended here, inside f's directory
            yield f


def test_status_reports_scan_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def main() -> None:
        write_docs(tmp_path, {f"a{i}.md": "alpha" for i in range(5)}, sub="one")
        write_docs(tmp_path, {f"b{i}.md": "beta" for i in range(5)}, sub="two")
        (tmp_path / "Documents" / "two" / "notes.bin").write_bytes(b"x")  # seen, not indexable
        monkeypatch.setattr(daemon_mod, "Scanner", SlowScanner)
        d = make_daemon(make_cfg(tmp_path))
        d.start()
        try:
            await wait_until(lambda: d._scanner is not None and d._scanner.stats.matched >= 2, what="scanning")
            st = await d.status()
            assert st["state"] == "scanning"
            assert Path(st["scan_dir"]).parent == tmp_path / "Documents"
            assert st["scan_dirs"] >= 2 and st["scan_files"] >= st["scan_matched"] >= 2
            await until_idle(d, 10)
            st = await d.status()
            assert (st["scan_dir"], st["scan_dirs"], st["scan_files"], st["scan_matched"]) == ("", 0, 0, 0)
        finally:
            await d.stop()

    asyncio.run(main())


class DownloadingEmbedder(LifecycleEmbedder):
    def __init__(self, model_dir: Path) -> None:
        super().__init__()
        self.model_dir = model_dir

    def download_progress(self):
        from semma import embedder as emb

        return emb.download_progress(self.model_dir)


def test_status_reports_model_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    from semma import embedder as emb

    monkeypatch.setattr(emb, "GGUF_SPECS", {"text": ("t.gguf", 1000, "0" * 64), "mmproj": ("m.gguf", 3000, "0" * 64)})
    models = tmp_path / "models"
    models.mkdir()
    (models / "t.gguf.part").write_bytes(b"x" * 250)

    async def main() -> None:
        d = make_daemon(make_cfg(tmp_path), embedder=DownloadingEmbedder(models))
        try:
            st = await d.status()  # an interrupted download is not running
            assert st["state"] == "idle" and st["download_file"] == ""
            assert (st["download_done"], st["download_total"], st["download_all_total"]) == (0, 0, 0)
            with (models / "t.gguf.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                st = await d.status()
            assert st["state"] == "downloading" and st["download_file"] == "t.gguf"
            assert (st["download_done"], st["download_total"]) == (250, 1000)
            assert (st["download_all_done"], st["download_all_total"]) == (250, 4000)
            st = await d.status()
            assert st["state"] == "idle" and st["download_file"] == ""
            # A probe that fails only hides the download.
            d.embedder.download_progress = lambda: 1 / 0
            assert (await d.status())["download_file"] == ""
        finally:
            await d.stop()

    asyncio.run(main())


def test_embedder_download_progress_uses_its_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    from semma import embedder as emb

    monkeypatch.delenv("SEMMA_MODEL_DIR", raising=False)
    monkeypatch.setattr(emb, "GGUF_SPECS", {"text": ("t.gguf", 1000, "0" * 64)})
    cfg = make_cfg(tmp_path)
    cfg.model_dir = tmp_path / "models"
    cfg.model_dir.mkdir()
    (cfg.model_dir / "t.gguf.part").write_bytes(b"x" * 10)
    e = emb.Embedder(cfg)
    assert e.download_progress() is None
    with (cfg.model_dir / "t.gguf.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert e.download_progress() == {"file": "t.gguf", "done": 10, "total": 1000,
                                         "all_done": 10, "all_total": 1000}


# -- D-Bus (session bus, throwaway names) --------------------------------------------------

@needs_bus
def test_control_interface_over_dbus(home: Path) -> None:
    async def main() -> None:
        name = test_bus_name()
        cfg = make_cfg(home)
        ext = FakeExtractor({"bread.txt": RuntimeError("corrupt file")})
        d = make_daemon(cfg, extractor=ext, bus_name=name)
        stop = asyncio.Event()
        task = asyncio.ensure_future(serve(d, stop_event=stop))
        try:
            await wait_until(lambda: d._running, what="daemon running")
            await until_idle(d)
            st = await control_call("Status", bus_name=name)
            assert st["state"] == "idle" and st["indexed"] == 2 and st["failed"] == 1
            assert set(st) >= {"state", "model_loaded", "query_model_loaded", "index_model_loaded", "paused",
                               "pause_reason", "pending", "indexed", "failed", "chunks", "current_file",
                               "last_scan_at", "uptime"}
            assert st["last_scan_at"] > 0 and st["chunks"] == 2
            assert set(st) == set(daemon_mod.STATUS_SIGNATURES)
            assert st["indexed_by_kind"]["text"] == 2 and st["failed_by_kind"]["text"] == 1
            assert st["pending_by_kind"] == dict.fromkeys(daemon_mod.ft.KINDS, 0)
            assert st["download_file"] == "" and st["run_total"] == 0 and st["scan_dir"] == ""
            assert isinstance(st["run_rate"], float)
            bread = str(home / "Documents" / "bread.txt")
            failed = await control_call("Failed", 10, signature="u", bus_name=name)
            assert len(failed) == 1 and failed[0][0] == bread and "corrupt file" in failed[0][1]
            assert await control_call("Failed", 0, signature="u", bus_name=name) == failed
            rows = await asyncio.to_thread(daemon_failed, 5, bus_name=name)
            assert rows == [(bread, failed[0][1])]
            await control_call("Pause", bus_name=name)
            st = await control_call("Status", bus_name=name)
            assert st["paused"] is True
            await control_call("Resume", bus_name=name)
            ext.contents.clear()
            assert await control_call("RetryFailed", bus_name=name) == 1
            await wait_until(lambda: counts(cfg.db_path).get("indexed") == 3, what="retried file indexed")
            await control_call("Rescan", bus_name=name)
            await until_idle(d)
            # A second daemon cannot take the name.
            d2 = make_daemon(cfg, bus_name=name)
            assert await serve(d2) == 1
        finally:
            stop.set()
            assert await asyncio.wait_for(task, 10) == 0
        assert await asyncio.to_thread(daemon_status, bus_name=name) is None
        assert await asyncio.to_thread(daemon_failed, bus_name=name) is None
        with pytest.raises(DaemonNotRunning):
            await control_call("Status", bus_name=name)

    asyncio.run(main())


@needs_bus
def test_krunner_match_over_dbus(home: Path) -> None:
    pytest.importorskip("semma.krunner")
    from dbus_fast import Message, MessageType
    from dbus_fast.aio import MessageBus

    from semma import dbus_names
    from semma.krunner import KRunnerService

    async def main() -> None:
        name = test_bus_name()
        d = Daemon(make_cfg(home), embedder=LifecycleEmbedder(), extract_fn=FakeExtractor(), gpu_monitor=None,
                   watcher_factory=None, runner_factory=KRunnerService, bus_name=name)
        stop = asyncio.Event()
        task = asyncio.ensure_future(serve(d, stop_event=stop))
        bus = await MessageBus().connect()
        try:
            await wait_until(lambda: d._running, what="daemon running")
            await until_idle(d)
            reply = await bus.call(Message(destination=name, path=dbus_names.RUNNER_PATH,
                                           interface=dbus_names.RUNNER_INTERFACE, member="Match",
                                           signature="s", body=["rocket launch"]))
            assert reply.message_type == MessageType.METHOD_RETURN, reply.body
            matches = reply.body[0]
            assert matches and "rockets.md" in matches[0][1]
        finally:
            bus.disconnect()
            stop.set()
            assert await asyncio.wait_for(task, 10) == 0

    asyncio.run(main())


@needs_bus
@pytest.mark.parametrize("broken", [None, "runner", "gnome"])
def test_serve_exports_krunner_and_gnome_independently(home: Path, broken: str | None) -> None:
    from dbus_fast import Message, MessageType
    from dbus_fast.aio import MessageBus

    from semma import dbus_names
    from semma.gnome_search import GnomeSearchService
    from semma.krunner import KRunnerService

    def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("factory failed")

    async def main() -> None:
        name = test_bus_name()
        d = make_daemon(make_cfg(home), watcher=False, bus_name=name,
                        runner_factory=boom if broken == "runner" else KRunnerService,
                        gnome_factory=boom if broken == "gnome" else GnomeSearchService)
        stop = asyncio.Event()
        task = asyncio.ensure_future(serve(d, stop_event=stop))
        bus = await MessageBus().connect()
        try:
            await wait_until(lambda: d._running, what="daemon running")
            await until_idle(d)
            runner = await bus.call(Message(destination=name, path=dbus_names.RUNNER_PATH,
                                            interface=dbus_names.RUNNER_INTERFACE, member="Match",
                                            signature="s", body=["rocket launch"]))
            gnome = await bus.call(Message(destination=name, path=dbus_names.GNOME_PATH,
                                           interface=dbus_names.GNOME_INTERFACE, member="GetInitialResultSet",
                                           signature="as", body=[["rocket", "launch"]]))
            ok = {"runner": runner.message_type == MessageType.METHOD_RETURN,
                  "gnome": gnome.message_type == MessageType.METHOD_RETURN}
            assert ok == {"runner": broken != "runner", "gnome": broken != "gnome"}
            if ok["gnome"]:
                ids = gnome.body[0]
                assert ids and ids[0].endswith("rockets.md")
                metas = await bus.call(Message(destination=name, path=dbus_names.GNOME_PATH,
                                               interface=dbus_names.GNOME_INTERFACE, member="GetResultMetas",
                                               signature="as", body=[ids[:1]]))
                assert metas.body[0][0]["name"].value == "rockets.md"
            assert (await control_call("Status", bus_name=name))["state"]
        finally:
            bus.disconnect()
            stop.set()
            assert await asyncio.wait_for(task, 10) == 0

    asyncio.run(main())


@needs_bus
def test_sigterm_exits_cleanly(tmp_path: Path) -> None:
    """The real entry point: start, answer Status, exit 0 on SIGTERM and release the name."""
    name = test_bus_name()
    (tmp_path / "empty").mkdir()
    conf = tmp_path / "config.toml"
    conf.write_text(f'db_path = "{tmp_path}/db/index.db"\n[scan]\nroots = ["{tmp_path}/empty"]\n'
                    f"include_mounts = false\n", encoding="utf-8")
    code = ("import sys\nfrom pathlib import Path\nfrom semma.config import load_config\n"
            "from semma.daemon import run_daemon\n"
            f"cfg = load_config(path=Path({str(conf)!r}), home=Path({str(tmp_path)!r}))\n"
            f"sys.exit(run_daemon(cfg, bus_name={name!r}))\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        st = None
        while st is None and time.monotonic() < deadline and proc.poll() is None:
            st = daemon_status(bus_name=name, timeout=2)
            if st is None:
                time.sleep(0.1)
        assert st is not None, proc.stderr.read() if proc.poll() is not None else "no status"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(15) == 0
        err = proc.stderr.read()
        assert "shutting down" in err and "Traceback" not in err
        assert daemon_status(bus_name=name, timeout=2) is None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
