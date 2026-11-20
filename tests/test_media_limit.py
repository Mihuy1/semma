"""Changing the media limit / segment length brings already indexed audio and video in line."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmbedder

from semma.config import Config, load_config
from semma.extract.base import Chunk
from semma.indexer import Indexer
from semma.scanner import Scanner
from semma.store import Store

DURATIONS = {"long.mp3": 50 * 60, "short.mp3": 10 * 60, "film.mp4": 45 * 60}


class FakeMedia:
    """One chunk per segment up to min(duration, limit), like the real media extractors."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: Path, kind: str, cfg: Config) -> Iterator[Chunk]:
        self.calls.append(path.name)
        seg = cfg.media_segment_seconds
        end = min(DURATIONS[path.name], cfg.max_media_seconds)
        start = 0.0
        while start < end:
            yield Chunk("text", f"{path.stem} words at {int(start)}", f"{int(start)}", "", start)
            start += seg


@pytest.fixture
def env(tmp_path: Path):
    cfg = load_config(path=tmp_path / "nonexistent.toml", home=tmp_path)
    cfg.include_mounts = False
    cfg.db_path = tmp_path / "db" / "index.db"
    cfg.roots = [tmp_path / "Media"]
    cfg.max_media_seconds = 30 * 60
    cfg.media_segment_seconds = 60
    for name in DURATIONS:
        p = tmp_path / "Media" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * 4096)
    store = Store(cfg.db_path)
    scanner = Scanner(cfg)
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(), gen)
    store.finish_scan(gen, scanner.stats.roots)
    extractor = FakeMedia()
    ix = Indexer(store, cfg, FakeEmbedder(), extract_fn=extractor, prefetch=False)
    ix.run()
    assert sorted(extractor.calls) == sorted(DURATIONS)
    extractor.calls.clear()
    yield cfg, store, ix, extractor
    store.close()


def segments(store: Store) -> dict[str, int]:
    rows = store.db.execute("SELECT f.path, count(c.id) FROM files f LEFT JOIN chunks c ON c.file_id = f.id "
                            "GROUP BY f.id")
    return {Path(p).name: n for p, n in rows}


def pending(store: Store) -> set[str]:
    return {Path(r[0]).name for r in store.db.execute("SELECT path FROM files WHERE status = 'pending'")}


def test_built_with_settings_are_recorded(env) -> None:
    _, store, _, _ = env
    assert segments(store) == {"long.mp3": 30, "short.mp3": 10, "film.mp4": 30}
    assert store.get_meta("media_max_seconds") == "1800.0"
    assert store.get_meta("media_segment_seconds") == "60.0"


def test_longer_limit_reindexes_only_files_cut_off(env) -> None:
    cfg, store, ix, extractor = env
    cfg.max_media_seconds = 60 * 60
    assert ix.reconcile_media() == 2
    assert pending(store) == {"long.mp3", "film.mp4"}
    assert segments(store)["long.mp3"] == 30
    ix.run()
    assert sorted(extractor.calls) == ["film.mp4", "long.mp3"]
    assert segments(store) == {"long.mp3": 50, "short.mp3": 10, "film.mp4": 45}
    assert ix.reconcile_media() == 0 and pending(store) == set()


def test_shorter_limit_trims_without_reembedding(env) -> None:
    cfg, store, ix, extractor = env
    cache = store.warm_vector_cache()
    cfg.max_media_seconds = 20 * 60
    assert ix.reconcile_media() == 0
    assert pending(store) == set() and extractor.calls == []
    assert segments(store) == {"long.mp3": 20, "short.mp3": 10, "film.mp4": 20}
    assert store.db.execute("SELECT max(position) FROM chunks").fetchone()[0] == 19 * 60
    n_chunks = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert store.db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == n_chunks
    alive = cache.alive[: cache.n]
    assert int(alive.sum()) == n_chunks
    live_ids = set(cache.chunk_ids[: cache.n][alive].tolist())
    assert live_ids == {r[0] for r in store.db.execute("SELECT id FROM chunks")}
    row = cache.row_of[next(iter(live_ids))]
    stored = store.db.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?",
                              (int(cache.chunk_ids[row]),)).fetchone()[0]
    assert np.allclose(cache.mat[row], np.frombuffer(stored, dtype=np.float32), atol=1e-6)


def test_new_segment_length_reindexes_all_media(env) -> None:
    cfg, store, ix, extractor = env
    cfg.media_segment_seconds = 30
    assert ix.reconcile_media() == 3
    ix.run()
    assert segments(store) == {"long.mp3": 60, "short.mp3": 20, "film.mp4": 60}


def test_index_from_before_recording_assumes_the_old_30_minute_default(env) -> None:
    cfg, store, ix, _ = env
    store.db.execute("DELETE FROM meta WHERE key IN ('media_max_seconds', 'media_segment_seconds')")
    store.db.commit()
    cfg.max_media_seconds = 60 * 60
    assert ix.reconcile_media() == 2
    assert pending(store) == {"long.mp3", "film.mp4"}


def test_empty_index_just_records(tmp_path: Path) -> None:
    cfg = load_config(path=tmp_path / "nonexistent.toml", home=tmp_path)
    cfg.db_path = tmp_path / "index.db"
    store = Store(cfg.db_path)
    try:
        ix = Indexer(store, cfg, FakeEmbedder(), extract_fn=FakeMedia(), prefetch=False)
        assert ix.reconcile_media() == 0
        assert store.get_meta("media_max_seconds") == str(float(cfg.max_media_seconds))
    finally:
        store.close()
