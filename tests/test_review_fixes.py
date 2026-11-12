"""Regression tests for the indexing-core review findings and the search-quality fixes."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fakes import DIM, FakeEmbedder, FakeExtractor, bag_of_words
from test_index import make_cfg, run, scan, statuses, table_counts, write_docs

from gemmasearch import config as config_mod
from gemmasearch import indexer as indexer_mod
from gemmasearch.cli import main as cli_main
from gemmasearch.config import MAX_KNN_K, load_config
from gemmasearch.embedder import Embedder, GpuLock
from gemmasearch.extract.base import Chunk
from gemmasearch.indexer import INDEX_FORMAT_VERSION, Indexer, ModelMismatch
from gemmasearch.search import Searcher
from gemmasearch.store import SCHEMA_V1, SCHEMA_V2, Store, dir_terms


def names(results) -> list[str]:
    return [Path(r.path).name for r in results]


def make_env(tmp_path: Path, docs: dict[str, str]):
    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, docs)
    store = Store(cfg.db_path)
    scan(store, cfg)
    return cfg, store


# -- 1. one un-embeddable file must not abort the run ---------------------------------------------

def test_poison_file_fails_alone_and_run_continues(tmp_path: Path) -> None:
    docs = {f"good{i}.md": f"good document number {i}" for i in range(5)} | {"bad.md": "x"}
    cfg, store = make_env(tmp_path, docs)
    now = time.time() + 10
    os.utime(tmp_path / "Documents/bad.md", (now, now))  # newest -> head of the queue
    scan(store, cfg)
    bad_chunks = [f"BAD chunk {i}" for i in range(16)]  # two full batches, all from one file
    emb = FakeEmbedder()
    emb.fail_on = set(bad_chunks)
    for _ in range(2):  # the second run must not trip over it either
        Indexer(store, cfg, emb, extract_fn=FakeExtractor({"bad.md": bad_chunks})).run()
    st = statuses(store)
    assert st.pop("bad.md") == "failed"
    assert set(st.values()) == {"indexed"}


def test_consecutive_failed_files_still_abort(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.batch_sizes["text"] = 1  # every batch holds one file, so only the streak rule can trigger
    docs = {f"f{i}.md": f"text {i}" for i in range(10)}
    write_docs(tmp_path, docs)
    store = Store(cfg.db_path)
    scan(store, cfg)
    emb = FakeEmbedder()
    emb.fail_on = set(docs.values())
    with pytest.raises(RuntimeError):
        Indexer(store, cfg, emb, extract_fn=FakeExtractor()).run()
    st = statuses(store)
    assert list(st.values()).count("failed") == indexer_mod.MAX_CONSECUTIVE_EMBED_FAILURES
    assert list(st.values()).count("pending") == 10 - indexer_mod.MAX_CONSECUTIVE_EMBED_FAILURES


# -- 2. one connection per thread ---------------------------------------------------------------

def test_other_thread_neither_sees_nor_commits_an_open_transaction(tmp_path: Path) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha beta gamma"})
    run(cfg, store)
    seen: list[tuple] = []

    def other() -> None:
        seen.append(table_counts_on(store.db))
        store.set_meta("last_scan_at", "1")  # waits for the open write transaction to end

    t = threading.Thread(target=other)
    with pytest.raises(RuntimeError), store.write() as (db, _):
        db.execute("DELETE FROM chunks")
        t.start()
        t.join(0.3)
        raise RuntimeError("the commit fails")
    t.join()
    assert seen == [(1, 1)]  # committed state only
    assert table_counts(store) == (1, 1, 1)  # rolled back, not committed by the other thread
    assert store.get_meta("last_scan_at") == "1"
    store.close()


def table_counts_on(db: sqlite3.Connection) -> tuple[int, int]:
    return (db.execute("SELECT count(*) FROM chunks").fetchone()[0],
            db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0])


def test_threads_get_their_own_connections_and_search_is_read_only(tmp_path: Path) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "x"})
    other: list[sqlite3.Connection] = []
    t = threading.Thread(target=lambda: other.append(store.db))
    t.start()
    t.join()
    assert other[0] is not store.db
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store.reader().execute("DELETE FROM files")
    store.close()


@pytest.mark.parametrize("cache", [False, True])
def test_concurrent_index_search_and_scan(tmp_path: Path, cache: bool) -> None:
    cfg = make_cfg(tmp_path)
    cfg.batch_sizes["text"] = 4
    docs = {f"doc{i:03d}.md": f"note {i} about rockets and bread {i % 7}" for i in range(150)}
    write_docs(tmp_path, docs)
    store = Store(cfg.db_path, vector_cache=cache)
    scan(store, cfg)
    errors: list[BaseException] = []
    done = threading.Event()

    def guard(fn):
        def inner() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
                errors.append(exc)
        return inner

    def searching() -> None:
        s = Searcher(store, cfg, FakeEmbedder())
        while not done.is_set():
            for r in s.search("rockets bread", limit=5):
                assert r.path.endswith(".md")

    def scanning() -> None:
        for i in range(5):
            (tmp_path / f"Documents/new{i}.md").write_text(f"new note {i}")
            scan(store, cfg)
            (tmp_path / f"Documents/doc{140 + i}.md").unlink()

    threads = [threading.Thread(target=guard(searching)) for _ in range(2)] + [threading.Thread(target=guard(scanning))]
    for t in threads:
        t.start()
    try:
        Indexer(store, cfg, FakeEmbedder(), extract_fn=FakeExtractor()).run()
    finally:
        done.set()
        for t in threads:
            t.join()
    assert not errors, errors
    chunks, vectors, fts = table_counts(store)
    assert chunks == vectors == fts
    run(cfg, store)  # whatever was skipped meanwhile
    scan(store, cfg)
    run(cfg, store)
    assert set(statuses(store).values()) == {"indexed"}
    if cache:
        s = Searcher(store, cfg, FakeEmbedder())
        with_cache = [(r.path, r.similarity) for r in s.search("note 3 rockets", limit=10)]
        store.vectors = None
        assert with_cache == [(r.path, r.similarity) for r in s.search("note 3 rockets", limit=10)]
    store.close()


# -- 3. retries, attempts, index format and model changes -----------------------------------------

def test_failed_files_retry(tmp_path: Path) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha", "b.md": "beta"})
    run(cfg, store, FakeExtractor({"a.md": RuntimeError("ffmpeg not found on PATH")}))
    assert statuses(store)["a.md"] == "failed"
    assert store.failed_files()[0]["attempts"] == 1
    # A plain re-run leaves it alone (the backoff has not passed)...
    stats, _ = run(cfg, store)
    assert stats.total == 0 and statuses(store)["a.md"] == "failed"
    # ...--retry-failed queues it now.
    emb = FakeEmbedder()
    ix = Indexer(store, cfg, emb, extract_fn=FakeExtractor())
    assert ix.prepare(retry_failed=True) == 1
    assert ix.run().indexed == 1
    assert statuses(store)["a.md"] == "indexed"
    assert store.db.execute("SELECT attempts FROM files WHERE path LIKE '%a.md'").fetchone()[0] == 0


def test_automatic_retry_with_backoff(tmp_path: Path) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha"})
    run(cfg, store, FakeExtractor({"a.md": RuntimeError("locked")}))
    day = indexer_mod.RETRY_BACKOFF_S
    assert store.requeue_failed_for_retry(3, day, now=time.time() + 0.5 * day) == 0
    assert store.requeue_failed_for_retry(3, day, now=time.time() + 1.1 * day) == 1
    store.db.execute("UPDATE files SET status = 'failed', attempts = 3")
    store.db.commit()
    assert store.requeue_failed_for_retry(3, day, now=time.time() + 100 * day) == 0  # gave up


def test_extractor_version_change_retries_failed(tmp_path: Path, monkeypatch) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha", "b.md": "beta"})
    run(cfg, store, FakeExtractor({"a.md": RuntimeError("unsupported")}))
    monkeypatch.setattr(indexer_mod, "extractor_version", lambda: "999")
    stats, _ = run(cfg, store)
    assert stats.total == 1 and set(statuses(store).values()) == {"indexed"}


def test_index_format_change_requeues_everything_keeping_old_chunks(tmp_path: Path, monkeypatch) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha", "b.md": "beta"})
    run(cfg, store)
    assert store.get_meta("index_format") == str(INDEX_FORMAT_VERSION)
    monkeypatch.setattr(indexer_mod, "INDEX_FORMAT_VERSION", INDEX_FORMAT_VERSION + 1)
    ix = Indexer(store, cfg, FakeEmbedder(), extract_fn=FakeExtractor())
    assert ix.prepare() == 2
    assert set(statuses(store).values()) == {"pending"}
    assert table_counts(store) == (2, 2, 2)  # still searchable until replaced
    assert ix.run().indexed == 2
    assert table_counts(store) == (2, 2, 2)


def test_model_change_requires_rebuild(tmp_path: Path) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha"})
    run(cfg, store)
    emb = FakeEmbedder()
    emb.model_id = "another-model"
    ix = Indexer(store, cfg, emb, extract_fn=FakeExtractor())
    with pytest.raises(ModelMismatch):
        ix.run()
    assert table_counts(store) == (1, 1, 1) and store.get_meta("model_id") == "fake-bow"
    ix.prepare(rebuild=True)
    assert table_counts(store) == (0, 0, 0) and store.get_meta("model_id") == "another-model"
    assert ix.run().indexed == 1


def test_cli_index_refuses_model_change(tmp_path: Path, capsys) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha"})
    run(cfg, store)
    store.close()
    conf = tmp_path / "config.toml"
    conf.write_text(f'db_path = "{cfg.db_path}"\n[model]\nname = "some/other-model"\n')
    assert cli_main(["--config", str(conf), "index"]) == 1
    assert "--rebuild" in capsys.readouterr().err
    assert Store(cfg.db_path).get_meta("model_id") == "fake-bow"


def test_migration_v2_to_v3(tmp_path: Path) -> None:
    from gemmasearch.store import connect

    path = tmp_path / "v2.db"
    db = connect(path)
    db.executescript(f"BEGIN; {SCHEMA_V1}; {SCHEMA_V2}; PRAGMA user_version = 2; COMMIT;")
    db.execute("INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
               "VALUES ('/home/someone/Documents/thesis/notes.md', 'text', 1, 1, 0, 50, 'indexed', 1)")
    db.execute("INSERT INTO chunks (file_id, seq, modality) VALUES (1, 0, 'text')")
    db.execute("INSERT INTO vec_chunks VALUES (1, ?, 'text', 'text')", (bag_of_words("x"),))
    db.execute("INSERT INTO meta VALUES ('model_id', 'fake-bow')")
    db.commit()
    db.close()

    store = Store(path)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == 3
    assert store.db.execute("SELECT attempts FROM files").fetchone()[0] == 0
    assert store.db.execute("SELECT dirs FROM fts_files").fetchone()[0] == "Documents thesis"
    sql = store.db.execute("SELECT sql FROM sqlite_master WHERE name = 'fts_chunks'").fetchone()[0]
    assert "prefix='3'" in sql
    # The old index has no format version: the indexer re-queues it (fts_chunks lost its text).
    ix = Indexer(store, make_cfg(tmp_path), FakeEmbedder(), extract_fn=FakeExtractor())
    assert ix.prepare() == 1
    # Two processes opening an old database at once must not both migrate.
    store.close()


def test_concurrent_migration(tmp_path: Path) -> None:
    from gemmasearch.store import connect

    path = tmp_path / "v1.db"
    db = connect(path)
    db.executescript(f"BEGIN; {SCHEMA_V1}; PRAGMA user_version = 1; COMMIT;")
    db.close()
    errors: list[BaseException] = []

    def open_store() -> None:
        try:
            Store(path).close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


# -- 4. folder words must not flood the results ---------------------------------------------------

def test_dir_terms_drop_home_and_mount_prefixes() -> None:
    assert dir_terms("/home/patrik/Documents/thesis/v3/a.md") == "Documents thesis v3"
    assert dir_terms("/var/home/patrik/a.md") == ""
    assert dir_terms("/home/patrik/Documents/code/proj/data/coco/x.jpg") == "proj data coco"
    assert dir_terms("/mnt/games/code/thesisProject/a.md") == "code thesisProject"
    assert dir_terms("/run/media/patrik/USB/photos/a.jpg") == "photos"


def test_folder_word_does_not_flood(tmp_path: Path) -> None:
    docs = {f"note{i:02d}.md": f"entry {i} about gardening tomatoes" for i in range(60)}
    docs["paper.md"] = "the documents describing the merger were signed in march"
    cfg, store = make_env(tmp_path, docs)
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    assert names(s.search("documents", limit=10)) == ["paper.md"]
    assert s.search(tmp_path.name, limit=10) == []


def test_folder_match_still_helps_relevant_files(tmp_path: Path) -> None:
    cfg, store = make_env(tmp_path, {"rockets/notes.md": "rockets", "garden.md": "tomatoes"})
    _, emb = run(cfg, store)
    res = Searcher(store, cfg, emb).search("rockets")
    assert names(res) == ["notes.md"] and "folder" in res[0].matched


# -- 6. a lone chunk in a rare modality buffer --------------------------------------------------

def test_lone_chunk_is_flushed_after_other_files(tmp_path: Path) -> None:
    docs = {f"n{i:03d}.md": f"note {i}" for i in range(100)}
    cfg, store = make_env(tmp_path, docs | {"scan.md": "x"})
    now = time.time() + 10
    os.utime(tmp_path / "Documents/scan.md", (now, now))
    scan(store, cfg)

    def extract_fn(path: Path, kind: str, cfg):
        if path.name == "scan.md":
            yield Chunk("text", "page one text")
            yield Chunk("image", "<a scanned page>")
        else:
            yield Chunk("text", path.read_text())

    seen: dict[int, str] = {}

    def progress(st) -> None:
        seen[st.done] = store.db.execute("SELECT status FROM files WHERE path LIKE '%scan.md'").fetchone()[0]

    Indexer(store, cfg, FakeEmbedder(), extract_fn=extract_fn, prefetch=False).run(progress=progress)
    first_indexed = min(d for d, s in seen.items() if s == "indexed")
    assert first_indexed <= indexer_mod.FLUSH_AFTER_FILES + 2


def test_aged_buffer_is_flushed_by_time(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(indexer_mod, "FLUSH_AFTER_S", 0.0)
    cfg, store = make_env(tmp_path, {"a.md": "x", "b.md": "y"})
    stats, emb = run(cfg, store)
    assert stats.indexed == 2 and emb.calls == [("text", 1), ("text", 1)]


# -- 7. kind filter inside the text query, vanished files -----------------------------------------

def test_kind_filter_applies_before_the_fts_limit(tmp_path: Path) -> None:
    docs = {f"n{i:03d}.md": "budget budget budget budget" for i in range(230)}
    cfg, store = make_env(tmp_path, docs)
    (tmp_path / "Documents/annual.pdf").write_text("x" * 3000)
    scan(store, cfg)
    ex = FakeExtractor({"annual.pdf": ["the annual budget was approved by the board " * 3]})
    _, emb = run(cfg, store, ex)
    cfg.min_similarity = 0.99
    res = Searcher(store, cfg, emb).search("budget", kinds=["pdf"])
    assert names(res) == ["annual.pdf"]


def test_deleted_files_are_not_returned(env_docs) -> None:
    cfg, store, tmp_path = env_docs
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    assert names(s.search("sourdough bread"))[0] == "bread.txt"
    (tmp_path / "Documents/bread.txt").unlink()
    assert "bread.txt" not in names(s.search("sourdough bread"))


@pytest.fixture
def env_docs(tmp_path: Path):
    from test_index import DOCS

    cfg, store = make_env(tmp_path, DOCS)
    yield cfg, store, tmp_path
    store.close()


# -- 8-9. embedder: OOM recovery, query priority, query cache -------------------------------------

class FakeModel:
    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.sizes: list[int] = []
        self.queries = 0

    def encode(self, inputs, **kw):
        import torch

        self.sizes.append(len(inputs))
        if len(inputs) > self.max_batch:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return np.full((len(inputs), DIM), 1.0, dtype=np.float32)

    def encode_query(self, inputs, **kw):
        self.queries += 1
        return np.stack([bag_of_words(q) * 1.01 for q in inputs])


def fake_embedder(max_batch: int = 64) -> Embedder:
    cfg = load_config(path=Path("/nonexistent.toml"))
    e = Embedder(cfg)
    e._model = FakeModel(max_batch)
    return e


def test_oom_halves_and_remembers_the_batch_size() -> None:
    e = fake_embedder(max_batch=2)
    e.batch_sizes["text"] = 8
    out = e.embed_batch("text", ["t"] * 8, ["x"] * 8)
    assert out.shape == (8, DIM)
    assert e._model.sizes[:3] == [8, 4, 2]
    assert e.batch_limit("text") == 2
    e._model.sizes.clear()
    e.embed_batch("text", ["t"] * 4, ["x"] * 4)
    assert e._model.sizes == [2, 2]  # no OOM round-trip again
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_regrows_after_many_clean_batches(monkeypatch) -> None:
    from gemmasearch import embedder as emb_mod

    monkeypatch.setattr(emb_mod, "OOM_REGROW_AFTER", 3)
    e = fake_embedder(max_batch=2)
    e.batch_sizes["text"] = 4
    e.embed_batch("text", ["t"] * 4, ["x"] * 4)
    assert e.batch_limit("text") == 2
    e._model.max_batch = 100
    e.embed_batch("text", ["t"] * 6, ["x"] * 6)
    assert e.batch_limit("text") == 4


def test_non_oom_errors_are_not_retried() -> None:
    from gemmasearch.embedder import _is_oom

    assert not _is_oom(RuntimeError("out of memory"))  # e.g. a host-side allocator message
    e = fake_embedder()
    e._model.encode = lambda inputs, **kw: (_ for _ in ()).throw(RuntimeError("out of memory"))
    with pytest.raises(RuntimeError):
        e.embed_batch("text", ["a", "b"], ["x", "y"])
    assert e.batch_limit("text") == e.batch_sizes["text"]


def test_query_cache_and_float32_normalisation() -> None:
    e = fake_embedder()
    a = e.embed_query("dog")
    b = e.embed_query("dog")
    assert a is b and e._model.queries == 1
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-6  # the model's 1.01 norm is fixed
    assert not a.flags.writeable


def test_embedder_query_path_loads_text_only_and_reuses_full_model(monkeypatch) -> None:
    import sentence_transformers
    import torch

    st_calls: list[dict] = []

    class StubST:
        def __init__(self, model_name: str, **kwargs):
            st_calls.append({"model_name": model_name, **kwargs})
            self.queries = 0
            self.batches = 0

        def eval(self):
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return DIM

        def encode_query(self, inputs, **kw):
            self.queries += 1
            return np.stack([bag_of_words(q) for q in inputs])

        def encode(self, inputs, **kw):
            self.batches += 1
            return np.stack([bag_of_words(str(x)) for x in inputs])

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", StubST)
    cfg = load_config(path=Path("/nonexistent.toml"))
    cfg.device = "cuda"
    e = Embedder(cfg)

    # 1. Query path loads the text-only model with vision_config=None, audio_config=None, local_files_only=True.
    v1 = e.embed_query("hello world")
    assert v1.shape == (DIM,)
    assert e.query_model_loaded and not e.index_model_loaded and e.loaded
    assert len(st_calls) == 1
    assert st_calls[0]["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert st_calls[0]["local_files_only"] is True
    assert st_calls[0]["device"] == "cuda"
    assert st_calls[0]["model_kwargs"] == {"dtype": torch.bfloat16}

    # 2. Indexing loads the full multimodal model (no config_kwargs) and drops the GPU query model.
    st_calls.clear()
    b = e.embed_batch("text", ["some text"], ["doc.md"])
    assert b.shape == (1, DIM)
    assert e.index_model_loaded and not e.query_model_loaded and e.loaded
    assert len(st_calls) == 1
    assert "config_kwargs" not in st_calls[0]

    # 3. While the full model is loaded, a new query reuses it without loading the query model.
    st_calls.clear()
    full_model = e._model
    queries_before = full_model.queries
    e.embed_query("another query")
    assert st_calls == []
    assert not e.query_model_loaded and e.index_model_loaded
    assert full_model.queries == queries_before + 1

    # 4. Unloading the index model leaves nothing loaded; next query loads the text-only model again.
    assert e.unload_index() is True
    assert not e.loaded
    e.embed_query("third query")
    assert e.query_model_loaded and not e.index_model_loaded
    assert len(st_calls) == 1
    assert st_calls[0]["config_kwargs"] == {"vision_config": None, "audio_config": None}


def test_query_device_cpu_selects_float32_cpu_and_invalid_warns(tmp_path: Path, monkeypatch, capsys) -> None:
    import sentence_transformers
    import torch

    st_calls: list[dict] = []

    class StubST:
        def __init__(self, model_name: str, **kwargs):
            st_calls.append({"model_name": model_name, **kwargs})

        def eval(self):
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return DIM

        def encode_query(self, inputs, **kw):
            return np.stack([bag_of_words(q) for q in inputs])

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", StubST)

    conf = tmp_path / "c.toml"
    conf.write_text('[daemon]\nquery_device = "cpu"\nindex_model_unload_seconds = 45\n')
    cfg = load_config(conf, home=tmp_path)
    assert cfg.query_device == "cpu"
    assert cfg.index_model_unload_seconds == 45.0
    cfg.device = "cuda"  # even when indexing uses CUDA, query_device="cpu" uses CPU + float32

    e = Embedder(cfg)
    e.embed_query("cpu query")
    assert len(st_calls) == 1
    assert st_calls[0]["device"] == "cpu"
    assert st_calls[0]["model_kwargs"] == {"dtype": torch.float32}
    assert st_calls[0]["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert e.query_device == "cpu"

    # Unknown query_device warns and falls back to "auto".
    conf2 = tmp_path / "c2.toml"
    conf2.write_text('[daemon]\nquery_device = "tpu"\n')
    cfg2 = load_config(conf2, home=tmp_path)
    assert cfg2.query_device == "auto"
    assert "query_device" in capsys.readouterr().err


def test_waiting_query_goes_before_waiting_batches() -> None:
    lock = GpuLock()
    order: list[str] = []
    release_first = threading.Event()

    def first_batch() -> None:
        with lock.batch():
            order.append("batch1")
            release_first.wait()

    def batch(name: str) -> None:
        with lock.batch():
            order.append(name)

    def query() -> None:
        with lock.query():
            order.append("query")

    t1 = threading.Thread(target=first_batch)
    t1.start()
    while not order:
        time.sleep(0.001)
    waiting_batch = threading.Thread(target=batch, args=("batch2",))
    waiting_batch.start()
    time.sleep(0.05)
    q = threading.Thread(target=query)
    q.start()
    while lock.queries_waiting == 0:
        time.sleep(0.001)
    release_first.set()
    for t in (t1, waiting_batch, q):
        t.join()
    assert order == ["batch1", "query", "batch2"]


# -- 10. transient storage errors ---------------------------------------------------------------

def test_locked_database_is_retried_then_left_pending(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(indexer_mod, "STORE_BACKOFF_S", 0.001)
    cfg, store = make_env(tmp_path, {"a.md": "alpha", "b.md": "beta"})
    real = store.replace_chunks
    calls = {"n": 0}

    def flaky(file_id, *a):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise sqlite3.OperationalError("database is locked")
        if Path(store.db.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()[0]).name == "b.md":
            raise sqlite3.OperationalError("database is locked")
        return real(file_id, *a)

    monkeypatch.setattr(store, "replace_chunks", flaky)
    monkeypatch.setattr(store, "mark_failed", lambda *a: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    stats, _ = run(cfg, store)
    assert stats.indexed == 1 and stats.deferred == 1 and stats.failed == 0
    assert statuses(store) == {"a.md": "indexed", "b.md": "pending"}


def test_unstorable_failure_does_not_kill_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(indexer_mod, "STORE_BACKOFF_S", 0.001)
    cfg, store = make_env(tmp_path, {"a.md": "alpha", "b.md": "beta"})
    monkeypatch.setattr(store, "mark_failed", lambda *a: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    stats, _ = run(cfg, store, FakeExtractor({"a.md": RuntimeError("bad")}))
    assert (stats.indexed, stats.failed, stats.deferred) == (1, 0, 1)
    assert statuses(store)["a.md"] == "pending"


# -- 11. stored vectors are unit length in float32 ------------------------------------------------

def test_stored_vectors_are_renormalised(tmp_path: Path) -> None:
    class Sloppy(FakeEmbedder):
        def embed_batch(self, modality, contents, titles):
            return super().embed_batch(modality, contents, titles) * 1.01

    cfg, store = make_env(tmp_path, {"a.md": "alpha"})
    Indexer(store, cfg, Sloppy(), extract_fn=FakeExtractor()).run()
    blob = store.db.execute("SELECT embedding FROM vec_chunks").fetchone()[0]
    assert abs(float(np.linalg.norm(np.frombuffer(blob, dtype=np.float32))) - 1.0) < 1e-6


# -- 12. config and CLI validation ---------------------------------------------------------------

def test_config_validation(tmp_path: Path, capsys) -> None:
    conf = tmp_path / "c.toml"
    conf.write_text('[search]\nknn_k = 10000\nmin_similarty = 0.5\n[scan]\nroot = ["/x"]\n[typo]\na = 1\n')
    cfg = load_config(conf, home=tmp_path)
    assert cfg.knn_k == MAX_KNN_K
    err = capsys.readouterr().err
    for key in ("search.min_similarty", "scan.root", "'typo'", "knn_k"):
        assert key in err, key
    assert config_mod.check_unknown_keys({"search": {"knn_k": 1}}) == []


def test_search_n_must_be_positive(capsys) -> None:
    with pytest.raises(SystemExit):
        cli_main(["search", "x", "-n", "0"])
    assert "positive" in capsys.readouterr().err


def test_reset_refuses_while_the_database_is_open(tmp_path: Path, capsys) -> None:
    cfg, store = make_env(tmp_path, {"a.md": "alpha"})
    store.close()
    conf = tmp_path / "config.toml"
    conf.write_text(f'db_path = "{cfg.db_path}"\n')
    holder = subprocess.Popen([sys.executable, "-c",
                               "import sqlite3, sys, time; c = sqlite3.connect(sys.argv[1]); "
                               "c.execute('select 1 from files').fetchall(); print('ok', flush=True); time.sleep(30)",
                               str(cfg.db_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ok"
        assert cli_main(["--config", str(conf), "reset", "--yes", "--all"]) == 1
        assert str(holder.pid) in capsys.readouterr().err
        assert cfg.db_path.exists()
    finally:
        holder.kill()
        holder.wait()
    assert cli_main(["--config", str(conf), "reset", "--yes", "--all"]) == 0
    assert not cfg.db_path.exists()


# -- 13. an interrupted extraction is not committed -----------------------------------------------

@pytest.mark.parametrize("prefetch", [True, False])
def test_stop_during_extraction_leaves_file_pending(tmp_path: Path, prefetch: bool) -> None:
    cfg, store = make_env(tmp_path, {"video.md": "x"})
    stop = threading.Event()

    def truncated(path: Path, kind: str, cfg):
        yield Chunk("text", "first minute")
        stop.set()  # Ctrl-C: ffmpeg dies and the extractor ends early without an error

    stats = Indexer(store, cfg, FakeEmbedder(), extract_fn=truncated, prefetch=prefetch).run(stop=stop)
    assert stats.stopped and stats.indexed == 0
    assert statuses(store) == {"video.md": "pending"}


# -- 14-16. ranking: duplicates, location prior, short queries ------------------------------------

def put_vectors(store: Store, files: list[tuple[str, int, np.ndarray]], kind: str = "image",
                modality: str = "image") -> None:
    """Index synthetic files directly: (path, priority, vector)."""
    with store.write() as (db, _):
        for path, priority, vec in files:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"x")
            cur = db.execute("INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
                             "VALUES (?, ?, 1, ?, 0, ?, 'indexed', 1)", (path, kind, 1.0, priority))
            cid = db.execute("INSERT INTO chunks (file_id, seq, modality) VALUES (?, 0, ?)",
                             (cur.lastrowid, modality)).lastrowid
            v = np.asarray(vec, dtype=np.float32)
            db.execute("INSERT INTO vec_chunks VALUES (?, ?, ?, ?)", (cid, v / np.linalg.norm(v), modality, kind))


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def with_cos(q: np.ndarray, cos: float, seed: int) -> np.ndarray:
    """A unit vector with exactly this cosine to q."""
    r = unit(seed)
    r = r - (r @ q) * q
    r /= np.linalg.norm(r)
    return cos * q + np.sqrt(1 - cos**2) * r


@pytest.mark.parametrize("cache", [False, True])
def test_near_duplicates_collapse_to_the_preferred_copy(tmp_path: Path, cache: bool) -> None:
    cfg = make_cfg(tmp_path)
    store = Store(cfg.db_path, vector_cache=cache)
    q = unit(1)
    photo = with_cos(q, 0.8, 2)
    copy = photo + 0.05 * unit(3)  # a re-encoded copy: cos ~0.999
    put_vectors(store, [
        (str(tmp_path / "data/prepared/dog.jpg"), 10, photo),
        (str(tmp_path / "data/prepared_standalone/dog.jpg"), 10, copy),
        (str(tmp_path / "Pictures/dog.jpg"), 70, copy),
        (str(tmp_path / "Pictures/cat.jpg"), 70, with_cos(q, 0.75, 4)),
    ])
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "", limit=10)
    assert [(Path(r.path).parent.name, Path(r.path).name, r.duplicates) for r in res] == [
        ("Pictures", "dog.jpg", 2), ("Pictures", "cat.jpg", 0)]
    assert len(res[0].duplicate_paths) == 2
    cfg.duplicate_similarity = {}
    assert len(Searcher(store, cfg, FakeEmbedder()).search_vector(q, "", limit=10)) == 4


def test_location_prior_only_breaks_near_ties(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.duplicate_similarity = {}
    store = Store(cfg.db_path)
    q = unit(1)
    put_vectors(store, [
        (str(tmp_path / "proj/data/a.jpg"), 5, with_cos(q, 0.705, 2)),   # marginally better, deep in a project
        (str(tmp_path / "Pictures/b.jpg"), 80, with_cos(q, 0.700, 3)),   # in the user's own folder
        (str(tmp_path / "proj/data/c.jpg"), 5, with_cos(q, 0.760, 4)),   # clearly better
    ])
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "", limit=10)
    assert names(res) == ["c.jpg", "b.jpg", "a.jpg"]
    cfg.location_prior = 0.0
    assert names(Searcher(store, cfg, FakeEmbedder()).search_vector(q, "", limit=10)) == ["c.jpg", "a.jpg", "b.jpg"]


def test_single_word_queries_use_the_stricter_floor(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.duplicate_similarity = {}
    cfg.min_similarity, cfg.similarity_margin = 0.5, 0.2
    cfg.min_similarity_short, cfg.similarity_margin_short = 0.5, 0.05
    store = Store(cfg.db_path)
    q = unit(1)
    put_vectors(store, [(str(tmp_path / "x/a.jpg"), 10, with_cos(q, 0.70, 2)),
                        (str(tmp_path / "x/b.jpg"), 10, with_cos(q, 0.62, 3))])
    s = Searcher(store, cfg, FakeEmbedder())
    assert names(s.search_vector(q, "zzzz", limit=10)) == ["a.jpg"]
    assert names(s.search_vector(q, "zzzz yyyy", limit=10)) == ["a.jpg", "b.jpg"]


# -- 17. the in-memory vector cache --------------------------------------------------------------

def test_vector_cache_tracks_local_and_foreign_writes(tmp_path: Path) -> None:
    from test_index import DOCS

    cfg, store = make_env(tmp_path, DOCS)
    store.close()
    store = Store(cfg.db_path, vector_cache=True)
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    assert names(s.search("sourdough bread"))[0] == "bread.txt"
    cache = store.vectors
    assert cache is not None and cache.loads == 1 and cache.size == 3
    assert cache.fast_loads == 1  # the sqlite-vec storage layout check passed

    # Local writes patch the cache without a reload.
    (tmp_path / "Documents/pizza.md").write_text("pizza dough with tomato and cheese")
    scan(store, cfg)
    run(cfg, store)
    assert names(s.search("pizza tomato cheese"))[0] == "pizza.md"
    (tmp_path / "Documents/cats.md").unlink()
    scan(store, cfg)
    assert "cats.md" not in names(s.search("cat sleeps on the sofa"))
    assert cache.loads == 1 and cache.size == 3

    # Another process (here: another Store) changes the index: the cache reloads.
    other = Store(cfg.db_path)
    (tmp_path / "Documents/soup.md").write_text("tomato soup with basil")
    scan(other, cfg)
    run(cfg, other)
    assert names(s.search("tomato soup basil"))[0] == "soup.md"
    assert cache.loads == 2 and cache.size == 4
    other.reset_index()
    assert [r.matched for r in s.search("soup")] == [["name"]]  # lexical only now
    assert cache.size == 0
    other.close()
    store.close()


def test_vector_cache_falls_back_when_too_big(tmp_path: Path) -> None:
    from test_index import DOCS

    cfg, store = make_env(tmp_path, DOCS)
    _, emb = run(cfg, store)
    store.enable_vector_cache(max_chunks=2)
    res = Searcher(store, cfg, emb).search("sourdough bread")
    assert store.vectors is not None and store.vectors.too_big
    assert names(res)[0] == "bread.txt"


def test_common_text_terms_are_dropped_at_scale(tmp_path: Path, monkeypatch) -> None:
    import gemmasearch.search as search_mod

    monkeypatch.setattr(search_mod, "TEXT_MAX_DF", 3)
    docs = {f"d{i}.md": f"the common words and {w}" for i, w in enumerate(["rocket", "bread", "cat", "dog", "fish"])}
    cfg, store = make_env(tmp_path, docs)
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    assert s._selective(store.reader(), ['"the"', '"rocket"']) == '"rocket"'
    assert s._selective(store.reader(), ['"the"', '"common"']) is None
    assert names(s.search("the rocket"))[0] == "d0.md"


def test_versions_collapse_by_first_chunk(tmp_path: Path) -> None:
    """Versions share their opening (title page/abstract) but the query matches different sections."""
    cfg = make_cfg(tmp_path)
    opening = "thesis title page abstract about semantic image search and embeddings"
    docs = {f"thesis_v{i}.md": [opening, f"chapter {i} about rockets rockets {i} version only words v{i}"]
            for i in range(3, 6)}
    write_docs(tmp_path, {n: "x" for n in docs} | {"other.md": "x"})
    store = Store(cfg.db_path)
    scan(store, cfg)
    _, emb = run(cfg, store, FakeExtractor(docs | {"other.md": ["rockets rockets launch window"]}))
    res = Searcher(store, cfg, emb).search("rockets chapter", limit=10)
    assert "other.md" in names(res)
    thesis = [r for r in res if "thesis" in r.path]
    assert len(thesis) == 1 and thesis[0].duplicates == 2


def test_single_word_text_hits_rank_after_strong_matches(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.duplicate_similarity = {}
    cfg.min_similarity_short, cfg.similarity_margin_short = 0.5, 0.1
    store = Store(cfg.db_path)
    q = unit(1)
    put_vectors(store, [(str(tmp_path / "x/photo.jpg"), 10, with_cos(q, 0.70, 2))])
    put_vectors(store, [(str(tmp_path / "x/thesis.md"), 90, with_cos(q, 0.30, 3))], kind="text", modality="text")
    with store.write() as (db, _):
        cid = db.execute("SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path LIKE '%thesis.md'").fetchone()[0]
        db.execute("INSERT INTO fts_chunks (rowid, text) VALUES (?, 'a dog appears in figure 3')", (cid,))
    s = Searcher(store, cfg, FakeEmbedder())
    assert names(s.search_vector(q, "dog", limit=10)) == ["photo.jpg", "thesis.md"]
