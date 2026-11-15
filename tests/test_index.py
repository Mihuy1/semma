from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest
from fakes import ExtractionError, FakeEmbedder, FakeExtractor

from semma.config import Config, load_config
from semma.indexer import Indexer
from semma.scanner import Scanner
from semma.search import Searcher, _file_patterns, _patterns, query_terms
from semma.store import MIGRATIONS, SCHEMA_V1, Store, connect

DOCS = {
    "cats.md": "the cat sleeps on the warm sofa all afternoon",
    "rockets.md": "rocket engines burn liquid oxygen and kerosene during launch",
    "bread.txt": "knead the dough then bake the sourdough bread in a hot oven",
}


def make_cfg(home: Path) -> Config:
    cfg = load_config(path=home / "nonexistent.toml", home=home)
    cfg.include_mounts = False
    cfg.db_path = home / "db" / "index.db"
    cfg.roots = [home / "Documents"]
    # Tiny indexes: with n vectors no z-score can exceed sqrt(n - 1), so the default gate (2.0)
    # and floor (4.5) are out of reach for 3-5 files.
    cfg.z_gate = 1.0
    cfg.z_floor = 0.0
    return cfg


def write_docs(home: Path, docs: dict[str, str]) -> None:
    for name, text in docs.items():
        p = home / "Documents" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def scan(store: Store, cfg: Config) -> None:
    scanner = Scanner(cfg)
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(), gen)
    store.finish_scan(gen, scanner.stats.roots)


@pytest.fixture
def env(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, DOCS)
    store = Store(cfg.db_path)
    scan(store, cfg)
    yield cfg, store
    store.close()


def statuses(store: Store) -> dict[str, str]:
    return {Path(r[0]).name: r[1] for r in store.db.execute("SELECT path, status FROM files")}


def table_counts(store: Store) -> tuple[int, int, int]:
    db = store.db
    chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vectors = db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    # fts_chunks is contentless and cannot be scanned; its docsize shadow table has a row per document.
    fts = db.execute("SELECT count(*) FROM fts_chunks_docsize").fetchone()[0]
    return chunks, vectors, fts


def run(cfg: Config, store: Store, extractor: FakeExtractor | None = None, prefetch: bool = True, **kw):
    emb = FakeEmbedder()
    ix = Indexer(store, cfg, emb, extract_fn=extractor or FakeExtractor(), prefetch=prefetch)
    return ix.run(**kw), emb


# -- schema --------------------------------------------------------------------

def test_migration_from_v1_backfills_file_names(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(f"BEGIN; {SCHEMA_V1}; PRAGMA user_version = 1; COMMIT;")
    db.execute("INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
               "VALUES ('/home/u/Documents/Opinnäytetyö.odt', 'office', 1, 1, 0, 50, 'indexed', 1)")
    db.commit()
    db.close()

    store = Store(path)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS) == 4
    assert store.embedding_dim() == 768
    hits = store.db.execute("SELECT rowid FROM fts_files WHERE fts_files MATCH ?", ['"opinnaytetyo"']).fetchall()
    assert len(hits) == 1
    store.close()


def test_every_connection_loads_sqlite_vec(tmp_path: Path) -> None:
    db = connect(tmp_path / "x.db")
    assert db.execute("SELECT vec_version()").fetchone()[0].startswith("v")


# -- indexing ------------------------------------------------------------------

@pytest.mark.parametrize("prefetch", [True, False])
def test_index_and_search(env, prefetch: bool) -> None:
    cfg, store = env
    stats, emb = run(cfg, store, prefetch=prefetch)
    assert (stats.indexed, stats.failed, stats.skipped, stats.chunks) == (3, 0, 0, 3)
    assert set(statuses(store).values()) == {"indexed"}
    assert table_counts(store) == (3, 3, 3)
    # One text chunk per file, batched across files into a single model call.
    assert emb.calls == [("text", 3)]
    assert store.get_meta("model_id") == "fake-bow"

    results = Searcher(store, cfg, emb).search("sourdough bread oven")
    assert results[0].path.endswith("bread.txt")
    assert results[0].similarity is not None and results[0].similarity > 0.4


def test_failed_files_are_recorded_and_others_continue(env) -> None:
    cfg, store = env
    extractor = FakeExtractor({"rockets.md": ExtractionError("corrupt file"),
                               "bread.txt": ValueError("boom")})
    stats, _ = run(cfg, store, extractor)
    assert (stats.indexed, stats.failed) == (1, 2)
    errors = {Path(r["path"]).name: r["error"] for r in store.failed_files()}
    assert errors == {"rockets.md": "corrupt file", "bread.txt": "ValueError: boom"}
    assert table_counts(store) == (1, 1, 1)


def test_embedding_failure_only_fails_that_file(env) -> None:
    cfg, store = env
    emb = FakeEmbedder()
    emb.fail_on = {DOCS["cats.md"]}
    Indexer(store, cfg, emb, extract_fn=FakeExtractor()).run()
    assert statuses(store) == {"cats.md": "failed", "rockets.md": "indexed", "bread.txt": "indexed"}
    assert "embedding: RuntimeError" in store.failed_files()[0]["error"]


@pytest.mark.parametrize("prefetch", [True, False])
def test_embedding_failure_mid_file_is_not_overwritten(tmp_path: Path, prefetch: bool) -> None:
    """Chunks still arriving for a file that already failed must not resurrect it as indexed."""
    cfg = make_cfg(tmp_path)
    cfg.batch_sizes["text"] = 1
    write_docs(tmp_path, {"big.md": "x", "ok.md": "y"})
    store = Store(cfg.db_path)
    scan(store, cfg)
    emb = FakeEmbedder()
    emb.fail_on = {"bad chunk"}
    extractor = FakeExtractor({"big.md": ["bad chunk", "fine", "also fine"], "ok.md": ["good"]})
    Indexer(store, cfg, emb, extract_fn=extractor, prefetch=prefetch).run()
    assert statuses(store) == {"big.md": "failed", "ok.md": "indexed"}
    assert table_counts(store) == (1, 1, 1)
    store.close()


def test_systemic_embedding_failure_aborts_run(env) -> None:
    cfg, store = env
    emb = FakeEmbedder()
    emb.fail_on = set(DOCS.values())
    with pytest.raises(RuntimeError):
        Indexer(store, cfg, emb, extract_fn=FakeExtractor()).run()
    assert set(statuses(store).values()) == {"pending"}


def test_changed_and_vanished_files_are_skipped(env, tmp_path: Path) -> None:
    cfg, store = env
    (tmp_path / "Documents/cats.md").write_text("changed after the scan, longer than before")
    (tmp_path / "Documents/rockets.md").unlink()
    stats, _ = run(cfg, store)
    assert (stats.indexed, stats.skipped) == (1, 2)
    assert statuses(store) == {"cats.md": "pending", "rockets.md": "pending", "bread.txt": "indexed"}


def test_zero_chunk_file_is_indexed_and_findable_by_name(env) -> None:
    cfg, store = env
    stats, emb = run(cfg, store, FakeExtractor({"rockets.md": []}))
    assert statuses(store)["rockets.md"] == "indexed"
    assert table_counts(store) == (2, 2, 2)
    results = Searcher(store, cfg, emb).search("rockets")
    assert [Path(r.path).name for r in results][:1] == ["rockets.md"]
    assert results[0].matched == ["name"] and results[0].similarity is None


def test_limit_kinds_and_priority_order(env, tmp_path: Path) -> None:
    cfg, store = env
    store.db.execute("UPDATE files SET priority = 90 WHERE path LIKE '%bread.txt'")
    store.db.commit()
    extractor = FakeExtractor()
    stats, _ = run(cfg, store, extractor, limit=2)
    assert stats.total == 2 and extractor.calls[0] == "bread.txt" and len(extractor.calls) == 2
    stats, _ = run(cfg, store, FakeExtractor(), kinds=["pdf"])
    assert stats.total == 0 and stats.done == 0


def test_reindex_replaces_chunks_and_vectors(env, tmp_path: Path) -> None:
    cfg, store = env
    run(cfg, store, FakeExtractor({"cats.md": ["a", "b", "c"]}))
    assert table_counts(store) == (5, 5, 5)

    p = tmp_path / "Documents/cats.md"
    p.write_text("a new version of the cat note")
    scan(store, cfg)
    assert statuses(store)["cats.md"] == "pending"
    run(cfg, store)
    assert table_counts(store) == (3, 3, 3)
    # Chunk ids can be reused, so check that every vector belongs to a live chunk.
    orphans = store.db.execute("SELECT count(*) FROM vec_chunks v LEFT JOIN chunks c ON c.id = v.chunk_id "
                               "WHERE c.id IS NULL").fetchone()[0]
    assert orphans == 0
    emb = FakeEmbedder()
    top = Searcher(store, cfg, emb).search("a new version of the cat note")[0]
    assert top.path.endswith("cats.md") and top.similarity is not None and top.similarity > 0.99


def test_removed_files_drop_vectors_and_fts(env, tmp_path: Path) -> None:
    cfg, store = env
    run(cfg, store, FakeExtractor({"cats.md": ["one", "two"]}))
    assert table_counts(store) == (4, 4, 4)
    (tmp_path / "Documents/cats.md").unlink()
    scan(store, cfg)  # finish_scan deletes the row; the cascade + triggers do the rest
    assert table_counts(store) == (2, 2, 2)
    assert store.db.execute("SELECT count(*) FROM fts_files").fetchone()[0] == 2
    emb = FakeEmbedder()
    assert all("cats" not in r.path for r in Searcher(store, cfg, emb).search("cat sofa"))


def test_model_change_resets_index(env) -> None:
    cfg, store = env
    run(cfg, store)
    assert store.model_matches("fake-bow", 768)
    assert not store.model_matches("other-model", 768)
    store.set_model("other-model", 768)
    assert set(statuses(store).values()) == {"pending"}
    assert table_counts(store) == (0, 0, 0)
    assert store.model_matches("other-model", 768)
    store.set_model("small-model", 256)
    assert store.embedding_dim() == 256
    store.db.execute("INSERT INTO vec_chunks (chunk_id, embedding, modality, kind) VALUES (1, ?, 'text', 'text')",
                     (np.zeros(256, dtype=np.float32),))


@pytest.mark.parametrize("prefetch", [True, False])
def test_stop_leaves_database_consistent(tmp_path: Path, prefetch: bool) -> None:
    cfg = make_cfg(tmp_path)
    cfg.batch_sizes["text"] = 4
    write_docs(tmp_path, {f"doc{i:02d}.md": f"document number {i}" for i in range(30)})
    store = Store(cfg.db_path)
    scan(store, cfg)
    stop = threading.Event()
    seen = 0

    def on_chunk(path: Path, i: int) -> None:
        nonlocal seen
        seen += 1
        if seen == 23:  # mid-file: doc with 3 chunks, stop on its 2nd chunk
            stop.set()

    extractor = FakeExtractor({f"doc{i:02d}.md": [f"part {j} of {i}" for j in range(3)] for i in range(30)},
                              on_chunk=on_chunk)
    stats, _ = run(cfg, store, extractor, prefetch=prefetch, stop=stop)
    assert stats.stopped
    st = statuses(store)
    indexed = [n for n, s in st.items() if s == "indexed"]
    assert 0 < len(indexed) < 30 and set(st.values()) <= {"indexed", "pending"}
    # Every indexed file has all of its chunks, pending ones none.
    per_file = dict(store.db.execute(
        "SELECT f.status, count(c.id) FROM files f LEFT JOIN chunks c ON c.file_id = f.id GROUP BY f.status"))
    assert per_file.get("pending", 0) == 0
    assert per_file["indexed"] == 3 * len(indexed)
    chunks, vectors, fts = table_counts(store)
    assert chunks == vectors == fts == 3 * len(indexed)
    # A second run picks up exactly the rest.
    stats, _ = run(cfg, store, FakeExtractor({n: ["x", "y", "z"] for n in st}))
    assert stats.indexed == 30 - len(indexed)
    assert set(statuses(store).values()) == {"indexed"}
    store.close()


def test_crash_in_extract_module_propagates(env) -> None:
    cfg, store = env

    def broken(path: Path, kind: str, cfg: Config):
        raise SystemExit("extractor import failed")  # not an Exception: must not be swallowed per file

    with pytest.raises(SystemExit):
        Indexer(store, cfg, FakeEmbedder(), extract_fn=broken).run()
    assert set(statuses(store).values()) == {"pending"}


# -- search --------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    '"', '""', "'", "a AND", "OR", "NOT x", "NEAR(a b)", "-", "--x", "foo-bar", "c++", "*", "x*",
    "(", ")", "^a", ":", "name:cat", "{a}", "a\"b", "  ", "_", "日本語", "😀", "a" * 500,
])
def test_fts_escaping_never_errors(env, query: str) -> None:
    cfg, store = env
    run(cfg, store)
    terms = query_terms(query)
    if terms:
        store.db.execute("SELECT rowid FROM fts_files WHERE fts_files MATCH ?",
                         [" ".join(_file_patterns(terms))]).fetchall()
        store.db.execute("SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?",
                         [" ".join(_patterns(terms))]).fetchall()
    Searcher(store, cfg, FakeEmbedder()).search(query)


def test_fts_prefix_rules(tmp_path: Path) -> None:
    # File names/dirs prefix every term >= 3 chars (>= 2 for last); chunk text prefixes only the
    # last term >= 3 chars, and only when not already in chunks as typed.
    assert _file_patterns(query_terms('rocket-engine "launch" OR x')) == [
        '"rocket"*', '"engine"*', '"launch"*', '"OR"', '"x"']
    assert _file_patterns(query_terms("thes pr")) == ['"thes"*', '"pr"*']
    assert _file_patterns(query_terms("!!! ---")) == []
    assert _patterns(query_terms('rocket-engine "launch"')) == ['"rocket"', '"engine"', '"launch"*']
    assert _patterns(query_terms("the re")) == ['"the"', '"re"']
    assert _patterns(query_terms("!!!")) == []

    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, {
        "a.md": "transformer attention mechanism",
        "b.md": "installing transformers with pip",
    })
    store = Store(cfg.db_path)
    scan(store, cfg)
    _, emb = run(store=store, cfg=cfg)
    s = Searcher(store, cfg, emb)
    # "transfo" is not indexed whole, so chunk text prefixes it -> matches "transformer" and "transformers".
    assert {"a.md", "b.md"} <= {Path(r.path).name for r in s.search("transfo")}
    # "transformer" IS indexed whole, so chunk text drops the wildcard -> "b.md" ("transformers") is not text-matched.
    res = s.search("transformer")
    assert Path(res[0].path).name == "a.md" and "text" in res[0].matched
    b_res = [r for r in res if Path(r.path).name == "b.md"]
    assert not b_res or "text" not in b_res[0].matched
    store.close()


def test_finnish_diacritics_both_ways(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, {"Opinnäytetyö_luonnos.md": "tämä on ensimmäinen versio",
                          "kauppalista.txt": "maito, leipä ja juusto"})
    store = Store(cfg.db_path)
    scan(store, cfg)
    stats, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    for q in ("opinnaytetyo", "Opinnäytetyö", "OPINNÄYTE", "opinnäytetyöni"[:10]):
        assert Path(s.search(q)[0].path).name == "Opinnäytetyö_luonnos.md", q
    assert Path(s.search("leipa")[0].path).name == "kauppalista.txt"  # chunk text, folded
    store.close()


def test_camel_case_and_prefix_names(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, {"MyThesisDraft2024.md": "zzz", "other.md": "yyy"})
    store = Store(cfg.db_path)
    scan(store, cfg)
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    for q in ("thesis", "draft", "mythes", "2024", "thesis draft"):
        assert Path(s.search(q)[0].path).name == "MyThesisDraft2024.md", q
    store.close()


def test_name_and_content_combine(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, {
        # Matches by content only.
        "notes.md": "orbital rocket launch window calculations",
        # Matches by name and content: should win.
        "rocket_launch.md": "orbital rocket launch checklist",
        # Matches by name only.
        "rocket_launch_photo_list.md": "unrelated words entirely",
        "garden.md": "tomatoes and cucumbers in the greenhouse",
    })
    store = Store(cfg.db_path)
    scan(store, cfg)
    _, emb = run(cfg, store)
    results = Searcher(store, cfg, emb).search("rocket launch", limit=10)
    names = [Path(r.path).name for r in results]
    assert names[0] == "rocket_launch.md"
    assert set(results[0].matched) == {"vector", "name", "text"}
    assert "rocket_launch_photo_list.md" in names and "notes.md" in names
    # The unrelated file is dropped by the similarity floor (it has no lexical match).
    assert "garden.md" not in names
    store.close()


def test_kind_filter(env, tmp_path: Path) -> None:
    cfg, store = env
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    assert s.search("bread", kinds=["image"]) == []
    assert [r.kind for r in s.search("bread", kinds=["text"])] == ["text"]


def test_z_gate_drops_vector_only_matches(env) -> None:
    cfg, store = env
    _, emb = run(cfg, store)
    # Only "sofa" occurs in a file (1 of 4 words: too few to count), so only meaning can find it.
    query = "sofa feline kitten whiskers"
    cfg.z_gate = 10.0
    assert Searcher(store, cfg, emb).search(query) == []
    cfg.z_gate = 1.0
    res = Searcher(store, cfg, emb).search(query)
    assert [Path(r.path).name for r in res] == ["cats.md"] and "vector" in res[0].matched
    # Half the words found in the text keep a file whatever its meaning score.
    cfg.z_gate = 10.0
    res = Searcher(store, cfg, emb).search("warm sofa feline")
    assert [Path(r.path).name for r in res] == ["cats.md"] and res[0].matched == ["text"]


def test_reset_index(env) -> None:
    cfg, store = env
    run(cfg, store)
    store.reset_index()
    assert table_counts(store) == (0, 0, 0)
    assert set(statuses(store).values()) == {"pending"}
    assert store.db.execute("SELECT count(*) FROM fts_files").fetchone()[0] == 3


def test_search_latency_on_synthetic_index(tmp_path: Path) -> None:
    """20k chunks: the post-embedding pipeline should stay well under the 50 ms target."""
    import time

    cfg = make_cfg(tmp_path)
    store = Store(cfg.db_path)
    rng = np.random.default_rng(0)
    with store.db:
        for i in range(2000):
            store.db.execute("INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
                             "VALUES (?, 'text', 1, 1, 0, 1, 'indexed', 1)", (f"/d/file{i}.md",))
        for i in range(20_000):
            fid = i % 2000 + 1
            cur = store.db.execute("INSERT INTO chunks (file_id, seq, modality) VALUES (?, 0, 'text')", (fid,))
            v = rng.standard_normal(768).astype(np.float32)
            store.db.execute("INSERT INTO vec_chunks VALUES (?, ?, 'text', 'text')", (cur.lastrowid, v / np.linalg.norm(v)))
            store.db.execute("INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)", (cur.lastrowid, f"word{i % 97} common"))
    s = Searcher(store, cfg, FakeEmbedder())
    q = rng.standard_normal(768).astype(np.float32)
    s.search_vector(q, "common word5", 10)
    t = time.perf_counter()
    for _ in range(5):
        s.search_vector(q, "common word5", 10)
    ms = (time.perf_counter() - t) / 5 * 1000
    assert ms < 200, ms  # generous for slow CI; observed numbers are in the report
    print(f"\nsearch over 20k chunks: {ms:.1f} ms")
