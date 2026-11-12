"""SQLite storage.

Schema v1: the `files` table filled by the scanner (the indexing queue).
Schema v2: `chunks` (one row per embedded piece of a file), `vec_chunks` (sqlite-vec vectors),
`fts_files` (file names and folders) and `fts_chunks` (text chunk contents) for lexical matching.

Deletion is driven by triggers so every path that removes rows (re-indexing, `finish_scan`,
`reset`) keeps the virtual tables in step: deleting a file cascades (FK) to its chunks, and the
chunk trigger removes the vector and the FTS row. vec0/FTS5 tables do not take part in FK
cascades themselves, but triggers fire for cascaded deletes.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import sqlite_vec

from .scanner import ScannedFile

SCHEMA_V1 = """
CREATE TABLE files (
    id          INTEGER PRIMARY KEY,
    path        TEXT    NOT NULL UNIQUE,
    kind        TEXT    NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL    NOT NULL,
    in_project  INTEGER NOT NULL,
    priority    REAL    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | indexed | failed
    error       TEXT,
    indexed_at  REAL,
    last_seen   INTEGER NOT NULL                     -- scan generation that last saw the file
);
CREATE INDEX files_queue ON files (status, priority DESC, mtime DESC);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

EMBEDDING_DIM = 768

# The FTS tokenizer folds case and strips diacritics ("opinnäytetyö" matches "opinnaytetyo").
# Queries add a prefix wildcard to every term (see search.py), which covers type-as-you-go and
# Finnish inflection ("opinnäytetyö" finds "opinnäytetyöni"); trigram would also match
# mid-word substrings, but triples the index size for chunk text and makes BM25 noisy.
FTS_TOKENIZE = "unicode61 remove_diacritics 2"


def vec_table_sql(dim: int) -> str:
    # `kind` lets `search --kind` filter inside the KNN instead of after it.
    return (f"CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, "
            f"embedding float[{dim}] distance_metric=cosine, modality TEXT, kind TEXT)")


SCHEMA_V2 = f"""
CREATE TABLE chunks (
    id        INTEGER PRIMARY KEY,
    file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,
    modality  TEXT    NOT NULL,
    locator   TEXT    NOT NULL DEFAULT '',
    snippet   TEXT    NOT NULL DEFAULT '',
    position  REAL
);
CREATE INDEX chunks_file ON chunks (file_id);
{vec_table_sql(EMBEDDING_DIM)};
CREATE VIRTUAL TABLE fts_files USING fts5(name, dirs, tokenize="{FTS_TOKENIZE}", prefix='2 3');
CREATE VIRTUAL TABLE fts_chunks USING fts5(text, content='', contentless_delete=1,
                                           tokenize="{FTS_TOKENIZE}");
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM vec_chunks WHERE chunk_id = old.id;
    DELETE FROM fts_chunks WHERE rowid = old.id;
END;
-- File names are searchable as soon as the scanner sees a file, before it is indexed.
-- A row's path never changes (it is the upsert key), so insert and delete are enough.
CREATE TRIGGER files_ai AFTER INSERT ON files BEGIN
    INSERT INTO fts_files (rowid, name, dirs) VALUES (new.id, gs_name_terms(new.path), gs_dir_terms(new.path));
END;
CREATE TRIGGER files_ad AFTER DELETE ON files BEGIN
    DELETE FROM fts_files WHERE rowid = old.id;
END;
INSERT INTO fts_files (rowid, name, dirs) SELECT id, gs_name_terms(path), gs_dir_terms(path) FROM files;
INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_dim', '{EMBEDDING_DIM}');
"""

MIGRATIONS: list[str] = [SCHEMA_V1, SCHEMA_V2]

# Word boundaries inside file names that unicode61 does not see: camelCase and letter/digit.
_CAMEL = re.compile(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Þ])|(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")


def name_terms(path: str) -> str:
    """FTS text for a file name: the name itself plus a camelCase/digit-split copy.

    "MyThesis2024.pdf" -> "MyThesis2024.pdf My Thesis 2024.pdf", so both "mythesis" and
    "thesis" match.
    """
    name = os.path.basename(path)
    split = _CAMEL.sub(" ", name)
    return name if split == name else f"{name} {split}"


def dir_terms(path: str) -> str:
    return os.path.dirname(path)

UPSERT = """
INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen)
VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
ON CONFLICT (path) DO UPDATE SET
    status = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                       OR files.kind != excluded.kind
                  THEN 'pending' ELSE files.status END,
    error  = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                  THEN NULL ELSE files.error END,
    kind = excluded.kind, size = excluded.size, mtime = excluded.mtime,
    in_project = excluded.in_project, priority = excluded.priority, last_seen = excluded.last_seen
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and the functions the triggers call."""
    db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.create_function("gs_name_terms", 1, name_terms, deterministic=True)
    db.create_function("gs_dir_terms", 1, dir_terms, deterministic=True)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = connect(path)
        # Serialises use of the connection between the indexer's threads (queue reads in the
        # extraction thread, write transactions in the embedding thread).
        self.lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.db:
                self.db.executescript(f"BEGIN; {script}; PRAGMA user_version = {i}; COMMIT;")

    def close(self) -> None:
        self.db.close()

    # -- meta -----------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # -- scanning -------------------------------------------------------------
    def begin_scan(self) -> int:
        gen = int(self.get_meta("scan_generation", "0")) + 1
        self.set_meta("scan_generation", str(gen))
        return gen

    def upsert_scanned(self, files: Iterable[ScannedFile], generation: int, batch: int = 2000) -> int:
        n = 0
        rows: list[tuple] = []
        for f in files:
            rows.append((f.path, f.kind, f.size, f.mtime, int(f.in_project), f.score.total, generation))
            if len(rows) >= batch:
                n += self._flush(rows)
        n += self._flush(rows)
        return n

    def _flush(self, rows: list[tuple]) -> int:
        if not rows:
            return 0
        with self.db:
            self.db.executemany(UPSERT, rows)
        count = len(rows)
        rows.clear()
        return count

    def finish_scan(self, generation: int, scanned_roots: Iterable[Path]) -> int:
        """Forget files under the scanned roots that this scan did not see (deleted or now excluded).

        Only roots that were actually scanned are touched, so an unmounted drive keeps its index.
        Their chunks cascade away, and the triggers drop the vectors and FTS rows with them.
        """
        removed = 0
        with self.db:
            for root in scanned_roots:
                prefix = str(root).rstrip("/") + "/"
                cur = self.db.execute(
                    "DELETE FROM files WHERE last_seen < ? AND substr(path, 1, ?) = ?",
                    (generation, len(prefix), prefix),
                )
                removed += cur.rowcount
        self.set_meta("last_scan_at", str(time.time()))
        return removed

    # -- queue ----------------------------------------------------------------
    def counts(self) -> dict[tuple[str, str], int]:
        rows = self.db.execute("SELECT kind, status, count(*) FROM files GROUP BY kind, status")
        return {(r[0], r[1]): r[2] for r in rows}

    # -- index ----------------------------------------------------------------
    def embedding_dim(self) -> int:
        return int(self.get_meta("embedding_dim", str(EMBEDDING_DIM)) or EMBEDDING_DIM)

    def ensure_model(self, model_id: str, dim: int) -> bool:
        """Record which model the vectors come from. If it changed, drop all vectors and queue
        every file again (vectors from different models are not comparable). Returns True if
        the index was reset."""
        stored = self.get_meta("model_id")
        if stored == model_id and self.embedding_dim() == dim:
            return False
        has_chunks = self.db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None
        reset = stored is not None or has_chunks or self.embedding_dim() != dim
        if reset:
            self.reset_index(dim)
        self.set_meta("model_id", model_id)
        return reset

    def reset_index(self, dim: int | None = None) -> None:
        """Delete all chunks, vectors and chunk text and mark every file pending again.
        File rows (the scan queue) and the file-name FTS are kept."""
        with self.lock, self.db:
            # Bulk-clear the virtual tables first so the per-row chunk trigger finds nothing.
            self.db.execute("DELETE FROM vec_chunks")
            self.db.execute("INSERT INTO fts_chunks (fts_chunks) VALUES ('delete-all')")
            self.db.execute("DELETE FROM chunks")
            self.db.execute("UPDATE files SET status = 'pending', error = NULL, indexed_at = NULL "
                            "WHERE status != 'pending'")
            if dim is not None and dim != self.embedding_dim():
                self.db.execute("DROP TABLE vec_chunks")
                self.db.execute(vec_table_sql(dim))
                self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_dim', ?)",
                                (str(dim),))

    def index_counts(self) -> dict[str, int]:
        """Chunk counts per modality, plus total vectors (should equal total chunks)."""
        out = {r[0]: r[1] for r in self.db.execute("SELECT modality, count(*) FROM chunks GROUP BY modality")}
        out["vectors"] = self.db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
        return out

    def failed_files(self, limit: int | None = None) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT path, kind, error FROM files WHERE status = 'failed' ORDER BY priority DESC LIMIT ?",
            (-1 if limit is None else limit,),
        ).fetchall()
