"""SQLite storage.

Schema v1: the `files` table filled by the scanner (the indexing queue).
Schema v2: `chunks` (one row per embedded piece of a file), `vec_chunks` (sqlite-vec vectors),
`fts_files` (file names and folders) and `fts_chunks` (text chunk contents) for lexical matching.
Schema v3: `files.attempts`; `fts_files.dirs` holds only the last folder names (not the absolute
path); `fts_chunks` gets a 3-character prefix index (it is contentless, so it is recreated empty
and the indexer re-queues every file through the index-format version); `counters.chunks`
counts every chunk insert/delete so an in-memory vector cache can tell when another process
changed the index.

Deletion is driven by triggers so every path that removes rows (re-indexing, `finish_scan`,
`reset`) keeps the virtual tables in step: deleting a file cascades (FK) to its chunks, and the
chunk trigger removes the vector and the FTS row. vec0/FTS5 tables do not take part in FK
cascades themselves, but triggers fire for cascaded deletes.

Connections: one per thread (`Store.db`), plus a read-only one per thread for searches
(`Store.reader()`). WAL gives every reader a consistent snapshot while one writer commits.
Writes go through `Store.write()`, an IMMEDIATE transaction (takes the write lock up front, so
a busy database waits in the busy handler instead of failing with SQLITE_BUSY mid-transaction).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from .scanner import ScannedFile
from .vectors import Delta, VectorCache, chunks_version

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
BUSY_TIMEOUT_S = 10.0

# The FTS tokenizer folds case and strips diacritics ("opinnäytetyö" matches "opinnaytetyo").
# Queries add a prefix wildcard to the last term (see search.py), which covers type-as-you-go
# and Finnish inflection ("opinnäytetyö" finds "opinnäytetyöni"); trigram would also match
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

SCHEMA_V3 = f"""
ALTER TABLE files ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
-- gs_dir_terms now returns only the last folder names: rebuild the file-name index.
DELETE FROM fts_files;
INSERT INTO fts_files (rowid, name, dirs) SELECT id, gs_name_terms(path), gs_dir_terms(path) FROM files;
DROP TRIGGER chunks_ad;
DROP TABLE fts_chunks;
CREATE VIRTUAL TABLE fts_chunks USING fts5(text, content='', contentless_delete=1,
                                           tokenize="{FTS_TOKENIZE}", prefix='3');
CREATE TABLE counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT INTO counters (name, value) VALUES ('chunks', 0);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    UPDATE counters SET value = value + 1 WHERE name = 'chunks';
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM vec_chunks WHERE chunk_id = old.id;
    DELETE FROM fts_chunks WHERE rowid = old.id;
    UPDATE counters SET value = value + 1 WHERE name = 'chunks';
END;
"""

MIGRATIONS: list[str] = [SCHEMA_V1, SCHEMA_V2, SCHEMA_V3]

# Word boundaries inside file names that unicode61 does not see: camelCase and letter/digit.
_CAMEL = re.compile(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Þ])|(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")

# Leading path parts that say nothing about a file: the home directory (its user name matches
# every file) and mount points of data drives.
_ROOT_PREFIX = re.compile(r"^(?:/var)?/home/[^/]+|^/root(?=/|$)|^/run/media/[^/]+/[^/]+"
                          r"|^/media/[^/]+/[^/]+|^/mnt/[^/]+")
DIR_TERMS_LEVELS = 3


def name_terms(path: str) -> str:
    """FTS text for a file name: the name itself plus a camelCase/digit-split copy.

    "MyThesis2024.pdf" -> "MyThesis2024.pdf My Thesis 2024.pdf", so both "mythesis" and
    "thesis" match.
    """
    name = os.path.basename(path)
    split = _CAMEL.sub(" ", name)
    return name if split == name else f"{name} {split}"


def dir_terms(path: str) -> str:
    """FTS text for a file's folder: the last few folder names below home / the mount point.

    "/home/u/Documents/thesis/v3/a.md" -> "Documents thesis v3". The absolute path would make
    "home", the user name or "documents" match every file. Search treats folder-only matches
    as weak (they must still pass the similarity floor).
    """
    rest = _ROOT_PREFIX.sub("", os.path.dirname(path), count=1)
    parts = [p for p in rest.split("/") if p]
    return " ".join(parts[-DIR_TERMS_LEVELS:])


UPSERT = """
INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen)
VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
ON CONFLICT (path) DO UPDATE SET
    status = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                       OR files.kind != excluded.kind
                  THEN 'pending' ELSE files.status END,
    error  = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                  THEN NULL ELSE files.error END,
    attempts = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                    THEN 0 ELSE files.attempts END,
    kind = excluded.kind, size = excluded.size, mtime = excluded.mtime,
    in_project = excluded.in_project, priority = excluded.priority, last_seen = excluded.last_seen
"""


def connect(path: Path | str, readonly: bool = False, autocommit: bool | None = None) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and the functions the triggers call.

    `check_same_thread=False` only so the owning Store can close it from any thread; each
    connection is used by a single thread (see Store.db / Store.reader).
    """
    kwargs = {} if autocommit is None else {"autocommit": autocommit}
    db = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_S, check_same_thread=False, **kwargs)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.create_function("gs_name_terms", 1, name_terms, deterministic=True)
    db.create_function("gs_dir_terms", 1, dir_terms, deterministic=True)
    if str(path) != ":memory:":
        db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    if readonly:
        db.execute("PRAGMA query_only=1")
    return db


@dataclass(slots=True)
class ChunkRecord:
    modality: str
    locator: str
    snippet: str
    position: float | None
    text: str | None  # full text for the FTS (text chunks only)
    vector: np.ndarray


class SchemaTooNew(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path | str, vector_cache: bool = False, cache_max_chunks: int | None = None):
        self.path = path
        self._memory = str(path) == ":memory:"
        if not self._memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        # Serialises this process's write transactions so the vector cache sees them in commit
        # order. (Other processes are serialised by SQLite's own write lock.)
        self._write_lock = threading.RLock()
        self._shared: sqlite3.Connection | None = None
        if self._memory:  # tests only: one database, so one connection for every thread
            self._shared = connect(path)
            self._conns.append(self._shared)
        self._migrate()
        self.vectors: VectorCache | None = None
        if vector_cache:
            self.enable_vector_cache(cache_max_chunks)

    # -- connections --------------------------------------------------------------
    def connect(self, readonly: bool = False) -> sqlite3.Connection:
        """A new connection owned by the caller."""
        return connect(self.path, readonly=readonly)

    def _thread_conn(self, attr: str, readonly: bool) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, attr, None)
        if conn is None:
            conn = connect(self.path, readonly=readonly)
            setattr(self._local, attr, conn)
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    @property
    def db(self) -> sqlite3.Connection:
        """This thread's read/write connection."""
        return self._thread_conn("rw", False)

    def reader(self) -> sqlite3.Connection:
        """This thread's read-only connection (PRAGMA query_only), for searches."""
        return self._thread_conn("ro", True)

    def release_thread(self) -> None:
        """Close the calling thread's connections (a thread that is about to exit; otherwise
        they would stay open until `close()`, one per short-lived thread in a daemon)."""
        if self._shared is not None:
            return
        for attr in ("rw", "ro"):
            conn = getattr(self._local, attr, None)
            if conn is None:
                continue
            delattr(self._local, attr)
            with self._conns_lock:
                if conn in self._conns:
                    self._conns.remove(conn)
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def close(self) -> None:
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()

    def _migrate(self) -> None:
        if self._memory:
            db = self._shared
            assert db is not None
            version = db.execute("PRAGMA user_version").fetchone()[0]
            for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
                db.executescript(f"BEGIN; {script}; PRAGMA user_version = {i}; COMMIT;")
            return
        db = connect(self.path, autocommit=True)
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS):
                return
            # Re-read the version under the write lock: another process may be migrating too.
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > len(MIGRATIONS):
                raise SchemaTooNew(f"{self.path} has schema v{version}; this gemmasearch knows v{len(MIGRATIONS)}")
            for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
                db.executescript(script)
                db.execute(f"PRAGMA user_version = {i}")
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    @contextmanager
    def write(self) -> Iterator[tuple[sqlite3.Connection, Delta]]:
        """An IMMEDIATE write transaction on this thread's connection; commits on success.

        Yields (connection, delta); methods that change chunks record the change in `delta` so
        the vector cache can be patched in step with the commit.
        """
        db = self.db
        cache = self.vectors
        delta = Delta()
        before = after = None
        committed = False
        with self._write_lock:
            if cache is not None:
                cache.begin_local_write()
            try:
                db.execute("BEGIN IMMEDIATE")
                try:
                    if cache is not None:
                        before = chunks_version(db)
                    yield db, delta
                    if cache is not None:
                        after = chunks_version(db)
                    db.commit()
                    committed = True
                except BaseException:
                    db.rollback()
                    raise
            finally:
                if cache is not None:
                    cache.end_local_write(before, after, delta if committed else None)

    def enable_vector_cache(self, max_chunks: int | None = None) -> VectorCache:
        """Keep an in-memory copy of all vectors for fast KNN (loaded lazily on first search)."""
        if self.vectors is None:
            self.vectors = VectorCache(self.embedding_dim(), max_chunks)
        return self.vectors

    def warm_vector_cache(self, max_chunks: int | None = None) -> VectorCache:
        """Enable the in-memory vector cache and load it now (for CLI/eval timings)."""
        cache = self.enable_vector_cache(max_chunks)
        with cache.lock:
            cache.ensure_fresh(self.reader())
        return cache

    # -- meta -----------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.write() as (db, _):
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # -- scanning -------------------------------------------------------------
    def begin_scan(self) -> int:
        with self.write() as (db, _):
            row = db.execute("SELECT value FROM meta WHERE key = 'scan_generation'").fetchone()
            gen = int(row[0] if row else 0) + 1
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('scan_generation', ?)", (str(gen),))
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
        with self.write() as (db, _):
            db.executemany(UPSERT, rows)
        count = len(rows)
        rows.clear()
        return count

    def finish_scan(self, generation: int, scanned_roots: Iterable[Path],
                    failed_dirs: Iterable[str] = ()) -> int:
        """Forget files under the scanned roots that this scan did not see (deleted or now excluded).

        Files below a directory that could not be listed (permission error, I/O error, stale
        network mount, unmounted root) are kept: not seeing them proves nothing.
        Their chunks cascade away, and the triggers drop the vectors and FTS rows with them.
        """
        failed = tuple(d.rstrip("/") + "/" for d in failed_dirs)
        removed = 0
        with self.write() as (db, delta):
            for root in scanned_roots:
                prefix = str(root).rstrip("/") + "/"
                if prefix in failed:
                    continue
                rows = db.execute(
                    "SELECT id, path FROM files WHERE last_seen < ? AND substr(path, 1, ?) = ?",
                    (generation, len(prefix), prefix),
                ).fetchall()
                doomed = [r[0] for r in rows if not r[1].startswith(failed)]
                db.executemany("DELETE FROM files WHERE id = ?", [(d,) for d in doomed])
                delta.removed_files.update(doomed)
                removed += len(doomed)
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_scan_at', ?)", (str(time.time()),))
        return removed

    # -- queue ----------------------------------------------------------------
    def counts(self) -> dict[tuple[str, str], int]:
        rows = self.db.execute("SELECT kind, status, count(*) FROM files GROUP BY kind, status")
        return {(r[0], r[1]): r[2] for r in rows}

    def requeue(self, statuses: Sequence[str] = ("indexed", "failed"), reset_attempts: bool = True) -> int:
        """Mark files with these statuses pending again. Their chunks stay searchable until the
        indexer replaces them."""
        marks = ",".join("?" * len(statuses))
        with self.write() as (db, _):
            cur = db.execute(f"UPDATE files SET status = 'pending', error = NULL"
                             f"{', attempts = 0' if reset_attempts else ''} WHERE status IN ({marks})",
                             list(statuses))
            return cur.rowcount

    def requeue_failed_for_retry(self, max_attempts: int, backoff_s: float, now: float | None = None) -> int:
        """Automatic retry: failed files with fewer than `max_attempts` attempts whose last attempt
        is at least `backoff_s * attempts` old (transient causes: a file locked by another
        program, a missing tool installed since, a network share that was offline)."""
        now = time.time() if now is None else now
        with self.write() as (db, _):
            return db.execute("UPDATE files SET status = 'pending' WHERE status = 'failed' AND attempts < ? "
                              "AND coalesce(indexed_at, 0) <= ? - ? * attempts",
                              (max_attempts, now, backoff_s)).rowcount

    # -- index ----------------------------------------------------------------
    def embedding_dim(self) -> int:
        return int(self.get_meta("embedding_dim", str(EMBEDDING_DIM)) or EMBEDDING_DIM)

    def has_chunks(self) -> bool:
        return self.db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None

    def model_matches(self, model_id: str, dim: int) -> bool:
        """True if the stored vectors come from this model, or there are none yet."""
        stored = self.get_meta("model_id")
        if stored is None:
            return not self.has_chunks() and self.embedding_dim() == dim
        return stored == model_id and self.embedding_dim() == dim

    def set_model(self, model_id: str, dim: int) -> None:
        """Record the model the vectors come from, dropping every vector if it changed."""
        if not self.model_matches(model_id, dim):
            self.reset_index(dim)
        self.set_meta("model_id", model_id)

    def reset_index(self, dim: int | None = None) -> None:
        """Delete all chunks, vectors and chunk text and mark every file pending again.
        File rows (the scan queue) and the file-name FTS are kept."""
        with self.write() as (db, delta):
            # Bulk-clear the virtual tables first so the per-row chunk trigger finds nothing.
            db.execute("DELETE FROM vec_chunks")
            db.execute("INSERT INTO fts_chunks (fts_chunks) VALUES ('delete-all')")
            db.execute("DELETE FROM chunks")
            db.execute("UPDATE files SET status = 'pending', error = NULL, indexed_at = NULL, attempts = 0 "
                       "WHERE status != 'pending' OR attempts != 0")
            current = int((db.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
                           or [EMBEDDING_DIM])[0])
            if dim is not None and dim != current:
                db.execute("DROP TABLE vec_chunks")
                db.execute(vec_table_sql(dim))
                db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_dim', ?)", (str(dim),))
            delta.cleared = True
        if self.vectors is not None and dim is not None and dim != self.vectors.dim:
            self.vectors = VectorCache(dim, self.vectors.max_chunks)

    def replace_chunks(self, file_id: int, size: int, mtime: float, kind: str,
                       records: Sequence[ChunkRecord]) -> bool:
        """Replace a file's chunks, vectors and FTS rows and mark it indexed, atomically.

        Returns False (and changes nothing) if the file row is gone or a newer scan saw a
        different version of the file (it stays pending for the next run).
        """
        with self.write() as (db, delta):
            row = db.execute("SELECT size, mtime FROM files WHERE id = ?", (file_id,)).fetchone()
            if row is None or row[0] != size or row[1] != mtime:
                return False
            db.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))  # triggers clear vec/fts
            delta.removed_files.add(file_id)
            for seq, rec in enumerate(records):
                cur = db.execute(
                    "INSERT INTO chunks (file_id, seq, modality, locator, snippet, position) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (file_id, seq, rec.modality, rec.locator, rec.snippet, rec.position),
                )
                chunk_id = cur.lastrowid
                assert chunk_id is not None
                vec = np.ascontiguousarray(rec.vector, dtype=np.float32)
                db.execute("INSERT INTO vec_chunks (chunk_id, embedding, modality, kind) VALUES (?, ?, ?, ?)",
                           (chunk_id, vec, rec.modality, kind))
                if rec.text:
                    db.execute("INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)", (chunk_id, rec.text))
                delta.added.append((chunk_id, file_id, vec, rec.modality, kind))
            db.execute("UPDATE files SET status = 'indexed', error = NULL, indexed_at = ?, attempts = 0 "
                       "WHERE id = ?", (time.time(), file_id))
        return True

    def mark_failed(self, file_id: int, error: str) -> None:
        with self.write() as (db, _):
            db.execute("UPDATE files SET status = 'failed', error = ?, indexed_at = ?, attempts = attempts + 1 "
                       "WHERE id = ?", (error[:2000], time.time(), file_id))

    def index_counts(self) -> dict[str, int]:
        """Chunk counts per modality, plus total vectors (should equal total chunks)."""
        out = {r[0]: r[1] for r in self.db.execute("SELECT modality, count(*) FROM chunks GROUP BY modality")}
        out["vectors"] = self.db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
        return out

    def failed_files(self, limit: int | None = None) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT path, kind, error, attempts FROM files WHERE status = 'failed' ORDER BY priority DESC LIMIT ?",
            (-1 if limit is None else limit,),
        ).fetchall()


def db_users(path: Path | str) -> list[int]:
    """PIDs of other processes that have the database (or its WAL) open. Linux only, best effort:
    processes of other users are invisible."""
    targets = set()
    for suffix in ("", "-wal", "-shm"):
        try:
            targets.add(os.path.realpath(f"{path}{suffix}"))
        except OSError:
            pass
    me = os.getpid()
    pids: list[int] = []
    try:
        procs = os.listdir("/proc")
    except OSError:
        return pids
    for p in procs:
        if not p.isdigit() or int(p) == me:
            continue
        try:
            for fd in os.listdir(f"/proc/{p}/fd"):
                try:
                    if os.readlink(f"/proc/{p}/fd/{fd}") in targets:
                        pids.append(int(p))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids
