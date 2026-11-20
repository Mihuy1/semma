"""In-memory copy of the chunk vectors for a long-lived process (the search daemon).

sqlite-vec stays the storage format and the fallback search path; its KNN is a brute-force scan
that reads every vector from SQLite (~110 ms at 100k chunks). This cache holds the same vectors
as one float32 matrix, so a query is a single matrix-vector product (~5 ms at 100k chunks,
307 MB of RAM).

Consistency:
- Writes made through this process's `Store.write()` are applied to the cache right after they
  commit (`apply`), so the indexer's own commits never force a reload.
- Every insert/delete on `chunks` bumps `counters.chunks` (a trigger), whoever makes it. The
  cache remembers the counter value its contents correspond to; when a search sees another value
  and no local write is in flight, another process changed the index and the cache reloads.
- A local write records the counter before and after its changes (inside its IMMEDIATE
  transaction, so nobody else can interleave). If the cache was not at the "before" value, it
  missed a foreign change and is marked stale instead of being patched.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

_GROW_MIN = 1024


@dataclass
class Delta:
    """What one committed local write changed (filled in by the Store's write methods)."""

    cleared: bool = False
    removed_files: set[int] = field(default_factory=set)
    added: list[tuple[int, int, np.ndarray, str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.cleared or bool(self.removed_files) or bool(self.added)


@dataclass
class VectorScores:
    """One query scored against every row of the cache (see `VectorCache.scores`)."""

    sims: np.ndarray
    chunk_ids: np.ndarray
    file_ids: np.ndarray
    modality: np.ndarray
    kind: np.ndarray
    valid: np.ndarray | None
    names: list[str]
    files: np.ndarray


def chunks_version(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT value FROM counters WHERE name = 'chunks'").fetchone()
    return int(row[0]) if row else 0


class VectorCache:
    def __init__(self, dim: int, max_chunks: int | None = None):
        self.dim = dim
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.version: int | None = None
        self.too_big = False
        self.loads = 0
        self.fast_loads = 0
        self._local_writes = 0
        self._codes: dict[str, int] = {}
        self._names: list[str] = []
        self._reset(0)

    def _reset(self, capacity: int) -> None:
        capacity = max(capacity, _GROW_MIN)
        self.n = 0
        self.dead = 0
        self.mat = np.zeros((capacity, self.dim), dtype=np.float32)
        self.chunk_ids = np.zeros(capacity, dtype=np.int64)
        self.file_ids = np.zeros(capacity, dtype=np.int64)
        self.modality = np.zeros(capacity, dtype=np.int16)
        self.kind = np.zeros(capacity, dtype=np.int16)
        self.alive = np.zeros(capacity, dtype=bool)
        self.row_of: dict[int, int] = {}
        self._files: np.ndarray | None = None

    def _code(self, name: str) -> int:
        code = self._codes.get(name)
        if code is None:
            code = self._codes[name] = len(self._names)
            self._names.append(name)
        return code

    def _grow(self, need: int) -> None:
        cap = len(self.chunk_ids)
        if self.n + need <= cap:
            return
        new = max(cap * 2, self.n + need)
        for attr in ("mat", "chunk_ids", "file_ids", "modality", "kind", "alive"):
            old = getattr(self, attr)
            arr = np.zeros((new, *old.shape[1:]), dtype=old.dtype)
            arr[: self.n] = old[: self.n]
            setattr(self, attr, arr)

    def _append(self, chunk_ids: np.ndarray, file_ids: np.ndarray, vecs: np.ndarray,
                modalities: Sequence[str], kinds: Sequence[str]) -> None:
        k = len(chunk_ids)
        if not k:
            return
        self._grow(k)
        s = slice(self.n, self.n + k)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.mat[s] = vecs / np.where(norms > 0, norms, 1)
        self.chunk_ids[s] = chunk_ids
        self.file_ids[s] = file_ids
        self.modality[s] = [self._code(m) for m in modalities]
        self.kind[s] = [self._code(x) for x in kinds]
        self.alive[s] = True
        for i, cid in enumerate(chunk_ids.tolist()):
            old = self.row_of.get(cid)
            if old is not None and self.alive[old]:
                self.alive[old] = False
                self.dead += 1
            self.row_of[cid] = self.n + i
        self.n += k
        self._files = None

    def _remove_rows(self, rows: np.ndarray) -> None:
        rows = rows[self.alive[rows]]
        if not len(rows):
            return
        self.alive[rows] = False
        self.dead += len(rows)
        for row, cid in zip(rows.tolist(), self.chunk_ids[rows].tolist(), strict=True):
            if self.row_of.get(cid) == row:
                del self.row_of[cid]
        self._files = None

    def _maybe_compact(self) -> None:
        if self.dead < max(_GROW_MIN, self.n // 4):
            return
        keep = np.flatnonzero(self.alive[: self.n])
        mat, cids, fids = self.mat[keep].copy(), self.chunk_ids[keep].copy(), self.file_ids[keep].copy()
        mods, kinds = self.modality[keep].copy(), self.kind[keep].copy()
        self._reset(len(keep) * 2)
        k = len(keep)
        self.mat[:k], self.chunk_ids[:k], self.file_ids[:k] = mat, cids, fids
        self.modality[:k], self.kind[:k], self.alive[:k] = mods, kinds, True
        self.row_of = {c: i for i, c in enumerate(cids.tolist())}
        self.n = k
        self._files = None

    @property
    def size(self) -> int:
        return self.n - self.dead

    def load(self, db: sqlite3.Connection) -> None:
        """(Re)read every vector in one read transaction (call with `lock` held)."""
        own_txn = not db.in_transaction
        if own_txn:
            db.execute("BEGIN")
        try:
            version = chunks_version(db)
            count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if self.max_chunks is not None and count > self.max_chunks:
                self.too_big = True
                self._reset(0)
                self.version = version
                return
            self.too_big = False
            self._reset(count + count // 8)
            if self._load_shadow(db, count):
                self.fast_loads += 1
            else:
                self._reset(count + count // 8)
                cur = db.execute("SELECT v.chunk_id, v.embedding, v.modality, v.kind, c.file_id "
                                 "FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id")
                while rows := cur.fetchmany(4096):
                    vecs = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), self.dim)
                    self._append(np.fromiter((r[0] for r in rows), np.int64, len(rows)),
                                 np.fromiter((r[4] for r in rows), np.int64, len(rows)),
                                 vecs, [r[2] for r in rows], [r[3] for r in rows])
            self.version = version
            self.loads += 1
            self._files = None
        finally:
            if own_txn:
                db.rollback()

    def _load_shadow(self, db: sqlite3.Connection, count: int) -> bool:
        """Fast path: read sqlite-vec's storage tables directly (0.25 s instead of 15 s at 100k
        chunks; a vec0 full scan decodes row by row). This relies on the vec0 layout of
        sqlite-vec 0.1.x (vec_chunks_rowids maps a row to (storage chunk, offset);
        vec_chunks_vector_chunks00 holds one blob of `size` vectors per storage chunk), so the
        result is checked against vec0 itself and the slow scan is used on any mismatch."""
        try:
            info = dict(db.execute("SELECT key, value FROM vec_chunks_info").fetchall())
            if (info.get("CREATE_VERSION_MAJOR"), info.get("CREATE_VERSION_MINOR")) != (0, 1):
                return False
            blobs = {r[0]: r[1] for r in db.execute("SELECT rowid, vectors FROM vec_chunks_vector_chunks00")}
            rows = db.execute(
                "SELECT r.rowid, r.chunk_id, r.chunk_offset, c.file_id, c.modality, f.kind "
                "FROM vec_chunks_rowids r JOIN chunks c ON c.id = r.rowid JOIN files f ON f.id = c.file_id "
                "ORDER BY r.chunk_id, r.chunk_offset").fetchall()
            if len(rows) != count:
                return False
            row_bytes = self.dim * 4
            buf = bytearray(len(rows) * row_bytes)
            for i, r in enumerate(rows):
                blob = blobs[r[1]]
                start = r[2] * row_bytes
                buf[i * row_bytes:(i + 1) * row_bytes] = blob[start:start + row_bytes]
            vecs = np.frombuffer(bytes(buf), dtype=np.float32).reshape(len(rows), self.dim)
            ids = np.fromiter((r[0] for r in rows), np.int64, len(rows))
            if len(rows):
                sample = np.unique(np.linspace(0, len(rows) - 1, num=min(8, len(rows))).astype(int))
                for i in sample.tolist():
                    got = db.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (int(ids[i]),)).fetchone()
                    if got is None or got[0] != vecs[i].tobytes():
                        log.warning("vector cache: sqlite-vec layout check failed, using the slow load")
                        return False
            self._append(ids, np.fromiter((r[3] for r in rows), np.int64, len(rows)), vecs,
                         [r[4] for r in rows], [r[5] for r in rows])
            return True
        except (sqlite3.Error, KeyError, ValueError) as exc:
            log.warning("vector cache: fast load failed (%s), using the slow load", exc)
            return False

    def ensure_fresh(self, db: sqlite3.Connection) -> None:
        """Reload if another process changed the index (call with `lock` held)."""
        if self.version is None:
            self.load(db)
            return
        if self._local_writes:
            return
        if chunks_version(db) != self.version:
            log.info("vector cache: index changed by another process, reloading")
            self.load(db)

    def begin_local_write(self) -> None:
        with self.lock:
            self._local_writes += 1

    def end_local_write(self, before: int | None, after: int | None, delta: Delta | None) -> None:
        with self.lock:
            self._local_writes -= 1
            if delta is None or before is None or after is None or before == after:
                return
            if self.version == after:
                return
            if self.version != before:
                self.version = None
                return
            if self.too_big:
                self.version = after
                return
            self.apply(delta)
            self.version = after

    def apply(self, delta: Delta) -> None:
        if delta.cleared:
            self._reset(0)
        if delta.removed_files and self.n:
            mask = np.isin(self.file_ids[: self.n], np.fromiter(delta.removed_files, np.int64))
            self._remove_rows(np.flatnonzero(mask))
        if delta.added:
            self._append(np.array([a[0] for a in delta.added], dtype=np.int64),
                         np.array([a[1] for a in delta.added], dtype=np.int64),
                         np.stack([a[2] for a in delta.added]).astype(np.float32, copy=False),
                         [a[3] for a in delta.added], [a[4] for a in delta.added])
        self._maybe_compact()

    def _file_counts(self) -> np.ndarray:
        if self._files is None:
            n, k = self.n, max(len(self._names), 1)
            files = np.zeros(k, dtype=np.int64)
            rows = np.flatnonzero(self.alive[:n])
            if len(rows):
                key = self.file_ids[rows].astype(np.int64) * k + self.modality[rows].astype(np.int64)
                files = np.bincount(np.unique(key) % k, minlength=k)
            self._files = files
        return self._files

    def scores(self, q: np.ndarray) -> VectorScores:
        """Cosine similarity of q with every stored vector, with each row's chunk id, file id,
        modality code and kind code (views into the cache: use them with `lock` held). `valid`
        masks out deleted rows (None when there are none)."""
        n = self.n
        q = np.asarray(q, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        return VectorScores(
            sims=self.mat[:n] @ q, chunk_ids=self.chunk_ids[:n], file_ids=self.file_ids[:n],
            modality=self.modality[:n], kind=self.kind[:n], valid=self.alive[:n] if self.dead else None,
            names=list(self._names), files=self._file_counts())

    def codes(self, names: Iterable[str]) -> list[int]:
        """Codes of these modality/kind names (unknown names are skipped)."""
        return [self._codes[x] for x in names if x in self._codes]

    def rows(self, chunk_ids: Iterable[int]) -> np.ndarray:
        """Rows of these chunks (chunks not in the cache are skipped)."""
        got = [r for r in (self.row_of.get(c) for c in chunk_ids) if r is not None]
        return np.asarray(got, dtype=np.int64)

    def vectors(self, chunk_ids: Iterable[int]) -> dict[int, np.ndarray]:
        out = {}
        for cid in chunk_ids:
            row = self.row_of.get(cid)
            if row is not None:
                out[cid] = self.mat[row].copy()
        return out
