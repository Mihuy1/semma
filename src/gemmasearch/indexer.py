"""Phase 2: extract and embed pending files in priority order.

Pipeline:
  extraction thread                         calling thread
  ──────────────────                        ──────────────
  read the queue page by page   ─ msgs ─▶   buffer chunks per modality (across files) and embed
  stat (skip changed/vanished)  (bounded)   a buffer when it reaches the batch size; a file is
  extract_fn(path) -> chunks                committed once it has ended and all its chunks
                                            have vectors: one transaction replaces its chunks,
                                            vectors and FTS rows and marks it indexed.

Batching across files matters: most files (photos, notes) are a single chunk. The bounded queue
keeps extraction a few chunks ahead of the GPU without holding many decoded videos in memory.
A stop request leaves every file either fully committed or untouched (still pending).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import Config
from .store import Store

if TYPE_CHECKING:
    from .embedder import EmbedderLike
    from .extract.base import Chunk

ExtractFn = Callable[[Path, str, Config], Iterator["Chunk"]]
SNIPPET_CHARS = 200
QUEUE_PAGE = 256


@dataclass(slots=True)
class FileRow:
    id: int
    path: str
    kind: str
    size: int
    mtime: float
    priority: float


@dataclass
class IndexStats:
    total: int = 0  # files queued for this run (pending, after kind filter and limit)
    indexed: int = 0
    failed: int = 0
    skipped: int = 0  # changed or vanished since the scan; left pending for the next scan
    chunks: int = 0
    embed_seconds: float = 0.0
    seconds: float = 0.0
    stopped: bool = False
    current: str = ""

    @property
    def done(self) -> int:
        return self.indexed + self.failed + self.skipped


ProgressFn = Callable[[IndexStats], None]


@dataclass(slots=True)
class _Record:
    """A chunk waiting for (or holding) its vector; the content itself lives only in the buffer."""

    modality: str
    locator: str
    snippet: str
    position: float | None
    text: str | None  # full text for the FTS, text chunks only
    vector: np.ndarray | None = None


@dataclass
class _Job:
    row: FileRow
    records: list[_Record] = field(default_factory=list)
    unembedded: int = 0
    ended: bool = False
    dead: bool = False  # failed or abandoned; its remaining chunks are ignored


# Messages from the extraction thread.
_CHUNK, _END, _SKIP, _FAIL, _DONE, _CRASH = "chunk", "end", "skip", "fail", "done", "crash"


def _snippet(chunk: Chunk) -> str:
    if chunk.snippet:
        return chunk.snippet
    if chunk.modality == "text" and isinstance(chunk.content, str):
        return " ".join(chunk.content[: SNIPPET_CHARS * 2].split())[:SNIPPET_CHARS]
    return ""


def _error_text(exc: BaseException) -> str:
    from .extract.base import ExtractionError

    msg = str(exc).strip()
    if isinstance(exc, ExtractionError):
        return msg or "extraction failed"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


class Indexer:
    def __init__(self, store: Store, cfg: Config, embedder: EmbedderLike, extract_fn: ExtractFn | None = None,
                 prefetch: bool = True):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.extract_fn = extract_fn
        self.prefetch = prefetch  # False: extract and embed on one thread (for comparison/debugging)

    # -- queue ----------------------------------------------------------------
    def _queue_where(self, kinds: Sequence[str] | None) -> tuple[str, list[Any]]:
        where, args = "status = 'pending'", []
        if kinds:
            where += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        return where, args

    def count_pending(self, kinds: Sequence[str] | None = None) -> int:
        where, args = self._queue_where(kinds)
        with self.store.lock:
            return self.store.db.execute(f"SELECT count(*) FROM files WHERE {where}", args).fetchone()[0]

    def iter_queue(self, kinds: Sequence[str] | None = None, limit: int | None = None) -> Iterator[FileRow]:
        """Pending files by priority, read a page at a time (keyset pagination, so rows being
        marked indexed meanwhile neither shift nor repeat the order)."""
        where, args = self._queue_where(kinds)
        sql = (f"SELECT id, path, kind, size, mtime, priority FROM files WHERE {where} {{after}} "
               f"ORDER BY priority DESC, mtime DESC, id DESC LIMIT ?")
        last: tuple[float, float, int] | None = None
        remaining = limit if limit is not None else -1
        while remaining != 0:
            page = QUEUE_PAGE if remaining < 0 else min(QUEUE_PAGE, remaining)
            after = "AND (priority, mtime, id) < (?, ?, ?)" if last else ""
            with self.store.lock:
                rows = self.store.db.execute(sql.format(after=after), [*args, *(last or ()), page]).fetchall()
            if not rows:
                return
            for r in rows:
                yield FileRow(r["id"], r["path"], r["kind"], r["size"], r["mtime"], r["priority"])
            last = (rows[-1]["priority"], rows[-1]["mtime"], rows[-1]["id"])
            if remaining > 0:
                remaining -= len(rows)

    # -- extraction (producer) --------------------------------------------------
    def _produce(self, rows: Iterator[FileRow], stop: threading.Event, cancelled: set[int]) -> Iterator[tuple]:
        extract_fn = self.extract_fn
        if extract_fn is None:
            from .extract import extract as extract_fn
        for row in rows:
            if stop.is_set():
                return
            try:
                st = os.stat(row.path)
            except FileNotFoundError:
                yield (_SKIP, row, "vanished")
                continue
            except OSError as exc:
                yield (_FAIL, row, _error_text(exc))
                continue
            if st.st_size != row.size or st.st_mtime != row.mtime:
                yield (_SKIP, row, "changed")
                continue
            chunks = None
            try:
                chunks = extract_fn(Path(row.path), row.kind, self.cfg)
                for chunk in chunks:
                    if stop.is_set():
                        return
                    if row.id in cancelled:
                        break
                    yield (_CHUNK, row, chunk)
            except Exception as exc:  # per-file failure, keep going
                yield (_FAIL, row, _error_text(exc))
                continue
            finally:
                close = getattr(chunks, "close", None)
                if close:
                    close()  # lets extractors clean up (e.g. kill ffmpeg) when abandoned
            yield (_END, row)

    def _threaded(self, rows: Iterator[FileRow], stop: threading.Event, cancelled: set[int]) -> Iterator[tuple]:
        """Run `_produce` on a worker thread, `prefetch_chunks` messages ahead."""
        q: queue.Queue[tuple] = queue.Queue(maxsize=max(1, self.cfg.prefetch_chunks))
        abort = threading.Event()  # consumer is gone; producer must not block on a full queue

        def put(msg: tuple) -> bool:
            while not abort.is_set():
                try:
                    q.put(msg, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def work() -> None:
            try:
                for msg in self._produce(rows, stop, cancelled):
                    if not put(msg):
                        return
            except BaseException as exc:  # e.g. the extract module failing to import
                put((_CRASH, exc))
                return
            put((_DONE,))

        thread = threading.Thread(target=work, name="gemmasearch-extract", daemon=True)
        thread.start()
        try:
            while True:
                try:
                    msg = q.get(timeout=0.2)
                except queue.Empty:
                    if stop.is_set():  # don't wait for a slow extraction to yield
                        return
                    continue
                if msg[0] == _DONE:
                    return
                if msg[0] == _CRASH:
                    raise msg[1]
                yield msg
        finally:
            abort.set()
            thread.join(timeout=30)

    # -- run (consumer) ---------------------------------------------------------
    def run(self, limit: int | None = None, kinds: Sequence[str] | None = None,
            progress: ProgressFn | None = None, stop: threading.Event | None = None) -> IndexStats:
        stop = stop or threading.Event()
        self.store.ensure_model(self.embedder.model_id, self.embedder.dim)
        stats = IndexStats()
        pending = self.count_pending(kinds)
        stats.total = pending if limit is None else min(limit, pending)
        started = time.monotonic()
        jobs: dict[int, _Job] = {}
        buffers: dict[str, list[tuple[_Job, _Record, Any]]] = {}
        cancelled: set[int] = set()

        def report() -> None:
            stats.seconds = time.monotonic() - started
            if progress:
                progress(stats)

        def finish(job: _Job) -> None:
            jobs.pop(job.row.id, None)
            self._commit(job, stats)
            report()

        def fail(job: _Job, error: str) -> None:
            job.dead = True
            cancelled.add(job.row.id)
            jobs.pop(job.row.id, None)
            self._mark_failed(job.row, error)
            stats.failed += 1
            report()

        def flush(modality: str) -> None:
            items = buffers.pop(modality, [])
            items = [it for it in items if not it[0].dead]
            if not items:
                return
            t = time.monotonic()
            try:
                vecs = self.embedder.embed_batch(modality, [it[2] for it in items],
                                                 [os.path.basename(it[0].row.path) for it in items])
                results: list[np.ndarray | BaseException] = list(vecs)
            except Exception:
                # Isolate the bad input: retry one by one so only its file fails.
                results = []
                for job, _, content in items:
                    try:
                        results.append(self.embedder.embed_batch(modality, [content],
                                                                 [os.path.basename(job.row.path)])[0])
                    except Exception as exc:
                        results.append(exc)
                if len(items) > 1 and all(isinstance(r, BaseException) for r in results):
                    # Every input failing points at the model/GPU, not the files: stop the run
                    # instead of marking the whole queue failed.
                    raise results[-1]  # type: ignore[misc]
            stats.embed_seconds += time.monotonic() - t
            for (job, rec, _), res in zip(items, results, strict=True):
                if job.dead:
                    continue
                if isinstance(res, BaseException):
                    fail(job, f"embedding: {_error_text(res)}")
                    continue
                rec.vector = res
                job.unembedded -= 1
                if job.ended and job.unembedded == 0:
                    finish(job)

        rows = self.iter_queue(kinds, limit)
        source = self._threaded(rows, stop, cancelled) if self.prefetch else self._produce(rows, stop, cancelled)
        try:
            for msg in source:
                kind, row = msg[0], msg[1]
                stats.current = row.path
                if kind == _SKIP:
                    stats.skipped += 1
                    report()
                    continue
                if row.id in cancelled:  # failed already; the producer may still be sending its chunks
                    continue
                job = jobs.get(row.id)
                if job is None:
                    job = jobs[row.id] = _Job(row)
                if kind == _CHUNK:
                    chunk: Chunk = msg[2]
                    rec = _Record(chunk.modality, chunk.locator, _snippet(chunk), chunk.position,
                                  chunk.content if chunk.modality == "text" else None)
                    job.records.append(rec)
                    job.unembedded += 1
                    buf = buffers.setdefault(chunk.modality, [])
                    buf.append((job, rec, chunk.content))
                    if len(buf) >= max(1, self.cfg.batch_sizes.get(chunk.modality, 8)):
                        flush(chunk.modality)
                elif kind == _FAIL:
                    fail(job, msg[2])
                elif kind == _END:
                    job.ended = True
                    if job.unembedded == 0:
                        finish(job)
                if stop.is_set():
                    break
            # Embed what is buffered: on a stop this finishes files that were fully extracted
            # (a partial batch is quick); files still mid-extraction stay pending.
            for job in jobs.values():
                if not job.ended:
                    job.dead = True
            for modality in list(buffers):
                flush(modality)
        finally:
            close = getattr(source, "close", None)
            if close:
                close()
        stats.stopped = stop.is_set()
        stats.current = ""
        report()
        return stats

    # -- database writes --------------------------------------------------------
    def _commit(self, job: _Job, stats: IndexStats) -> None:
        """Replace the file's chunks, vectors and FTS rows and mark it indexed, atomically."""
        row = job.row
        db = self.store.db
        try:
            with self.store.lock, db:
                db.execute("DELETE FROM chunks WHERE file_id = ?", (row.id,))  # triggers clear vec/fts
                for seq, rec in enumerate(job.records):
                    cur = db.execute(
                        "INSERT INTO chunks (file_id, seq, modality, locator, snippet, position) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (row.id, seq, rec.modality, rec.locator, rec.snippet, rec.position),
                    )
                    chunk_id = cur.lastrowid
                    assert rec.vector is not None
                    db.execute("INSERT INTO vec_chunks (chunk_id, embedding, modality, kind) VALUES (?, ?, ?, ?)",
                               (chunk_id, np.ascontiguousarray(rec.vector, dtype=np.float32), rec.modality, row.kind))
                    if rec.text:
                        db.execute("INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)", (chunk_id, rec.text))
                # Guarded by size/mtime: if a concurrent scan saw a newer version, it stays pending.
                db.execute("UPDATE files SET status = 'indexed', error = NULL, indexed_at = ? "
                           "WHERE id = ? AND size = ? AND mtime = ?", (time.time(), row.id, row.size, row.mtime))
        except Exception as exc:
            # Typically the file row was removed by a concurrent scan (FK violation).
            self._mark_failed(row, f"storing: {_error_text(exc)}")
            stats.failed += 1
            return
        stats.indexed += 1
        stats.chunks += len(job.records)

    def _mark_failed(self, row: FileRow, error: str) -> None:
        with self.store.lock, self.store.db:
            self.store.db.execute("UPDATE files SET status = 'failed', error = ?, indexed_at = ? WHERE id = ?",
                                  (error[:2000], time.time(), row.id))
