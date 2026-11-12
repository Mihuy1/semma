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

A buffer of a rarely used modality (one scanned PDF page among thousands of notes) is flushed
after FLUSH_AFTER_S seconds or FLUSH_AFTER_FILES other files, so its file does not wait for the
end of a long (in a daemon: endless) run.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import Config
from .store import ChunkRecord, Store

if TYPE_CHECKING:
    from .embedder import EmbedderLike
    from .extract.base import Chunk

log = logging.getLogger(__name__)

ExtractFn = Callable[[Path, str, Config], Iterator["Chunk"]]
SNIPPET_CHARS = 200
QUEUE_PAGE = 256

# Bump when chunking, the document prompt format or the FTS layout changes: every file is then
# re-queued (its old chunks stay searchable until replaced). Stored in meta 'index_format'.
# 2: schema v3 (fts_chunks recreated with a prefix index), vectors renormalised in float32.
INDEX_FORMAT_VERSION = 3  # 3: tiny text chunks are no longer embedded

FLUSH_AFTER_S = 3.0
FLUSH_AFTER_FILES = 16
# Embedding failures of this many different files in a row (with no success in between) mean
# the model/GPU is broken, not the files: the run stops instead of failing the whole queue.
MAX_CONSECUTIVE_EMBED_FAILURES = 5
# Transient storage errors (database locked by another process for longer than the busy
# timeout): retry with backoff, then leave the file pending rather than recording a failure.
STORE_RETRIES = 4
STORE_BACKOFF_S = 0.5
# Failed files are retried automatically up to this many attempts in total, the n-th retry no
# sooner than n * RETRY_BACKOFF_S after the last attempt. `index --retry-failed` retries now.
AUTO_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 86_400.0


class ModelMismatch(RuntimeError):
    def __init__(self, stored: str | None, configured: str):
        super().__init__(f"the index was built with {stored}, but the configured model is {configured}")
        self.stored = stored
        self.configured = configured


def extractor_version() -> str:
    """Failed files are retried once whenever this changes (an upgrade may have fixed them)."""
    from . import __version__

    try:
        from . import extract

        ext = getattr(extract, "EXTRACTOR_VERSION", None)
    except Exception:  # pragma: no cover - reported when indexing actually starts
        ext = None
    return f"{__version__}/{ext}" if ext is not None else __version__


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
    deferred: int = 0  # could not be stored (database busy); left pending for the next run
    chunks: int = 0
    embed_seconds: float = 0.0
    seconds: float = 0.0
    stopped: bool = False
    current: str = ""

    @property
    def done(self) -> int:
        return self.indexed + self.failed + self.skipped + self.deferred


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


@dataclass
class _Buffer:
    items: list[tuple[_Job, _Record, Any]] = field(default_factory=list)
    started: float = 0.0  # monotonic time of the oldest item
    files_at_start: int = 0  # files finished (ended/failed/skipped) when the oldest item arrived


# Messages from the extraction thread. _TICK: nothing arrived for a while (lets the consumer
# flush aged buffers while a slow file is being extracted).
_CHUNK, _END, _SKIP, _FAIL, _DONE, _CRASH, _TICK = "chunk", "end", "skip", "fail", "done", "crash", "tick"


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

    # -- preparation -------------------------------------------------------------
    def prepare(self, rebuild: bool = False, retry_failed: bool = False) -> int:
        """Check the stored model and index format and re-queue files as needed.

        - A different model never wipes the index silently: ModelMismatch unless `rebuild`.
        - `rebuild`: re-embed everything (with the same model, old vectors stay searchable
          until replaced).
        - A new INDEX_FORMAT_VERSION re-queues everything; a new extractor version re-queues
          failed files; `retry_failed` re-queues failed files now.
        Returns the number of files re-queued.
        """
        store = self.store
        model, dim = self.embedder.model_id, self.embedder.dim
        requeued = 0
        if not store.model_matches(model, dim):
            if not rebuild:
                raise ModelMismatch(store.get_meta("model_id"), model)
            store.set_model(model, dim)  # drops every vector, marks everything pending
        elif store.get_meta("model_id") != model:
            store.set_meta("model_id", model)
        if rebuild:
            requeued += store.requeue(("indexed", "failed"))

        if store.get_meta("index_format") != str(INDEX_FORMAT_VERSION):
            requeued += store.requeue(("indexed", "failed"))
            store.set_meta("index_format", str(INDEX_FORMAT_VERSION))
        version = extractor_version()
        if store.get_meta("extractor_version") != version:
            requeued += store.requeue(("failed",))
            store.set_meta("extractor_version", version)
        if retry_failed:
            requeued += store.requeue(("failed",), reset_attempts=False)
        else:
            requeued += store.requeue_failed_for_retry(AUTO_RETRY_ATTEMPTS, RETRY_BACKOFF_S)
        return requeued

    # -- queue ----------------------------------------------------------------
    def _queue_where(self, kinds: Sequence[str] | None) -> tuple[str, list[Any]]:
        where, args = "status = 'pending'", []
        if kinds:
            where += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        return where, args

    def count_pending(self, kinds: Sequence[str] | None = None) -> int:
        where, args = self._queue_where(kinds)
        return self.store.db.execute(f"SELECT count(*) FROM files WHERE {where}", args).fetchone()[0]

    def iter_queue(self, kinds: Sequence[str] | None = None, limit: int | None = None) -> Iterator[FileRow]:
        """Pending files by priority, read a page at a time (keyset pagination, so rows being
        marked indexed meanwhile neither shift nor repeat the order). Uses the calling thread's
        own connection."""
        where, args = self._queue_where(kinds)
        sql = (f"SELECT id, path, kind, size, mtime, priority FROM files WHERE {where} {{after}} "
               f"ORDER BY priority DESC, mtime DESC, id DESC LIMIT ?")
        last: tuple[float, float, int] | None = None
        remaining = limit if limit is not None else -1
        while remaining != 0:
            page = QUEUE_PAGE if remaining < 0 else min(QUEUE_PAGE, remaining)
            after = "AND (priority, mtime, id) < (?, ?, ?)" if last else ""
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
                if stop.is_set():
                    return  # e.g. Ctrl-C killed ffmpeg: not the file's fault, leave it pending
                yield (_FAIL, row, _error_text(exc))
                continue
            finally:
                close = getattr(chunks, "close", None)
                if close:
                    close()  # lets extractors clean up (e.g. kill ffmpeg) when abandoned
            if stop.is_set():
                # The extraction may have been cut short by the interrupt (Ctrl-C also reaches
                # ffmpeg); never commit a possibly truncated file as complete.
                return
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
            finally:
                self.store.release_thread()  # the queue reader's connection (one per run)
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
                    yield (_TICK,)
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
        self.prepare()
        stats = IndexStats()
        pending = self.count_pending(kinds)
        stats.total = pending if limit is None else min(limit, pending)
        started = time.monotonic()
        jobs: dict[int, _Job] = {}
        buffers: dict[str, _Buffer] = {}
        cancelled: set[int] = set()
        files_finished = 0  # files whose extraction ended, failed or was skipped
        consecutive_failures = 0  # files failing at embedding with no success in between

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
            if self._mark_failed(job.row, error):
                stats.failed += 1
            else:
                stats.deferred += 1
            report()

        def flush(modality: str) -> None:
            nonlocal consecutive_failures
            buf = buffers.pop(modality, None)
            items = [it for it in (buf.items if buf else []) if not it[0].dead]
            if not items:
                return
            t = time.monotonic()
            titles = [os.path.basename(it[0].row.path) for it in items]
            try:
                vecs = self.embedder.embed_batch(modality, [it[2] for it in items], titles)
                results: list[np.ndarray | BaseException] = list(vecs)
            except Exception:
                # Isolate the bad input: retry one by one so only its file fails.
                results = []
                for (_, _, content), title in zip(items, titles, strict=True):
                    try:
                        results.append(self.embedder.embed_batch(modality, [content], [title])[0])
                    except Exception as exc:
                        results.append(exc)
                failed_files = {job.row.id for (job, _, _), r in zip(items, results, strict=True)
                                if isinstance(r, BaseException)}
                if len(failed_files) >= 2 and all(isinstance(r, BaseException) for r in results):
                    # Inputs of several files all failing points at the model/GPU, not the files:
                    # stop the run instead of marking the whole queue failed. (All chunks of ONE
                    # file failing is that file's problem: it fails below and the run goes on.)
                    raise results[-1]  # type: ignore[misc]
            stats.embed_seconds += time.monotonic() - t
            for (job, rec, _), res in zip(items, results, strict=True):
                if job.dead:
                    continue
                if isinstance(res, BaseException):
                    consecutive_failures += 1
                    fail(job, f"embedding: {_error_text(res)}")
                    if consecutive_failures >= MAX_CONSECUTIVE_EMBED_FAILURES:
                        raise res
                    continue
                consecutive_failures = 0
                # Renormalise in float32: the model normalises in bfloat16 (norm error ~1%).
                v = np.asarray(res, dtype=np.float32)
                norm = float(np.linalg.norm(v))
                rec.vector = v / norm if norm > 0 else v
                job.unembedded -= 1
                if job.ended and job.unembedded == 0:
                    finish(job)

        def flush_aged() -> None:
            now = time.monotonic()
            for modality, buf in list(buffers.items()):
                if now - buf.started >= FLUSH_AFTER_S or files_finished - buf.files_at_start >= FLUSH_AFTER_FILES:
                    flush(modality)

        rows = self.iter_queue(kinds, limit)
        source = self._threaded(rows, stop, cancelled) if self.prefetch else self._produce(rows, stop, cancelled)
        try:
            for msg in source:
                kind = msg[0]
                if kind == _TICK:
                    flush_aged()
                    continue
                row = msg[1]
                stats.current = row.path
                if kind == _SKIP:
                    stats.skipped += 1
                    files_finished += 1
                    report()
                    flush_aged()
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
                    buf = buffers.get(chunk.modality)
                    if buf is None:
                        buf = buffers[chunk.modality] = _Buffer(started=time.monotonic(),
                                                                files_at_start=files_finished)
                    buf.items.append((job, rec, chunk.content))
                    if len(buf.items) >= max(1, self.cfg.batch_sizes.get(chunk.modality, 8)):
                        flush(chunk.modality)
                elif kind == _FAIL:
                    files_finished += 1
                    fail(job, msg[2])
                elif kind == _END:
                    files_finished += 1
                    job.ended = True
                    if job.unembedded == 0:
                        finish(job)
                if stop.is_set():
                    break
                flush_aged()
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
    def _with_retry(self, what: str, fn: Callable[[], Any]) -> tuple[bool, Any]:
        """Run a store write, retrying transient OperationalErrors ("database is locked").
        Returns (ok, result); ok is False if it still failed after the retries."""
        delay = STORE_BACKOFF_S
        for attempt in range(STORE_RETRIES):
            try:
                return True, fn()
            except sqlite3.OperationalError as exc:
                if attempt == STORE_RETRIES - 1:
                    log.warning("%s: %s; leaving the file pending", what, exc)
                    return False, None
                time.sleep(delay)
                delay *= 2
        return False, None  # pragma: no cover

    def _commit(self, job: _Job, stats: IndexStats) -> None:
        """Replace the file's chunks, vectors and FTS rows and mark it indexed, atomically."""
        row = job.row
        records = []
        for rec in job.records:
            assert rec.vector is not None
            records.append(ChunkRecord(rec.modality, rec.locator, rec.snippet, rec.position, rec.text, rec.vector))
        try:
            ok, stored = self._with_retry(f"storing {row.path}", lambda: self.store.replace_chunks(
                row.id, row.size, row.mtime, row.kind, records))
        except Exception as exc:  # a real problem with this file's data (not a busy database)
            if self._mark_failed(row, f"storing: {_error_text(exc)}"):
                stats.failed += 1
            else:
                stats.deferred += 1
            return
        if not ok:
            stats.deferred += 1
        elif not stored:
            stats.skipped += 1  # removed or changed by a concurrent scan; it is pending (or gone)
        else:
            stats.indexed += 1
            stats.chunks += len(job.records)

    def _mark_failed(self, row: FileRow, error: str) -> bool:
        """Record a failure. Returns False if even that could not be stored (left pending)."""
        try:
            ok, _ = self._with_retry(f"marking {row.path} failed", lambda: self.store.mark_failed(row.id, error))
        except Exception:
            log.exception("marking %s failed", row.path)
            return False
        return ok
