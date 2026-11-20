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

from . import filetypes as ft
from .config import Config
from .store import ChunkRecord, Store

if TYPE_CHECKING:
    from .embedder import EmbedderLike
    from .extract.base import Chunk

log = logging.getLogger(__name__)

ExtractFn = Callable[[Path, str, Config], Iterator["Chunk"]]
SNIPPET_CHARS = 200
QUEUE_PAGE = 256

INDEX_FORMAT_VERSION = 3

FLUSH_AFTER_S = 3.0
FLUSH_AFTER_FILES = 16
MAX_CONSECUTIVE_EMBED_FAILURES = 5
STORE_RETRIES = 4
STORE_BACKOFF_S = 0.5
AUTO_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 86_400.0

MEDIA_KINDS = (ft.AUDIO, ft.VIDEO)
LEGACY_MEDIA_MAX_SECONDS = 30 * 60


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
    total: int = 0
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    deferred: int = 0
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
    text: str | None
    vector: np.ndarray | None = None


@dataclass
class _Job:
    row: FileRow
    records: list[_Record] = field(default_factory=list)
    unembedded: int = 0
    ended: bool = False
    dead: bool = False


@dataclass
class _Buffer:
    items: list[tuple[_Job, _Record, Any]] = field(default_factory=list)
    started: float = 0.0
    files_at_start: int = 0


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


def _release_extraction() -> None:
    """Free the calling thread's extraction resources (see extract.release_thread)."""
    try:
        from .extract import release_thread

        release_thread()
    except Exception:  # pragma: no cover - never let cleanup fail a run
        log.debug("releasing extraction resources failed", exc_info=True)


class Indexer:
    def __init__(self, store: Store, cfg: Config, embedder: EmbedderLike, extract_fn: ExtractFn | None = None,
                 prefetch: bool = True):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.extract_fn = extract_fn
        self.prefetch = prefetch
        self.ocr_setup: str | None = None

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
            store.set_model(model, dim)
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
        requeued += self.reconcile_media()
        requeued += self.reconcile_ocr()
        if retry_failed:
            requeued += store.requeue(("failed",), reset_attempts=False)
        else:
            requeued += store.requeue_failed_for_retry(AUTO_RETRY_ATTEMPTS, RETRY_BACKOFF_S)
        return requeued

    def reconcile_media(self) -> int:
        """Bring indexed audio/video in line with the media settings (`media.max_minutes`,
        `media.segment_seconds`) when they differ from those the index was built with:

        - a new segment length re-queues every audio/video file (all positions change);
        - a longer limit re-queues only the files cut off at the old one (a segment reaching
          into the last two segments before it); shorter files have nothing more to index;
        - a shorter limit deletes the segments past it, without re-embedding anything.
        Returns the number of files re-queued.
        """
        from .extract.audio import segment_seconds

        store = self.store
        limit, seg = float(self.cfg.max_media_seconds), segment_seconds(self.cfg)
        old_limit_s, old_seg_s = store.get_meta("media_max_seconds"), store.get_meta("media_segment_seconds")
        if old_limit_s is None or old_seg_s is None:
            if not self._has_media_chunks():
                old_limit_s, old_seg_s = str(limit), str(seg)
            else:
                old_limit_s, old_seg_s = old_limit_s or str(LEGACY_MEDIA_MAX_SECONDS), old_seg_s or str(seg)
        old_limit, old_seg = float(old_limit_s), float(old_seg_s)
        requeued = trimmed = 0
        if abs(seg - old_seg) > 1e-6:
            requeued = store.requeue_kinds(MEDIA_KINDS)
        elif limit > old_limit + 1e-6:
            requeued = store.requeue_kinds(MEDIA_KINDS, min_last_position=max(0.0, old_limit - 2 * old_seg))
        if limit < old_limit - 1e-6:
            trimmed = store.trim_positions(MEDIA_KINDS, limit)
        if requeued or trimmed:
            log.info("media settings changed (limit %.0f -> %.0f s, segments %.0f -> %.0f s): "
                     "%d audio/video files re-queued, %d segments past the limit deleted",
                     old_limit, limit, old_seg, seg, requeued, trimmed)
        for key, value in (("media_max_seconds", str(limit)), ("media_segment_seconds", str(seg))):
            if store.get_meta(key) != value:
                store.set_meta(key, value)
        return requeued

    def reconcile_ocr(self) -> int:
        """Bring indexed screenshots in line with the OCR setup (`[ocr]` and the installed
        languages: ocr.effective_languages, recorded in meta 'ocr_languages' joined with "+",
        "" when off; an index from before OCR existed has no OCR text, i.e. "off"):

        - OCR turned on or other languages: re-queue the indexed screenshots (old chunks stay
          searchable until replaced);
        - OCR off (or no longer possible): delete the text chunks of images, re-embedding nothing.
        Returns the number of files re-queued.
        """
        from . import ocr

        store = self.store
        setup = "+".join(ocr.effective_languages(self.cfg))
        old = store.get_meta("ocr_languages") or ""
        requeued = removed = 0
        if setup != old:
            if setup:
                rows = store.db.execute("SELECT id, path FROM files WHERE kind = ? AND status = 'indexed'",
                                        (ft.IMAGE,)).fetchall()
                requeued = store.requeue_files(r[0] for r in rows if ocr.is_screenshot(r[1]))
            else:
                removed = store.delete_modality((ft.IMAGE,), "text")
            if requeued or removed:
                log.info("OCR changed (%s -> %s): %d screenshots re-queued, %d text chunks of images deleted",
                         old or "off", setup or "off", requeued, removed)
        if store.get_meta("ocr_languages") != setup:
            store.set_meta("ocr_languages", setup)
        self.ocr_setup = setup
        return requeued

    def _has_media_chunks(self) -> bool:
        marks = ",".join("?" * len(MEDIA_KINDS))
        return self.store.db.execute(
            f"SELECT 1 FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.kind IN ({marks}) LIMIT 1",
            MEDIA_KINDS).fetchone() is not None

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
            except Exception as exc:
                if stop.is_set():
                    return
                yield (_FAIL, row, _error_text(exc))
                continue
            finally:
                close = getattr(chunks, "close", None)
                if close:
                    close()
            if stop.is_set():
                return
            yield (_END, row)

    def _threaded(self, rows: Iterator[FileRow], stop: threading.Event, cancelled: set[int]) -> Iterator[tuple]:
        """Run `_produce` on a worker thread, `prefetch_chunks` messages ahead."""
        q: queue.Queue[tuple] = queue.Queue(maxsize=max(1, self.cfg.prefetch_chunks))
        abort = threading.Event()

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
            except BaseException as exc:
                put((_CRASH, exc))
                return
            finally:
                self.store.release_thread()
                _release_extraction()
            put((_DONE,))

        thread = threading.Thread(target=work, name="semma-extract", daemon=True)
        thread.start()
        try:
            while True:
                try:
                    msg = q.get(timeout=0.2)
                except queue.Empty:
                    if stop.is_set():
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
        files_finished = 0
        consecutive_failures = 0

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
                results = []
                for (_, _, content), title in zip(items, titles, strict=True):
                    try:
                        results.append(self.embedder.embed_batch(modality, [content], [title])[0])
                    except Exception as exc:
                        results.append(exc)
                failed_files = {job.row.id for (job, _, _), r in zip(items, results, strict=True)
                                if isinstance(r, BaseException)}
                if len(failed_files) >= 2 and all(isinstance(r, BaseException) for r in results):
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
                if row.id in cancelled:
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
            for job in jobs.values():
                if not job.ended:
                    job.dead = True
            for modality in list(buffers):
                flush(modality)
        finally:
            close = getattr(source, "close", None)
            if close:
                close()
            if not self.prefetch:
                _release_extraction()
        stats.stopped = stop.is_set()
        stats.current = ""
        report()
        return stats

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
        except Exception as exc:
            if self._mark_failed(row, f"storing: {_error_text(exc)}"):
                stats.failed += 1
            else:
                stats.deferred += 1
            return
        if not ok:
            stats.deferred += 1
        elif not stored:
            stats.skipped += 1
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
