"""EmbeddingGemma 2 wrapper: chunks and queries -> L2-normalised float32 vectors.

Verified with sentence-transformers 6.3 (see extract/base.py for the content types):
- The model has no default prompt, so `encode(..., prompt="")` adds nothing; text documents are
  formatted here as "title: <file name> | text: <chunk>" (the built-in "document" prompt
  hard-codes "title: none"). `encode_query` adds "task: search result | query: ".
- bfloat16 on CUDA; float16 produces NaNs. CPU falls back to float32.
- Multimodal inputs are {"image": PIL}, {"audio": float32 16 kHz}, {"video": [PIL, ...]} dicts.
  Mixed-modality batches are untested, so every model call is single-modality.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import DEFAULT_BATCH_SIZES, DEFAULT_MODEL

if TYPE_CHECKING:
    from .config import Config
    from .extract.base import Chunk

log = logging.getLogger(__name__)

# The context is 8192 tokens; extractors chunk well below that, this only guards against a
# runaway chunk (Finnish averages roughly 3 characters per token).
MAX_TEXT_CHARS = 16_000

QUERY_CACHE_SIZE = 256  # type-as-you-go sends the same prefixes again (backspace, re-typing)
# After a CUDA OOM the batch size of that modality stays reduced; it is doubled again (up to the
# configured size) only after this many successful batches in a row.
OOM_REGROW_AFTER = 200


class EmbedderLike(Protocol):
    """What the indexer and search need; implemented by Embedder and by the tests' fake."""

    model_id: str
    dim: int

    def embed_query(self, query: str) -> np.ndarray: ...

    def embed_batch(self, modality: str, contents: Sequence[Any], titles: Sequence[str]) -> np.ndarray: ...


def format_document(text: str, title: str) -> str:
    title = title.replace("\n", " ").strip() or "none"
    return f"title: {title} | text: {text[:MAX_TEXT_CHARS]}"


def model_input(modality: str, content: Any, title: str) -> Any:
    """Turn a chunk's content into what `SentenceTransformer.encode` accepts."""
    if modality == "text":
        return format_document(content, title)
    if modality == "image":
        return {"image": content.convert("RGB") if content.mode != "RGB" else content}
    if modality == "audio":
        return {"audio": np.ascontiguousarray(content, dtype=np.float32)}
    if modality == "video":
        return {"video": list(content)}
    raise ValueError(f"unknown modality {modality!r}")


def embed_chunks(embedder: EmbedderLike, chunks: Sequence[Chunk], title: str | Sequence[str]) -> np.ndarray:
    """Embed chunks of any modality mix, one model call per modality; rows follow input order."""
    titles = [title] * len(chunks) if isinstance(title, str) else list(title)
    out = np.empty((len(chunks), embedder.dim), dtype=np.float32)
    by_modality: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_modality.setdefault(c.modality, []).append(i)
    for modality, idx in by_modality.items():
        out[idx] = embedder.embed_batch(modality, [chunks[i].content for i in idx], [titles[i] for i in idx])
    return out


class GpuLock:
    """A lock where waiting queries go before waiting index batches.

    `threading.Lock` is unfair: a query arriving during a long video batch could lose the lock
    to the next batch again and again. Here a batch only takes the GPU when no query is
    waiting, so a query waits for at most the one batch already running.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._busy = False
        self._queries_waiting = 0

    @property
    def queries_waiting(self) -> int:
        return self._queries_waiting

    @contextmanager
    def query(self) -> Iterator[None]:
        with self._cv:
            self._queries_waiting += 1
            try:
                self._cv.wait_for(lambda: not self._busy)
            finally:
                self._queries_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def batch(self) -> Iterator[None]:
        with self._cv:
            self._cv.wait_for(lambda: not self._busy and self._queries_waiting == 0)
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise rows in float32 (the model normalises in bfloat16: norms off by up to ~1%)."""
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1)


def quiet_model_loading() -> None:
    """No "Loading weights: 100%|..." progress bars from transformers / huggingface_hub.
    Warnings and errors still go through logging."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import transformers.utils.logging as tlog

        tlog.disable_progress_bar()
    except Exception:  # pragma: no cover - transformers missing or moved the helper
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # pragma: no cover
        pass


class Embedder:
    """Lazily loaded, thread-safe model wrapper. A daemon can serve queries from one thread
    while another indexes: queries take priority over index batches (GpuLock), so a query waits
    for at most the batch already running.

    Queries are always text, so `embed_query` / `load_query` load a text-only model
    (`vision_config=None, audio_config=None`, ~271M params) unless the full multimodal index
    model (`_model`, ~740M params) is already loaded, in which case queries reuse it.

    Loading and unloading also hold the lock (GPU lock when on CUDA, CPU query lock when
    `query_device == "cpu"`), so concurrent callers share a single load.
    """

    def __init__(self, cfg: Config | None = None, model_name: str | None = None, device: str | None = None,
                 query_device: str | None = None):
        self.model_name = model_name or (cfg.model_name if cfg else DEFAULT_MODEL)
        self.model_id = self.model_name
        self.requested_device = device or (cfg.device if cfg else None)
        self.requested_query_device = query_device or (cfg.query_device if cfg else "auto")
        self.batch_sizes = dict(cfg.batch_sizes if cfg else DEFAULT_BATCH_SIZES)
        # Current batch size per modality: reduced after a CUDA OOM, regrown slowly.
        self._limits: dict[str, int] = {}
        self._ok_batches: dict[str, int] = {}
        self.dim = 768  # replaced with the model's own value on load
        self.device: str | None = None
        self.query_device: str | None = None
        self.load_seconds = 0.0
        self.query_load_seconds = 0.0
        self.index_load_seconds = 0.0
        self.last_used = 0.0  # time.monotonic() of the last model call, load or unload
        self.query_last_used = 0.0
        self.index_last_used = 0.0
        self._loading = False
        self._query_loading = False
        self._index_loading = False
        self._loaded_once = False
        self._model: Any = None  # full multimodal index model
        self._query_model: Any = None  # text-only query model
        self._gpu = GpuLock()
        self._cpu_query_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------------
    @property
    def index_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def query_model_loaded(self) -> bool:
        return self._query_model is not None

    @property
    def loaded(self) -> bool:
        return self._model is not None or self._query_model is not None

    @property
    def query_ready(self) -> bool:
        return self._model is not None or self._query_model is not None

    @property
    def loading(self) -> bool:
        return self._loading or self._query_loading or self._index_loading

    def _resolve_index_device(self) -> str:
        if self.requested_device:
            return self.requested_device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_query_device(self) -> str:
        req = (self.requested_query_device or "auto").strip().lower()
        if req not in ("", "auto"):
            return req
        return self._resolve_index_device()

    @contextmanager
    def _query_lock(self, device: str | None = None) -> Iterator[None]:
        dev = device or self.query_device or self._resolve_query_device()
        if dev == "cpu":
            with self._cpu_query_lock:
                yield
        else:
            with self._gpu.query():
                yield

    def load(self) -> None:
        """Load the full multimodal index model."""
        with self._gpu.query():
            self._load_locked()

    def load_query(self) -> None:
        """Load the text-only query model (unless the full index model is already loaded)."""
        if self._model is not None or self._query_model is not None:
            return
        dev = self._resolve_query_device()
        with self._query_lock(dev):
            self._load_query_locked(dev)

    def _load_locked(self) -> None:
        """Load the full model; the caller holds the GPU lock."""
        if self._model is not None:
            return
        self._loading = True
        self._index_loading = True
        try:
            self._load_model()
            # If a GPU text-only query model was loaded, drop it so we don't hold both in VRAM.
            if self._query_model is not None and (self.query_device or "").startswith("cuda"):
                self._query_model = None
                gc.collect()
                self._empty_cache()
        finally:
            self._loading = False
            self._index_loading = False
        now = time.monotonic()
        self.last_used = now
        self.index_last_used = now

    def _load_query_locked(self, device: str | None = None) -> None:
        """Load the text-only query model; the caller holds the appropriate query lock."""
        if self._model is not None or self._query_model is not None:
            return
        self._loading = True
        self._query_loading = True
        try:
            self._load_query_model(device)
        finally:
            self._loading = False
            self._query_loading = False
        now = time.monotonic()
        self.last_used = now
        self.query_last_used = now

    def _instantiate_st(self, device: str, text_only: bool) -> tuple[Any, Any]:
        quiet_model_loading()
        import torch
        from sentence_transformers import SentenceTransformer

        # Never float16: EmbeddingGemma overflows to NaN in it.
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        kwargs: dict[str, Any] = {"device": device, "model_kwargs": {"dtype": dtype}}
        if text_only:
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        model = None
        if self._loaded_once or text_only:
            # Try local cache first: no Hub round-trips (~1 s), works offline, and cannot
            # switch to a newer model revision. Falls back to the Hub if not cached yet.
            try:
                model = SentenceTransformer(self.model_name, local_files_only=True, **kwargs)
            except Exception as exc:
                log.info("loading %s from the local cache failed (%s); trying the Hub", self.model_name, exc)
        if model is None:
            model = SentenceTransformer(self.model_name, **kwargs)
        model.eval()
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        # Warm up: the first call compiles/loads kernels (~0.5 s on CUDA), which would otherwise
        # land on the first user query.
        model.encode_query(["warm up"], convert_to_numpy=True, show_progress_bar=False)
        self._loaded_once = True
        return model, dtype

    def _load_model(self) -> None:
        started = time.monotonic()
        device = self._resolve_index_device()
        model, dtype = self._instantiate_st(device, text_only=False)
        self.device = device
        self._model = model
        self.index_load_seconds = time.monotonic() - started
        self.load_seconds = self.index_load_seconds
        log.info("loaded %s (full) on %s (%s) in %.1fs", self.model_name, device, dtype, self.index_load_seconds)

    def _load_query_model(self, device: str | None = None) -> None:
        started = time.monotonic()
        device = device or self._resolve_query_device()
        model, dtype = self._instantiate_st(device, text_only=True)
        self.query_device = device
        if self.device is None:
            self.device = device
        self._query_model = model
        self.query_load_seconds = time.monotonic() - started
        self.load_seconds = self.query_load_seconds
        log.info("loaded %s (text-only) on %s (%s) in %.1fs", self.model_name, device, dtype, self.query_load_seconds)

    def unload_index(self) -> bool:
        """Drop the full index model and return its VRAM to the driver.
        Returns False if it was not loaded."""
        with self._gpu.query():
            if self._model is None:
                return False
            started = time.monotonic()
            self._model = None
            gc.collect()  # the model's modules hold reference cycles
            self._empty_cache()
            now = time.monotonic()
            self.index_last_used = now
            self.last_used = now
            log.info("unloaded %s (full) in %.2fs", self.model_name, now - started)
            return True

    def unload_query(self) -> bool:
        """Drop the text-only query model and free its memory/VRAM.
        Returns False if it was not loaded."""
        dev = self.query_device or self._resolve_query_device()
        with self._query_lock(dev):
            if self._query_model is None:
                return False
            started = time.monotonic()
            self._query_model = None
            gc.collect()
            if dev.startswith("cuda"):
                self._empty_cache(dev)
            now = time.monotonic()
            self.query_last_used = now
            self.last_used = now
            log.info("unloaded %s (text-only) in %.2fs", self.model_name, now - started)
            return True

    def unload(self) -> bool:
        """Drop any loaded models (index and/or query) and return VRAM to the driver.
        Waits for a running batch or query; the next model call loads again.
        Returns False if nothing was loaded."""
        u_idx = self.unload_index()
        u_qry = self.unload_query()
        return u_idx or u_qry

    def release_cache(self) -> None:
        """Return cached-but-unused CUDA memory to the driver (e.g. after a run of video batches)."""
        with self._gpu.query():
            self._empty_cache()

    def _empty_cache(self, device: str | None = None) -> None:
        dev = device or self.device or self.query_device
        if dev and dev.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    # -- encoding -------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """Query vector (read-only array; repeated queries come from a small LRU cache)."""
        with self._query_cache_lock:
            hit = self._query_cache.get(query)
            if hit is not None:
                self._query_cache.move_to_end(query)
                return hit
        # Reuse the full model if it is already loaded (and we don't already have a CPU query model).
        if self._query_model is None and self._model is not None:
            with self._gpu.query():
                model = self._model or self._query_model
                if model is None:
                    dev = self._resolve_query_device()
                    self._load_query_locked(dev)
                    model = self._query_model or self._model
                vec = model.encode_query([query], convert_to_numpy=True, normalize_embeddings=True,
                                         show_progress_bar=False)
                now = time.monotonic()
                self.last_used = now
                self.query_last_used = now
        else:
            dev = self.query_device or self._resolve_query_device()
            with self._query_lock(dev):
                if self._query_model is None and self._model is None:
                    self._load_query_locked(dev)
                model = self._query_model if self._query_model is not None else self._model
                vec = model.encode_query([query], convert_to_numpy=True, normalize_embeddings=True,
                                         show_progress_bar=False)
                now = time.monotonic()
                self.last_used = now
                self.query_last_used = now
        out = _normalise(vec[0])
        out.flags.writeable = False
        with self._query_cache_lock:
            self._query_cache[query] = out
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return out

    def embed_chunks(self, chunks: Sequence[Chunk], title: str | Sequence[str]) -> np.ndarray:
        return embed_chunks(self, chunks, title)

    def batch_limit(self, modality: str) -> int:
        return self._limits.get(modality, max(1, self.batch_sizes.get(modality, 8)))

    def embed_batch(self, modality: str, contents: Sequence[Any], titles: Sequence[str]) -> np.ndarray:
        """Embed single-modality contents in batches of the configured size.

        On a CUDA OOM the batch is halved and retried. The recovery runs after the `except`
        block: inside it the traceback still references the failed call's frames and their
        tensors, so emptying the cache there frees little. The reduced size is remembered for
        the modality and only regrown after OOM_REGROW_AFTER clean batches.
        """
        inputs = [model_input(modality, c, t) for c, t in zip(contents, titles, strict=True)]
        parts: list[np.ndarray] = []
        i = 0
        while i < len(inputs):
            size = self.batch_limit(modality)
            batch = inputs[i : i + size]
            oom = False
            try:
                parts.append(self._encode(batch))
            except Exception as exc:
                if not _is_oom(exc) or len(batch) == 1:
                    raise
                oom = True
            if oom:
                del batch
                smaller = max(1, size // 2)
                self._limits[modality] = smaller
                self._ok_batches[modality] = 0
                log.warning("CUDA out of memory on a %s batch of %d, continuing with %d", modality, size, smaller)
                self.release_cache()
                continue
            i += len(batch)
            self._regrow(modality, size)
        if not parts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _regrow(self, modality: str, size: int) -> None:
        if modality not in self._limits:
            return
        ok = self._ok_batches.get(modality, 0) + 1
        self._ok_batches[modality] = ok
        if ok >= OOM_REGROW_AFTER:
            configured = max(1, self.batch_sizes.get(modality, 8))
            grown = min(configured, size * 2)
            if grown >= configured:
                del self._limits[modality]
            else:
                self._limits[modality] = grown
            self._ok_batches[modality] = 0

    def _encode(self, inputs: list[Any]) -> np.ndarray:
        with self._gpu.batch():
            self._load_locked()
            vecs = self._model.encode(inputs, prompt="", batch_size=len(inputs), convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
            now = time.monotonic()
            self.last_used = now
            self.index_last_used = now
        vecs = np.asarray(vecs, dtype=np.float32)
        if not np.isfinite(vecs).all():
            raise FloatingPointError("model produced non-finite embeddings")
        return _normalise(vecs)


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)
