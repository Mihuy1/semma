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

import logging
import threading
from collections.abc import Sequence
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


class Embedder:
    """Lazily loaded, thread-safe model wrapper. One lock guards the GPU, so a daemon can serve
    queries from another thread while indexing (each call waits for at most one batch)."""

    def __init__(self, cfg: Config | None = None, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or (cfg.model_name if cfg else DEFAULT_MODEL)
        self.model_id = self.model_name
        self.requested_device = device or (cfg.device if cfg else None)
        self.batch_sizes = dict(cfg.batch_sizes if cfg else DEFAULT_BATCH_SIZES)
        self.dim = 768  # replaced with the model's own value on load
        self.device: str | None = None
        self.load_seconds = 0.0
        self._model: Any = None
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        import time

        import torch
        from sentence_transformers import SentenceTransformer

        started = time.monotonic()
        device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Never float16: EmbeddingGemma overflows to NaN in it.
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = SentenceTransformer(self.model_name, device=device, model_kwargs={"dtype": dtype})
        model.eval()
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        # Warm up: the first CUDA call compiles/loads kernels (~0.5 s), which would otherwise
        # land on the first user query.
        model.encode_query(["warm up"], convert_to_numpy=True)
        self.device = device
        self._model = model
        self.load_seconds = time.monotonic() - started
        log.info("loaded %s on %s (%s) in %.1fs", self.model_name, device, dtype, self.load_seconds)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._empty_cache()

    def release_cache(self) -> None:
        """Return cached-but-unused CUDA memory to the driver (e.g. after a run of video batches)."""
        with self._lock:
            self._empty_cache()

    def _empty_cache(self) -> None:
        if self.device and self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    # -- encoding -------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            self._load_locked()
            vec = self._model.encode_query([query], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    def embed_chunks(self, chunks: Sequence[Chunk], title: str | Sequence[str]) -> np.ndarray:
        return embed_chunks(self, chunks, title)

    def embed_batch(self, modality: str, contents: Sequence[Any], titles: Sequence[str]) -> np.ndarray:
        """Embed single-modality contents in batches of the configured size."""
        inputs = [model_input(modality, c, t) for c, t in zip(contents, titles, strict=True)]
        size = max(1, self.batch_sizes.get(modality, 8))
        parts: list[np.ndarray] = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + size]
            try:
                parts.append(self._encode(batch))
            except Exception as exc:
                if not _is_oom(exc) or size == 1:
                    raise
                size = max(1, size // 2)
                log.warning("CUDA out of memory on %s batch of %d, retrying with %d", modality, len(batch), size)
                self.release_cache()
                continue
            i += len(batch)
        if not parts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _encode(self, inputs: list[Any]) -> np.ndarray:
        import torch

        with self._lock:
            self._load_locked()
            try:
                vecs = self._model.encode(inputs, prompt="", batch_size=len(inputs), convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False)
            except torch.cuda.OutOfMemoryError:
                self._empty_cache()
                raise
        vecs = np.asarray(vecs, dtype=np.float32)
        if not np.isfinite(vecs).all():
            raise FloatingPointError("model produced non-finite embeddings")
        return vecs


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return "out of memory" in str(exc).lower()
