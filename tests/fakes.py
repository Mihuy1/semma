"""Test doubles: a deterministic embedder and an in-memory extractor."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gemmasearch.config import Config
from gemmasearch.extract.base import Chunk, ExtractionError

DIM = 768


def _word_vec(word: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


def bag_of_words(text: str) -> np.ndarray:
    """Sum of per-word random vectors: identical text gives identical vectors, and texts sharing
    words are more similar, so ranking can be asserted without a model."""
    words = re.findall(r"\w+", text.lower()) or ["<empty>"]
    v = np.sum([_word_vec(w) for w in words], axis=0)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmbedder:
    model_id = "fake-bow"
    dim = DIM

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []  # (modality, batch size)
        self.fail_on: set[str] = set()  # contents that raise, to test per-item isolation

    def embed_query(self, query: str) -> np.ndarray:
        return bag_of_words(query)

    def embed_batch(self, modality: str, contents: Sequence[Any], titles: Sequence[str]) -> np.ndarray:
        self.calls.append((modality, len(contents)))
        out = []
        for c in contents:
            if isinstance(c, str) and c in self.fail_on:
                raise RuntimeError(f"cannot embed {c!r}")
            # Titles are ignored so identical chunk text in two files gives identical vectors.
            out.append(bag_of_words(c if isinstance(c, str) else repr(c)))
        return np.stack(out)


class FakeExtractor:
    """Text chunks from a dict path -> list of strings (or an exception to raise)."""

    def __init__(self, contents: dict[str, list[str] | Exception] | None = None,
                 on_chunk: Callable[[Path, int], None] | None = None) -> None:
        self.contents = contents or {}
        self.on_chunk = on_chunk
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, path: Path, kind: str, cfg: Config) -> Iterator[Chunk]:
        with self.lock:
            self.calls.append(path.name)
        data = self.contents.get(path.name)
        if data is None:
            data = [path.read_text(encoding="utf-8")]
        if isinstance(data, Exception):
            raise data
        for i, text in enumerate(data):
            if self.on_chunk:
                self.on_chunk(path, i)
            yield Chunk("text", text, locator=f"part {i + 1}" if len(data) > 1 else "", position=None)


__all__ = ["DIM", "ExtractionError", "FakeEmbedder", "FakeExtractor", "bag_of_words"]
