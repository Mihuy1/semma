"""Hybrid search: vector KNN over chunks + FTS5 BM25 over names/folders and chunk text, fused
with reciprocal rank fusion.

1. embed the query, KNN over vec_chunks (k = cfg.knn_k, filtered by kind inside the KNN)
2. group chunks by file, the best chunk wins (its similarity, locator and snippet)
3. BM25 rankings from fts_files (name, folders) and fts_chunks (text), also grouped by file
4. RRF: score = sum over rankings of weight / (rrf_k + rank)
5. drop files that only matched the vector search and are weak: similarity below
   cfg.min_similarity, or more than cfg.similarity_margin below the query's best similarity

EmbeddingGemma similarities are compressed (cosine of unrelated query/document pairs is far
from 0), so the thresholds are calibrated on real data; see the defaults in config.py.
Lexical matches are kept regardless: a file name containing every query term is a hit even
when the content (or a photo) is semantically unrelated.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import Config
from .store import Store

if TYPE_CHECKING:
    from .embedder import EmbedderLike

# RRF weights per ranking. File names are a strong signal for desktop search; chunk text BM25 is
# weaker because the vector ranking already covers the same text semantically.
W_VECTOR, W_NAME, W_TEXT = 1.0, 1.0, 0.5
FTS_LIMIT = 50
MAX_QUERY_TERMS = 16
_TERM = re.compile(r"[^\W_]+")  # letters/digits; drops every FTS5 operator, quote and hyphen


@dataclass(slots=True)
class SearchResult:
    path: str
    kind: str
    score: float  # fused RRF score (only meaningful for ordering)
    similarity: float | None  # cosine similarity of the best chunk, None for lexical-only hits
    modality: str | None
    locator: str
    snippet: str
    position: float | None
    mtime: float
    matched: list[str] = field(default_factory=list)  # which rankings found it: vector/name/text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchTimings:
    embed_ms: float = 0.0
    search_ms: float = 0.0  # everything after the query embedding


def fts_query(text: str) -> str | None:
    """Turn arbitrary user input into a safe FTS5 query: every letter/digit run becomes a quoted
    prefix term ("term"*), implicitly ANDed. Returns None when nothing searchable remains."""
    terms = _TERM.findall(text)[:MAX_QUERY_TERMS]
    if not terms:
        return None
    # \\w never contains '"', so quoting cannot be broken out of; the quotes also neutralise
    # AND/OR/NOT/NEAR. Single characters get no wildcard (they would match half the index).
    return " ".join(f'"{t}"*' if len(t) > 1 else f'"{t}"' for t in terms)


class Searcher:
    def __init__(self, store: Store, cfg: Config, embedder: EmbedderLike):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.last_timings = SearchTimings()

    def search(self, query: str, limit: int = 10, kinds: Sequence[str] | None = None) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        results = self.search_vector(qvec, query, limit, kinds)
        t2 = time.perf_counter()
        self.last_timings = SearchTimings((t1 - t0) * 1000, (t2 - t1) * 1000)
        return results

    def search_vector(self, qvec: np.ndarray, query: str, limit: int = 10,
                      kinds: Sequence[str] | None = None) -> list[SearchResult]:
        """Everything after the query embedding (separate so it can be timed and tested alone)."""
        db = self.store.db
        kinds = list(kinds) if kinds else None
        kind_in = f"IN ({','.join('?' * len(kinds))})" if kinds else ""

        # 1-2. KNN, best chunk per file.
        knn_sql = "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        args: list[Any] = [np.ascontiguousarray(qvec, dtype=np.float32), self.cfg.knn_k]
        if kinds:
            knn_sql += f" AND kind {kind_in}"
            args += kinds
        hits = db.execute(knn_sql, args).fetchall()
        best: dict[int, tuple[float, int]] = {}  # file_id -> (similarity, chunk_id)
        chunk_file = self._chunk_files([h[0] for h in hits])
        for chunk_id, distance in hits:
            file_id = chunk_file.get(chunk_id)
            if file_id is None:
                continue
            sim = 1.0 - distance
            if file_id not in best or sim > best[file_id][0]:
                best[file_id] = (sim, chunk_id)
        vector_rank = sorted(best, key=lambda f: -best[f][0])

        # 3. Lexical rankings.
        name_rank: list[int] = []
        text_best: dict[int, int] = {}  # file_id -> chunk_id of its best text match
        match = fts_query(query)
        if match:
            kind_filter = f" AND f.kind {kind_in}" if kinds else ""
            name_rank = [r[0] for r in self._fts(
                f"SELECT fts_files.rowid FROM fts_files JOIN files f ON f.id = fts_files.rowid "
                f"WHERE fts_files MATCH ?{kind_filter} ORDER BY bm25(fts_files, 10.0, 1.0) LIMIT {FTS_LIMIT}",
                [match, *(kinds or [])])]
            for chunk_id, file_id in self._fts(
                f"SELECT c.id, c.file_id FROM (SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? "
                f"ORDER BY rank LIMIT {FTS_LIMIT * 4}) m JOIN chunks c ON c.id = m.rowid "
                f"JOIN files f ON f.id = c.file_id WHERE 1{kind_filter} ORDER BY m.rank",
                [match, *(kinds or [])]):
                text_best.setdefault(file_id, chunk_id)
        text_rank = list(text_best)[:FTS_LIMIT]

        # 4. Reciprocal rank fusion.
        k = self.cfg.rrf_k
        fused: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for label, weight, ranking in (("vector", W_VECTOR, vector_rank), ("name", W_NAME, name_rank),
                                       ("text", W_TEXT, text_rank)):
            for rank, file_id in enumerate(ranking):
                fused[file_id] = fused.get(file_id, 0.0) + weight / (k + rank + 1)
                matched.setdefault(file_id, []).append(label)

        # 5. Drop weak vector-only matches; lexical matches are kept regardless.
        top_sim = max((b[0] for b in best.values()), default=0.0)
        floor = max(self.cfg.min_similarity, top_sim - self.cfg.similarity_margin)
        candidates = [f for f in fused if matched[f] != ["vector"] or best[f][0] >= floor]
        candidates.sort(key=lambda f: (-fused[f], -(best[f][0] if f in best else 0.0)))
        top = candidates[:limit]
        return self._materialise(top, fused, best, text_best, matched)

    # -- helpers --------------------------------------------------------------
    def _chunk_files(self, chunk_ids: list[int]) -> dict[int, int]:
        if not chunk_ids:
            return {}
        rows = self.store.db.execute(
            f"SELECT id, file_id FROM chunks WHERE id IN ({','.join('?' * len(chunk_ids))})", chunk_ids)
        return {r[0]: r[1] for r in rows}

    def _fts(self, sql: str, args: list[Any]) -> list[sqlite3.Row]:
        try:
            return self.store.db.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            # fts_query only emits quoted terms, so this is a last line of defence.
            return []

    def _materialise(self, file_ids: list[int], fused: dict[int, float], best: dict[int, tuple[float, int]],
                     text_best: dict[int, int], matched: dict[int, list[str]]) -> list[SearchResult]:
        if not file_ids:
            return []
        db = self.store.db
        marks = ",".join("?" * len(file_ids))
        files = {r["id"]: r for r in db.execute(f"SELECT id, path, kind, mtime FROM files WHERE id IN ({marks})",
                                                file_ids)}
        chunk_ids = [best[f][1] if f in best else text_best.get(f) for f in file_ids]
        wanted = [c for c in chunk_ids if c is not None]
        chunks = {}
        if wanted:
            chunks = {r["id"]: r for r in db.execute(
                f"SELECT id, modality, locator, snippet, position FROM chunks "
                f"WHERE id IN ({','.join('?' * len(wanted))})", wanted)}
        out: list[SearchResult] = []
        for file_id, chunk_id in zip(file_ids, chunk_ids, strict=True):
            f = files.get(file_id)
            if f is None:
                continue
            c = chunks.get(chunk_id) if chunk_id is not None else None
            out.append(SearchResult(
                path=f["path"], kind=f["kind"], score=round(fused[file_id], 6),
                similarity=round(best[file_id][0], 4) if file_id in best else None,
                modality=c["modality"] if c else None, locator=c["locator"] if c else "",
                snippet=c["snippet"] if c else "", position=c["position"] if c else None,
                mtime=f["mtime"], matched=matched[file_id],
            ))
        return out
