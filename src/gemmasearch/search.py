"""Hybrid search: vector KNN over chunks + FTS5 BM25 over names/folders and chunk text, fused
with reciprocal rank fusion, then near-duplicates collapsed.

1. embed the query; KNN over the chunk vectors (k = cfg.knn_k, filtered by kind inside the
   KNN) from the in-memory VectorCache when the Store has one, else sqlite-vec
2. group chunks by file, the best chunk wins (its similarity, locator and snippet)
3. BM25 rankings from fts_files (file name; folder names as a separate, weak ranking) and
   fts_chunks (text), all filtered by kind inside the query
4. RRF: score = sum over rankings of weight / (rrf_k + rank). The vector ranking orders files
   by similarity plus a small location prior (cfg.location_prior * priority / 100), which only
   reorders near-ties: ~/Documents beats a dataset folder when the content is equally relevant.
5. drop weak matches: a file needs a name or chunk-text match, or a similarity of at least
   cfg.min_similarity and within cfg.similarity_margin of the query's best similarity.
   Single-word queries use stricter values (*_short), and their chunk-text-only hits below the
   floor rank after all other results. Folder-only matches are not enough.
6. collapse near-duplicates (copies of a photo, versions of a document): files whose matching
   chunks, or (same kind) whose first chunks, are near-identical (cos >=
   cfg.duplicate_similarity[modality]) become one result, represented by the copy with the
   highest priority, then the best-ranked; `duplicates` says how many were folded into it.
7. drop results whose file no longer exists (lstat; deleted since the last scan)

EmbeddingGemma similarities are compressed (cosine of unrelated query/document pairs is far
from 0), so the thresholds are calibrated on real data; see the defaults in config.py.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import MAX_KNN_K, Config
from .store import Store

if TYPE_CHECKING:
    from .embedder import EmbedderLike

# RRF weights per ranking. File names are a strong signal for desktop search; chunk text BM25 is
# weaker because the vector ranking already covers the same text semantically; folder names
# are weaker still (a folder word matches every file inside).
W_VECTOR, W_NAME, W_TEXT, W_FOLDER = 1.0, 1.0, 0.5, 0.25
FTS_LIMIT = 50
MAX_QUERY_TERMS = 16
TEXT_PREFIX_MIN = 3  # chunk text: only the last term gets a wildcard, and only from 3 letters
# A chunk-text term matching more chunks than this (or than TEXT_MAX_DF_RATIO of all chunks,
# whichever is larger) is dropped from the text query: ranking every chunk that contains "the"
# costs ~60 ms at 100k chunks and says nothing the vector ranking does not.
TEXT_MAX_DF = 5000
TEXT_MAX_DF_RATIO = 0.05
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
    matched: list[str] = field(default_factory=list)  # rankings that found it: vector/name/text/folder
    duplicates: int = 0  # near-identical files collapsed into this result
    duplicate_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchTimings:
    embed_ms: float = 0.0
    search_ms: float = 0.0  # everything after the query embedding


def query_terms(text: str) -> list[str]:
    return _TERM.findall(text)[:MAX_QUERY_TERMS]


def fts_query(text: str) -> str | None:
    """FTS5 query for file names: every letter/digit run becomes a quoted prefix term ("term"*),
    implicitly ANDed. Returns None when nothing searchable remains. (The names table is small
    and has 2- and 3-character prefix indexes, so prefixing every term is cheap.)"""
    terms = query_terms(text)
    if not terms:
        return None
    # \\w never contains '"', so quoting cannot be broken out of; the quotes also neutralise
    # AND/OR/NOT/NEAR. Single characters get no wildcard (they would match half the index).
    return " ".join(f'"{t}"*' if len(t) > 1 else f'"{t}"' for t in terms)


def fts_text_terms(text: str) -> list[str]:
    """FTS5 terms for chunk text: exact terms, plus a wildcard on the last term (the word being
    typed) when it has at least TEXT_PREFIX_MIN characters. A short prefix over a large text
    index expands to thousands of terms (`w1*` took ~1 s at 100k chunks)."""
    terms = query_terms(text)
    out = [f'"{t}"' for t in terms]
    if terms and len(terms[-1]) >= TEXT_PREFIX_MIN:
        out[-1] += "*"
    return out


def fts_text_query(text: str) -> str | None:
    return " ".join(fts_text_terms(text)) or None


def _exists(path: str) -> bool:
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True  # unreadable, unmounted for a moment...: keep it
    return True


class Searcher:
    def __init__(self, store: Store, cfg: Config, embedder: EmbedderLike):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.last_timings = SearchTimings()

    def search(self, query: str, limit: int = 10, kinds: Sequence[str] | None = None) -> list[SearchResult]:
        query = query.strip()
        if not query or limit <= 0:
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
        db = self.store.reader()
        own_txn = not db.in_transaction
        if own_txn:
            db.execute("BEGIN")  # one snapshot for every query below (WAL)
        try:
            return self._search(db, qvec, query, limit, list(kinds) if kinds else None)
        finally:
            if own_txn:
                db.rollback()

    # -- pipeline -------------------------------------------------------------
    def _search(self, db: sqlite3.Connection, qvec: np.ndarray, query: str, limit: int,
                kinds: list[str] | None) -> list[SearchResult]:
        cfg = self.cfg
        kind_in = f"IN ({','.join('?' * len(kinds))})" if kinds else ""
        kind_args = kinds or []

        # 1-2. KNN, best chunk per file.
        hits, vectors = self._knn(db, qvec, kinds, kind_in)
        best: dict[int, tuple[float, int]] = {}  # file_id -> (similarity, chunk_id)
        for chunk_id, file_id, sim in hits:
            if file_id not in best or sim > best[file_id][0]:
                best[file_id] = (sim, chunk_id)

        # 3. Lexical rankings: names first (small table), then folders, then chunk text.
        name_rank: list[int] = []
        folder_rank: list[int] = []
        text_best: dict[int, int] = {}  # file_id -> chunk_id of its best text match
        kind_join = f" JOIN files f ON f.id = fts_files.rowid AND f.kind {kind_in}" if kinds else ""
        match = fts_query(query)
        if match:
            name_rank = [r[0] for r in self._fts(
                db, f"SELECT fts_files.rowid FROM fts_files{kind_join} WHERE fts_files MATCH ? "
                    f"ORDER BY rank LIMIT {FTS_LIMIT}", [*kind_args, f"{{name}} : ({match})"])]
            named = set(name_rank)
            folder_rank = [r[0] for r in self._fts(
                db, f"SELECT fts_files.rowid FROM fts_files{kind_join} WHERE fts_files MATCH ? "
                    f"ORDER BY rank LIMIT {FTS_LIMIT * 2}", [*kind_args, match]) if r[0] not in named][:FTS_LIMIT]
        text_match = self._selective(db, fts_text_terms(query))
        if text_match:
            if kinds:
                sql = (f"SELECT c.id, c.file_id FROM fts_chunks JOIN chunks c ON c.id = fts_chunks.rowid "
                       f"JOIN files f ON f.id = c.file_id WHERE fts_chunks MATCH ? AND f.kind {kind_in} "
                       f"ORDER BY fts_chunks.rank LIMIT {FTS_LIMIT * 4}")
                args = [text_match, *kind_args]
            else:
                sql = (f"SELECT c.id, c.file_id FROM (SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? "
                       f"ORDER BY rank LIMIT {FTS_LIMIT * 4}) m JOIN chunks c ON c.id = m.rowid ORDER BY m.rank")
                args = [text_match]
            for chunk_id, file_id in self._fts(db, sql, args):
                text_best.setdefault(file_id, chunk_id)
        text_rank = list(text_best)[:FTS_LIMIT]

        all_ids = set(best) | set(name_rank) | set(folder_rank) | set(text_rank)
        if not all_ids:
            return []
        files = self._files(db, all_ids)

        # 4. Reciprocal rank fusion; the vector ranking includes the location prior.
        def prior(f: int) -> float:
            row = files.get(f)
            return cfg.location_prior * (row["priority"] / 100.0) if row else 0.0

        vector_rank = sorted(best, key=lambda f: -(best[f][0] + prior(f)))
        k = cfg.rrf_k
        fused: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for label, weight, ranking in (("vector", W_VECTOR, vector_rank), ("name", W_NAME, name_rank),
                                       ("text", W_TEXT, text_rank), ("folder", W_FOLDER, folder_rank)):
            for rank, file_id in enumerate(ranking):
                fused[file_id] = fused.get(file_id, 0.0) + weight / (k + rank + 1)
                matched.setdefault(file_id, []).append(label)

        # 5. Drop weak matches: name/text matches are kept regardless, the rest needs similarity.
        short = len(query_terms(query)) <= 1
        min_sim = cfg.min_similarity_short if short else cfg.min_similarity
        margin = cfg.similarity_margin_short if short else cfg.similarity_margin
        top_sim = max((b[0] for b in best.values()), default=0.0)
        floor = max(min_sim, top_sim - margin)

        def strong(f: int) -> bool:
            return "name" in matched[f] or (f in best and best[f][0] >= floor)

        def keep(f: int) -> bool:
            # A file name containing the query is a hit whatever the content, and so is chunk
            # text containing every query word; anything else needs the similarity.
            return f in files and (strong(f) or "text" in matched[f])

        # Single words: a chunk-text hit below the similarity floor (one mention of "dog" in a
        # 90-page thesis) is kept but ranked after every strong match (the dog photos).
        def tier(f: int) -> int:
            return 0 if not short or strong(f) else 1

        candidates = [f for f in fused if keep(f)]
        candidates.sort(key=lambda f: (tier(f), -fused[f], -(best[f][0] if f in best else 0.0), -prior(f)))

        # 6-7. Collapse near-duplicates, drop vanished files.
        groups = self._group_duplicates(db, candidates, files, best, text_best, vectors)
        out: list[SearchResult] = []
        picks: list[tuple[int, list[int]]] = []
        rank = {f: i for i, f in enumerate(candidates)}
        for members in groups:
            # The copy in the most important location; among equals the best-ranked one.
            ordered = sorted(members, key=lambda f: (-files[f]["priority"], rank[f]))
            rep = next((f for f in ordered if _exists(files[f]["path"])), None)
            if rep is None:
                continue
            picks.append((rep, [f for f in ordered if f != rep]))
            if len(picks) >= limit:
                break
        chunk_ids = [best[f][1] if f in best else text_best.get(f) for f, _ in picks]
        chunks = self._chunks(db, [c for c in chunk_ids if c is not None])
        for (f, others), chunk_id in zip(picks, chunk_ids, strict=True):
            row = files[f]
            c = chunks.get(chunk_id) if chunk_id is not None else None
            labels = list(dict.fromkeys(m for g in (f, *others) for m in matched[g]))
            out.append(SearchResult(
                path=row["path"], kind=row["kind"], score=round(max(fused[g] for g in (f, *others)), 6),
                similarity=round(best[f][0], 4) if f in best else None,
                modality=c["modality"] if c else None, locator=c["locator"] if c else "",
                snippet=c["snippet"] if c else "", position=c["position"] if c else None,
                mtime=row["mtime"], matched=labels if others else matched[f],
                duplicates=len(others), duplicate_paths=[files[g]["path"] for g in others[:20]],
            ))
        return out

    # -- steps ----------------------------------------------------------------
    def _knn(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None,
             kind_in: str) -> tuple[list[tuple[int, int, float]], dict[int, np.ndarray]]:
        """(chunk_id, file_id, similarity) best first. (Selecting the embedding column in the
        sqlite-vec KNN adds ~50 ms at 100k chunks; the few vectors dedup needs are fetched later.)"""
        k = max(1, min(self.cfg.knn_k, MAX_KNN_K))
        cache = self.store.vectors
        if cache is not None:
            with cache.lock:
                cache.ensure_fresh(db)
                if not cache.too_big:
                    return cache.knn(qvec, k, kinds), {}
        sql = "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        args: list[Any] = [np.ascontiguousarray(qvec, dtype=np.float32), k]
        if kinds:
            sql += f" AND kind {kind_in}"
            args += kinds
        rows = db.execute(sql, args).fetchall()
        chunk_file = self._chunk_files(db, [r[0] for r in rows])
        return [(r[0], chunk_file[r[0]], 1.0 - r[1]) for r in rows if r[0] in chunk_file], {}

    def _group_duplicates(self, db: sqlite3.Connection, candidates: list[int], files: dict[int, sqlite3.Row],
                          best: dict[int, tuple[float, int]], text_best: dict[int, int],
                          vectors: dict[int, np.ndarray]) -> list[list[int]]:
        """Greedy grouping in rank order: a file joins the first earlier group whose head is a
        near-duplicate (cos >= duplicate_similarity[modality]) by either signature:
        - the chunk that matched the query (best vector chunk, else best text chunk): the same
          photo in three folders, the same section in two versions of a document;
        - the first chunk, for files of the same kind: versions of a document that share their
          title page and abstract but whose best-matching sections differ.
        Files without chunks are never grouped."""
        thresholds = self.cfg.duplicate_similarity
        if not candidates or not thresholds:
            return [[f] for f in candidates]
        marks = ",".join("?" * len(candidates))
        first: dict[int, tuple[int, str]] = {r[0]: (r[1], r[2]) for r in db.execute(
            f"SELECT file_id, id, modality FROM chunks WHERE file_id IN ({marks}) AND seq = 0", candidates)}
        hit: dict[int, int] = {f: best[f][1] if f in best else text_best[f]
                               for f in candidates if f in best or f in text_best}
        modality = {cid: mod for cid, mod in first.values()}
        extra = [c for c in hit.values() if c not in modality]
        if extra:
            modality.update({r[0]: r[1] for r in db.execute(
                f"SELECT id, modality FROM chunks WHERE id IN ({','.join('?' * len(extra))})", extra)})
        need = list({c for c in [*hit.values(), *(c for c, _ in first.values())] if c not in vectors})
        if need:
            vectors = {**vectors, **self._vectors(db, need)}

        def unit(cid: int | None) -> np.ndarray | None:
            v = vectors.get(cid) if cid is not None else None
            return None if v is None else v / (np.linalg.norm(v) or 1.0)

        groups: list[list[int]] = []
        heads: list[tuple[int, str, str, np.ndarray | None, np.ndarray | None]] = []  # group, modality, kind, hit, first
        for f in candidates:
            fc = first.get(f)
            hc = hit.get(f, fc[0] if fc else None)
            mod = modality.get(hc) if hc is not None else None
            thr = thresholds.get(mod) if mod else None
            hv, fv = unit(hc), unit(fc[0] if fc else None)
            kind = files[f]["kind"]
            joined = False
            if thr is not None and (hv is not None or fv is not None):
                for g, gmod, gkind, ghv, gfv in heads:
                    if gmod != mod:
                        continue
                    if (hv is not None and ghv is not None and float(hv @ ghv) >= thr) or \
                            (gkind == kind and fv is not None and gfv is not None and float(fv @ gfv) >= thr):
                        groups[g].append(f)
                        joined = True
                        break
            if not joined:
                if thr is not None and mod is not None:
                    heads.append((len(groups), mod, kind, hv, fv))
                groups.append([f])
        return groups

    # -- helpers --------------------------------------------------------------
    def _selective(self, db: sqlite3.Connection, terms: list[str]) -> str | None:
        """The chunk-text query without terms that match too many chunks (see TEXT_MAX_DF).
        Each probe stops after the cap, so it costs well under 2 ms even for "the"."""
        if not terms:
            return None
        top = db.execute("SELECT max(id) FROM chunks").fetchone()[0] or 0
        cap = max(TEXT_MAX_DF, int(top * TEXT_MAX_DF_RATIO))
        if top <= cap:
            return " ".join(terms)  # small index: nothing can be too common
        keep = []
        for t in terms:
            rows = self._fts(db, "SELECT count(*) FROM (SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? "
                                 "LIMIT ?)", [t, cap + 1])
            if rows and rows[0][0] <= cap:
                keep.append(t)
        return " ".join(keep) or None

    def _vectors(self, db: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, np.ndarray]:
        cache = self.store.vectors
        if cache is not None and not cache.too_big:
            with cache.lock:
                got = cache.vectors(chunk_ids)
            if len(got) == len(chunk_ids):
                return got
        rows = db.execute(f"SELECT chunk_id, embedding FROM vec_chunks WHERE chunk_id IN "
                          f"({','.join('?' * len(chunk_ids))})", chunk_ids)
        return {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}

    @staticmethod
    def _chunk_files(db: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, int]:
        if not chunk_ids:
            return {}
        rows = db.execute(f"SELECT id, file_id FROM chunks WHERE id IN ({','.join('?' * len(chunk_ids))})", chunk_ids)
        return {r[0]: r[1] for r in rows}

    @staticmethod
    def _files(db: sqlite3.Connection, file_ids: set[int]) -> dict[int, sqlite3.Row]:
        ids = list(file_ids)
        return {r["id"]: r for r in db.execute(
            f"SELECT id, path, kind, mtime, priority FROM files WHERE id IN ({','.join('?' * len(ids))})", ids)}

    @staticmethod
    def _chunks(db: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
        if not chunk_ids:
            return {}
        return {r["id"]: r for r in db.execute(
            f"SELECT id, modality, locator, snippet, position FROM chunks "
            f"WHERE id IN ({','.join('?' * len(chunk_ids))})", chunk_ids)}

    @staticmethod
    def _fts(db: sqlite3.Connection, sql: str, args: list[Any]) -> list[sqlite3.Row]:
        try:
            return db.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            # The query builders only emit quoted terms, so this is a last line of defence.
            return []
