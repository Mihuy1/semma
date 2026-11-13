"""Hybrid search: how far each file's best chunk stands out in meaning for this query (a z-score
over every stored vector), plus how many of the query's words the file contains.

1. query words: `query_terms` minus stopwords and type words ("screenshot", "pdf", "video",
   "kuva", ...); type words give a bonus instead (step 4), they are not filters. A query made only
   of type words or stopwords keeps them as words. Words matching too many chunks are dropped from
   the chunk-text queries and from word coverage (`_selective`); the last word gets a prefix wildcard (the word being
   typed) from TEXT_PREFIX_MIN characters, in chunk text only while it is in no chunk as typed.
2. lexical candidates (all filtered by kind): an OR query over chunk text (bm25 order, its best
   chunk per file is the file's text match), over file names and over folder names, and a name
   query requiring every word (the name match).
3. meaning: the query's cosine with EVERY stored vector (one matrix-vector product over the
   in-memory VectorCache), minus the mean cosine of that vector's modality for this query (text,
   image, audio and video vectors sit at different levels: image cosines run ~0.05 below text
   ones; a modality with < MODALITY_MIN vectors uses the overall mean), divided by the standard
   deviation of those centred values over all vectors (floored at STD_FLOOR) after subtracting
   their mean. That is a z-score: how many standard deviations a chunk stands out from the whole
   index for this query. The scale always comes from all vectors; `kinds` only restricts which
   files are candidates. A file's meaning is its best chunk's z (its cosine is reported as
   `similarity`). Without the cache (sqlite-vec fallback, rare) the scale comes from a fixed
   sample of SCALE_SAMPLE vectors and only the KNN top-k (cfg.knn_k) is scored.
4. candidates: z >= cfg.z_gate, or at least half the words in the file's chunk text or name
   (folder words alone are too weak: one folder word matches every file inside), or the name
   match. score = max(z, 0) + W_WORDS * coverage**1.5 (coverage counts chunk text, name and
   folders) + NAME_BONUS for the name match + TYPE_BONUS for the kind a type word asks for
   (+ SCREENSHOT_BONUS for screenshots when asked for) + the location prior: cfg.location_prior
   is a cosine bonus at priority 100, converted to z units by the query's standard deviation, so
   it only reorders near-ties (~/Documents beats a dataset folder when equally relevant).
   (Ported from DigUp's Searcher, where it beat reciprocal rank fusion: R@1 0.88 vs 0.73 on a
   synthetic eval and 0.86 vs 0.82 on a real one. RRF erased similarity gaps: the best vector
   match ranked 13th behind files that merely also contained the words.)
5. collapse near-duplicates (copies of a photo, versions of a document): files whose matching
   chunks, or (same kind) whose first chunks, are near-identical (cos >=
   cfg.duplicate_similarity[modality]) become one result, represented by the copy with the
   highest priority, then the best-ranked; `duplicates` says how many were folded into it.
6. drop results whose file no longer exists (lstat; deleted since the last scan)

The snippet/locator is the best meaning chunk if it passes the gate, else the best text match.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import filetypes as ft
from .config import MAX_KNN_K, Config
from .store import Store
from .vectors import chunks_version

if TYPE_CHECKING:
    from .embedder import EmbedderLike

FTS_LIMIT = 50
MAX_QUERY_TERMS = 16
TEXT_PREFIX_MIN = 3  # chunk text: only the last term gets a wildcard, and only from 3 letters
# A chunk-text term matching more chunks than this (or than TEXT_MAX_DF_RATIO of all chunks,
# whichever is larger) is dropped from the text query: ranking every chunk that contains "the"
# costs ~60 ms at 100k chunks and says nothing the meaning score does not.
TEXT_MAX_DF = 5000
TEXT_MAX_DF_RATIO = 0.05
_TERM = re.compile(r"[^\W_]+")  # letters/digits; drops every FTS5 operator, quote and hyphen

# -- scoring (z units: standard deviations of this query's cosines over the whole index) ------
MODALITY_MIN = 20  # fewer vectors of a modality: centre them on the overall mean instead
STD_FLOOR = 0.02  # a tiny or uniform index must not blow small cosine gaps up into huge z
MAX_MEANING_ROWS = 4096  # most chunks above the gate looked at (best z first)
# Coverage of every query word adds 2 (DigUp's weight): enough to lift a file that has the
# words and a middling meaning score over one with a slightly better meaning score and none of
# the words, while one common word (coverage 0.5 -> +0.71) can't beat a clear meaning match.
W_WORDS = 2.0
# The file name contains every query word. Typing a file's name must put it first: with the
# coverage it also gets (+2) it beats any file whose z is less than ~5 above its own, and z > 5
# over the name file's is rare (the best match of a query usually sits at z 4-7).
NAME_BONUS = 3.0
TYPE_BONUS = 1.0  # the file's kind is what a type word asks for ("screenshot of ...", "pdf")
SCREENSHOT_BONUS = 0.5  # ... and "screenshot" asks for images that are screenshots in particular
SCALE_SAMPLE = 2048  # sqlite-vec fallback: vectors sampled to estimate the scale
SCALE_SAMPLE_MAX_AGE = 300.0  # seconds a sample is reused after the index changed

STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "my", "me", "that",
    "this", "these", "those", "from", "about", "where", "what", "which", "who", "when", "is", "was",
    "are", "were", "be", "it", "its", "some", "any", "find", "show", "file", "files",
    # Finnish
    "ja", "tai", "ei", "se", "ne", "että", "kun", "joka", "jossa", "mikä", "missä", "minun",
    "mun", "tämä", "tuo", "nämä", "jotka", "kanssa", "etsi", "näytä", "tiedosto", "tiedostot",
})

_DOC = (ft.OFFICE, ft.PDF)
TYPE_WORDS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(("screenshot", "screenshots", "photo", "photos", "picture", "pictures", "image",
                     "images", "pic", "pics"), (ft.IMAGE,)),
    **dict.fromkeys(("pdf", "pdfs", "paper", "papers"), (ft.PDF,)),
    **dict.fromkeys(("document", "documents", "doc", "docs"), _DOC),
    **dict.fromkeys(("note", "notes"), (ft.TEXT, ft.OFFICE)),
    **dict.fromkeys(("video", "videos", "clip", "clips", "movie", "movies"), (ft.VIDEO,)),
    **dict.fromkeys(("audio", "song", "songs", "music", "podcast", "podcasts", "recording",
                     "recordings", "voice", "sound"), (ft.AUDIO,)),
    # Finnish (common inflected forms; no stemming here)
    **dict.fromkeys(("kuvakaappaus", "kuvakaappaukset", "kuvakaappauksen", "kuvakaappauksia",
                     "kuvakaappauksessa", "kuva", "kuvat", "kuvan", "kuvia", "kuvassa", "valokuva",
                     "valokuvat", "valokuvan", "valokuvia", "valokuvassa"), (ft.IMAGE,)),
    **dict.fromkeys(("asiakirja", "asiakirjat", "asiakirjan", "asiakirjoja", "dokumentti",
                     "dokumentit", "dokumentin", "dokumentteja"), _DOC),
    **dict.fromkeys(("muistiinpano", "muistiinpanot", "muistiinpanon", "muistiinpanoja"),
                    (ft.TEXT, ft.OFFICE)),
    **dict.fromkeys(("videot", "videon", "videoita", "videolla", "videossa"), (ft.VIDEO,)),
    **dict.fromkeys(("ääni", "äänet", "äänen", "äänite", "äänitteet", "äänitteen", "kappale",
                     "kappaleet", "kappaleen"), (ft.AUDIO,)),
}
SCREENSHOT_WORDS = frozenset({"screenshot", "screenshots", "kuvakaappaus", "kuvakaappaukset",
                              "kuvakaappauksen", "kuvakaappauksia", "kuvakaappauksessa"})


@dataclass(slots=True)
class SearchResult:
    path: str
    kind: str
    score: float  # z of the best chunk + word coverage + bonuses (see the module docstring)
    similarity: float | None  # cosine similarity of the best chunk, None for lexical-only hits
    modality: str | None
    locator: str
    snippet: str
    position: float | None
    mtime: float
    matched: list[str] = field(default_factory=list)  # signals that found it: vector/name/text/folder
    duplicates: int = 0  # near-identical files collapsed into this result
    duplicate_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchTimings:
    embed_ms: float = 0.0
    search_ms: float = 0.0  # everything after the query embedding


@dataclass
class QueryWords:
    terms: list[str]  # content words matched lexically
    kinds: set[str]  # kinds asked for by type words
    screenshot: bool  # a screenshot word was used


def query_terms(text: str) -> list[str]:
    return _TERM.findall(text)[:MAX_QUERY_TERMS]


def query_words(text: str) -> QueryWords:
    """Content words, and the kinds the type words ask for. Content words of one character are
    dropped (except digits); a query of nothing but type words/stopwords keeps them as words."""
    words = query_terms(text)
    folded = [w.casefold() for w in words]
    kinds = {k for w in folded for k in TYPE_WORDS.get(w, ())}
    screenshot = any(w in SCREENSHOT_WORDS for w in folded)
    seen: set[str] = set()

    def pick(pred: Callable[[str], bool]) -> list[str]:
        out = []
        for w, f in zip(words, folded, strict=True):
            if pred(f) and f not in seen and (len(f) > 1 or f.isdigit()):
                seen.add(f)
                out.append(w)
        return out

    terms = pick(lambda f: f not in STOPWORDS and f not in TYPE_WORDS)
    if not terms:
        terms = pick(lambda f: f not in STOPWORDS) or pick(lambda f: True)
    return QueryWords(terms, kinds, screenshot)


def _patterns(terms: list[str]) -> list[str]:
    """Quoted FTS5 terms; the last one (being typed) also as a prefix from TEXT_PREFIX_MIN chars."""
    out = [f'"{t}"' for t in terms]
    if terms and len(terms[-1]) >= TEXT_PREFIX_MIN:
        out[-1] += "*"
    return out


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
    return _patterns(query_terms(text))


def fts_text_query(text: str) -> str | None:
    return " ".join(fts_text_terms(text)) or None


@dataclass
class Scale:
    """This query's meaning scale: z = (cos - mean[modality] - center) / std."""

    means: dict[str, float]  # per modality
    overall: float  # for modalities with fewer than MODALITY_MIN vectors (or unseen)
    center: float
    std: float

    def z(self, cos: float, modality: str | None) -> float:
        return (cos - self.means.get(modality or "", self.overall) - self.center) / self.std


def meaning_scale(sims: np.ndarray, modality: np.ndarray, names: Sequence[str]) -> Scale:
    """The scale from cosines `sims` of vectors with modality codes `modality` (names[code])."""
    if not len(sims):
        return Scale({}, 0.0, 0.0, 1.0)
    codes = modality.astype(np.int64, copy=False)
    counts = np.bincount(codes, minlength=len(names))
    sums = np.bincount(codes, weights=sims, minlength=len(names))
    overall = float(sums.sum() / len(sims))
    means = np.where(counts >= MODALITY_MIN, sums / np.maximum(counts, 1), overall)
    centered = sims - means[codes].astype(np.float32)
    center = float(centered.mean(dtype=np.float64))
    std = max(float(centered.std(dtype=np.float64)), STD_FLOOR)
    return Scale({names[i]: float(means[i]) for i in range(len(names)) if counts[i] >= MODALITY_MIN},
                 overall, center, std)


def _exists(path: str) -> bool:
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True  # unreadable, unmounted for a moment...: keep it
    return True


def _is_screenshot(path: str) -> bool:
    name = os.path.basename(path).casefold()
    folder = os.path.basename(os.path.dirname(path)).casefold()
    return name.startswith(("screenshot", "kuvakaappaus")) or folder in ("screenshots", "kuvakaappaukset")


@dataclass
class _Meaning:
    best: dict[int, tuple[float, float, int]]  # file_id -> (z, cosine, chunk_id) of its best chunk
    scale: Scale


class Searcher:
    def __init__(self, store: Store, cfg: Config, embedder: EmbedderLike):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.last_timings = SearchTimings()
        # sqlite-vec fallback: (chunks version, time, unit vectors, modality names) of the sample
        self._sample: tuple[int, float, np.ndarray, list[str]] | None = None

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
        gate = cfg.z_gate
        kind_in = f"IN ({','.join('?' * len(kinds))})" if kinds else ""
        kind_args = kinds or []
        words = query_words(query)
        terms = words.terms
        patterns = _patterns(terms)
        text_patterns = self._selective(db, self._whole_if_indexed(db, patterns))

        # 2. Lexical candidates: chunk text (OR, bm25 order), names and folders (OR), full name match.
        text_best: dict[int, int] = {}  # file_id -> chunk_id of its best text match
        if text_patterns:
            match = " OR ".join(text_patterns)
            if kinds:
                sql = (f"SELECT c.id, c.file_id FROM fts_chunks JOIN chunks c ON c.id = fts_chunks.rowid "
                       f"JOIN files f ON f.id = c.file_id WHERE fts_chunks MATCH ? AND f.kind {kind_in} "
                       f"ORDER BY fts_chunks.rank LIMIT {FTS_LIMIT * 4}")
                args = [match, *kind_args]
            else:
                sql = (f"SELECT c.id, c.file_id FROM (SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? "
                       f"ORDER BY rank LIMIT {FTS_LIMIT * 4}) m JOIN chunks c ON c.id = m.rowid ORDER BY m.rank")
                args = [match]
            for chunk_id, file_id in self._fts(db, sql, args):
                text_best.setdefault(file_id, chunk_id)
        kind_join = f" JOIN files f ON f.id = fts_files.rowid AND f.kind {kind_in}" if kinds else ""

        def files_matching(match: str, limit: int) -> list[int]:
            return [r[0] for r in self._fts(
                db, f"SELECT fts_files.rowid FROM fts_files{kind_join} WHERE fts_files MATCH ? "
                    f"ORDER BY rank LIMIT {limit}", [*kind_args, match])]

        named: set[int] = set()  # the name contains every query word
        lexical: set[int] = set(text_best)
        if patterns:
            named = set(files_matching(f"{{name}} : ({' '.join(patterns)})", FTS_LIMIT))
            lexical |= named
            lexical.update(files_matching(f"{{name}} : ({' OR '.join(patterns)})", FTS_LIMIT * 2))
            lexical.update(files_matching(f"{{dirs}} : ({' OR '.join(patterns)})", FTS_LIMIT * 2))

        # 3. Meaning.
        meaning = self._meaning(db, qvec, kinds, gate, lexical)
        best = meaning.best
        pool = lexical | set(best)
        if not pool:
            return []

        # Word coverage over the pool, one query per word and place.
        text_hits: dict[int, int] = {}  # file_id -> words found in its chunk text
        name_hits: dict[int, int] = {}
        strong_hits: dict[int, int] = {}  # ... in its chunk text or name
        dir_hit: set[int] = set()  # some word is in its folder names
        any_hits: dict[int, int] = {}  # ... anywhere (text, name or folders)
        text_of = {p.rstrip("*"): p for p in text_patterns}
        # Words too common to search in chunk text (large indexes only) don't count at all.
        cov_patterns = [p for p in patterns if p.rstrip("*") in text_of] or patterns
        if cov_patterns:
            ids = ",".join(map(str, pool))  # integers from SQLite: safe to inline
            for p in cov_patterns:
                # (FTS5 with `rowid IN (...)` probes row by row: ~5 ms for 700 ids. Matching the
                # whole small files table and intersecting is far cheaper.)
                in_text = {r[0] for r in self._fts(
                    db, f"SELECT DISTINCT c.file_id FROM (SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?) m "
                        f"JOIN chunks c ON c.id = m.rowid WHERE c.file_id IN ({ids})", [text_of[p.rstrip("*")]])} \
                    if p.rstrip("*") in text_of else set()
                in_name = pool.intersection(r[0] for r in self._fts(
                    db, "SELECT rowid FROM fts_files WHERE fts_files MATCH ?", [f"{{name}} : {p}"]))
                in_dirs = pool.intersection(r[0] for r in self._fts(
                    db, "SELECT rowid FROM fts_files WHERE fts_files MATCH ?", [f"{{dirs}} : {p}"]))
                for f in in_text:
                    text_hits[f] = text_hits.get(f, 0) + 1
                for f in in_name:
                    name_hits[f] = name_hits.get(f, 0) + 1
                for f in in_text | in_name:
                    strong_hits[f] = strong_hits.get(f, 0) + 1
                dir_hit |= in_dirs
                for f in in_text | in_name | in_dirs:
                    any_hits[f] = any_hits.get(f, 0) + 1
        n_terms = max(len(cov_patterns), 1)
        files = self._files(db, pool)

        # 4. Candidates and score.
        scores: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        std = meaning.scale.std
        for f in pool:
            row = files.get(f)
            if row is None:
                continue
            z = best[f][0] if f in best else None
            coverage = any_hits.get(f, 0) / n_terms
            # Folder words count for the score, but alone they don't make a file a candidate.
            strong = strong_hits.get(f, 0) / n_terms
            if not ((z is not None and z >= gate) or strong >= 0.5 or f in named):
                continue
            score = max(z or 0.0, 0.0) + W_WORDS * coverage ** 1.5
            if f in named:
                score += NAME_BONUS
            if row["kind"] in words.kinds:
                score += TYPE_BONUS
                if words.screenshot and row["kind"] == ft.IMAGE and _is_screenshot(row["path"]):
                    score += SCREENSHOT_BONUS
            score += cfg.location_prior * (row["priority"] / 100.0) / std
            scores[f] = score
            labels = []
            if z is not None and z >= gate:
                labels.append("vector")
            if f in named or name_hits.get(f):
                labels.append("name")
            if text_hits.get(f) or f in text_best:
                labels.append("text")
            if f in dir_hit:
                labels.append("folder")
            matched[f] = labels

        candidates = sorted(scores, key=lambda f: (-scores[f], -(best[f][1] if f in best else -1.0),
                                                   -files[f]["priority"], f))
        # Enough for `limit` results after duplicates and vanished files are taken out.
        candidates = candidates[:max(limit * 5, 50)]

        # The chunk to show: the best meaning chunk if it stands out, else the best text match.
        hit: dict[int, int] = {}
        for f in candidates:  # (both orders of preference fall back to whatever exists)
            if f in best and (best[f][0] >= gate or f not in text_best):
                hit[f] = best[f][2]
            elif f in text_best:
                hit[f] = text_best[f]

        # 5-6. Collapse near-duplicates, drop vanished files.
        groups = self._group_duplicates(db, candidates, files, hit)
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
        chunk_ids = [hit.get(f) for f, _ in picks]
        chunks = self._chunks(db, [c for c in chunk_ids if c is not None])
        for (f, others), chunk_id in zip(picks, chunk_ids, strict=True):
            row = files[f]
            c = chunks.get(chunk_id) if chunk_id is not None else None
            labels = list(dict.fromkeys(m for g in (f, *others) for m in matched[g]))
            out.append(SearchResult(
                path=row["path"], kind=row["kind"], score=round(max(scores[g] for g in (f, *others)), 6),
                similarity=round(best[f][1], 4) if f in best else None,
                modality=c["modality"] if c else None, locator=c["locator"] if c else "",
                snippet=c["snippet"] if c else "", position=c["position"] if c else None,
                mtime=row["mtime"], matched=labels if others else matched[f],
                duplicates=len(others), duplicate_paths=[files[g]["path"] for g in others[:20]],
            ))
        return out

    # -- steps ----------------------------------------------------------------
    def _meaning(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None, gate: float,
                 lexical: set[int]) -> _Meaning:
        """Each candidate file's best chunk (z, cosine, chunk id): files with a chunk at
        z >= gate (at most MAX_MEANING_ROWS chunks, best first) and the `lexical` files."""
        cache = self.store.vectors
        if cache is not None:
            with cache.lock:
                cache.ensure_fresh(db)
                if not cache.too_big:
                    return self._meaning_cached(db, qvec, kinds, gate, lexical)
        return self._meaning_sqlite(db, qvec, kinds, gate, lexical)

    def _meaning_cached(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None,
                        gate: float, lexical: set[int]) -> _Meaning:
        cache = self.store.vectors
        assert cache is not None
        sc = cache.scores(qvec)
        sims, mods = sc.sims, sc.modality
        if sc.valid is not None:
            scale = meaning_scale(sims[sc.valid], mods[sc.valid], sc.names)
        else:
            scale = meaning_scale(sims, mods, sc.names)
        best: dict[int, tuple[float, float, int]] = {}
        if not len(sims):
            return _Meaning(best, scale)
        # z for every row (float32): (cos - mean[modality] - center) / std
        offset = np.array([scale.means.get(n, scale.overall) + scale.center for n in sc.names] or [0.0],
                          dtype=np.float32)
        z = (sims - offset[mods]) * np.float32(1.0 / scale.std)
        ok = z >= gate
        if sc.valid is not None:
            ok &= sc.valid
        if kinds is not None:
            ok &= np.isin(sc.kind, cache.codes(kinds))
        rows = np.flatnonzero(ok)
        if len(rows) > MAX_MEANING_ROWS:
            rows = rows[np.argpartition(-z[rows], MAX_MEANING_ROWS - 1)[:MAX_MEANING_ROWS]]
        rows = rows[np.argsort(-z[rows], kind="stable")]

        def take(rows: np.ndarray) -> None:
            for r, f, c, zz, cos in zip(rows.tolist(), sc.file_ids[rows].tolist(), sc.chunk_ids[rows].tolist(),
                                       z[rows].tolist(), sims[rows].tolist(), strict=True):
                if f not in best or zz > best[f][0]:
                    best[f] = (zz, cos, c)

        take(rows)
        extra = [f for f in lexical if f not in best]
        if extra:
            take(cache.rows(self._file_chunks(db, extra)))
        return _Meaning(best, scale)

    def _meaning_sqlite(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None,
                        gate: float, lexical: set[int]) -> _Meaning:
        """Fallback without the vector cache (index above vector_cache_max_chunks): only the KNN
        top-k (cfg.knn_k) is scored, on a scale estimated from a fixed sample of the index; a
        lexical candidate outside the top-k counts as z = 0 (fetching its vectors from vec0 costs
        ~0.15 ms each). Selecting the embedding column in the KNN adds ~50 ms at 100k chunks."""
        scale = self._sample_scale(db, qvec)
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        k = max(1, min(self.cfg.knn_k, MAX_KNN_K))
        sql = "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        args: list[Any] = [np.ascontiguousarray(q), k]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        cos = {r[0]: 1.0 - r[1] for r in db.execute(sql, args)}
        best: dict[int, tuple[float, float, int]] = {}
        if cos:
            ids = list(cos)
            for cid, fid, mod in db.execute(f"SELECT id, file_id, modality FROM chunks WHERE id IN "
                                            f"({','.join('?' * len(ids))})", ids):
                zz = scale.z(cos[cid], mod)
                if (zz >= gate or fid in lexical) and (fid not in best or zz > best[fid][0]):
                    best[fid] = (zz, cos[cid], cid)
        return _Meaning(best, scale)

    def _sample_scale(self, db: sqlite3.Connection, qvec: np.ndarray) -> Scale:
        """The scale from SCALE_SAMPLE vectors spread evenly over the chunk ids (reloaded when the
        index changed and the sample is older than SCALE_SAMPLE_MAX_AGE, or at once when the
        index is small enough for the sample to be all of it)."""
        version = chunks_version(db)
        now = time.monotonic()
        s = self._sample
        if s is None or (s[0] != version and (now - s[1] > SCALE_SAMPLE_MAX_AGE or len(s[2]) < SCALE_SAMPLE)):
            top = db.execute("SELECT max(id) FROM chunks").fetchone()[0] or 0
            step = max(1, top // SCALE_SAMPLE)
            rows = db.execute("SELECT id, modality FROM chunks WHERE id % ? = 0 LIMIT ?",
                              (step, SCALE_SAMPLE)).fetchall()
            mod = {r[0]: r[1] for r in rows}
            vecs = self._vectors(db, list(mod)) if mod else {}
            ids = [c for c in mod if c in vecs]
            mat = np.stack([vecs[c] for c in ids]).astype(np.float32) if ids else np.zeros((0, len(qvec)), np.float32)
            if len(mat):
                mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
            s = self._sample = (version, now, mat, [mod[c] for c in ids])
        _, _, mat, mods = s
        q = np.asarray(qvec, dtype=np.float32)
        names = sorted(set(mods))
        code = {n: i for i, n in enumerate(names)}
        return meaning_scale(mat @ (q / (np.linalg.norm(q) or 1.0)), np.array([code[m] for m in mods], np.int64),
                             names)

    def _group_duplicates(self, db: sqlite3.Connection, candidates: list[int], files: dict[int, sqlite3.Row],
                          hit: dict[int, int]) -> list[list[int]]:
        """Greedy grouping in rank order: a file joins the first earlier group whose head is a
        near-duplicate (cos >= duplicate_similarity[modality]) by either signature:
        - the chunk that matched the query (`hit`: the chunk shown for it): the same
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
        modality = {cid: mod for cid, mod in first.values()}
        extra = [c for c in hit.values() if c not in modality]
        if extra:
            modality.update({r[0]: r[1] for r in db.execute(
                f"SELECT id, modality FROM chunks WHERE id IN ({','.join('?' * len(extra))})", extra)})
        need = list({c for c in [*hit.values(), *(c for c, _ in first.values())]})
        vectors = self._vectors(db, need) if need else {}

        def unit(cid: int | None) -> np.ndarray | None:
            v = vectors.get(cid) if cid is not None else None
            return None if v is None else v / (np.linalg.norm(v) or 1.0)

        groups: list[list[int]] = []
        # Group heads per modality, as matrices so each candidate is one matrix-vector product.
        # A missing vector is a zero row (it never reaches a threshold > 0).
        dim = next((len(v) for v in vectors.values()), 0)
        kind_code = {k: i for i, k in enumerate(sorted({files[f]["kind"] for f in candidates}))}
        heads: dict[str, tuple[list[int], np.ndarray, np.ndarray, np.ndarray]] = {}  # group, kind, hit, first
        for f in candidates:
            fc = first.get(f)
            hc = hit.get(f, fc[0] if fc else None)
            mod = modality.get(hc) if hc is not None else None
            thr = thresholds.get(mod) if mod else None
            hv, fv = unit(hc), unit(fc[0] if fc else None)
            kind = kind_code[files[f]["kind"]]
            if thr is None or mod is None or (hv is None and fv is None):
                groups.append([f])
                continue
            h = heads.get(mod)
            if h is not None:
                gs, kinds, hm, fm = h
                n = len(gs)
                same = np.zeros(n, dtype=bool)
                if hv is not None:
                    same |= hm[:n] @ hv >= thr
                if fv is not None:
                    same |= (fm[:n] @ fv >= thr) & (kinds[:n] == kind)
                hits = np.flatnonzero(same)
                if len(hits):
                    groups[gs[int(hits[0])]].append(f)
                    continue
            else:
                h = heads[mod] = ([], np.zeros(len(candidates), np.int16), np.zeros((len(candidates), dim), np.float32),
                                  np.zeros((len(candidates), dim), np.float32))
            gs, kinds, hm, fm = h
            i = len(gs)
            if hv is not None:
                hm[i] = hv
            if fv is not None:
                fm[i] = fv
            kinds[i] = kind
            gs.append(len(groups))
            groups.append([f])
        return groups

    # -- helpers --------------------------------------------------------------
    def _whole_if_indexed(self, db: sqlite3.Connection, patterns: list[str]) -> list[str]:
        """Chunk text: the last word keeps its prefix wildcard only while it is in no chunk as
        typed (DigUp's rule): "transfo" finds "transformer", but a finished "transformer" does not
        also pull in every "pip install transformers" (file names keep the wildcard: they are
        short, and "dog" should find dogs.jpg)."""
        if not patterns or not patterns[-1].endswith("*"):
            return patterns
        whole = patterns[-1][:-1]
        if self._fts(db, "SELECT 1 FROM fts_chunks WHERE fts_chunks MATCH ? LIMIT 1", [whole]):
            return [*patterns[:-1], whole]
        return patterns

    def _selective(self, db: sqlite3.Connection, terms: list[str]) -> list[str]:
        """The chunk-text terms without those that match too many chunks (see TEXT_MAX_DF).
        Each probe stops after the cap, so it costs well under 2 ms even for "the"."""
        if not terms:
            return []
        top = db.execute("SELECT max(id) FROM chunks").fetchone()[0] or 0
        cap = max(TEXT_MAX_DF, int(top * TEXT_MAX_DF_RATIO))
        if top <= cap:
            return list(terms)  # small index: nothing can be too common
        keep = []
        for t in terms:
            rows = self._fts(db, "SELECT count(*) FROM (SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? "
                                 "LIMIT ?)", [t, cap + 1])
            if rows and rows[0][0] <= cap:
                keep.append(t)
        return keep

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
    def _file_chunks(db: sqlite3.Connection, file_ids: list[int]) -> list[int]:
        return [r[0] for r in db.execute(
            f"SELECT id FROM chunks WHERE file_id IN ({','.join(map(str, file_ids))}) ORDER BY file_id, seq")]

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
