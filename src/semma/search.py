"""Hybrid search: how far each file's best chunk stands out in meaning for this query (a z-score
over every stored vector), plus how many of the query's words the file contains.

1. query words: `query_terms` minus stopwords and type words used as kind hints. Unambiguous type
   words ("screenshot", "photo", "pdf", "video", "kuvakaappaus", "valokuva", ...) are always
   hints; the others ("image", "paper", "notes", "music", "sound", ...) only where the phrasing
   says so ("image of a dog", "notes about X", "... in a recording"); elsewhere they are ordinary
   words and ask for no kind ("image segmentation", "music theory"). Hints give a bonus (step 4),
   they are not filters. A query made only of type words or stopwords keeps them as words. Words
   matching too many chunks are dropped from the chunk-text queries and from word coverage
   (`_selective`); in chunk text only the last word gets a prefix wildcard from TEXT_PREFIX_MIN
   characters while it is in no chunk as typed, while file names and folders prefix every term
   from FILE_PREFIX_MIN characters (FILE_LAST_PREFIX_MIN for the last term).
2. lexical candidates (all filtered by kind): an OR query over chunk text (bm25 order, its best
   chunk per file is the file's text match), over file names and over folder names, and a name
   query requiring every word.
3. meaning: the query's cosine with EVERY stored vector (one matrix-vector product over the
   in-memory VectorCache), minus the mean cosine of that vector's modality for this query (text,
   image, audio and video vectors sit at different levels: image cosines run ~0.05 below text
   ones; a modality with < MODALITY_MIN vectors uses the overall mean), divided by the standard
   deviation of those centred values over all vectors (floored at STD_FLOOR) after subtracting
   their mean. That is a z-score: how many standard deviations a chunk stands out from the whole
   index for this query. The scale always comes from all vectors; `kinds` only restricts which
   files are candidates. A file's meaning is its best chunk's z (its cosine is reported as
   `similarity`). Without the cache (sqlite-vec fallback, rare) the scale comes from a sample of
   about SCALE_SAMPLE vectors (stratified by modality, weighted back to the index's proportions),
   the KNN top-k (cfg.knn_k) is scored, and so is one chunk of each of the first
   SQLITE_LEXICAL_VECTORS lexical candidates outside it.
4. candidates: a file with at least half the words in its chunk text or name (folder words alone
   are too weak: one folder word matches every file inside), or every query word in its name.
   A file without them (a vector-only match) needs z >= cfg.z_gate, and z within Z_MARGIN
   (SINGLE_WORD_MARGIN for one word) of the query's best match, and that best match must reach
   cfg.z_floor; the kind bonus counts in both comparisons. z is relative, so every query has a
   tail of ~2-5% of the index above the gate; the margin cuts that tail and the floor makes a
   query whose best match stands out no further than noise ("asdfgh qwerty") find nothing.
   score = w_m * max(z, 0) + w_w * coverage**1.5 (coverage counts chunk text and name matches as
   1, folder-only matches as 0.5; w_m = 1 and w_w = W_WORDS, but lookups (is_lookup: every term
   a code like "b0bbce05d5") weigh meaning 0.5 and words 2 * W_WORDS, and one non-lookup word
   found outside the file's name counts w_w * SINGLE_TEXT_FACTOR) + NAME_BONUS for matching every
   query word in the file name + TYPE_BONUS for the kind a type word asks for (+
   SCREENSHOT_BONUS for screenshots when asked for) + the location prior: cfg.location_prior is
   a z-unit bonus at priority 100, so it only reorders near-ties (~/Documents beats a dataset
   folder when equally relevant).
   (Ported from DigUp's Searcher, where it beat reciprocal rank fusion: R@1 0.88 vs 0.73 on a
   synthetic eval and 0.86 vs 0.82 on a real one. RRF erased similarity gaps: the best vector
   match ranked 13th behind files that merely also contained the words.)
5. collapse near-duplicates (copies of a photo, versions of a document): files whose matching
   chunks, or (same kind) whose first chunks, are near-identical (cos >=
   cfg.duplicate_similarity[modality]) become one result, represented by the copy with the
   highest priority, then the best-ranked; `duplicates` says how many were folded into it.
6. drop results whose file no longer exists (lstat; deleted since the last scan)

The snippet/locator is the best meaning chunk if it passes the gate, else the best text match;
for an image (whose picture chunk has no snippet) the text match whenever there is one: the
recognised text of a screenshot (semma.ocr) that contains the query's words.
"""

from __future__ import annotations

import itertools
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
from .ocr import is_screenshot
from .store import Store
from .vectors import chunks_version

if TYPE_CHECKING:
    from .embedder import EmbedderLike

FTS_LIMIT = 50
MAX_QUERY_TERMS = 16
TEXT_PREFIX_MIN = 3  # chunk text: only the last term gets a wildcard, and only from 3 letters
FILE_PREFIX_MIN = 3  # fts_files (names/dirs): prefix='2 3' indexes make per-term prefixes cheap
FILE_LAST_PREFIX_MIN = 2
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
# Lookups (is_lookup: "b0bbce05d5", "0xff32:0x0074"): meaning counts half, words double (DigUp).
LOOKUP_MEANING_WEIGHT = 0.5
LOOKUP_WORDS_FACTOR = 2.0
# One (non-lookup) query word found only in a file's text adds W_WORDS * 0.5: "dog" mentioned in a
# thesis (z 3.6 + 1) no longer beats the dog photo (z 5.2), while a file named after the word
# keeps the full weight.
SINGLE_TEXT_FACTOR = 0.5
# A file without the query's words (a vector-only match) must come within this many z of the
# query's best meaning match. Its tail is otherwise ~2-5% of the index for every query (z is
# relative): on the local eval the tails behind a z 9-11 top sat at z 2-4.5. One word gets a
# tighter margin (like the old 0.04 cosine margin for short queries): its meaning is vaguer and
# its tail longer ("firefox": the Firefox screenshot at z 5.2, then console.txt at 3.5).
Z_MARGIN = 3.0
SINGLE_WORD_MARGIN = 1.5
TYPE_BONUS = 1.0  # the file's kind is what a type word asks for ("screenshot of ...", "pdf")
SCREENSHOT_BONUS = 0.5  # ... and "screenshot" asks for images that are screenshots in particular
SCALE_SAMPLE = 2048  # sqlite-vec fallback: vectors sampled to estimate the scale
MODALITY_SAMPLE_MIN = 64  # ... at least this many of each modality with MODALITY_MIN chunks
SQLITE_LEXICAL_VECTORS = 50  # ... lexical files outside the KNN top-k scored by one chunk each
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
# Type words that hardly mean anything else: always a kind hint, never searched as words. The
# rest ("image segmentation", "music theory", "sound card", "paper towel", "voice notes") are also
# ordinary words: they stay content words and ask for their kind only where the phrasing says so
# (HINT_AFTER/HINT_BEFORE). Finnish locatives ("kuvassa", "videolla": in a picture/video) are hints.
UNAMBIGUOUS_TYPE_WORDS = SCREENSHOT_WORDS | frozenset({
    "photo", "photos", "pdf", "pdfs", "video", "videos",
    "valokuva", "valokuvat", "valokuvan", "valokuvia", "valokuvassa", "kuvassa",
    "videot", "videon", "videoita", "videolla", "videossa",
})
HINT_AFTER = frozenset({"of", "about"})  # "image of a dog", "paper about attention"
HINT_BEFORE = frozenset({"in", "on", "from"})  # "... in a recording", "... from the paper"
ARTICLES = frozenset({"a", "an", "the", "my", "some", "this", "that"})


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
    z: float | None = None  # meaning z of the best chunk (None: not scored, lexical-only)

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


def _is_hint(folded: list[str], i: int) -> bool:
    """Whether type word folded[i] asks for a kind: always for an unambiguous one, else only when
    followed by "of"/"about" or preceded (articles skipped) by "in"/"on"/"from"."""
    w = folded[i]
    if w in UNAMBIGUOUS_TYPE_WORDS:
        return True
    if i + 1 < len(folded) and folded[i + 1] in HINT_AFTER:
        return True
    j = i - 1
    while j >= 0 and folded[j] in ARTICLES:
        j -= 1
    return j >= 0 and folded[j] in HINT_BEFORE


def query_words(text: str) -> QueryWords:
    """Content words, and the kinds the type words ask for. A type word used as a hint is not a
    content word; an ambiguous one used otherwise is only a content word (no kind bonus). Content
    words of one character are dropped (except digits); a query of nothing but type words or
    stopwords keeps them as words."""
    words = query_terms(text)
    folded = [w.casefold() for w in words]
    hints = {i for i, f in enumerate(folded) if f in TYPE_WORDS and _is_hint(folded, i)}
    kinds = {k for i in hints for k in TYPE_WORDS[folded[i]]}
    screenshot = any(folded[i] in SCREENSHOT_WORDS for i in hints)
    seen: set[str] = set()

    def pick(pred: Callable[[int, str], bool]) -> list[str]:
        out = []
        for i, (w, f) in enumerate(zip(words, folded, strict=True)):
            if pred(i, f) and f not in seen and (len(f) > 1 or f.isdigit()):
                seen.add(f)
                out.append(w)
        return out

    terms = pick(lambda i, f: f not in STOPWORDS and i not in hints)
    if not terms:
        terms = pick(lambda i, f: f not in STOPWORDS) or pick(lambda i, f: True)
    return QueryWords(terms, kinds, screenshot)


def is_lookup(terms: Sequence[str]) -> bool:
    """Codes, IDs and long numbers ("b0bbce05d5", "0xff32", "TRX50", "4815162342"): every term has
    a digit and a letter or 4+ characters. They mean little to an embedding and everything to
    word search (DigUp's `isLookup`)."""
    return bool(terms) and all(any(c.isdigit() for c in t) and (any(c.isalpha() for c in t) or len(t) >= 4)
                               for t in terms)


def _patterns(terms: list[str]) -> list[str]:
    """Quoted FTS5 terms; the last one (being typed) also as a prefix from TEXT_PREFIX_MIN chars."""
    out = [f'"{t}"' for t in terms]
    if terms and len(terms[-1]) >= TEXT_PREFIX_MIN:
        out[-1] += "*"
    return out


def _file_patterns(terms: list[str]) -> list[str]:
    """Quoted FTS5 terms for fts_files (name/dirs): prefix every term >= 3 chars (>= 2 for the last)."""
    if not terms:
        return []
    last = len(terms) - 1
    return [f'"{t}"*' if len(t) >= (FILE_LAST_PREFIX_MIN if i == last else FILE_PREFIX_MIN) else f'"{t}"'
            for i, t in enumerate(terms)]


@dataclass
class Scale:
    """This query's meaning scale: z = (cos - mean[modality] - center) / std."""

    means: dict[str, float]  # per modality
    overall: float  # for modalities with fewer than MODALITY_MIN vectors (or unseen)
    center: float
    std: float

    def offset(self, modality: str | None) -> float:
        return self.means.get(modality or "", self.overall) + self.center

    def offsets(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.offset(n) for n in names] or [0.0], dtype=np.float32)

    def z(self, cos: float, modality: str | None) -> float:
        return (cos - self.offset(modality)) / self.std


def meaning_scale(sims: np.ndarray, modality: np.ndarray, names: Sequence[str],
                  weights: np.ndarray | None = None, counts: Sequence[int] | None = None) -> Scale:
    """The scale from cosines `sims` of vectors with modality codes `modality` (names[code]).
    A sample passes `weights` (how many index vectors each sampled one stands for) and the
    index's per-modality `counts` (which decide MODALITY_MIN). Everything comes from per-modality
    sums of w, w*cos and w*cos^2 (no full-length temporaries besides the squares)."""
    if not len(sims):
        return Scale({}, 0.0, 0.0, 1.0)
    codes = modality.astype(np.intp, copy=False)
    k = len(names)
    if weights is None:
        w0 = np.bincount(codes, minlength=k).astype(np.float64)
        s1 = np.bincount(codes, weights=sims, minlength=k)
        s2 = np.bincount(codes, weights=sims * sims, minlength=k)
    else:
        w0 = np.bincount(codes, weights=weights, minlength=k)
        s1 = np.bincount(codes, weights=weights * sims, minlength=k)
        s2 = np.bincount(codes, weights=weights * sims * sims, minlength=k)
    have = (np.asarray(counts if counts is not None else w0, dtype=np.float64) >= MODALITY_MIN) & (w0 > 0)
    total = float(w0.sum())
    overall = float(s1.sum() / total)
    means = np.where(have, s1 / np.maximum(w0, 1e-12), overall)
    # Centred values c = cos - means[modality]: sum c = s1 - w0*m, sum c^2 = s2 - 2*m*s1 + w0*m^2.
    center = float((s1 - w0 * means).sum() / total)
    var = float((s2 - 2 * means * s1 + w0 * means * means).sum() / total) - center * center
    std = max(float(np.sqrt(max(var, 0.0))), STD_FLOOR)
    return Scale({names[i]: float(means[i]) for i in range(k) if have[i]}, overall, center, std)


def _exists(path: str) -> bool:
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True  # unreadable, unmounted for a moment...: keep it
    return True


@dataclass
class _Meaning:
    best: dict[int, tuple[float, float, int]]  # file_id -> (z, cosine, chunk_id) of its best chunk
    scale: Scale


@dataclass
class _Sample:
    """sqlite-vec fallback: a weighted sample of the index's vectors for the meaning scale."""

    version: int  # chunks version it was taken at
    time: float  # time.monotonic() when taken
    mat: np.ndarray  # unit vectors (rows)
    codes: np.ndarray  # modality code per row (names[code])
    weights: np.ndarray  # chunks of its modality in the index each row stands for
    names: list[str]
    counts: list[int]  # chunks per modality in the index
    full: bool  # the sample is the whole index (so any growth reloads it at once)


def _vector_min(z_max: float, gate: float, margin: float, bonus: float, floor: float) -> float:
    """The lowest z at which a chunk without the query's words can still pass (see
    `Searcher._search`), given the best gated chunk's z: within `margin` plus the kind `bonus`
    a file may get of it, and only if that best chunk plus the bonus reaches the floor."""
    return max(gate, z_max - margin - bonus) if z_max + bonus >= floor else float("inf")


@dataclass
class _WordCoverage:
    named: set[int]             # file_id -> every query word matched in the file name
    coverage: dict[int, float]  # weighted coverage (text/name = 1.0, folder-only = 0.5)
    strong: dict[int, float]    # fraction of coverage terms matched in text or name
    text_hit: set[int]
    name_hit: set[int]
    dir_hit: set[int]


class Searcher:
    def __init__(self, store: Store, cfg: Config, embedder: EmbedderLike):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.last_timings = SearchTimings()
        self._sample: _Sample | None = None  # sqlite-vec fallback

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
        # Check/reload the vector cache first so `chunks_version` pins the WAL read snapshot
        # before any FTS query runs (prevents reloading from a stale snapshot on concurrent writes).
        cache = self.store.vectors
        if cache is not None:
            with cache.lock:
                cache.ensure_fresh(db)

        cfg = self.cfg
        gate = cfg.z_gate
        kind_in = f"IN ({','.join('?' * len(kinds))})" if kinds else ""
        kind_args = kinds or []
        words = query_words(query)
        terms = words.terms
        file_patterns = _file_patterns(terms)
        text_patterns = self._selective(db, self._whole_if_indexed(db, _patterns(terms)))

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

        def files_matching(match: str, lim: int) -> list[int]:
            return [r[0] for r in self._fts(
                db, f"SELECT fts_files.rowid FROM fts_files{kind_join} WHERE fts_files MATCH ? "
                    f"ORDER BY rank LIMIT {lim}", [*kind_args, match])]

        # file id -> its best text chunk (None: matched by name/folder), in lexical rank order
        lexical: dict[int, int | None] = dict(text_best)
        if file_patterns:
            for f in itertools.chain(
                    files_matching(f"{{name}} : ({' '.join(file_patterns)})", FTS_LIMIT * 4),
                    files_matching(f"{{name}} : ({' OR '.join(file_patterns)})", FTS_LIMIT * 2)
                    if len(file_patterns) > 1 else (),
                    files_matching(f"{{dirs}} : ({' OR '.join(file_patterns)})", FTS_LIMIT * 2)):
                lexical.setdefault(f, None)

        # 3. Meaning.
        lookup = is_lookup(terms)
        single = len(terms) == 1 and not lookup
        margin = SINGLE_WORD_MARGIN if single else Z_MARGIN
        bonus = TYPE_BONUS if words.kinds else 0.0
        meaning = self._meaning(db, qvec, kinds, gate, lexical, margin, bonus, cfg.z_floor)
        best = meaning.best
        pool = set(lexical) | set(best)
        if not pool:
            return []

        cov = self._word_coverage(db, pool, file_patterns, text_patterns)
        files = self._files(db, pool)

        # 4. Candidates and score.
        w_meaning, w_words = (LOOKUP_MEANING_WEIGHT, W_WORDS * LOOKUP_WORDS_FACTOR) if lookup else (1.0, W_WORDS)
        # A file without the words must stand out absolutely (the query's best meaning match
        # reaches cfg.z_floor) and relatively (within `margin` of that best match). Both count
        # the kind bonus: "image of a dog" asks for images, whose z run lower.
        def z_kind(f: int) -> float:
            row = files.get(f)
            return best[f][0] + (bonus if row is not None and row["kind"] in words.kinds else 0.0)

        z_top = max(map(z_kind, best), default=0.0)
        z_min = z_top - margin if z_top >= cfg.z_floor else float("inf")
        scores: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for f in pool:
            row = files.get(f)
            if row is None:
                continue
            z = best[f][0] if f in best else None
            coverage = cov.coverage.get(f, 0.0)
            # Folder words count for the score, but alone they don't make a file a candidate.
            lexical_ok = cov.strong.get(f, 0.0) >= 0.5 or f in cov.named
            if not (lexical_ok or (z is not None and z >= gate and z_kind(f) >= z_min)):
                continue
            # One word found only in a file's text is weak evidence ("dog" mentioned once in a
            # thesis): half the word weight, unless the file is named after it.
            ww = w_words * SINGLE_TEXT_FACTOR if single and f not in cov.name_hit else w_words
            score = w_meaning * max(z or 0.0, 0.0) + ww * coverage ** 1.5
            if f in cov.named:
                score += NAME_BONUS
            if row["kind"] in words.kinds:
                score += TYPE_BONUS
                if words.screenshot and row["kind"] == ft.IMAGE and is_screenshot(row["path"]):
                    score += SCREENSHOT_BONUS
            score += cfg.location_prior * (row["priority"] / 100.0)
            scores[f] = score
            labels = []
            if z is not None and z >= gate:
                labels.append("vector")
            if f in cov.name_hit:
                labels.append("name")
            if f in cov.text_hit or f in text_best:
                labels.append("text")
            if f in cov.dir_hit:
                labels.append("folder")
            matched[f] = labels

        candidates = sorted(scores, key=lambda f: (-scores[f], -(best[f][1] if f in best else -1.0),
                                                   -files[f]["priority"], f))
        # Enough for `limit` results after duplicates and vanished files are taken out.
        candidates = candidates[:max(limit * 5, 50)]

        # The chunk to show: the best meaning chunk if it stands out, else the best text match.
        # An image's own chunk has nothing to show: its OCR text with the words comes first.
        hit: dict[int, int] = {}
        for f in candidates:  # (both orders of preference fall back to whatever exists)
            if f in text_best and files[f]["kind"] == ft.IMAGE:
                hit[f] = text_best[f]
            elif f in best and (best[f][0] >= gate or f not in text_best):
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
                mtime=row["mtime"], matched=labels,
                duplicates=len(others), duplicate_paths=[files[g]["path"] for g in others[:20]],
                z=round(best[f][0], 4) if f in best else None,
            ))
        return out

    # -- steps ----------------------------------------------------------------
    def _word_coverage(self, db: sqlite3.Connection, pool: set[int], file_patterns: list[str],
                       text_patterns: list[str]) -> _WordCoverage:
        """Per-word hits over `pool` for coverage, strong matches, and full-name matches."""
        if not file_patterns:
            return _WordCoverage(set(), {}, {}, set(), set(), set())
        text_of = {p.rstrip("*"): p for p in text_patterns}
        # Words too common to search in chunk text (large indexes only) don't count for coverage,
        # unless every word was dropped; `named` still requires every word in the file name.
        use_all = not text_of
        cov_count = len(file_patterns) if use_all else sum(
            1 for fp in file_patterns if fp.rstrip("*") in text_of)
        n_terms = max(cov_count, 1)

        all_name_hits: dict[int, int] = {}
        strong_hits: dict[int, int] = {}
        dir_only_hits: dict[int, int] = {}
        text_hit: set[int] = set()
        name_hit: set[int] = set()
        dir_hit: set[int] = set()
        ids = ",".join(map(str, pool))  # integers from SQLite: safe to inline

        for fp in file_patterns:
            tp = text_of.get(fp.rstrip("*"))
            # (FTS5 with `rowid IN (...)` probes row by row: ~5 ms for 700 ids. Matching the
            # whole small files table and intersecting is far cheaper.)
            in_name = pool.intersection(r[0] for r in self._fts(
                db, "SELECT rowid FROM fts_files WHERE fts_files MATCH ?", [f"{{name}} : {fp}"]))
            for f in in_name:
                all_name_hits[f] = all_name_hits.get(f, 0) + 1
            name_hit |= in_name

            if not use_all and tp is None:
                continue
            in_text = {r[0] for r in self._fts(
                db, f"SELECT DISTINCT c.file_id FROM (SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?) m "
                    f"JOIN chunks c ON c.id = m.rowid WHERE c.file_id IN ({ids})", [tp])} \
                if tp is not None else set()
            in_dirs = pool.intersection(r[0] for r in self._fts(
                db, "SELECT rowid FROM fts_files WHERE fts_files MATCH ?", [f"{{dirs}} : {fp}"]))
            text_hit |= in_text
            dir_hit |= in_dirs
            in_strong = in_text | in_name
            for f in in_strong:
                strong_hits[f] = strong_hits.get(f, 0) + 1
            for f in in_dirs - in_strong:
                dir_only_hits[f] = dir_only_hits.get(f, 0) + 1

        named = {f for f, count in all_name_hits.items() if count == len(file_patterns)}
        coverage = {
            f: (strong_hits.get(f, 0) + 0.5 * dir_only_hits.get(f, 0)) / n_terms
            for f in strong_hits.keys() | dir_only_hits.keys()
        }
        strong = {f: count / n_terms for f, count in strong_hits.items()}
        return _WordCoverage(named, coverage, strong, text_hit, name_hit, dir_hit)

    def _meaning(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None, gate: float,
                 lexical: dict[int, int | None], margin: float, bonus: float, floor: float) -> _Meaning:
        """Each candidate file's best chunk (z, cosine, chunk id): the `lexical` files (file id ->
        its best text chunk or None, in lexical rank order), and the files with a chunk at
        z >= gate that can still pass the margin and floor (`_vector_min`; at most
        MAX_MEANING_ROWS chunks, best first). Pruning here keeps the per-file work (file rows,
        word coverage) small on big indexes."""
        cache = self.store.vectors
        if cache is not None:
            with cache.lock:
                if not cache.too_big:
                    return self._meaning_cached(db, qvec, kinds, gate, lexical, margin, bonus, floor)
        return self._meaning_sqlite(db, qvec, kinds, gate, lexical, margin, bonus, floor)

    def _meaning_cached(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None,
                        gate: float, lexical: dict[int, int | None], margin: float, bonus: float, floor: float) -> _Meaning:
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
        # The gate as one cosine threshold per modality: z >= gate <=> cos >= offset + gate * std.
        offset = scale.offsets(sc.names)
        ok = sims >= (offset + np.float32(gate * scale.std))[mods]
        if sc.valid is not None:
            ok &= sc.valid
        if kinds is not None:
            ok &= np.isin(sc.kind, cache.codes(kinds))
        rows = np.flatnonzero(ok)
        z = (sims[rows] - offset[mods[rows]]) * np.float32(1.0 / scale.std)
        if len(rows):
            z_min = _vector_min(float(z.max()), gate, margin, bonus, floor)
            keep = np.flatnonzero(z >= z_min)
            if len(keep) > MAX_MEANING_ROWS:
                keep = keep[np.argpartition(-z[keep], MAX_MEANING_ROWS - 1)[:MAX_MEANING_ROWS]]
            rows, z = rows[keep], z[keep]

        def take(rows: np.ndarray, z: np.ndarray) -> None:
            for r, f, c, zz, cos in zip(rows.tolist(), sc.file_ids[rows].tolist(), sc.chunk_ids[rows].tolist(),
                                       z.tolist(), sims[rows].tolist(), strict=True):
                if f not in best or zz > best[f][0]:
                    best[f] = (zz, cos, c)

        take(rows, z)
        extra = [f for f in lexical if f not in best]
        if extra:
            rows = cache.rows(self._file_chunks(db, extra))
            take(rows, (sims[rows] - offset[mods[rows]]) * np.float32(1.0 / scale.std))
        return _Meaning(best, scale)

    def _meaning_sqlite(self, db: sqlite3.Connection, qvec: np.ndarray, kinds: list[str] | None,
                        gate: float, lexical: dict[int, int | None], margin: float, bonus: float, floor: float) -> _Meaning:
        """Fallback without the vector cache (index above vector_cache_max_chunks): the KNN
        top-k (cfg.knn_k) is scored on a scale estimated from a sample of the index. A lexical
        candidate outside the top-k is scored by one chunk (its best text match, else its first
        chunk), for the first SQLITE_LEXICAL_VECTORS of them in lexical order (~0.15 ms each
        from vec0; the rest count as z = 0). Selecting the embedding column in the KNN would add
        ~50 ms at 100k chunks."""
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
            rows = [(cid, fid, scale.z(cos[cid], mod), cos[cid]) for cid, fid, mod in db.execute(
                f"SELECT id, file_id, modality FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids)]
            z_max = max((zz for _, _, zz, _ in rows if zz >= gate), default=None)
            z_min = _vector_min(z_max, gate, margin, bonus, floor) if z_max is not None else float("inf")
            for cid, fid, zz, c in rows:
                if (zz >= z_min or fid in lexical) and (fid not in best or zz > best[fid][0]):
                    best[fid] = (zz, c, cid)
        extra = list(itertools.islice((f for f in lexical if f not in best), SQLITE_LEXICAL_VECTORS))
        if extra:
            first = {r[0]: r[1] for r in db.execute(
                f"SELECT file_id, id FROM chunks WHERE file_id IN ({','.join('?' * len(extra))}) AND seq = 0", extra)}
            chunk_of = {f: c for f in extra if (c := lexical[f] or first.get(f)) is not None}
            mods = {r[0]: r[1] for r in db.execute(
                f"SELECT id, modality FROM chunks WHERE id IN ({','.join('?' * len(chunk_of))})",
                list(chunk_of.values()))} if chunk_of else {}
            vecs = self._vectors(db, list(chunk_of.values())) if chunk_of else {}
            for f, c in chunk_of.items():
                v = vecs.get(c)
                if v is not None:
                    cc = float(v @ q / (np.linalg.norm(v) or 1.0))
                    best[f] = (scale.z(cc, mods.get(c)), cc, c)
        return _Meaning(best, scale)

    def _sample_scale(self, db: sqlite3.Connection, qvec: np.ndarray) -> Scale:
        """The scale from a sample of about SCALE_SAMPLE vectors, spread evenly in chunk-id order
        within each modality and weighted by how many chunks each stands for; every modality
        with MODALITY_MIN chunks in the index gets at least MODALITY_SAMPLE_MIN of them, so a
        rare modality is still centred on its own mean (reloaded when the index changed and the
        sample is older than SCALE_SAMPLE_MAX_AGE, or at once while the sample is the whole
        index)."""
        version = chunks_version(db)
        now = time.monotonic()
        s = self._sample
        if s is None or s.version != version:
            if s is None or now - s.time > SCALE_SAMPLE_MAX_AGE or s.full:
                s = self._sample = self._load_sample(db, version, now, len(qvec))
        names = s.names
        q = np.asarray(qvec, dtype=np.float32)
        sims = s.mat @ (q / (np.linalg.norm(q) or 1.0))
        return meaning_scale(sims, s.codes, names, s.weights, s.counts)

    def _load_sample(self, db: sqlite3.Connection, version: int, now: float, dim: int) -> _Sample:
        per_mod = dict(db.execute("SELECT modality, count(*) FROM chunks GROUP BY modality").fetchall())
        names = sorted(per_mod)
        total = max(sum(per_mod.values()), 1)
        target = {m: min(n, max(round(SCALE_SAMPLE * n / total), MODALITY_SAMPLE_MIN if n >= MODALITY_MIN else 1))
                  for m, n in per_mod.items()}
        # Positions (within each modality, in chunk-id order) spread evenly over all of it.
        wanted = {m: {per_mod[m] * j // target[m] for j in range(target[m])} for m in names}
        seen = dict.fromkeys(names, 0)
        picked: list[tuple[int, str]] = []
        for cid, mod in db.execute("SELECT id, modality FROM chunks ORDER BY id"):
            if seen[mod] in wanted[mod]:
                picked.append((cid, mod))
            seen[mod] += 1
        vecs = self._vectors(db, [c for c, _ in picked]) if picked else {}
        picked = [(c, m) for c, m in picked if c in vecs]
        mat = np.stack([vecs[c] for c, _ in picked]).astype(np.float32) if picked else np.zeros((0, dim), np.float32)
        if len(mat):
            mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        code = {m: i for i, m in enumerate(names)}
        codes = np.array([code[m] for _, m in picked], dtype=np.intp)
        got = np.bincount(codes, minlength=len(names)) if len(codes) else np.zeros(len(names), np.int64)
        weights = np.array([per_mod[m] / got[code[m]] for _, m in picked], dtype=np.float64)
        return _Sample(version, now, mat, codes, weights, names, [per_mod[m] for m in names],
                       all(target[m] >= n for m, n in per_mod.items()))

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
