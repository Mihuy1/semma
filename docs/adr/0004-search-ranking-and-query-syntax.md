# ADR 0004: Search Ranking, Per-Modality Z-Scores, and Query Syntax

## Status

Accepted

## Context

EmbeddingGemma 2 produces compressed cosine similarity distributions (unrelated pairs commonly score ~0.45–0.70), and different modalities (text vs. image/video) have distinct baseline cosine means. A fixed cosine threshold fails across queries and modalities.

## Decisions

### 1. Per-Modality Z-Score Centering and Fallbacks
- Each chunk's semantic ("meaning") score is converted to a z-score relative to the distribution of stored vectors for the query, after centering each modality on its own mean so text does not systematically outscore visual modalities.
- `MODALITY_MIN` is counted by distinct files, not chunks (counting chunks caused a small number of multi-chunk videos to form their own high mean and drop below `z_floor` / `z_gate`).
- If `video` has fewer than `MODALITY_MIN` files, it falls back to the `image` modality mean (both share the visual SigLIP encoder) when `image` has >= `MODALITY_MIN` files, and otherwise falls back to the overall mean.
- The scale sample is cached across index changes within `SCALE_SAMPLE_MAX_AGE` to avoid refetching on every minor update.
- `video` and `audio` vectors count per file, not per chunk, in the scale sums (`PER_FILE_MODALITIES`: each modality's weight is scaled by `min(1, files / chunks)`). A single multi-chunk video in a small index otherwise makes the distribution bimodal, pulling the mean up and inflating the std so the video falls below `z_floor`. Its chunks are still scored individually; only their weight in the mean/std changes.
- On small indexes (total effective weight `<= CLIP_MAX_TOTAL`, currently `500`), strong matches are clipped from the scale: vectors above `CLIP_Z = 3.0` are removed from the sums and the scale is re-estimated, for at most `CLIP_ITERS = 2` passes (stopping early when the clipped set no longer changes or when no weight would remain). Without this a few strong matches on a small index inflate the std they are measured against and cap their own z at `sqrt(n - 1)`. Clipping is skipped above `CLIP_MAX_TOTAL` because single-match std inflation is already tiny at that size, whereas one-sided tail clipping on large right-skewed distributions would shrink std and push noise queries toward `z_floor`. Both the VectorCache path and the sqlite-vec sample path (weights + per-modality file counts) go through the same `meaning_scale`, so they agree when the sample is the whole index.

### 2. Candidate Gating, Z-Floor, and Tail Trimming
- A file whose best chunk reaches `z_gate` (default `2.0`, roughly the top 2–3% of chunks) is a semantic candidate; below `z_gate`, a file requires at least half the query's content words (or a filename match) to qualify.
- Vector-only matches (files without the query's words) are shown only if the query's best meaning match reaches `z_floor` (default `4.0`, separating nonsense queries topping out at z ~3.0–3.5 from real visual/video matches starting at z ~4.1–4.4) AND the file sits within `margin` of that best match. Single-word queries use a tighter margin (`SINGLE_WORD_MARGIN = 1.5`) because their meaning is vaguer and their tail longer.
- Both `z_floor` and relative margin include the kind bonus (e.g., `"image of a dog"`).
- Note on tiny indexes: with `n` vectors and no clipping, no z-score can exceed `sqrt(n - 1)`; clipping (section 1) lifts that cap for matches more than `CLIP_Z` above the rest, so tests that need a match to stay below the gate should keep it under z ~3.

### 3. Lexical Scoring, Selectivity, and Lookups
- Full coverage of query words adds `W_WORDS = 2.0`, lifting files with exact words and middling meaning over slightly higher vector-only matches, while partial coverage (e.g., 0.5 -> +0.71) cannot beat a strong meaning match.
- A filename containing every query word receives `NAME_BONUS` on top of coverage (+2), placing the named file first unless another file exceeds it by z > ~5. Partial filename matches do not earn `NAME_BONUS`.
- A single non-lookup query word appearing only in body text receives half weight (`W_WORDS * 0.5`) so a passing mention in a long document does not outrank a directly matching photo, unless the file is named after the word.
- Folder-name words contribute to score, but alone do not admit a file as a candidate.
- Lookup queries (hashes, hex codes like `"b0bbce05d5"` or `"0xff32:0x0074"`) halve semantic weight and double lexical weight.
- High-DF terms exceeding `TEXT_MAX_DF` (or `TEXT_MAX_DF_RATIO`) are dropped from chunk-text FTS queries (avoiding ~60 ms scans on 100k chunks for stopwords like `"the"`), and dropped terms do not count toward coverage unless all terms were dropped (`named` still requires every word).
- Matching the small `files` FTS table directly and intersecting in Python is far faster than `FTS5` `rowid IN (...)` probing (~5 ms for 700 IDs).

### 4. Prefix Matching, Type Words, and Explicit Filters
- Filenames/directories prefix-match every term >= 3 chars (>= 2 for the last term); chunk text prefixes only the last term >= 3 chars, and only when not already present verbatim.
- Explicit prefix filters (`image:`, `screenshot:`, etc.) restrict results strictly (e.g., `screenshot:` enforces `ocr.is_screenshot`) and intersect with any explicit `kinds` argument.
- Unambiguous type words (e.g., `"screenshot"`, Finnish locatives `"kuvassa"`, `"videolla"`) act purely as kind hints and are not searched as content words (unless the query consists solely of type words). Ambiguous words (`"image segmentation"`, `"music theory"`, `"paper towel"`) stay content words and act as hints only in explicit phrasings (`HINT_AFTER` / `HINT_BEFORE`).

### 5. Near-Duplicate Collapsing, Location Priors, Snippets, and Snapshots
- Near-duplicate thresholds were calibrated on a 1,190-file index: images use cosine `0.96` (re-encoded copies median 0.98, p5 0.955 vs. closest distinct photos 0.95); text uses `0.99` (unchanged sections >= 0.998, cross-format same doc 0.97–0.98, separate notes up to 0.987).
- Location priorities match longest-path-first so the most specific folder rule wins, and are bounded (e.g., max 0.3 z prior) so priority never overrides a 1.0 z relevance lead.
- Displayed snippet picks the best meaning chunk if it stands out, otherwise the best text match; for images, OCR text containing the query words takes precedence over the image's own empty chunk.
- Read queries pin a single SQLite WAL snapshot by checking/reloading the vector cache (`chunks_version`) before running FTS queries, preventing stale-snapshot reloads under concurrent writes.
- Search debouncing returns immediately when a newer query supersedes an in-flight debounced query.
