# ADR 0003: SQLite Storage, FTS5 Tokenization, and In-Memory Vector Cache

## Status

Accepted

## Context

Semma stores file metadata, extracted chunk text, FTS5 lexical indexes, and 768-dimensional float32 embeddings in a single SQLite database (`sqlite-vec`) accessed concurrently by the indexing worker, search thread, and CLI/status commands.

## Decisions

### 1. FTS5 Tokenizer and Path Normalization
- The FTS5 tokenizer folds case and strips diacritics (e.g., `"opinnäytetyö"` matches `"opinnaytetyo"`).
- Search queries append a prefix wildcard to the last term to support type-as-you-go and inflected languages (such as Finnish). A trigram tokenizer was rejected because it triples index size for chunk text and degrades BM25 precision.
- `fts_chunks` is contentless (cannot be scanned directly; its `docsize` shadow table tracks document counts).
- Leading path segments with no discriminative signal—such as the user's home directory (whose username matches every file) and data-drive mount points—are stripped from indexed path tokens.

### 2. Concurrency, Transactions, and Migrations
- Each thread uses its own SQLite connection so rollbacks on one thread never interfere with commits on another.
- An in-process write lock serializes write transactions in commit order so the in-memory vector cache observes mutations in the exact order they commit; cross-process writes are serialized by SQLite's own write lock.
- Schema migrations re-read the schema version after acquiring the write lock so two processes opening an older database simultaneously never both run the migration.
- Bulk index resets clear virtual tables (`fts_chunks`, `vec_chunks`) before deleting rows from `chunks` so per-row delete triggers do not execute redundant work.
- `finish_scan` relies on `ON DELETE CASCADE` and triggers to clean up chunks and vectors when a file row is deleted.

### 3. Index Format Versioning
- `meta['index_format']` tracks semantic index compatibility (chunking, document prompt format, FTS layout). Bumping it re-queues every file while leaving old chunks searchable until replaced:
  - Version 2: schema v3 (`fts_chunks` recreated with a prefix index) and float32 vector renormalization.
  - Version 3: tiny text chunks are no longer embedded.
- An older index with no `index_format` triggers re-queueing because `fts_chunks` lacked text under the older layout.
- Media settings (`media_max_seconds`, `media_segment_seconds`) are stored in `meta`; indexes created before they were recorded assume the legacy 30-minute default.

### 4. In-Memory Vector Cache and `sqlite-vec` Fallback
- `sqlite-vec` rejects KNN queries with `k > 4096`.
- Long-running processes keep all vectors in RAM (~3 KiB per 768-dim float32 chunk) up to the configured chunk limit; beyond that limit, search falls back to `sqlite-vec` directly.
- Chunk IDs can be reused after deletions, so cache integrity checks verify that every cached vector belongs to a live chunk and sample-check results against `vec0`.
- Cache synchronization with WAL snapshots:
  - If a local commit lands while a read is active, the commit patches or invalidates the cache directly (patching in flight avoids reloading from an older WAL snapshot).
  - If the cache was loaded after our commit, the commit is already included; if a foreign process committed in between, the cache reloads on next use.
  - Because the cache drops entries by whole file, partial chunk updates re-add the retained chunks for those files.

### 5. Forgetting Unconfigured Files
- `forget_unconfigured` removes files outside active roots, inside excluded subtrees (unless a configured root lies inside the excluded path), or matching disabled kinds/extensions (matched case-insensitively after the last dot, with `_` escaped rather than treated as a SQL wildcard).
- Files on configured but currently unplugged mount points (`include_mounts`) are preserved.
