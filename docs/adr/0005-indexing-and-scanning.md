# ADR 0005: File Scanning, Exclusion Heuristics, and Indexing Fault Tolerance

## Status

Accepted

## Context

Semma continuously scans user home directories and removable data drives. It must avoid indexing build artifacts, application bundles, and source code, must never wipe an index when a removable drive is unplugged, and must isolate single corrupt files from systemic GPU/model failures.

## Decisions

### 1. Scope and Exclusion Heuristics
- Source code files are deliberately excluded from `filetypes` indexing. `filetypes.classify` itself remains unfiltered for callers that need raw classification.
- OS system folders (such as `Windows` on an NTFS data drive) are pruned only directly below a scan root so a user's personal `~/Documents/Windows` folder is still indexed.
- Build/output directory names (such as `build`) are skipped only inside detected software projects; outside a project they are treated as ordinary user folders.
- User location anchors (`~`, `~/Documents`, `~/Downloads`, etc.) are never classified as software projects or application installations: a dotfiles `.git` in `~` or a stray `.exe` + `.dll` installer in `~/Downloads` must not demote or prune them.

### 2. Unplugged Drives and Deletion Safety
- Directories or roots that cannot be listed are recorded as unavailable; their previously indexed files are never deleted simply because a scan could not read them.
- An empty scan root is treated as an unmounted mount point (unavailable) rather than as "every file was deleted".
- With `include_mounts` enabled, mount-point parents are treated as configured roots so files on an unplugged drive stay in the index across daily scans and become immediately searchable again when plugged back in without re-indexing.
- Files not visited during a partial scan keep their prior scan generation; `finish_scan` only prunes beneath the roots actually walked, while `forget_unconfigured` removes files outside current settings.
- Watcher events inside a project or under an OS-folder child trigger a full scan so exclusion heuristics are re-evaluated accurately.

### 3. Indexing Fault Tolerance and Interrupts
- **Poison-file vs. broken-model streak rule**: if all chunks of a single file fail to embed, that file is marked failed and indexing continues. If embedding fails across a configured streak of distinct files in a row with no intervening success, the model/GPU is assumed broken and the indexing run halts rather than failing the entire queue.
- **Transient SQLite lock errors**: if SQLite remains locked past the busy timeout, indexing retries with backoff and then leaves the file pending instead of recording a permanent failure.
- **Automatic retry backoff**: failed files are retried automatically up to a configured attempt cap with linear backoff (`n * RETRY_BACKOFF_S`); `semma index --retry-failed` bypasses the backoff.
- **Non-`Exception` errors** (e.g., `BaseException` / interrupts) are never swallowed per file.
- **Interrupt safety**: when interrupted (e.g., `Ctrl-C` killing `ffmpeg`), any in-flight extraction may be truncated and is left pending rather than committed; already-extracted buffered chunks are embedded and committed before exiting.
- **Incremental media reconciliation**: when media limits change, old segments remain searchable until replaced, and deleting excess segments patches the in-memory vector cache directly.
