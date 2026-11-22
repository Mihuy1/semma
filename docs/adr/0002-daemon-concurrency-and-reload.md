# ADR 0002: Daemon Concurrency, Scheduling, and Live Config Reload

## Status

Accepted

## Context

`semma daemon` coordinates file scanning, inotify watching, background indexing, live D-Bus search queries, status polling, and settings reloads inside a single Python process mixing an `asyncio` event loop with dedicated worker threads.

## Decisions

### 1. Thread Ownership and Concurrency Invariants
- Worker state is guarded by `_cv`; single-attribute reads that do not require cross-field consistency are permitted without acquiring it.
- Slice progress `(done, total, failed, files_per_sec)` and the active scanner reference are replaced atomically as whole objects by the worker thread and read without locks by the event loop for `Status`.
- When updating slice state, progress attributes are written before the phase transition so `Status` observes both consistently.
- Requests setting `_slice_stop` before a clear are re-checked so stop signals are never lost.
- Config swaps for search happen on the search thread between queries so every query observes one consistent `Config` throughout its execution.
- Indexing lowers its I/O priority via Linux `ioprio_set(2)` syscall numbers where known; other architectures retain the default I/O scheduling class.

### 2. Cheap Status Polling and Shutdown Atomicity
- `light_state()` must not touch SQLite or enqueue work on the status executor so frequent tray polls stay cheap.
- Per-kind counts read only the `(status, kind)` SQLite index rather than scanning the `files` table.
- A failing download-progress probe only hides download progress rather than failing `Status`.
- On shutdown, every committed file is complete (nothing partial is written to the store).

### 3. Deferred-File Backoff and Watcher Design
- When an indexing slice only skips or defers files (e.g., modified since scan or database busy), the worker waits for the rescan triggered by the change (or a bounded timeout) instead of spinning.
- The database directory is excluded from inotify watches so Semma's own SQLite writes never trigger rescans.
- Hidden directory creation is treated as a directory event (not ignored as a temp file), and creating or moving subdirectories reports both the parent watched directory and the new subdirectory.
- Watcher events are debounced with a `max_delay` cap so continuous writes (e.g., every 80 ms under a 350 ms debounce) still flush within `max_delay`.

### 4. Live Config Reload vs. Restart-Required Settings
- `load_config` records the source config path and expanded home directory so the daemon reloads the exact same file.
- `Daemon.reload_config` classifies every `Config` field:
  - **Restart-required**: fields used to construct the embedder, store, or vector cache.
  - **Rescan-triggering**: fields deciding what is indexed (roots, exclusions, kinds, extensions) or priority scores (which rescore every file, changed or not).
  - **Per-file / reconciliation**: fields read when indexing starts; changing media limits/segment lengths calls `Indexer.reconcile_media`, and changing OCR settings calls `Indexer.reconcile_ocr`.
  - **Immediate / live**: fields read at use time or pushed to components that cache them.
- **Live device reload (`model.device` and `model.query_device`)**: device settings are hot-reloaded without restarting the daemon, even when a model is already loaded or loading. When `_apply_config` runs on the worker at a clean file/slice boundary (waking the worker promptly if idle), it updates `embedder.requested_device` / `embedder.requested_query_device` and serializes worker subprocess unloads through `_model_exec` so any in-flight load finishes before being unloaded. Exiting the worker subprocess reclaims VRAM/RAM and clears the resolved device state (`device` / `query_device`), and the next search or indexing pass lazily reloads on the new device.

### 5. First-Run Onboarding Pause
- `onboarding_done = None` (key absent) indexes only if the user already has an existing index.
- `onboarding_done = False` pauses scanning and indexing without deleting the index, and avoids loading the model (which would pin the device). Re-running setup keeps an existing index and continues answering queries if a model is already loaded.
- Setting `onboarding_done = True` triggers a full scan.
