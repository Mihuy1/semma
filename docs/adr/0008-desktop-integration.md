# ADR 0008: Desktop Integration (KRunner, GNOME Shell, Thumbnails, Viewer, MPRIS)

## Status

Accepted

## Context

Semma exposes search results to both KDE KRunner and GNOME Shell over D-Bus, renders thumbnails under tight latency budgets, and opens matched files directly at the matched PDF page or media timestamp.

## Decisions

### 1. GNOME Shell Search Provider
- GNOME Shell invokes `GetResultMetas` once per ID and caches the result actor until the search is cleared (`search.js _ensureResultActors`), so a thumbnail that misses the budget never appears. GNOME therefore uses a much longer thumbnail budget than KRunner.
- Only the top results that GNOME's list view actually displays have thumbnails prewarmed as soon as search completes (`max_results` itself is unchanged); `GetResultMetas` joins the in-flight deduplicated thumbnail task, and any remaining results generate on demand.
- When a thumbnail is available, it replaces `gicon` because GNOME Shell prefers `gicon` over `icon-data` if both are present.
- `GetSubsearchResultSet` re-runs the full search rather than filtering prior IDs because semantic search results do not narrow monotonically as the user types more characters.

### 2. KDE KRunner Provider
- Implements the KRunner D-Bus contract: `Match(s) -> a(sssida{sv})`, `Actions() -> a(sss)`, `Run(s, s)`, `SetActivationToken(s)`, and `Teardown()`.
- Relevance maps the search score linearly (`[0, 8] -> [0.3, 0.9]`) and strictly decreases by rank so ties never reorder.
- Category relevance assigns KRunner "Moderate" (50) only to strong matches (score threshold backed by lexical evidence or a standout vector-only z-score, avoiding z 3–4 tails) and "Low" (30) to the rest, never increasing down the list.
- Activation tokens are cleared immediately after `Run`.
- Even when the per-`Match` thumbnail budget is exhausted, KRunner still kicks off background thumbnail generation so the next `Match` call hits the cache.

### 3. Thumbnail Pipeline
- Freedesktop thumbnail cache hits return directly without decoding the underlying file.
- Decoders request at least 2x the target size because EXIF orientation may swap width and height.
- Tight caller timeouts abort waiting without cancelling background generation, so subsequent calls hit the LRU cache immediately.

### 4. Viewer Launching and MPRIS Seeking
- Known PDF viewers (including Chromium-based browsers via `#page=`) and media players with CLI start-time options are launched directly with spot arguments (browsers use HTML media fragments).
- If a viewer is unknown or contains a desktop-entry field code inside an argument, Semma falls back to plain opening rather than guessing.
- MIME detection falls back cleanly when the system `mimetypes` database does not recognize an audio/video file.
- For media players without a CLI start-time flag, Semma snapshots active MPRIS players before opening and seeks via MPRIS afterward (matching Flatpak player file paths as well). Players launched with a direct start option and PDF page spots never use MPRIS.

### 5. User Installation (`semma install`)
- Preserves the Python symlink path (e.g., `.venv/bin/python`) while expanding `~` and relative paths so virtualenv activation stays intact.
- The tray autostart desktop entry is optional so users can disable the tray by removing it.
- Application, tray autostart, and D-Bus/systemd unit files are shared between KRunner and GNOME providers and are retained while either provider remains installed.
