# ADR 0007: Screenshot OCR Pipeline (Tesseract)

## Status

Accepted

## Context

Semma optionally extracts text from screenshots using system-installed `libtesseract` via `ctypes`/`dlopen` so text inside UI screenshots is searchable alongside visual embeddings.

## Decisions

### 1. Screenshot Detection
- Screenshot classification is decided purely by file path (`SCREENSHOT_DIRS` and `SCREENSHOT_PREFIXES`, normalized via NFC, case-folding, typographic apostrophe -> `'`, and `_` -> space) to accommodate localized GNOME and KDE screenshot folder/file names.

### 2. Resolution Scaling and DPI
- OCR runs at full screenshot resolution (the 1536 px image embedding input makes small UI text illegible) up to 4K (`3840x2160` = 8.3 MP, ~2–3 s for a text-dense screen). Larger panoramas/8K images are scaled down to 8.3 MP, where high-DPI UI text remains legible.
- An explicit resolution is passed to Tesseract so it does not guess and emit stderr warnings.

### 3. Word and Text Quality Filtering
- Calibrated on 13 real GNOME and KDE screenshots (dialogs, browser task manager, tables, slides, diagrams, papers):
  - Real words score 60–97 confidence; icons, photos, and diagram strokes produce gibberish below 60 or high-confidence symbols (`@`, `|`, `©`).
  - Every kept word must contain at least one letter/digit and meet `MIN_WORD_CONFIDENCE` (and a stricter `MIN_SHORT_WORD_CONFIDENCE` for 1–2 alphanumeric character tokens so `"a"`/`"Wi"`/`"4"` noise is dropped while `"2"` in `"SigLIP 2"` or `"IP"` at 96 survives).
  - Overall extracted text must meet minimum counts of 3+ alphanumeric words and total alphanumeric characters; otherwise it is discarded so a lone caption does not embed as a generic vector matching every short query (the screenshot remains findable by visual embedding and filename).

### 4. Language Selection, Library Discovery, and Engine Lifecycle
- Each loaded Tesseract language costs ~25 MB RAM and slows recognition roughly linearly, so selection deduplicates and caps at three installed languages (missing languages are omitted rather than silently substituted; empty locale values count as unset).
- Locale-to-Tesseract mapping handles ISO 639-1 codes, glibc three-letter locales (e.g., `fil_PH`), Chinese territory variants, and Serbian `@latin`.
- `TESSDATA_PREFIX` supports both modern (`tessdata` path) and Tesseract 3 (parent directory) conventions.
- Library and `tessdata` discovery accounts for distribution differences: Fedora `/usr/lib64/libtesseract.so.5.5` + `/usr/share/tesseract/tessdata`, Debian/Ubuntu `/usr/lib/x86_64-linux-gnu/libtesseract.so.5` + `/usr/share/tesseract-ocr/5/tessdata` (where `libtesseract5` ships no language data, so install hints include English), Arch `/usr/share/tessdata`, and `/usr/local/share/tessdata`.
- Engine handles are guarded by a lock while in use so exit hooks never free them mid-call. Tesseract's `tprintf` output is suppressed to keep the systemd journal clean, and handles are properly ended because Tesseract retains dictionaries in a process-wide cache that logs `ObjectCache ... WARNING! LEAK!` at exit if a handle remains live.
- Reloading config picks up newly installed language packs even when settings text is unchanged, and OCR reconciliation keeps old chunks searchable until replaced or clears all OCR chunks when disabled.
