# ADR 0006: Content Extraction, Resource Bounds, and Error Containment

## Status

Accepted

## Context

Semma extracts text, images, audio, and video from arbitrary user files (often corrupt, huge, or adversarial). Extraction must bound memory and CPU time, prevent parser/subprocess exploits, and isolate per-page or per-file failures.

## Decisions

### 1. Extraction Versioning and Error Containment
- `EXTRACTION_VERSION` in `src/semma/extract/__init__.py` is bumped whenever extraction changes could fix previously failed files.
- Top-level extraction catches arbitrary parser exceptions because corrupt files fail in unpredictable ways; per-image EXIF/library quirks and per-page PDFium failures are contained so one broken page or odd image does not fail the rest of the document.

### 2. Text Limits and Linear Heading Detection
- Plain-text files whose entire content has fewer than `MIN_TEXT_ALNUM` letters/digits (e.g., stub files like `"@AGENTS.md"`) are not embedded because they produce generic vectors matching nearly every short query; they remain searchable by filename. Short chunks inside longer documents (such as slides) are still kept.
- Text, Office, and PDF files are capped at a maximum number of chunks (~2 MB of text) so huge logs, dumps, and generated files do not bloat the index.
- Heading detection operates line by line with non-backtracking patterns and skips lines longer than `MAX_HEADING_LINE`, keeping runtime linear on pathological inputs (a heading-like line followed by 100k spaces previously caused catastrophic backtracking).

### 3. Office and PDF Parsing
- Office XML parts enforce an uncompressed size cap (32 MiB, since `ElementTree` expands XML ~15–20x in RAM) checked during decompression rather than trusting zip header `file_size`, and reject DTDs while still supporting DTD-free UTF-16 XML parts.
- Slide iterators materialize inside the `with` block so the underlying zip archive stays open while reading slides.
- Office format routing lists template and macro variants alongside `odt`/`ott`/`odp`/`docx`/`pptx`/`rtf` so enabling them in `filetypes.py` requires no extractor changes.
- PDF text extraction strips/handles `U+0002`, which PDFium inserts for soft line-break hyphens.

### 4. Image, Audio, and Video Extraction
- Image decoding caps dimensions (bounding memory before the model's processor resizes) and enforces a 100 MP decompression-bomb limit checked from the header before decoding (JPEG draft/JPEG 2000 reduce decode cheaply; PNG/WebP/TIFF/GIF/BMP/HEIC/AVIF decode fully before downscaling, up to ~400 MB for 100 MP RGBA).
- Audio is segmented because EmbeddingGemma accepts at most ~320 s per audio chunk.
- `ffmpeg` supervision allows a minimum wall-clock timeout per process or a multiple of decoded media duration, whichever is larger. A non-zero `ffmpeg` exit (corrupt/truncated tail) is accepted only when decoded output covers the required fraction of expected duration; otherwise the file fails.
- Dangerous `ffmpeg` demuxers that read external files/URLs from inside a file (playlists, concat lists detected by content rather than filename) are disabled.
- `ffmpeg` subprocesses are isolated from the parent process group when required so interrupts are handled cleanly without leaving truncated output committed.
- Video decoding skips loop deblocking (~20–30% faster decode, invisible at 512 px).
- Video/audio track failure policy: an undecodable audio track is ignored so video frames are still indexed, and an undecodable video stream is tolerated if the audio track succeeded; however, a track that starts decoding and fails mid-stream (truncated decode) or a file where nothing decodes fails the file.
