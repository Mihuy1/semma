"""PDF documents via pypdfium2: page text through the text chunker, scanned pages as images.

PDFium is not thread-safe: use this extractor from one thread at a time (the indexer runs
extraction in a single worker). The document is opened lazily, kept open while the generator
is suspended, and closed in a ``finally`` block when the generator finishes or is closed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Chunk, ExtractionError
from .text import MAX_CHUNKS, chunk_text, make_snippet, normalize_newlines

if TYPE_CHECKING:
    from ..config import Config

MAX_PAGES = 500
MIN_TEXT_CHARS = 40
RENDER_LONG_SIDE = 1600
RENDER_MAX_PIXELS = 1600 * 1600
RENDER_MAX_SCALE = 4.0


def _page_text(page) -> str:
    textpage = page.get_textpage()
    try:
        raw = textpage.get_text_range()
    finally:
        textpage.close()
    raw = raw.replace("\x02\r\n", "").replace("\x02", "-")
    return normalize_newlines(raw).replace("\ufffe", "")


def _render(page):
    w, h = page.get_size()
    if w <= 0 or h <= 0:
        return None
    scale = min(RENDER_LONG_SIDE / max(w, h), RENDER_MAX_SCALE)
    if w * h * scale * scale > RENDER_MAX_PIXELS:
        scale = (RENDER_MAX_PIXELS / (w * h)) ** 0.5
    bitmap = page.render(scale=scale, draw_annots=True)
    try:
        return bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()


def extract_pdf(path: Path, cfg: Config | None = None) -> Iterator[Chunk]:
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        msg = str(e)
        if "password" in msg.lower():
            raise ExtractionError("PDF is encrypted (password required)") from e
        raise ExtractionError(f"cannot open PDF: {msg}") from e
    except OSError as e:
        raise ExtractionError(f"cannot read file: {e.strerror or e}") from e
    try:
        n_pages = min(len(doc), MAX_PAGES)
        failed = 0
        last_error: Exception | None = None
        count = 0
        for index in range(n_pages):
            number = index + 1
            locator = f"p. {number}"
            chunks: list[Chunk] = []
            page = None
            try:
                page = doc[index]
                text = _page_text(page)
                if sum(not c.isspace() for c in text) >= MIN_TEXT_CHARS:
                    for span in chunk_text(text):
                        chunks.append(Chunk("text", span.text, locator, make_snippet(span.text), float(number)))
                else:
                    image = _render(page)
                    if image is not None:
                        chunks.append(Chunk("image", image, locator, make_snippet(text), float(number)))
            except Exception as e:
                failed += 1
                last_error = e
                continue
            finally:
                if page is not None:
                    page.close()
            for chunk in chunks:
                if count >= MAX_CHUNKS:
                    return
                count += 1
                yield chunk
        if n_pages and failed == n_pages:
            raise ExtractionError(f"no page could be read: {type(last_error).__name__}: {last_error}"[:300])
    finally:
        doc.close()
