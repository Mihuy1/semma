"""Content extractors: file -> lazily generated Chunks (see base.py for the contract)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .. import filetypes as ft
from .base import Chunk, ExtractionError, Extractor
from .text import chunk_text, make_snippet

if TYPE_CHECKING:
    from ..config import Config

__all__ = ["Chunk", "ExtractionError", "EXTRACTORS", "chunk_text", "clean_str", "extract", "make_snippet"]


def clean_str(s: str) -> str:
    """`s` without NULs and lone surrogates, so it can always be UTF-8 encoded and stored.

    Surrogate pairs (e.g. from UTF-16 sources) are combined into the real character; lone
    surrogates become U+FFFD. The common case (already clean) costs one encode.
    """
    if "\x00" in s:
        s = s.replace("\x00", "")
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    return s


def _clean(chunk: Chunk) -> Chunk:
    if isinstance(chunk.content, str):
        chunk.content = clean_str(chunk.content)
    chunk.snippet = clean_str(chunk.snippet)
    chunk.locator = clean_str(chunk.locator)
    return chunk


def _text(path: Path, cfg: Config) -> Iterator[Chunk]:
    from .text import extract_text

    return extract_text(path, cfg)


def _pdf(path: Path, cfg: Config) -> Iterator[Chunk]:
    from .pdf import extract_pdf

    return extract_pdf(path, cfg)


def _office(path: Path, cfg: Config) -> Iterator[Chunk]:
    from .office import extract_office

    return extract_office(path, cfg)


def _image(path: Path, cfg: Config) -> Iterator[Chunk]:
    from .image import extract_image

    return extract_image(path, cfg)


def _audio(path: Path, cfg: Config) -> Iterator[Chunk]:
    from .audio import extract_audio

    return extract_audio(path, cfg)


def _video(path: Path, cfg: Config) -> Iterator[Chunk]:
    from .video import extract_video

    return extract_video(path, cfg)


EXTRACTORS: dict[str, Extractor] = {
    ft.TEXT: _text, ft.PDF: _pdf, ft.OFFICE: _office,
    ft.IMAGE: _image, ft.AUDIO: _audio, ft.VIDEO: _video,
}


def extract(path: Path, kind: str, cfg: Config) -> Iterator[Chunk]:
    """Lazily yield the chunks of `path`. Any failure surfaces as ExtractionError.

    Errors are raised while iterating (this is a generator), so callers should treat chunks
    already received as partial when an ExtractionError follows. Closing the generator early
    releases everything (files, pdfium documents, ffmpeg processes). Text fields (str content,
    snippet, locator) never contain NULs or lone surrogates.
    """
    fn = EXTRACTORS.get(kind)
    if fn is None:
        raise ExtractionError(f"no extractor for kind {kind!r}")
    gen = None
    try:
        gen = fn(Path(path), cfg)
        for chunk in gen:
            yield _clean(chunk)
    except ExtractionError:
        raise
    except MemoryError as e:
        raise ExtractionError("out of memory while extracting") from e
    except Exception as e:  # corrupt files make parsers fail in many creative ways
        msg = getattr(e, "strerror", None) or str(e) or type(e).__name__
        raise ExtractionError(f"{type(e).__name__}: {msg}"[:300]) from e
    finally:
        if gen is not None:
            gen.close()
