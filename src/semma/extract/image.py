"""Images via Pillow; HEIC/HEIF through ffmpeg (AVIF is native in Pillow 13).

HEIC is decoded by the system ffmpeg rather than pillow-heif, whose wheel bundles the GPL x265
encoder. ffmpeg composes multi-tile photos (as phones take them) and applies their rotation
from FFmpeg 7.1 on; it needs an HEVC decoder (Fedora's ffmpeg-free may lack one).

With OCR on (`[ocr]`, see semma.ocr), a screenshot also yields its recognised text as text
chunks after the image chunk. OCR decodes the file again at full resolution (the image chunk
is downscaled to MAX_SIDE for the model, too small for UI text); screenshots are small files.
Any OCR failure only loses the text: the image chunk is indexed regardless.
"""

from __future__ import annotations

import io
import logging
import subprocess
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Chunk, ExtractionError

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from ..config import Config

log = logging.getLogger(__name__)

MAX_SIDE = 1536  # px; the model's processor resizes anyway, this bounds memory
MIN_SIDE = 32  # smaller images (icons, spacer gifs) are not worth indexing
# Pillow raises DecompressionBombError above twice this many pixels: 100 MP (checked from the
# header, before decoding). JPEG (draft) and JPEG 2000 (reduce) decode at a reduced size, so
# their cost stays small; every other format (PNG, WebP, TIFF, GIF, BMP, HEIC, AVIF, ...) is
# fully decoded before downscaling, i.e. up to ~400 MB for a 100 MP RGBA image.
MAX_IMAGE_PIXELS = 50_000_000

HEIF_SUFFIXES = frozenset({".heic", ".heif", ".hif"})
HEIF_TIMEOUT = 60.0  # s for ffmpeg to decode one picture

_initialised = False


def _init() -> None:
    global _initialised
    if _initialised:
        return
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    _initialised = True


def is_heif(path: Path) -> bool:
    return path.suffix.lower() in HEIF_SUFFIXES


def decode_heif(path: Path, max_side: int, timeout: float = HEIF_TIMEOUT) -> PILImage.Image:
    """Decode a HEIC/HEIF picture with ffmpeg, rotated and scaled down to fit max_side.

    ffmpeg writes the full picture as uncompressed PPM (a scale filter there would conflict with
    the filter ffmpeg itself uses to join a multi-tile photo); Pillow scales it down, with its
    usual pixel limit. Raises ExtractionError when ffmpeg is missing, fails or times out."""
    from PIL import Image

    from .media import FFMPEG_FLAGS, input_args, tool

    try:
        res = subprocess.run(
            [tool("ffmpeg"), *FFMPEG_FLAGS, *input_args(path), "-frames:v", "1",
             "-f", "image2pipe", "-c:v", "ppm", "pipe:1"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, start_new_session=True,
        )
    except subprocess.TimeoutExpired as e:
        raise ExtractionError("ffmpeg timed out decoding the picture") from e
    except OSError as e:
        raise ExtractionError(f"cannot run ffmpeg: {e}") from e
    if res.returncode != 0 or not res.stdout:
        msg = res.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ExtractionError(f"cannot decode HEIF: {msg[-1] if msg else 'ffmpeg exit ' + str(res.returncode)}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(res.stdout))
            img.load()
    except Image.DecompressionBombError as e:
        raise ExtractionError(f"image too large: {e}") from e
    except (OSError, ValueError, SyntaxError) as e:
        raise ExtractionError(f"cannot decode HEIF: {e}") from e
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return img


def to_rgb(img: PILImage.Image) -> PILImage.Image:
    """Convert any mode to RGB, compositing transparency onto white. Always returns a copy."""
    from PIL import Image

    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        import numpy as np

        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(a)
        img = Image.fromarray(a.astype(np.uint8), "L")
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA", "PA", "La", "RGBa"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def load_image(path: Path) -> PILImage.Image | None:
    """Decode an image file to a bounded RGB image, or None if it is too small to matter."""
    _init()
    from PIL import Image, ImageOps, UnidentifiedImageError

    if is_heif(path):
        img = decode_heif(path, MAX_SIDE)
        return None if min(img.size) < MIN_SIDE else to_rgb(img)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                if min(img.size) < MIN_SIDE:
                    return None
                img.seek(0)  # first frame of animations
                if img.format == "JPEG":
                    img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # DCT-domain downscale: fast, low memory
                elif img.format == "JPEG2000":
                    # wavelet-level reduction while decoding (power of two, still >= MAX_SIDE)
                    factor = 0
                    while max(img.size) >> (factor + 1) >= MAX_SIDE and factor < 5:
                        factor += 1
                    if factor:
                        img.reduce = factor
                img.load()
                img = ImageOps.exif_transpose(img)
                if max(img.size) > MAX_SIDE:
                    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS, reducing_gap=3.0)
                return to_rgb(img)
    except Image.DecompressionBombError as e:
        raise ExtractionError(f"image too large: {e}") from e
    except UnidentifiedImageError as e:
        raise ExtractionError("not a recognised image") from e
    except OSError as e:
        raise ExtractionError(f"cannot decode image: {e.strerror or e}") from e
    except (ValueError, SyntaxError, EOFError) as e:  # Pillow plugins raise these on bad data
        raise ExtractionError(f"corrupt image: {e}") from e


def extract_image(path: Path, cfg: Config | None = None) -> Iterator[Chunk]:
    img = load_image(path)
    if img is None:
        return
    yield Chunk("image", img)
    del img
    if cfg is None or not cfg.ocr_enabled:
        return
    from .. import ocr

    if not ocr.is_screenshot(path):
        return
    languages = ocr.effective_languages(cfg)
    if not languages:
        return
    try:
        text = ocr_text(path, languages)
    except Exception as exc:  # a broken library or an odd image must not fail the file
        log.warning("OCR of %s failed: %s", path, exc)
        return
    if text:
        from .text import chunk_text, spans_to_chunks

        yield from spans_to_chunks(chunk_text(text), lambda span, i: ("", None))


def ocr_text(path: Path, languages: list[str]) -> str:
    """The recognised text of the image file at `path`, decoded at full resolution."""
    _init()
    from PIL import Image, ImageOps

    from .. import ocr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as img:
            img.seek(0)
            img.load()
            img = ImageOps.exif_transpose(img)
            return ocr.recognize(img, languages)
