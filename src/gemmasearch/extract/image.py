"""Images via Pillow (+ pillow-heif for HEIC/HEIF; AVIF is native in Pillow 13)."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Chunk, ExtractionError

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from ..config import Config

MAX_SIDE = 1536  # px; the model's processor resizes anyway, this bounds memory
MIN_SIDE = 32  # smaller images (icons, spacer gifs) are not worth indexing
# Pillow raises DecompressionBombError above twice this many pixels (here: 256 MP).
MAX_IMAGE_PIXELS = 128_000_000

_initialised = False


def _init() -> None:
    global _initialised
    if _initialised:
        return
    from PIL import Image

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:  # pragma: no cover - HEIC simply stays unsupported
        pass
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    _initialised = True


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

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                if min(img.size) < MIN_SIDE:
                    return None
                img.seek(0)  # first frame of animations
                if img.format == "JPEG":
                    img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # DCT-domain downscale: fast, low memory
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
    if img is not None:
        yield Chunk("image", img)
