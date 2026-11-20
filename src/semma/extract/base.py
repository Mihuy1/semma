"""The contract between extractors (file -> chunks) and the embedder (chunks -> vectors).

Content types were verified against google/embeddinggemma-2 with sentence-transformers 6.3:
  text   str                         raw chunk text; the embedder adds "title: <file name> | text: "
  image  PIL.Image.Image             RGB
  audio  numpy.ndarray               float32, mono, 16 kHz, at most ~320 s
  video  list[PIL.Image.Image]       RGB frames, at most 32 per chunk (frames only; the model
                                     ignores audio tracks, so videos also yield "audio" chunks)
Media is decoded with ffmpeg by the extractors; the model never receives file paths.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    from ..config import Config

Modality = Literal["text", "image", "audio", "video"]
Content = Union[str, "Image.Image", "np.ndarray", "list[Image.Image]"]

AUDIO_SAMPLE_RATE = 16_000
MAX_VIDEO_FRAMES = 32


@dataclass(slots=True)
class Chunk:
    modality: Modality
    content: Content
    locator: str = ""
    snippet: str = ""
    position: float | None = None


class ExtractionError(Exception):
    """The file cannot be read or decoded. The indexer marks it failed and moves on."""


Extractor = Callable[[Path, "Config"], Iterator[Chunk]]
