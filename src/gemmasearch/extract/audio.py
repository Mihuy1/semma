"""Audio via an ffmpeg pipe: float32 mono 16 kHz, one chunk per `media_segment_seconds`.

    ffmpeg -nostdin -hide_banner -loglevel error -i FILE -map 0:a:0 -vn -sn -dn
           -ac 1 -ar 16000 -t MAX -f f32le pipe:1

Only one segment of samples is held in memory at a time; the process is killed and reaped
when the generator is closed early.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import AUDIO_SAMPLE_RATE, Chunk, ExtractionError
from .media import Probe, ffmpeg_pipe, probe, time_range

if TYPE_CHECKING:
    from ..config import Config

MIN_TRAILING_SECONDS = 2.0  # a shorter last segment is dropped (unless it is the only one)
SILENCE_RMS = 1e-4  # ~ -80 dBFS: digital silence / dead air
MAX_SEGMENT_SECONDS = 300  # the model accepts at most ~320 s per audio chunk


def segment_seconds(cfg: Config) -> float:
    return float(min(max(cfg.media_segment_seconds, 1.0), MAX_SEGMENT_SECONDS))


def audio_segments(path: Path, cfg: Config) -> Iterator[tuple[float, np.ndarray]]:
    """Every segment (start second, samples) of the first audio stream, unfiltered."""
    seg = segment_seconds(cfg)
    seg_samples = int(round(seg * AUDIO_SAMPLE_RATE))
    args = [
        "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-t", f"{cfg.max_media_seconds:.3f}", "-f", "f32le", "pipe:1",
    ]
    index = 0
    with ffmpeg_pipe(args) as pipe:
        while True:
            data = pipe.read_exact(seg_samples * 4)
            usable = len(data) // 4 * 4
            if usable:
                yield index * seg, np.frombuffer(data[:usable], dtype="<f4").astype(np.float32)
                index += 1
            if len(data) < seg_samples * 4:
                break
        rc, err = pipe.finish()
    if rc != 0 and index == 0:
        raise ExtractionError(f"ffmpeg could not decode audio: {err or 'exit ' + str(rc)}")


def is_silent(samples: np.ndarray) -> bool:
    return samples.size == 0 or float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) < SILENCE_RMS


def audio_chunk(start: float, samples: np.ndarray, total: float | None) -> Chunk:
    end = start + samples.size / AUDIO_SAMPLE_RATE
    return Chunk("audio", samples, time_range(start, end, total), "", float(start))


def keep_segment(index: int, samples: np.ndarray) -> bool:
    """Drop digital silence and a short trailing segment (only a final segment can be short)."""
    if index > 0 and samples.size < MIN_TRAILING_SECONDS * AUDIO_SAMPLE_RATE:
        return False
    return not is_silent(samples)


def extract_audio(path: Path, cfg: Config, info: Probe | None = None) -> Iterator[Chunk]:
    info = info or probe(path)
    if not info.has_audio:
        raise ExtractionError("no audio stream")
    total = min(info.duration, cfg.max_media_seconds) if info.duration else None
    for i, (start, samples) in enumerate(audio_segments(path, cfg)):
        if keep_segment(i, samples):
            yield audio_chunk(start, samples, total)
