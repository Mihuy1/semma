"""Video via an ffmpeg pipe of raw RGB frames; audio tracks via audio.py in lockstep.

Per segment of `media_segment_seconds` (60 s by default) one "video" chunk with up to 32 frames
sampled uniformly (fps = 32 / segment, i.e. ~0.53 fps), longest side <= 512 px, and, when the
file has an audio track, one "audio" chunk for the same time range:

    ffmpeg -nostdin -hide_banner -loglevel error -skip_loop_filter all -i FILE
           -map 0:V -an -sn -dn -t MAX
           -vf fps=32/60,scale=W:H:flags=bilinear,setsar=1 -pix_fmt rgb24 -f rawvideo pipe:1

ffmpeg auto-rotates according to the display matrix, so W:H are computed from the rotated,
SAR-corrected display size reported by ffprobe and every frame is exactly W*H*3 bytes.
Only one segment of frames (~14 MB at 512x288) is in memory at a time.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from .audio import audio_chunk, audio_segments, extract_audio, keep_segment, segment_seconds
from .base import MAX_VIDEO_FRAMES, Chunk, ExtractionError
from .media import Probe, ffmpeg_pipe, probe, time_range

if TYPE_CHECKING:
    from ..config import Config

FRAME_MAX_SIDE = 512
MIN_TRAILING_SECONDS = 2.0


def output_size(width: int, height: int, max_side: int = FRAME_MAX_SIDE) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return max(2, round(width * scale)), max(2, round(height * scale))


def video_frames(path: Path, cfg: Config, info: Probe) -> Iterator[tuple[int, list]]:
    """(segment index, frames) for each segment; the last one may hold fewer frames."""
    from PIL import Image

    seg = segment_seconds(cfg)
    per_seg = MAX_VIDEO_FRAMES
    fps = Fraction(per_seg) / Fraction(seg).limit_denominator(1000)
    w, h = output_size(info.width, info.height)
    frame_bytes = w * h * 3
    args = [
        "-skip_loop_filter", "all",  # skip deblocking: ~20-30 % faster decode, invisible at 512 px
        "-i", str(path), "-map", f"0:{info.video_index}", "-an", "-sn", "-dn",
        "-t", f"{cfg.max_media_seconds:.3f}",
        "-vf", f"fps={fps.numerator}/{fps.denominator},scale={w}:{h}:flags=bilinear,setsar=1",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    index = 0
    with ffmpeg_pipe(args) as pipe:
        eof = False
        while not eof:
            frames = []
            while len(frames) < per_seg:
                data = pipe.read_exact(frame_bytes)
                if len(data) < frame_bytes:
                    eof = True
                    break
                frames.append(Image.frombytes("RGB", (w, h), data))
            if frames:
                yield index, frames
                index += 1
        rc, err = pipe.finish()
    if rc != 0 and index == 0:
        raise ExtractionError(f"ffmpeg could not decode video: {err or 'exit ' + str(rc)}")


def extract_video(path: Path, cfg: Config) -> Iterator[Chunk]:
    info = probe(path)
    if not info.has_video:
        if info.has_audio:  # e.g. an .mp4 holding only audio (or audio + cover art)
            yield from extract_audio(path, cfg, info)
            return
        raise ExtractionError("no video or audio stream")
    seg = segment_seconds(cfg)
    frame_step = seg / MAX_VIDEO_FRAMES
    total = min(info.duration, cfg.max_media_seconds) if info.duration else None
    audio = audio_segments(path, cfg) if info.has_audio else None
    audio_next = 0  # index of the next audio segment to pull
    video_error: ExtractionError | None = None
    yielded = 0

    def audio_upto(last: int) -> Iterator[Chunk]:
        """Pull (and yield the non-silent) audio segments with index <= last."""
        nonlocal audio, audio_next
        while audio is not None and audio_next <= last:
            try:
                start, samples = next(audio)
            except StopIteration:
                audio = None
                return
            except ExtractionError:
                audio = None  # undecodable audio track: keep the frames
                return
            i, audio_next = audio_next, audio_next + 1
            if keep_segment(i, samples):
                yield audio_chunk(start, samples, total)

    try:
        try:
            for index, frames in video_frames(path, cfg, info):
                start = index * seg
                if index == 0 or len(frames) * frame_step >= MIN_TRAILING_SECONDS:
                    end = start + seg if len(frames) == MAX_VIDEO_FRAMES else start + len(frames) * frame_step
                    if total:
                        end = min(start + seg, total)
                    yield Chunk("video", frames, time_range(start, end, total), "", float(start))
                    yielded += 1
                del frames
                for chunk in audio_upto(index):
                    yield chunk
                    yielded += 1
        except ExtractionError as e:
            video_error = e
        for chunk in audio_upto(10**9):  # audio running longer than the video stream
            yield chunk
            yielded += 1
    finally:
        if audio is not None:
            audio.close()
    if video_error is not None and yielded == 0:
        raise video_error
