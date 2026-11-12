"""ffmpeg/ffprobe plumbing shared by the audio and video extractors."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .base import ExtractionError

PROBE_TIMEOUT = 60  # s
FFMPEG_FLAGS = ["-nostdin", "-hide_banner", "-loglevel", "error"]


def tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise ExtractionError(f"{name} not found on PATH")
    return found


@dataclass(slots=True)
class Probe:
    duration: float | None
    has_audio: bool
    width: int = 0  # display size of the first real video stream (after rotation/SAR), 0 if none
    height: int = 0
    video_index: int | None = None  # absolute stream index of that video stream

    @property
    def has_video(self) -> bool:
        return self.video_index is not None and self.width > 0 and self.height > 0


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 0 else None  # drop NaN/0/negative


def _rotation(stream: dict) -> int:
    rot = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rot = int(float(side["rotation"]))
            break
    else:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            try:
                rot = int(float(tag))
            except ValueError:
                pass
    return rot % 360


def probe(path: Path) -> Probe:
    cmd = [
        tool("ffprobe"), "-v", "error", "-of", "json",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height,sample_aspect_ratio,duration"
        ":stream_tags=rotate:stream_side_data=rotation:stream_disposition=attached_pic",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise ExtractionError("ffprobe timed out") from e
    except OSError as e:
        raise ExtractionError(f"cannot run ffprobe: {e}") from e
    if res.returncode != 0:
        msg = res.stderr.decode(errors="replace").strip().splitlines()
        raise ExtractionError(f"ffprobe failed: {msg[-1] if msg else 'exit ' + str(res.returncode)}")
    try:
        info = json.loads(res.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise ExtractionError("ffprobe returned invalid output") from e
    streams = info.get("streams", []) or []
    duration = _float((info.get("format") or {}).get("duration"))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    p = Probe(duration=duration, has_audio=has_audio)
    for s in streams:
        if s.get("codec_type") != "video" or (s.get("disposition") or {}).get("attached_pic"):
            continue
        w, h = int(s.get("width") or 0), int(s.get("height") or 0)
        if w <= 0 or h <= 0:
            continue
        sar = s.get("sample_aspect_ratio") or "1:1"
        try:
            num, den = (int(x) for x in sar.split(":"))
            if num > 0 and den > 0:
                w = max(1, round(w * num / den))
        except ValueError:
            pass
        if _rotation(s) in (90, 270):
            w, h = h, w
        p.width, p.height, p.video_index = w, h, int(s.get("index", 0))
        if p.duration is None:
            p.duration = _float(s.get("duration"))
        break
    return p


def fmt_time(seconds: float, long: bool) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if long else f"{m + 60 * h:02d}:{sec:02d}"


def time_range(start: float, end: float, total: float | None) -> str:
    long = max(end, total or 0) >= 3600
    return f"{fmt_time(start, long)}–{fmt_time(end, long)}"


class FFmpegPipe:
    """An ffmpeg process writing raw data to stdout; stderr goes to a temp file (no deadlock)."""

    def __init__(self, args: list[str]) -> None:
        self.stderr: IO[bytes] = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(
                [tool("ffmpeg"), *FFMPEG_FLAGS, *args],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.stderr,
                bufsize=0,
            )
        except BaseException:
            self.stderr.close()
            raise

    def read_exact(self, n: int) -> bytes:
        """Read up to n bytes, blocking until n are available or EOF."""
        assert self.proc.stdout is not None
        buf = bytearray()
        while len(buf) < n:
            part = self.proc.stdout.read(n - len(buf))
            if not part:
                break
            buf += part
        return bytes(buf)

    def finish(self) -> tuple[int, str]:
        """Wait for exit after EOF; return (exit code, last stderr line)."""
        rc = self.proc.wait()
        return rc, self.error_tail()

    def error_tail(self) -> str:
        try:
            self.stderr.seek(0, 2)
            size = self.stderr.tell()
            self.stderr.seek(max(0, size - 2000))
            lines = self.stderr.read().decode(errors="replace").strip().splitlines()
        except (OSError, ValueError):
            return ""
        return lines[-1] if lines else ""

    def close(self) -> None:
        """Kill the process if still running and reap it (no zombie)."""
        proc = self.proc
        if proc.poll() is None:
            proc.kill()
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        finally:
            proc.wait()
            self.stderr.close()


@contextmanager
def ffmpeg_pipe(args: list[str]) -> Iterator[FFmpegPipe]:
    pipe = FFmpegPipe(args)
    try:
        yield pipe
    finally:
        pipe.close()
