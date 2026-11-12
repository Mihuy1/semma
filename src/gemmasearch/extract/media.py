"""ffmpeg/ffprobe plumbing shared by the audio and video extractors.

Safety of a long-running daemon:
- ffmpeg/ffprobe may only open local files (``-protocol_whitelist file,pipe``), and files that
  are playlists/concat lists (demuxers that open other files) are refused after probing.
- Reads from ffmpeg have a watchdog: no output for STALL_TIMEOUT seconds (stalled FUSE/NFS read,
  decoder stuck on a pathological stream) or more time spent waiting for ffmpeg than
  max(MIN_BUDGET, BUDGET_PER_SECOND * media seconds decoded so far) kills the process and
  raises ExtractionError. Only time spent blocked on ffmpeg counts, not time the consumer
  spends with the generator suspended (e.g. embedding).
- ffmpeg runs in its own session, so a Ctrl-C/SIGINT for the daemon's process group does not
  make it exit early with a truncated (seemingly valid) output.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .base import ExtractionError

PROBE_TIMEOUT = 60  # s
STALL_TIMEOUT = 60.0  # s without any output from ffmpeg
MIN_BUDGET = 120.0  # s of waiting for ffmpeg always allowed per process...
BUDGET_PER_SECOND = 4.0  # ...or this many seconds per media second decoded, if more
EXIT_TIMEOUT = 10.0  # s to wait for ffmpeg to exit after it closed its output
# A non-zero ffmpeg exit is accepted (corrupt or truncated tail) only if the decoded output
# covers this fraction of the expected duration; otherwise the file fails instead of being
# indexed as if it were complete.
COMPLETE_RATIO = 0.9
FFMPEG_FLAGS = ["-nostdin", "-hide_banner", "-loglevel", "error"]
PROTOCOLS = ["-protocol_whitelist", "file,pipe"]
# Demuxers that read other files/URLs listed inside the file (playlists, concat lists).
REFERENCE_FORMATS = frozenset({"hls", "applehttp", "concat", "ffconcat", "dash", "imf", "tee"})


def input_args(path: Path) -> list[str]:
    """`-i` for a local file: local protocols only; "file:" so a ':' in the name is no protocol."""
    return [*PROTOCOLS, "-i", f"file:{path}"]


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
        tool("ffprobe"), "-v", "error", "-of", "json", *PROTOCOLS,
        "-show_entries",
        "format=duration,format_name:stream=index,codec_type,width,height,sample_aspect_ratio,duration"
        ":stream_tags=rotate:stream_side_data=rotation:stream_disposition=attached_pic",
        f"file:{path}",
    ]
    try:
        res = subprocess.run(
            cmd, capture_output=True, timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL, start_new_session=True
        )
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
    fmt = info.get("format") or {}
    if set(str(fmt.get("format_name", "")).split(",")) & REFERENCE_FORMATS:
        raise ExtractionError(f"refusing playlist/concat file (format {fmt.get('format_name')})")
    duration = _float(fmt.get("duration"))
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
    """An ffmpeg process writing raw data to stdout; stderr goes to a temp file (no deadlock).

    `bytes_per_second` (output bytes per media second) sets the decode-time budget; reads raise
    ExtractionError (and kill ffmpeg) when the watchdog fires (see the module docstring).
    """

    def __init__(self, args: list[str], bytes_per_second: float | None = None) -> None:
        self.bytes_per_second = bytes_per_second
        self.bytes_read = 0
        self.waited = 0.0  # seconds spent blocked waiting for output
        self.stderr: IO[bytes] = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(
                [tool("ffmpeg"), *FFMPEG_FLAGS, *args],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.stderr,
                bufsize=0, start_new_session=True,
            )
        except BaseException:
            self.stderr.close()
            raise
        assert self.proc.stdout is not None
        self.fd = self.proc.stdout.fileno()
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.fd, selectors.EVENT_READ)

    @property
    def decoded_seconds(self) -> float:
        return self.bytes_read / self.bytes_per_second if self.bytes_per_second else 0.0

    def budget(self) -> float:
        if not self.bytes_per_second:
            return float("inf")
        return max(MIN_BUDGET, BUDGET_PER_SECOND * self.decoded_seconds)

    def _timeout(self, why: str) -> ExtractionError:
        self.proc.kill()
        return ExtractionError(f"ffmpeg {why}; killed")

    def read_exact(self, n: int) -> bytes:
        """Read up to n bytes, blocking until n are available or EOF (or the watchdog fires)."""
        buf = bytearray()
        while len(buf) < n:
            t0 = time.monotonic()
            remaining = self.budget() - self.waited
            ready = self.sel.select(min(STALL_TIMEOUT, max(remaining, 0.0)))
            if ready:
                part = os.read(self.fd, min(n - len(buf), 1 << 20))
            self.waited += time.monotonic() - t0
            if not ready:
                if self.waited >= self.budget():
                    raise self._timeout(f"too slow ({self.waited:.0f} s for {self.decoded_seconds:.0f} s of media)")
                raise self._timeout(f"produced no output for {STALL_TIMEOUT:.0f} s")
            if not part:
                break
            buf += part
            self.bytes_read += len(part)
        return bytes(buf)

    def finish(self) -> tuple[int, str]:
        """Wait for exit after EOF; return (exit code, last stderr line)."""
        try:
            rc = self.proc.wait(timeout=EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise self._timeout(f"did not exit within {EXIT_TIMEOUT:.0f} s after its output ended") from None
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
            self.sel.close()
            if proc.stdout is not None:
                proc.stdout.close()
        finally:
            proc.wait()
            self.stderr.close()


def check_exit(rc: int, err: str, decoded: float, expected: float | None, what: str) -> None:
    """Raise unless ffmpeg succeeded or (exit != 0) decoded nearly all of `expected` seconds.

    A non-zero exit after a complete-looking output happens with corrupt/truncated tails; those
    files are kept if >= COMPLETE_RATIO of the expected duration was decoded. A process killed
    by a signal, a decode that stopped early, or an unknown expected duration is a failure, so a
    truncated decode is never indexed as if it covered the whole file.
    """
    if rc == 0:
        return
    if rc > 0 and expected and decoded >= COMPLETE_RATIO * expected:
        return
    reason = err or (f"killed by signal {-rc}" if rc < 0 else f"exit {rc}")
    if decoded > 0:
        span = f"{expected:.0f}" if expected else "?"
        reason += f" (after {decoded:.0f} of {span} s)"
    raise ExtractionError(f"ffmpeg could not decode {what}: {reason}")


@contextmanager
def ffmpeg_pipe(args: list[str], bytes_per_second: float | None = None) -> Iterator[FFmpegPipe]:
    pipe = FFmpegPipe(args, bytes_per_second)
    try:
        yield pipe
    finally:
        pipe.close()
