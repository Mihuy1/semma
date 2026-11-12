"""Audio/video extractors against small ffmpeg-generated files."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gemmasearch.config import Config
from gemmasearch.extract import ExtractionError, extract
from gemmasearch.extract.base import AUDIO_SAMPLE_RATE, MAX_VIDEO_FRAMES
from gemmasearch.extract.media import fmt_time, probe, time_range

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("media")
    files = {
        "av": d / "av.mp4",
        "silent_video": d / "noaudio.mkv",
        "rotated": d / "rotated.mp4",
        "audio": d / "tone.flac",
        "short": d / "blip.ogg",
        "audio_mp4": d / "voice.mp4",
        "corrupt": d / "corrupt.mp4",
    }
    vcodec = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=25", "-f", "lavfi", "-i", "sine=frequency=440",
           "-t", "130", *vcodec, "-c:a", "aac", str(files["av"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "70", *vcodec, str(files["silent_video"]))
    ffmpeg("-display_rotation:v:0", "90", "-i", str(files["silent_video"]), "-t", "5", "-c", "copy",
           str(files["rotated"]))
    # 0-60 s tone, 60-120 s digital silence, 120-121 s tone (a 1 s tail: dropped)
    ffmpeg("-f", "lavfi", "-i",
           "aevalsrc='if(lt(t,60)+gte(t,120),0.5*sin(2*PI*440*t),0)':s=22050:d=121",
           "-c:a", "flac", str(files["audio"]))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=880:duration=1", "-c:a", "libvorbis", str(files["short"]))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=330:duration=8", "-c:a", "aac", str(files["audio_mp4"]))
    files["corrupt"].write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(20000))
    return files


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(home=tmp_path, db_path=tmp_path / "index.db", roots=[])


def child_pids() -> set[int]:
    pids: set[int] = set()
    for task in Path("/proc/self/task").iterdir():
        try:
            pids |= {int(x) for x in (task / "children").read_text().split()}
        except OSError:
            pass
    return pids


def test_time_format():
    assert fmt_time(75, False) == "01:15" and fmt_time(3725, True) == "1:02:05"
    assert time_range(60, 120, 130) == "01:00–02:00"
    assert time_range(0, 60, 4000) == "0:00:00–0:01:00"


def test_probe(media):
    p = probe(media["av"])
    assert p.has_audio and p.has_video and (p.width, p.height) == (320, 240)
    assert p.duration == pytest.approx(130, abs=0.5)
    r = probe(media["rotated"])
    assert (r.width, r.height) == (240, 320) and not r.has_audio
    a = probe(media["audio"])
    assert a.has_audio and not a.has_video


def test_video_with_audio(media, cfg):
    chunks = list(extract(media["av"], "video", cfg))
    assert [(c.modality, c.locator) for c in chunks] == [
        ("video", "00:00–01:00"), ("audio", "00:00–01:00"),
        ("video", "01:00–02:00"), ("audio", "01:00–02:00"),
        ("video", "02:00–02:10"), ("audio", "02:00–02:10"),
    ]
    assert [c.position for c in chunks] == [0, 0, 60, 60, 120, 120]
    v = chunks[0].content
    assert isinstance(v, list) and len(v) == MAX_VIDEO_FRAMES
    assert all(f.mode == "RGB" and f.size == (320, 240) for f in v)
    assert len(chunks[4].content) in (5, 6)  # 10 s at 32 frames/60 s
    a = chunks[1].content
    assert a.dtype == np.float32 and a.ndim == 1 and a.shape[0] == 60 * AUDIO_SAMPLE_RATE
    assert 0.05 < float(np.sqrt(np.mean(a**2))) < 1.0  # lavfi sine: amplitude 1/8
    assert chunks[5].content.shape[0] == pytest.approx(10 * AUDIO_SAMPLE_RATE, abs=2000)


def test_video_frames_change_over_time(media, cfg):
    first = next(iter(extract(media["av"], "video", cfg)))
    frames = [np.asarray(f, dtype=np.int16) for f in first.content]
    assert np.abs(frames[0] - frames[-1]).mean() > 1  # testsrc has a moving counter/gradient


def test_video_without_audio_and_trailing_segment(media, cfg):
    chunks = list(extract(media["silent_video"], "video", cfg))
    assert [c.modality for c in chunks] == ["video", "video"]
    assert chunks[1].locator == "01:00–01:10" and len(chunks[1].content) in (5, 6)


def test_rotated_video_dimensions(media, cfg):
    (c,) = list(extract(media["rotated"], "video", cfg))
    assert c.content[0].size == (240, 320)


def test_large_video_downscaled(tmp_path, cfg):
    p = tmp_path / "hd.webm"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=5", "-t", "3",
           "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8", str(p))
    (c,) = list(extract(p, "video", cfg))
    assert c.content[0].size == (512, 288)


def test_audio_segments_silence_and_tail(media, cfg):
    chunks = list(extract(media["audio"], "audio", cfg))
    # segment 2 (60-120 s) is digital silence, segment 3 (120-121 s) is a 1 s tail
    assert [c.locator for c in chunks] == ["00:00–01:00"]
    assert chunks[0].modality == "audio" and chunks[0].content.shape == (60 * AUDIO_SAMPLE_RATE,)


def test_short_audio_kept_when_only_segment(media, cfg):
    (c,) = list(extract(media["short"], "audio", cfg))
    assert c.locator == "00:00–00:01" and c.position == 0
    assert c.content.shape[0] == pytest.approx(AUDIO_SAMPLE_RATE, abs=1500)


def test_audio_only_mp4_as_video_kind(media, cfg):
    chunks = list(extract(media["audio_mp4"], "video", cfg))
    assert [c.modality for c in chunks] == ["audio"]


def test_max_media_seconds_caps_output(media, cfg):
    cfg.max_media_seconds = 90
    chunks = list(extract(media["av"], "video", cfg))
    assert [(c.modality, c.locator) for c in chunks] == [
        ("video", "00:00–01:00"), ("audio", "00:00–01:00"),
        ("video", "01:00–01:30"), ("audio", "01:00–01:30"),
    ]
    assert len(chunks[2].content) == 16
    assert chunks[3].content.shape[0] == pytest.approx(30 * AUDIO_SAMPLE_RATE, abs=100)
    cfg.max_media_seconds = 30
    cfg.media_segment_seconds = 10
    audio = list(extract(media["audio"], "audio", cfg))
    assert [c.locator for c in audio] == ["00:00–00:10", "00:10–00:20", "00:20–00:30"]


def test_corrupt_media(media, cfg, tmp_path):
    for kind in ("video", "audio"):
        with pytest.raises(ExtractionError):
            list(extract(media["corrupt"], kind, cfg))
    with pytest.raises(ExtractionError, match="no audio"):
        list(extract(media["silent_video"], "audio", cfg))
    with pytest.raises(ExtractionError):
        list(extract(tmp_path / "missing.mp4", "video", cfg))


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_early_close_kills_ffmpeg(media, cfg, kind):
    before = child_pids()
    gen = extract(media["av"], kind, cfg)
    next(gen)
    running = child_pids() - before
    assert running, "ffmpeg should be running while the generator is suspended"
    gen.close()
    assert child_pids() - before == set()  # killed and reaped: no zombie left behind
    for pid in running:
        assert not Path(f"/proc/{pid}").exists()
