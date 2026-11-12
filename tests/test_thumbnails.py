"""Tests for ``gemmasearch.thumbnails``."""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import time

from PIL import Image, PngImagePlugin
import pytest

from gemmasearch import filetypes as ft
from gemmasearch.thumbnails import (
    Thumbnailer,
    freedesktop_md5,
    make_icon_data,
    own_cache_key,
)


def _check_icon_data(icon, expected_longest: int = 64) -> None:
    assert icon is not None
    w, h, rowstride, has_alpha, bps, channels, raw = icon
    assert isinstance(w, int) and isinstance(h, int)
    assert 0 < w < 2048 and 0 < h < 2048
    assert max(w, h) == expected_longest
    assert rowstride == w * 4
    assert has_alpha is True
    assert bps == 8
    assert channels == 4
    assert isinstance(raw, bytes)
    assert len(raw) == rowstride * h


def test_non_media_kinds_return_none(tmp_path: Path) -> None:
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF-1.4")
    for kind in (ft.TEXT, ft.PDF, ft.OFFICE, ft.AUDIO):
        assert make_icon_data(doc, kind, 1000.0, cache_dir=tmp_path) is None


def test_freedesktop_cache_hit_and_mtime_mismatch(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    normal_dir = cache_dir / "thumbnails" / "normal"
    normal_dir.mkdir(parents=True)

    # Underlying file does not even need to be decoded if freedesktop cache hits!
    target = tmp_path / "photo with spaces.jpg"
    target.write_bytes(b"not-a-valid-jpeg-so-generation-would-fail")
    mtime = 1730000123.75

    digest = freedesktop_md5(target)
    fd_png = normal_dir / f"{digest}.png"

    # 1. Mismatching Thumb::MTime -> ignored, falls through to generation (which returns None).
    img = Image.new("RGBA", (128, 64), (10, 200, 30, 255))
    meta_wrong = PngImagePlugin.PngInfo()
    meta_wrong.add_text("Thumb::MTime", str(int(mtime) - 1))
    meta_wrong.add_text("Thumb::URI", target.as_uri())
    img.save(fd_png, format="PNG", pnginfo=meta_wrong)

    assert make_icon_data(target, ft.IMAGE, mtime, size=64, cache_dir=cache_dir) is None

    # 2. Matching Thumb::MTime -> hits freedesktop cache and returns scaled RGBA icon-data.
    meta_ok = PngImagePlugin.PngInfo()
    meta_ok.add_text("Thumb::MTime", str(int(mtime)))
    meta_ok.add_text("Thumb::URI", target.as_uri())
    img.save(fd_png, format="PNG", pnginfo=meta_ok)

    icon = make_icon_data(target, ft.IMAGE, mtime, size=64, cache_dir=cache_dir)
    _check_icon_data(icon, expected_longest=64)
    w, h, _, _, _, _, raw = icon
    assert (w, h) == (64, 32)
    # First pixel should be green-ish (10, 200, 30, 255)
    assert list(raw[:4]) == [10, 200, 30, 255]


def test_own_cache_and_exif_rotated_jpeg_generation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    jpg_path = tmp_path / "rotated.jpg"

    # Create a 200x100 image (wider than tall) with EXIF Orientation=6 (Rotate 90 CW -> 100x200).
    raw_img = Image.new("RGB", (200, 100), (220, 40, 50))
    exif = raw_img.getexif()
    exif[0x0112] = 6  # Orientation = 6 (90 CW)
    raw_img.save(jpg_path, format="JPEG", exif=exif.tobytes())
    mtime = jpg_path.stat().st_mtime

    icon = make_icon_data(jpg_path, ft.IMAGE, mtime, size=64, cache_dir=cache_dir)
    _check_icon_data(icon, expected_longest=64)
    w, h, _, _, _, _, _ = icon
    # Because of EXIF rotation 6, the resulting thumbnail must be taller than wide (32x64).
    assert (w, h) == (32, 64)

    # Check that our own cache file was written atomically.
    own_file = cache_dir / "gemmasearch" / "thumbnails" / f"{own_cache_key(str(jpg_path), mtime, 64)}.png"
    assert own_file.is_file()

    # Delete original image: second call with same path+mtime+size must hit our own cache!
    jpg_path.unlink()
    icon_cached = make_icon_data(jpg_path, ft.IMAGE, mtime, size=64, cache_dir=cache_dir)
    _check_icon_data(icon_cached, expected_longest=64)
    assert icon_cached[:6] == icon[:6]


def test_png_generation_and_lru(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    png_path = tmp_path / "sample.png"
    img = Image.new("RGBA", (300, 150), (10, 20, 30, 128))
    img.save(png_path, format="PNG")
    mtime = png_path.stat().st_mtime

    thumb = Thumbnailer(cache_dir=cache_dir, lru_capacity=4)
    assert thumb.get_cached(str(png_path), mtime, 64) is None

    icon = thumb.icon_data_sync(str(png_path), ft.IMAGE, mtime, 64)
    _check_icon_data(icon, expected_longest=64)
    assert (icon[0], icon[1]) == (64, 32)

    # Now in memory LRU
    assert thumb.get_cached(str(png_path), mtime, 64) == icon


def test_video_thumbnail_via_ffmpeg(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    vid_path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(vid_path),
        ],
        check=True,
    )
    mtime = vid_path.stat().st_mtime

    thumb = Thumbnailer(cache_dir=cache_dir)
    icon = asyncio.run(thumb.icon_data(str(vid_path), ft.VIDEO, mtime, size=64))
    _check_icon_data(icon, expected_longest=64)
    w, h = icon[0], icon[1]
    assert (w, h) == (64, 48)

    # Verify own cache was populated for video as well
    own_file = cache_dir / "gemmasearch" / "thumbnails" / f"{own_cache_key(str(vid_path), mtime, 64)}.png"
    assert own_file.is_file()


def test_slow_thumbnail_continues_in_background_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    img_path = tmp_path / "slow.png"
    Image.new("RGB", (128, 128), (100, 150, 200)).save(img_path, format="PNG")
    mtime = img_path.stat().st_mtime

    orig_make = make_icon_data
    calls = 0

    def slow_make(path, kind, mt, size=64, *, cache_dir=None):
        nonlocal calls
        calls += 1
        time.sleep(0.12)
        return orig_make(path, kind, mt, size, cache_dir=cache_dir)

    monkeypatch.setattr("gemmasearch.thumbnails.make_icon_data", slow_make)
    thumb = Thumbnailer(cache_dir=cache_dir)

    async def scenario() -> None:
        # Wait with a tight budget (20 ms): should time out, but generation continues in background.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(thumb.icon_data(str(img_path), ft.IMAGE, mtime), timeout=0.02)

        # Wait enough for background thread to finish.
        await asyncio.sleep(0.15)
        assert thumb.get_cached(str(img_path), mtime) is not None

        # Second call returns immediately from LRU without invoking make_icon_data again.
        t0 = time.perf_counter()
        icon = await thumb.icon_data(str(img_path), ft.IMAGE, mtime)
        dt = time.perf_counter() - t0
        _check_icon_data(icon, expected_longest=64)
        assert dt < 0.02
        assert calls == 1

    asyncio.run(scenario())
