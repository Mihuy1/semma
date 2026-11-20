"""Thumbnails for KRunner search results (images and videos).

Looks up thumbnails in:
1. An in-memory LRU cache keyed by ``(path, mtime, size)``.
2. The freedesktop thumbnail cache (``$XDG_CACHE_HOME/thumbnails/{normal,large}/<md5(uri)>.png``),
   accepted only when its ``Thumb::MTime`` PNG text chunk matches ``int(mtime)``.
3. Our own on-disk cache (``$XDG_CACHE_HOME/semma/thumbnails/<sha1(path|mtime|size)>.png``).
4. On-the-fly generation: Pillow for images (with EXIF transpose and JPEG draft mode; HEIC/HEIF
   through ``ffmpeg``), or a single frame via ``ffmpeg`` for videos. Newly generated thumbnails are
   saved atomically to our own cache.

Returns the ``(iiibiiay)`` struct expected by KRunner's ``icon-data`` property:
``(width, height, rowstride, has_alpha, bits_per_sample, channels, raw_rgba_bytes)``.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import warnings

from . import filetypes as ft
from .extract.base import ExtractionError
from .extract.image import decode_heif, is_heif
from .extract.media import probe

log = logging.getLogger(__name__)

type IconData = tuple[int, int, int, bool, int, int, bytes]

DEFAULT_SIZE = 64
MAX_KRUNNER_SIDE = 2047
DEFAULT_LRU_CAPACITY = 256
FFMPEG_TIMEOUT = 10.0
MAX_IMAGE_PIXELS = 50_000_000

_pillow_initialised = False


def _ensure_pillow_init() -> None:
    global _pillow_initialised
    if _pillow_initialised:
        return
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    _pillow_initialised = True


def xdg_cache_home() -> Path:
    val = os.environ.get("XDG_CACHE_HOME")
    if val and os.path.isabs(val):
        return Path(val)
    return Path.home() / ".cache"


def freedesktop_md5(path: str | Path) -> str:
    uri = Path(path).as_uri()
    return hashlib.md5(uri.encode("utf-8")).hexdigest()


def own_cache_key(path: str, mtime: float, size: int) -> str:
    raw = f"{path}|{mtime!r}|{size}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha1(raw).hexdigest()


def _resize_to_fit(img, size: int):
    from PIL import Image

    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    longest = max(w, h)
    if longest != size:
        if longest > size:
            img = img.copy()
            img.thumbnail((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
        else:
            scale = size / longest
            new_w = max(1, min(size, round(w * scale)))
            new_h = max(1, min(size, round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def _image_to_icon_data(img, size: int) -> IconData | None:
    size = max(1, min(size, MAX_KRUNNER_SIDE))
    img = _resize_to_fit(img, size)
    if img is None:
        return None
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w <= 0 or h <= 0 or w > MAX_KRUNNER_SIDE or h > MAX_KRUNNER_SIDE:
        return None
    rowstride = w * 4
    data = rgba.tobytes()
    if len(data) != rowstride * h:
        return None
    return (w, h, rowstride, True, 8, 4, data)


def _load_freedesktop_thumb(cache_root: Path, path: str, mtime: float):
    from PIL import Image

    try:
        digest = freedesktop_md5(path)
    except ValueError:
        return None
    expected_mtime = str(int(mtime))
    thumbs_dir = cache_root / "thumbnails"
    for subdir in ("normal", "large"):
        candidate = thumbs_dir / subdir / f"{digest}.png"
        if not candidate.is_file():
            continue
        try:
            with Image.open(candidate) as img:
                thumb_mtime = img.info.get("Thumb::MTime")
                if thumb_mtime is None or str(thumb_mtime).strip() != expected_mtime:
                    continue
                img.load()
                return img.copy()
        except Exception:
            continue
    return None


def _load_own_cache_thumb(own_dir: Path, path: str, mtime: float, size: int):
    from PIL import Image

    candidate = own_dir / f"{own_cache_key(path, mtime, size)}.png"
    if not candidate.is_file():
        return None
    try:
        with Image.open(candidate) as img:
            img.load()
            return img.copy()
    except Exception:
        return None


def _save_atomic_png(img, target: Path) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".thumb-", suffix=".png", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                img.save(fh, format="PNG")
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as e:
        log.debug("failed to cache thumbnail %s: %s", target, e)


def _generate_image_thumb(path: Path, size: int):
    _ensure_pillow_init()
    from PIL import Image, ImageOps

    if is_heif(path):
        try:
            img = decode_heif(path, max(size * 2, 128), timeout=FFMPEG_TIMEOUT).convert("RGBA")
        except ExtractionError as e:
            log.debug("cannot generate image thumbnail for %s: %s", path, e)
            return None
        img.thumbnail((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
        return img
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                img.seek(0)
                if img.format == "JPEG":
                    draft_side = max(size * 2, 128)
                    img.draft("RGB", (draft_side, draft_side))
                elif img.format == "JPEG2000":
                    factor = 0
                    while max(img.size) >> (factor + 1) >= size * 2 and factor < 5:
                        factor += 1
                    if factor:
                        img.reduce = factor
                img.load()
                img = ImageOps.exif_transpose(img)
                if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
                    import numpy as np

                    a = np.asarray(img, dtype=np.float32)
                    lo, hi = float(a.min()), float(a.max())
                    a = (a - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(a)
                    img = Image.fromarray(a.astype(np.uint8), "L")
                img = img.convert("RGBA")
                img.thumbnail((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
                return img
    except Exception as e:
        log.debug("cannot generate image thumbnail for %s: %s", path, e)
        return None


def _video_seek_seconds(path: Path) -> float:
    try:
        info = probe(path)
        if info.duration and info.duration > 0:
            return min(info.duration * 0.1, max(0.0, info.duration - 0.05))
    except Exception:
        pass
    return 1.0


def _run_ffmpeg_frame(ffmpeg_bin: str, path: Path, seek: float, size: int) -> bytes | None:
    vf = f"scale={size}:{size}:force_original_aspect_ratio=decrease,setsar=1"
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-ss",
        f"{seek:.3f}",
        "-i",
        f"file:{path}",
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        log.debug("ffmpeg thumbnail timed out for %s", path)
        return None
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0 or not stdout:
        return None
    return stdout


def _generate_video_thumb(path: Path, size: int):
    _ensure_pillow_init()
    from PIL import Image

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None
    try:
        seek = _video_seek_seconds(path)
        png_bytes = _run_ffmpeg_frame(ffmpeg_bin, path, seek, size)
        if not png_bytes and seek > 0.0:
            png_bytes = _run_ffmpeg_frame(ffmpeg_bin, path, 0.0, size)
        if not png_bytes:
            return None
        with Image.open(io.BytesIO(png_bytes)) as img:
            img.load()
            rgba = img.convert("RGBA")
            rgba.thumbnail((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
            return rgba
    except Exception as e:
        log.debug("cannot generate video thumbnail for %s: %s", path, e)
        return None


def make_icon_data(
    path: str | Path,
    kind: str,
    mtime: float,
    size: int = DEFAULT_SIZE,
    *,
    cache_dir: Path | None = None,
) -> IconData | None:
    """Synchronous core thumbnail lookup/generation function.

    Checks the freedesktop thumbnail cache, then our own cache, and otherwise generates a
    thumbnail (saving it atomically to our own cache) and returns KRunner ``icon-data``.
    """
    if kind not in (ft.IMAGE, ft.VIDEO):
        return None
    size = max(1, min(int(size), MAX_KRUNNER_SIDE))
    p = Path(path)
    if not p.is_absolute():
        p = p.resolve()
    path_str = str(p)
    root = cache_dir if cache_dir is not None else xdg_cache_home()
    own_dir = root / "semma" / "thumbnails"

    img = _load_freedesktop_thumb(root, path_str, mtime)
    if img is not None:
        return _image_to_icon_data(img, size)

    img = _load_own_cache_thumb(own_dir, path_str, mtime, size)
    if img is not None:
        return _image_to_icon_data(img, size)

    if kind == ft.IMAGE:
        img = _generate_image_thumb(p, size)
    else:
        img = _generate_video_thumb(p, size)
    if img is None:
        return None

    icon = _image_to_icon_data(img, size)
    if icon is None:
        return None

    from PIL import Image

    w, h, _, _, _, _, raw = icon
    cached_img = Image.frombytes("RGBA", (w, h), raw)
    target = own_dir / f"{own_cache_key(path_str, mtime, size)}.png"
    _save_atomic_png(cached_img, target)
    return icon


class Thumbnailer:
    """Async/sync thumbnail provider with in-memory LRU cache and deduplicated background work."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        lru_capacity: int = DEFAULT_LRU_CAPACITY,
        max_workers: int = 4,
    ) -> None:
        self._cache_dir = cache_dir
        self._lru_capacity = max(1, lru_capacity)
        self._lru: OrderedDict[tuple[str, int, int], IconData] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="semma-thumb")
        self._inflight: dict[tuple[str, int, int], asyncio.Future[IconData | None]] = {}

    def _lru_key(self, path: str, mtime: float, size: int) -> tuple[str, int, int]:
        return (path, int(mtime * 1_000_000), size)

    def get_cached(self, path: str, mtime: float, size: int = DEFAULT_SIZE) -> IconData | None:
        key = self._lru_key(path, mtime, size)
        with self._lock:
            val = self._lru.get(key)
            if val is not None:
                self._lru.move_to_end(key)
            return val

    def _put_cached(self, key: tuple[str, int, int], val: IconData) -> None:
        with self._lock:
            self._lru[key] = val
            self._lru.move_to_end(key)
            while len(self._lru) > self._lru_capacity:
                self._lru.popitem(last=False)

    def icon_data_sync(self, path: str, kind: str, mtime: float, size: int = DEFAULT_SIZE) -> IconData | None:
        """Synchronous lookup/generation (checks in-memory LRU first, then disk caches, then generates)."""
        if kind not in (ft.IMAGE, ft.VIDEO):
            return None
        size = max(1, min(int(size), MAX_KRUNNER_SIDE))
        key = self._lru_key(path, mtime, size)
        with self._lock:
            cached = self._lru.get(key)
            if cached is not None:
                self._lru.move_to_end(key)
                return cached
        res = make_icon_data(path, kind, mtime, size, cache_dir=self._cache_dir)
        if res is not None:
            self._put_cached(key, res)
        return res

    async def icon_data(self, path: str, kind: str, mtime: float, size: int = DEFAULT_SIZE) -> IconData | None:
        """Async lookup/generation.

        If shielded or awaited with a timeout that expires, the underlying thread-pool task
        continues and populates the LRU and disk cache for subsequent calls.
        """
        if kind not in (ft.IMAGE, ft.VIDEO):
            return None
        size = max(1, min(int(size), MAX_KRUNNER_SIDE))
        key = self._lru_key(path, mtime, size)
        with self._lock:
            cached = self._lru.get(key)
            if cached is not None:
                self._lru.move_to_end(key)
                return cached

        loop = asyncio.get_running_loop()
        fut = self._inflight.get(key)
        if fut is None:
            raw_fut = loop.run_in_executor(self._executor, self.icon_data_sync, path, kind, mtime, size)
            fut = asyncio.ensure_future(raw_fut)
            self._inflight[key] = fut

            def _cleanup(_f: asyncio.Future[IconData | None]) -> None:
                self._inflight.pop(key, None)

            fut.add_done_callback(_cleanup)

        return await asyncio.shield(fut)
