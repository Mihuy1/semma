"""EmbeddingGemma 2 wrapper backed by llama.cpp (Vulkan GPU / CPU) via ctypes + helper subprocess.

Produces L2-normalised float32 vectors (768-dim) matching google/embeddinggemma-2:
- Queries: prefixed with "task: search result | query: "
- Text documents: formatted as "title: <title> | text: <text>"
- Images: resized with fit_image_to_budget(..., budget=280) via PIL BICUBIC, fed to libmtmd
- Audio: 16 kHz mono float32 sliced to 30 s (480_000 samples), fed to libmtmd
- Video: up to 32 RGB frames resized with fit_image_to_budget(..., budget=140), fed to libmtmd

Each loaded model (text-only query worker or full multimodal index worker) lives in a spawned
helper subprocess so `unload()`, `unload_index()`, and `unload_query()` return 100% of host RAM
and Vulkan VRAM back to the OS.
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import functools
import hashlib
import logging
import math
import multiprocessing
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections import OrderedDict, deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import DEFAULT_BATCH_SIZES, DEFAULT_MODEL, xdg_data_home

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from .config import Config
    from .extract.base import Chunk

log = logging.getLogger(__name__)

# Guard against runaway chunks before tokenization.
MAX_TEXT_CHARS = 16_000
DEFAULT_MAX_TOKENS = 2048
EXPANDED_VIDEO_CTX = 4608
MAX_AUDIO_SAMPLES = 480_000  # 30 s at 16 kHz
MAX_VIDEO_FRAMES = 32

QUERY_PREFIX = "task: search result | query: "
QUERY_CACHE_SIZE = 256  # type-as-you-go sends the same prefixes again (backspace, re-typing)
# After an OOM the batch size of that modality stays reduced; it is doubled again (up to the
# configured size) only after this many successful batches in a row.
OOM_REGROW_AFTER = 200

GGUF_REVISION = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"
GGUF_BASE_URL = f"https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/{GGUF_REVISION}"
GGUF_SPECS: dict[str, tuple[str, int, str]] = {
    "text": (
        "embeddinggemma-2-Q8_0.gguf",
        309_855_456,
        "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135",
    ),
    "mmproj": (
        "mmproj-embeddinggemma-2-Q8_0.gguf",
        554_821_024,
        "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f",
    ),
}

_DOWNLOAD_LOCK = threading.Lock()
DOWNLOAD_CHUNK = 1024 * 1024
VULKAN_MODULE = "libsemma-ggml-vulkan.so"


class LlamaOOMError(RuntimeError):
    """Raised when the Vulkan / llama.cpp backend runs out of device or host memory."""

    is_oom = True


class LlamaWorkerCrashError(LlamaOOMError):
    """Raised when the llama.cpp worker process crashes mid-call (e.g. Vulkan OOM abort)."""


class EmbedderLike(Protocol):
    """What the indexer and search need; implemented by Embedder and by the tests' fake."""

    model_id: str
    dim: int

    def embed_query(self, query: str) -> np.ndarray: ...

    def embed_batch(self, modality: str, contents: Sequence[Any], titles: Sequence[str]) -> np.ndarray: ...


def format_document(text: str, title: str) -> str:
    title = title.replace("\n", " ").strip() or "none"
    return f"title: {title} | text: {text[:MAX_TEXT_CHARS]}"


def model_input(modality: str, content: Any, title: str) -> Any:
    """Turn a chunk's content into canonical input representation."""
    if modality == "text":
        return format_document(content, title)
    if modality == "image":
        return {"image": content.convert("RGB") if content.mode != "RGB" else content}
    if modality == "audio":
        return {"audio": np.ascontiguousarray(content, dtype=np.float32)}
    if modality == "video":
        return {"video": list(content)}
    raise ValueError(f"unknown modality {modality!r}")


def embed_chunks(embedder: EmbedderLike, chunks: Sequence[Chunk], title: str | Sequence[str]) -> np.ndarray:
    """Embed chunks of any modality mix, one model call per modality; rows follow input order."""
    titles = [title] * len(chunks) if isinstance(title, str) else list(title)
    out = np.empty((len(chunks), embedder.dim), dtype=np.float32)
    by_modality: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_modality.setdefault(c.modality, []).append(i)
    for modality, idx in by_modality.items():
        out[idx] = embedder.embed_batch(modality, [chunks[i].content for i in idx], [titles[i] for i in idx])
    return out


def fit_image_to_budget(
    width: int,
    height: int,
    patch_size: int = 16,
    pool: int = 3,
    budget: int = 280,
) -> tuple[int, int]:
    """Compute (target_width, target_height) matching Gemma4ImageProcessor aspect-ratio-preserving resize."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions {width}x{height}")
    max_patches = budget * (pool ** 2)
    total_px = width * height
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_h = factor * height
    ideal_w = factor * width
    side_mult = pool * patch_size

    target_h = int(math.floor(ideal_h / side_mult)) * side_mult
    target_w = int(math.floor(ideal_w / side_mult)) * side_mult

    max_side_length = (max_patches // (pool ** 2)) * side_mult
    if target_h == 0 and target_w == 0:
        target_h = side_mult
        target_w = side_mult
    elif target_h == 0:
        target_h = side_mult
        target_w = max(side_mult, min(int(math.floor(width / height)) * side_mult, max_side_length))
    elif target_w == 0:
        target_w = side_mult
        target_h = max(side_mult, min(int(math.floor(height / width)) * side_mult, max_side_length))

    return target_w, target_h


class GpuLock:
    """A lock where waiting queries go before waiting index batches.

    `threading.Lock` is unfair: a query arriving during a long video batch could lose the lock
    to the next batch again and again. Here a batch only takes the GPU when no query is
    waiting, so a query waits for at most the one batch already running.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._busy = False
        self._queries_waiting = 0

    @property
    def queries_waiting(self) -> int:
        return self._queries_waiting

    @contextmanager
    def query(self) -> Iterator[None]:
        with self._cv:
            self._queries_waiting += 1
            try:
                self._cv.wait_for(lambda: not self._busy)
            finally:
                self._queries_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def batch(self) -> Iterator[None]:
        with self._cv:
            self._cv.wait_for(lambda: not self._busy and self._queries_waiting == 0)
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise rows in float32."""
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1)


def quiet_model_loading() -> None:
    """Kept for backwards compatibility; llama.cpp logs are routed inside the worker."""
    return None


def resolve_model_dir(cfg: Config | None = None) -> Path:
    env = os.environ.get("SEMMA_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    if cfg is not None and getattr(cfg, "model_dir", None) is not None:
        return Path(cfg.model_dir)
    return xdg_data_home(Path.home()) / "semma" / "models"


def bundled_llama_lib_dir() -> Path | None:
    """The llama.cpp libraries an installed package carries beside its Python (/usr/lib/semma/llama),
    or those scripts/build-llama.sh built in a source checkout (build/llama/lib)."""
    candidates = (
        Path(sys.base_prefix).parent / "llama",
        Path(__file__).resolve().parents[2] / "build" / "llama" / "lib",
    )
    for candidate in candidates:
        if (candidate / "libllama.so").exists():
            return candidate
    return None


def resolve_llama_lib_dir(cfg: Config | None = None) -> Path:
    env = os.environ.get("SEMMA_LLAMA_LIB_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    if cfg is not None and getattr(cfg, "llama_lib_dir", None) is not None:
        return Path(cfg.llama_lib_dir)
    bundled = bundled_llama_lib_dir()
    if bundled is not None:
        return bundled
    return xdg_data_home(Path.home()) / "semma" / "lib"


def ensure_model_file(kind: str, model_dir: Path) -> Path:
    """Ensure a GGUF file exists in `model_dir` with the expected size, downloading it if missing.

    An interrupted download is kept as `<name>.part` and continued from where it stopped (an HTTP
    range request) the next time; the finished file is checked against its size and SHA-256 before it
    is moved into place. A lock file keeps other processes (the query and index workers, a second
    daemon) from writing the same partial file.
    """
    filename, expected_size, expected_sha256 = GGUF_SPECS[kind]
    dest = model_dir / filename
    if dest.is_file() and dest.stat().st_size == expected_size:
        return dest

    with _DOWNLOAD_LOCK:
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / f"{filename}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dest.is_file() and dest.stat().st_size == expected_size:
                return dest
            part = model_dir / f"{filename}.part"
            _download_resumable(f"{GGUF_BASE_URL}/{filename}", part, expected_size)

            hasher = hashlib.sha256()
            with part.open("rb") as fh:
                while chunk := fh.read(DOWNLOAD_CHUNK):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            total = part.stat().st_size
            if total < expected_size:
                # The server closed the connection early (urllib reports that as the end of the
                # data): the part is kept, and the next call continues after it.
                raise RuntimeError(f"download of {filename} stopped at {total} of {expected_size} bytes")
            if total != expected_size or digest != expected_sha256:
                part.unlink(missing_ok=True)
                raise RuntimeError(
                    f"downloaded {filename} failed verification: size={total} (expected {expected_size}), "
                    f"sha256={digest} (expected {expected_sha256})"
                )
            os.replace(part, dest)
            log.info("verified %s (sha256=%s)", filename, digest[:12])
            return dest


def _download_resumable(url: str, part: Path, expected_size: int) -> None:
    """Download `url` into `part`, continuing after the bytes `part` already holds.

    A failed transfer leaves `part` in place for the next call. A server that ignores the range
    request (200 instead of 206), answers with another range, or has a shorter file (416) restarts
    the file from zero.
    """
    have = part.stat().st_size if part.is_file() else 0
    if have > expected_size:
        part.unlink()
        have = 0
    if have == expected_size:
        return
    headers = {"User-Agent": "Semma/0.1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    mib = 1024 * 1024
    log.info("downloading %s (%d MiB%s) to %s", part.name.removesuffix(".part"), expected_size // mib,
             f", continuing at {have // mib} MiB" if have else "", part.parent)
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not have:
            raise
        # The part is longer than the server's file (left by another model revision): start again.
        part.unlink()
        return _download_resumable(url, part, expected_size)
    with resp:
        resumed = (have > 0 and resp.status == 206
                   and (resp.headers.get("Content-Range") or "").startswith(f"bytes {have}-"))
        if have and resp.status == 206 and not resumed:
            part.unlink()  # a range other than the one asked for
            return _download_resumable(url, part, expected_size)
        with part.open("ab" if resumed else "wb") as fh:
            while chunk := resp.read(DOWNLOAD_CHUNK):
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# llama.cpp / libmtmd ctypes definitions (verified against build b12347 ABI)
# ---------------------------------------------------------------------------

LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1
LLAMA_PROCESS_TYPE_DECODE = 1

MTMD_INPUT_CHUNK_TYPE_TEXT = 0
MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
MTMD_INPUT_CHUNK_TYPE_AUDIO = 2

_GGML_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class _LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("lazy_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class _LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("moe_cache_size", ctypes.c_size_t),
        ("moe_cache_cpu_share", ctypes.c_float),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class _LlamaEmbd(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("n_rows", ctypes.c_size_t),
        ("n_embd", ctypes.c_size_t),
    ]


class _MtmdContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("device", ctypes.c_void_p),
        ("print_timings", ctypes.c_bool),
        ("n_threads", ctypes.c_int),
        ("image_marker", ctypes.c_char_p),
        ("media_marker", ctypes.c_char_p),
        ("flash_attn_type", ctypes.c_int32),
        ("warmup", ctypes.c_bool),
        ("image_min_tokens", ctypes.c_int),
        ("image_max_tokens", ctypes.c_int),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("batch_max_tokens", ctypes.c_int32),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
    ]


class _MtmdInputText(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_size_t),
        ("add_special", ctypes.c_bool),
        ("parse_special", ctypes.c_bool),
    ]


class _MtmdDecoderPos(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_uint32),
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
        ("z", ctypes.c_uint32),
    ]


@functools.cache
def _system_has_vulkan() -> bool:
    """Whether the system's Vulkan loader is installed. The packages only recommend it: without it
    llama.cpp's Vulkan backend cannot be loaded, and "auto" means the processor."""
    try:
        ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return False
    return True


def _vulkan_module(lib_dir: Path) -> Path | None:
    """The Vulkan backend of a build with module backends. scripts/build-llama.sh gives it a name
    ggml does not look for when it loads its backends, so a worker that runs on the processor
    never touches Vulkan."""
    path = lib_dir / VULKAN_MODULE
    return path if path.exists() else None


def _backends_are_modules(lib_dir: Path) -> bool:
    """True for a llama.cpp build whose backends are loaded at run time (GGML_BACKEND_DL): the CPU
    backend then exists only as per-microarchitecture variants (libggml-cpu-haswell.so, ...)."""
    return not (lib_dir / "libggml-cpu.so").exists() and any(lib_dir.glob("libggml-cpu-*.so"))


def _load_shared_libs(
    lib_dir: Path,
    need_mtmd: bool,
    need_vulkan: bool = True,
) -> tuple[ctypes.CDLL, ctypes.CDLL, ctypes.CDLL | None]:
    # A build with ggml's backends as modules (GGML_BACKEND_DL, what the packages carry) has no
    # libggml-cpu.so: ggml picks the CPU variant for this machine itself (see _backends_are_modules).
    names = ["libggml-base.so"]
    if not _backends_are_modules(lib_dir):
        names.append("libggml-cpu.so")
        if need_vulkan and (lib_dir / "libggml-vulkan.so").exists():
            names.append("libggml-vulkan.so")
    names.extend(["libggml.so", "libllama.so"])
    if need_mtmd:
        names.append("libmtmd.so")

    loaded: dict[str, ctypes.CDLL] = {}
    for name in names:
        path = lib_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing llama.cpp shared library: {path}")
        loaded[name] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    ggml = loaded["libggml.so"]
    llama = loaded["libllama.so"]
    mtmd = loaded.get("libmtmd.so")

    if hasattr(ggml, "ggml_backend_load_all_from_path"):
        ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_load_all_from_path.restype = None
    if hasattr(ggml, "ggml_backend_load"):
        ggml.ggml_backend_load.argtypes = [ctypes.c_char_p]
        ggml.ggml_backend_load.restype = ctypes.c_void_p
    if hasattr(ggml, "ggml_backend_reg_count"):
        ggml.ggml_backend_reg_count.argtypes = []
        ggml.ggml_backend_reg_count.restype = ctypes.c_size_t

    _bind_llama_api(llama)
    if mtmd is not None:
        _bind_mtmd_api(mtmd)
    return ggml, llama, mtmd


def _bind_llama_api(llama: ctypes.CDLL) -> None:
    c_void_p = ctypes.c_void_p
    c_int32 = ctypes.c_int32
    c_uint32 = ctypes.c_uint32
    c_bool = ctypes.c_bool
    c_char_p = ctypes.c_char_p
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int32_p = ctypes.POINTER(ctypes.c_int32)

    llama.llama_log_set.argtypes = [_GGML_LOG_CALLBACK, c_void_p]
    llama.llama_log_set.restype = None

    llama.llama_backend_init.argtypes = []
    llama.llama_backend_init.restype = None
    llama.llama_backend_free.argtypes = []
    llama.llama_backend_free.restype = None
    llama.llama_supports_gpu_offload.argtypes = []
    llama.llama_supports_gpu_offload.restype = c_bool

    llama.llama_model_default_params.argtypes = []
    llama.llama_model_default_params.restype = _LlamaModelParams
    llama.llama_context_default_params.argtypes = []
    llama.llama_context_default_params.restype = _LlamaContextParams

    llama.llama_model_load_from_file.argtypes = [c_char_p, _LlamaModelParams]
    llama.llama_model_load_from_file.restype = c_void_p
    llama.llama_model_free.argtypes = [c_void_p]
    llama.llama_model_free.restype = None

    llama.llama_init_from_model.argtypes = [c_void_p, _LlamaContextParams]
    llama.llama_init_from_model.restype = c_void_p
    llama.llama_free.argtypes = [c_void_p]
    llama.llama_free.restype = None

    llama.llama_model_get_vocab.argtypes = [c_void_p]
    llama.llama_model_get_vocab.restype = c_void_p
    llama.llama_model_n_embd_inp.argtypes = [c_void_p]
    llama.llama_model_n_embd_inp.restype = c_int32
    llama.llama_model_n_embd_out.argtypes = [c_void_p]
    llama.llama_model_n_embd_out.restype = c_int32

    llama.llama_vocab_eos.argtypes = [c_void_p]
    llama.llama_vocab_eos.restype = c_int32

    llama.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, c_int32_p, c_int32, c_bool, c_bool]
    llama.llama_tokenize.restype = c_int32

    llama.llama_get_memory.argtypes = [c_void_p]
    llama.llama_get_memory.restype = c_void_p
    llama.llama_memory_clear.argtypes = [c_void_p, c_bool]
    llama.llama_memory_clear.restype = None

    llama.llama_batch_ext_init.argtypes = [c_void_p]
    llama.llama_batch_ext_init.restype = c_void_p
    llama.llama_batch_ext_free.argtypes = [c_void_p]
    llama.llama_batch_ext_free.restype = None
    llama.llama_batch_ext_clear.argtypes = [c_void_p]
    llama.llama_batch_ext_clear.restype = None
    llama.llama_batch_ext_add_token.argtypes = [c_void_p, c_int32, c_int32]
    llama.llama_batch_ext_add_token.restype = c_int32
    llama.llama_batch_ext_add_embd.argtypes = [c_void_p, c_int32, _LlamaEmbd]
    llama.llama_batch_ext_add_embd.restype = c_int32
    llama.llama_batch_ext_set_pos.argtypes = [c_void_p, c_int32, c_int32_p]
    llama.llama_batch_ext_set_pos.restype = c_bool
    llama.llama_batch_ext_set_output_embd.argtypes = [c_void_p, c_int32, c_bool]
    llama.llama_batch_ext_set_output_embd.restype = c_bool

    llama.llama_process.argtypes = [c_void_p, c_int32, c_void_p]
    llama.llama_process.restype = c_int32

    llama.llama_get_embeddings_seq.argtypes = [c_void_p, c_int32]
    llama.llama_get_embeddings_seq.restype = c_float_p
    llama.llama_n_ctx.argtypes = [c_void_p]
    llama.llama_n_ctx.restype = c_uint32


def _bind_mtmd_api(mtmd: ctypes.CDLL) -> None:
    c_void_p = ctypes.c_void_p
    c_int32 = ctypes.c_int32
    c_uint32 = ctypes.c_uint32
    c_size_t = ctypes.c_size_t
    c_bool = ctypes.c_bool
    c_char_p = ctypes.c_char_p
    c_float_p = ctypes.POINTER(ctypes.c_float)

    mtmd.mtmd_log_set.argtypes = [_GGML_LOG_CALLBACK, c_void_p]
    mtmd.mtmd_log_set.restype = None

    mtmd.mtmd_context_params_default.argtypes = []
    mtmd.mtmd_context_params_default.restype = _MtmdContextParams
    mtmd.mtmd_init_from_file.argtypes = [c_char_p, c_void_p, _MtmdContextParams]
    mtmd.mtmd_init_from_file.restype = c_void_p
    mtmd.mtmd_free.argtypes = [c_void_p]
    mtmd.mtmd_free.restype = None

    mtmd.mtmd_decode_use_mrope.argtypes = [c_void_p]
    mtmd.mtmd_decode_use_mrope.restype = c_bool

    mtmd.mtmd_bitmap_init.argtypes = [c_uint32, c_uint32, ctypes.POINTER(ctypes.c_ubyte)]
    mtmd.mtmd_bitmap_init.restype = c_void_p
    mtmd.mtmd_bitmap_init_from_audio.argtypes = [c_size_t, c_float_p]
    mtmd.mtmd_bitmap_init_from_audio.restype = c_void_p
    mtmd.mtmd_bitmap_free.argtypes = [c_void_p]
    mtmd.mtmd_bitmap_free.restype = None

    mtmd.mtmd_input_chunks_init.argtypes = []
    mtmd.mtmd_input_chunks_init.restype = c_void_p
    mtmd.mtmd_input_chunks_size.argtypes = [c_void_p]
    mtmd.mtmd_input_chunks_size.restype = c_size_t
    mtmd.mtmd_input_chunks_get.argtypes = [c_void_p, c_size_t]
    mtmd.mtmd_input_chunks_get.restype = c_void_p
    mtmd.mtmd_input_chunks_free.argtypes = [c_void_p]
    mtmd.mtmd_input_chunks_free.restype = None

    mtmd.mtmd_input_chunk_get_type.argtypes = [c_void_p]
    mtmd.mtmd_input_chunk_get_type.restype = c_int32
    mtmd.mtmd_input_chunk_get_tokens_text.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]
    mtmd.mtmd_input_chunk_get_tokens_text.restype = ctypes.POINTER(c_int32)
    mtmd.mtmd_input_chunk_get_tokens_image.argtypes = [c_void_p]
    mtmd.mtmd_input_chunk_get_tokens_image.restype = c_void_p
    mtmd.mtmd_input_chunk_get_n_tokens.argtypes = [c_void_p]
    mtmd.mtmd_input_chunk_get_n_tokens.restype = c_size_t
    mtmd.mtmd_input_chunk_get_n_pos.argtypes = [c_void_p]
    mtmd.mtmd_input_chunk_get_n_pos.restype = c_int32

    mtmd.mtmd_image_tokens_get_decoder_pos.argtypes = [c_void_p, c_int32, c_size_t]
    mtmd.mtmd_image_tokens_get_decoder_pos.restype = _MtmdDecoderPos

    mtmd.mtmd_tokenize.argtypes = [
        c_void_p,
        c_void_p,
        ctypes.POINTER(_MtmdInputText),
        ctypes.POINTER(c_void_p),
        c_size_t,
    ]
    mtmd.mtmd_tokenize.restype = c_int32

    mtmd.mtmd_encode_chunk.argtypes = [c_void_p, c_void_p]
    mtmd.mtmd_encode_chunk.restype = c_int32
    mtmd.mtmd_get_output_embd.argtypes = [c_void_p]
    mtmd.mtmd_get_output_embd.restype = c_float_p

    mtmd.mtmd_batch_init.argtypes = [c_void_p]
    mtmd.mtmd_batch_init.restype = c_void_p
    mtmd.mtmd_batch_free.argtypes = [c_void_p]
    mtmd.mtmd_batch_free.restype = None
    mtmd.mtmd_batch_add_chunk.argtypes = [c_void_p, c_void_p]
    mtmd.mtmd_batch_add_chunk.restype = c_int32
    mtmd.mtmd_batch_encode.argtypes = [c_void_p]
    mtmd.mtmd_batch_encode.restype = c_int32
    mtmd.mtmd_batch_get_output_embd.argtypes = [c_void_p, c_void_p]
    mtmd.mtmd_batch_get_output_embd.restype = c_float_p


# ---------------------------------------------------------------------------
# Subprocess worker implementation
# ---------------------------------------------------------------------------

def _looks_like_oom(msg: str) -> bool:
    low = msg.lower()
    return (
        "out of memory" in low
        or "memory allocation failed" in low
        or "vk::outofdevicememoryerror" in low
        or "vk::outofhostmemoryerror" in low
        or "failed to allocate" in low
    )


class _LlamaEngine:
    """In-worker llama.cpp + libmtmd inference engine."""

    def __init__(
        self,
        lib_dir: str,
        text_model_path: str,
        mmproj_path: str | None,
        use_gpu: bool,
        device_label: str | None = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.recent_logs: deque[str] = deque(maxlen=64)
        self._log_cb = _GGML_LOG_CALLBACK(self._on_log)
        self.ggml, self.llama, self.mtmd = _load_shared_libs(
            Path(lib_dir), need_mtmd=mmproj_path is not None, need_vulkan=use_gpu
        )
        self.llama.llama_log_set(self._log_cb, None)
        if self.mtmd is not None:
            self.mtmd.mtmd_log_set(self._log_cb, None)

        # Module backends are registered here: Vulkan only for a GPU worker (a missing or broken
        # libvulkan leaves that backend out), then the best CPU variant for this machine.
        modules = _backends_are_modules(Path(lib_dir))
        vulkan = _vulkan_module(Path(lib_dir))
        if use_gpu and modules and vulkan is not None:
            self.ggml.ggml_backend_load(str(vulkan).encode("utf-8"))
        if (use_gpu or modules) and hasattr(self.ggml, "ggml_backend_load_all_from_path"):
            self.ggml.ggml_backend_load_all_from_path(lib_dir.encode("utf-8"))
        self.llama.llama_backend_init()

        gpu_supported = bool(self.llama.llama_supports_gpu_offload())
        if use_gpu and not gpu_supported:
            if not allow_cpu_fallback:
                raise self._error(
                    f"requested GPU device {device_label or 'vulkan'!r} is unavailable (Vulkan GPU offload unsupported)"
                )
            self.use_gpu = False
            self.device = "cpu"
        else:
            self.use_gpu = bool(use_gpu and gpu_supported)
            self.device = (device_label or "vulkan") if self.use_gpu else "cpu"

        mparams = self.llama.llama_model_default_params()
        mparams.n_gpu_layers = -1 if self.use_gpu else 0
        mparams.split_mode = 0  # LLAMA_SPLIT_MODE_NONE: keep the entire model on one GPU
        mparams.main_gpu = 0
        self.model = self.llama.llama_model_load_from_file(text_model_path.encode("utf-8"), mparams)
        if not self.model:
            raise self._error("llama_model_load_from_file failed")

        self.vocab = self.llama.llama_model_get_vocab(self.model)
        self.eos_token = int(self.llama.llama_vocab_eos(self.vocab))
        self.n_embd_inp = int(self.llama.llama_model_n_embd_inp(self.model))
        self.n_embd_out = int(self.llama.llama_model_n_embd_out(self.model))

        self.n_seq_max = 16
        self.cur_ctx_size = 0
        self._ctx_target = 0
        self.lctx = None
        self.batch = None
        self._ensure_context(DEFAULT_MAX_TOKENS)

        self.mctx = None
        self.is_mrope = False
        if mmproj_path is not None:
            assert self.mtmd is not None
            mp = self.mtmd.mtmd_context_params_default()
            mp.use_gpu = self.use_gpu
            mp.print_timings = False
            mp.warmup = False
            mp.image_min_tokens = 1
            mp.image_max_tokens = 1120
            if not self.use_gpu:
                mp.n_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
            self.mctx = self.mtmd.mtmd_init_from_file(mmproj_path.encode("utf-8"), self.model, mp)
            if not self.mctx:
                raise self._error("mtmd_init_from_file failed")
            self.is_mrope = bool(self.mtmd.mtmd_decode_use_mrope(self.mctx))

    def _on_log(self, level: int, text: bytes | None, _user_data: Any) -> None:
        if text:
            msg = text.decode("utf-8", errors="replace").rstrip()
            if msg:
                self.recent_logs.append(msg)

    def _error(self, prefix: str) -> RuntimeError:
        tail = " | ".join(list(self.recent_logs)[-6:])
        msg = f"{prefix}: {tail}" if tail else prefix
        if _looks_like_oom(msg):
            return LlamaOOMError(msg)
        return RuntimeError(msg)

    def _ensure_context(self, min_tokens: int) -> None:
        target = DEFAULT_MAX_TOKENS if min_tokens <= DEFAULT_MAX_TOKENS else max(EXPANDED_VIDEO_CTX, min_tokens)
        if self.lctx is not None:
            if target == DEFAULT_MAX_TOKENS and self._ctx_target == DEFAULT_MAX_TOKENS:
                return
            if target > DEFAULT_MAX_TOKENS and min_tokens <= self._ctx_target:
                return
        if self.batch is not None:
            self.llama.llama_batch_ext_free(self.batch)
            self.batch = None
        if self.lctx is not None:
            self.llama.llama_free(self.lctx)
            self.lctx = None

        cp = self.llama.llama_context_default_params()
        cp.n_ctx = target
        cp.n_batch = target
        cp.n_ubatch = target
        cp.n_seq_max = self.n_seq_max
        cp.embeddings = True
        cp.pooling_type = LLAMA_POOLING_TYPE_MEAN
        cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL
        cp.offload_kqv = self.use_gpu
        cp.op_offload = self.use_gpu
        cp.no_perf = True
        if not self.use_gpu:
            threads = max(1, min(8, (os.cpu_count() or 4) // 2))
            cp.n_threads = threads
            cp.n_threads_batch = threads

        lctx = self.llama.llama_init_from_model(self.model, cp)
        if not lctx:
            raise self._error(f"llama_init_from_model(n_ctx={target}) failed")
        self.lctx = lctx
        self._ctx_target = target
        self.cur_ctx_size = int(self.llama.llama_n_ctx(lctx))
        self.batch = self.llama.llama_batch_ext_init(lctx)
        if not self.batch:
            raise self._error("llama_batch_ext_init failed")

    def _tokenize_text(self, text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> np.ndarray:
        raw = text.encode("utf-8")
        cap = max(64, len(raw) + 16)
        buf = (ctypes.c_int32 * cap)()
        n = int(self.llama.llama_tokenize(self.vocab, raw, len(raw), buf, cap, True, True))
        if n < 0:
            cap = -n
            buf = (ctypes.c_int32 * cap)()
            n = int(self.llama.llama_tokenize(self.vocab, raw, len(raw), buf, cap, True, True))
        if n <= 0:
            raise self._error(f"llama_tokenize returned {n}")
        toks = np.ctypeslib.as_array(buf)[:n].copy()
        if len(toks) > max_tokens:
            toks = toks[:max_tokens].copy()
            toks[-1] = self.eos_token
        return toks

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.n_embd_out), dtype=np.float32)

        self._ensure_context(DEFAULT_MAX_TOKENS)
        token_seqs = [self._tokenize_text(t, DEFAULT_MAX_TOKENS) for t in texts]
        out_rows: list[np.ndarray] = []

        i = 0
        while i < len(token_seqs):
            # Pack as many sequences as fit into _ctx_target (n_batch) and n_seq_max.
            group: list[np.ndarray] = []
            tok_sum = 0
            while i < len(token_seqs) and len(group) < self.n_seq_max:
                seq = token_seqs[i]
                if group and tok_sum + len(seq) > self._ctx_target:
                    break
                group.append(seq)
                tok_sum += len(seq)
                i += 1

            self.llama.llama_batch_ext_clear(self.batch)
            pos_val = ctypes.c_int32(0)
            for seq_id, seq in enumerate(group):
                for p, tok in enumerate(seq):
                    idx = int(self.llama.llama_batch_ext_add_token(self.batch, seq_id, int(tok)))
                    if idx < 0:
                        raise self._error(f"llama_batch_ext_add_token failed ({idx})")
                    pos_val.value = p
                    self.llama.llama_batch_ext_set_pos(self.batch, idx, ctypes.byref(pos_val))
                    self.llama.llama_batch_ext_set_output_embd(self.batch, idx, True)

            mem = self.llama.llama_get_memory(self.lctx)
            self.llama.llama_memory_clear(mem, True)
            rc = int(self.llama.llama_process(self.lctx, LLAMA_PROCESS_TYPE_DECODE, self.batch))
            if rc != 0:
                raise self._error(f"llama_process failed with rc={rc}")

            for seq_id in range(len(group)):
                ptr = self.llama.llama_get_embeddings_seq(self.lctx, seq_id)
                if not ptr:
                    raise self._error(f"llama_get_embeddings_seq({seq_id}) returned NULL")
                vec = np.ctypeslib.as_array(ptr, shape=(self.n_embd_out,)).copy()
                out_rows.append(vec)

        return _normalise(np.stack(out_rows, axis=0))

    def _encode_media_chunks(self, chunks_ptr: int, chunk_ptrs: list[int]) -> list[np.ndarray | None]:
        """Encode all non-text chunks using mtmd_batch_encode (splitting on batch_max_tokens)."""
        assert self.mtmd is not None
        encoded: list[np.ndarray | None] = [None] * len(chunk_ptrs)
        media_indices = [
            i for i, c in enumerate(chunk_ptrs)
            if int(self.mtmd.mtmd_input_chunk_get_type(c)) != MTMD_INPUT_CHUNK_TYPE_TEXT
        ]
        if not media_indices:
            return encoded

        pos = 0
        while pos < len(media_indices):
            mbatch = self.mtmd.mtmd_batch_init(self.mctx)
            if not mbatch:
                raise self._error("mtmd_batch_init failed")
            added: list[int] = []
            try:
                while pos < len(media_indices):
                    idx = media_indices[pos]
                    rc = int(self.mtmd.mtmd_batch_add_chunk(mbatch, chunk_ptrs[idx]))
                    if rc == 0:
                        added.append(idx)
                        pos += 1
                    elif rc in (2, 3) and added:
                        break
                    else:
                        raise self._error(f"mtmd_batch_add_chunk failed with rc={rc}")

                rc = int(self.mtmd.mtmd_batch_encode(mbatch))
                if rc != 0:
                    raise self._error(f"mtmd_batch_encode failed with rc={rc}")

                for idx in added:
                    c = chunk_ptrs[idx]
                    n_tok = int(self.mtmd.mtmd_input_chunk_get_n_tokens(c))
                    ptr = self.mtmd.mtmd_batch_get_output_embd(mbatch, c)
                    if not ptr:
                        raise self._error("mtmd_batch_get_output_embd returned NULL")
                    encoded[idx] = np.ctypeslib.as_array(ptr, shape=(n_tok, self.n_embd_inp)).copy()
            finally:
                self.mtmd.mtmd_batch_free(mbatch)

        return encoded

    def _require_mtmd(self) -> ctypes.CDLL:
        if self.mctx is None or self.mtmd is None:
            raise RuntimeError("multimodal projector is not loaded in this worker")
        return self.mtmd

    def encode_mtmd(self, prompt: str, bitmaps: list[int]) -> np.ndarray:
        self._require_mtmd()
        assert self.mtmd is not None

        chunks = self.mtmd.mtmd_input_chunks_init()
        if not chunks:
            raise self._error("mtmd_input_chunks_init failed")
        try:
            prompt_bytes = prompt.encode("utf-8")
            inp_txt = _MtmdInputText(prompt_bytes, len(prompt_bytes), True, True)
            bmp_arr = (ctypes.c_void_p * len(bitmaps))(*bitmaps) if bitmaps else None
            rc = int(self.mtmd.mtmd_tokenize(self.mctx, chunks, ctypes.byref(inp_txt), bmp_arr, len(bitmaps)))
            # chunks now owns preprocessed image/audio data; free bitmaps immediately.
            for b in bitmaps:
                self.mtmd.mtmd_bitmap_free(b)
            bitmaps.clear()
            if rc != 0:
                raise self._error(f"mtmd_tokenize failed with rc={rc}")

            n_chunks = int(self.mtmd.mtmd_input_chunks_size(chunks))
            chunk_ptrs = [self.mtmd.mtmd_input_chunks_get(chunks, i) for i in range(n_chunks)]
            total_tokens = sum(int(self.mtmd.mtmd_input_chunk_get_n_tokens(c)) for c in chunk_ptrs)
            self._ensure_context(total_tokens)

            encoded_chunks = self._encode_media_chunks(chunks, chunk_ptrs)

            self.llama.llama_batch_ext_clear(self.batch)
            pos_cur = 0
            n_tok_out = ctypes.c_size_t(0)
            pos_scalar = ctypes.c_int32(0)
            pos_mrope = (ctypes.c_int32 * 4)(0, 0, 0, 0)

            for i, c in enumerate(chunk_ptrs):
                ctype = int(self.mtmd.mtmd_input_chunk_get_type(c))
                if ctype == MTMD_INPUT_CHUNK_TYPE_TEXT:
                    tok_ptr = self.mtmd.mtmd_input_chunk_get_tokens_text(c, ctypes.byref(n_tok_out))
                    for j in range(int(n_tok_out.value)):
                        idx = int(self.llama.llama_batch_ext_add_token(self.batch, 0, int(tok_ptr[j])))
                        if idx < 0:
                            raise self._error(f"llama_batch_ext_add_token failed ({idx})")
                        pos_scalar.value = pos_cur
                        self.llama.llama_batch_ext_set_pos(self.batch, idx, ctypes.byref(pos_scalar))
                        self.llama.llama_batch_ext_set_output_embd(self.batch, idx, True)
                        pos_cur += 1
                else:
                    embd_arr = encoded_chunks[i]
                    assert embd_arr is not None
                    n_tok = embd_arr.shape[0]
                    img_tokens = (
                        self.mtmd.mtmd_input_chunk_get_tokens_image(c)
                        if (self.is_mrope and ctype == MTMD_INPUT_CHUNK_TYPE_IMAGE)
                        else None
                    )
                    for j in range(n_tok):
                        row_ptr = embd_arr[j].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        embd_struct = _LlamaEmbd(row_ptr, 1, self.n_embd_inp)
                        idx = int(self.llama.llama_batch_ext_add_embd(self.batch, 0, embd_struct))
                        if idx < 0:
                            raise self._error(f"llama_batch_ext_add_embd failed ({idx})")
                        if self.is_mrope:
                            if img_tokens:
                                dpos = self.mtmd.mtmd_image_tokens_get_decoder_pos(img_tokens, pos_cur, j)
                                pos_mrope[0], pos_mrope[1], pos_mrope[2], pos_mrope[3] = (
                                    dpos.t, dpos.y, dpos.x, dpos.z
                                )
                            else:
                                p = pos_cur + j
                                pos_mrope[0] = pos_mrope[1] = pos_mrope[2] = pos_mrope[3] = p
                            self.llama.llama_batch_ext_set_pos(self.batch, idx, pos_mrope)
                        else:
                            pos_scalar.value = pos_cur + j
                            self.llama.llama_batch_ext_set_pos(self.batch, idx, ctypes.byref(pos_scalar))
                        self.llama.llama_batch_ext_set_output_embd(self.batch, idx, True)
                    pos_cur += int(self.mtmd.mtmd_input_chunk_get_n_pos(c))

            mem = self.llama.llama_get_memory(self.lctx)
            self.llama.llama_memory_clear(mem, True)
            rc = int(self.llama.llama_process(self.lctx, LLAMA_PROCESS_TYPE_DECODE, self.batch))
            if rc != 0:
                raise self._error(f"llama_process failed with rc={rc}")

            ptr = self.llama.llama_get_embeddings_seq(self.lctx, 0)
            if not ptr:
                raise self._error("llama_get_embeddings_seq(0) returned NULL")
            vec = np.ctypeslib.as_array(ptr, shape=(self.n_embd_out,)).copy()
            return _normalise(vec)
        finally:
            for b in bitmaps:
                self.mtmd.mtmd_bitmap_free(b)
            self.mtmd.mtmd_input_chunks_free(chunks)

    def encode_image(self, width: int, height: int, rgb_bytes: bytes) -> np.ndarray:
        self._require_mtmd()
        assert self.mtmd is not None
        buf = (ctypes.c_ubyte * len(rgb_bytes)).from_buffer_copy(rgb_bytes)
        bmp = self.mtmd.mtmd_bitmap_init(width, height, buf)
        if not bmp:
            raise self._error("mtmd_bitmap_init failed")
        return self.encode_mtmd("<__media__>", [bmp])

    def encode_audio(self, pcm_bytes: bytes) -> np.ndarray:
        self._require_mtmd()
        assert self.mtmd is not None
        samples = np.frombuffer(pcm_bytes, dtype=np.float32)
        if len(samples) == 0:
            raise ValueError("empty audio buffer")
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        bmp = self.mtmd.mtmd_bitmap_init_from_audio(
            len(samples), samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if not bmp:
            raise self._error("mtmd_bitmap_init_from_audio failed")
        return self.encode_mtmd("<__media__>", [bmp])

    def encode_video(self, frames: list[tuple[int, int, bytes]]) -> np.ndarray:
        self._require_mtmd()
        assert self.mtmd is not None
        if not frames:
            raise ValueError("empty video frame list")
        bitmaps: list[int] = []
        try:
            for width, height, rgb_bytes in frames:
                buf = (ctypes.c_ubyte * len(rgb_bytes)).from_buffer_copy(rgb_bytes)
                bmp = self.mtmd.mtmd_bitmap_init(width, height, buf)
                if not bmp:
                    raise self._error("mtmd_bitmap_init failed for video frame")
                bitmaps.append(bmp)
            prompt = "<__media__>" * len(bitmaps)
            return self.encode_mtmd(prompt, bitmaps)
        except Exception:
            for b in bitmaps:
                self.mtmd.mtmd_bitmap_free(b)
            raise

    def close(self) -> None:
        if getattr(self, "mctx", None):
            assert self.mtmd is not None
            self.mtmd.mtmd_free(self.mctx)
            self.mctx = None
        if getattr(self, "batch", None):
            self.llama.llama_batch_ext_free(self.batch)
            self.batch = None
        if getattr(self, "lctx", None):
            self.llama.llama_free(self.lctx)
            self.lctx = None
        if getattr(self, "model", None):
            self.llama.llama_model_free(self.model)
            self.model = None
        if getattr(self, "mtmd", None) is not None:
            try:
                self.mtmd.mtmd_log_set(_GGML_LOG_CALLBACK(), None)
            except Exception:
                pass
        if getattr(self, "llama", None) is not None:
            try:
                self.llama.llama_log_set(_GGML_LOG_CALLBACK(), None)
            except Exception:
                pass
            self.llama.llama_backend_free()


def _worker_main(conn: Connection, init_cfg: dict[str, Any]) -> None:
    engine: _LlamaEngine | None = None
    try:
        if init_cfg.get("use_gpu") and init_cfg.get("gpu_index") is not None:
            if init_cfg.get("explicit_index") or not os.environ.get("GGML_VK_VISIBLE_DEVICES", "").strip():
                os.environ["GGML_VK_VISIBLE_DEVICES"] = str(init_cfg["gpu_index"])

        engine = _LlamaEngine(
            lib_dir=init_cfg["lib_dir"],
            text_model_path=init_cfg["text_model_path"],
            mmproj_path=init_cfg["mmproj_path"],
            use_gpu=init_cfg["use_gpu"],
            device_label=init_cfg.get("device_label"),
            allow_cpu_fallback=init_cfg.get("allow_cpu_fallback", True),
        )
        if init_cfg.get("warmup", True):
            engine.encode_texts([f"{QUERY_PREFIX}warm up"])
        conn.send(("ready", {"dim": engine.n_embd_out, "device": engine.device}))

        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            cmd, payload = msg
            if cmd == "shutdown":
                conn.send(("ok", None))
                break
            engine.recent_logs.clear()
            try:
                if cmd == "encode_texts":
                    res = engine.encode_texts(payload)
                elif cmd == "encode_mixed":
                    rows = []
                    for kind, data in payload:
                        if kind == "text":
                            rows.append(engine.encode_texts([data])[0])
                        elif kind == "image":
                            rows.append(engine.encode_image(*data))
                        elif kind == "audio":
                            rows.append(engine.encode_audio(data))
                        elif kind == "video":
                            rows.append(engine.encode_video(data))
                        else:
                            raise ValueError(f"unknown item kind {kind!r}")
                    res = np.stack(rows, axis=0) if rows else np.empty((0, engine.n_embd_out), dtype=np.float32)
                elif cmd == "get_ctx_size":
                    res = engine._ctx_target
                else:
                    raise ValueError(f"unknown worker command {cmd!r}")
                conn.send(("ok", res))
            except LlamaOOMError as exc:
                conn.send(("oom", str(exc)))
            except Exception as exc:
                conn.send(("err", f"{type(exc).__name__}: {exc}"))
    except LlamaOOMError as exc:
        try:
            conn.send(("oom", str(exc)))
        except Exception:
            pass
    except Exception as exc:
        try:
            conn.send(("err", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        if engine is not None:
            engine.close()
        try:
            conn.close()
        except Exception:
            pass


def _prepare_image(img: Any, budget: int) -> tuple[int, int, bytes]:
    from PIL import Image

    if img.mode != "RGB":
        img = img.convert("RGB")
    tw, th = fit_image_to_budget(img.width, img.height, patch_size=16, pool=3, budget=budget)
    if (tw, th) != img.size:
        img = img.resize((tw, th), Image.Resampling.BICUBIC)
    return img.width, img.height, img.tobytes()


def _prepare_audio(samples: Any) -> bytes:
    arr = np.ascontiguousarray(samples, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if len(arr) == 0:
        raise ValueError("empty audio buffer")
    arr = np.ascontiguousarray(arr[:MAX_AUDIO_SAMPLES], dtype=np.float32)
    return arr.tobytes()


def _prepare_video(frames: Sequence[Any]) -> list[tuple[int, int, bytes]]:
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("empty video frame list")
    if len(frame_list) > MAX_VIDEO_FRAMES:
        idx = np.linspace(0, len(frame_list) - 1, MAX_VIDEO_FRAMES, dtype=int)
        frame_list = [frame_list[int(i)] for i in idx]
    return [_prepare_image(f, budget=140) for f in frame_list]


class _WorkerHandle:
    """Parent-process handle to a spawned llama.cpp worker process."""

    def __init__(
        self,
        lib_dir: Path,
        text_model_path: Path,
        mmproj_path: Path | None,
        use_gpu: bool,
        gpu_index: int | None = None,
        explicit_index: bool = False,
        device_label: str | None = None,
        allow_cpu_fallback: bool = True,
        startup_timeout: float = 120.0,
        call_timeout: float = 300.0,
    ) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._call_timeout = float(call_timeout)
        mp = multiprocessing.get_context("spawn")
        parent_conn, child_conn = mp.Pipe(duplex=True)
        self._conn: Connection | None = parent_conn
        init_cfg = {
            "lib_dir": str(lib_dir),
            "text_model_path": str(text_model_path),
            "mmproj_path": str(mmproj_path) if mmproj_path is not None else None,
            "use_gpu": use_gpu,
            "gpu_index": gpu_index,
            "explicit_index": explicit_index,
            "device_label": device_label,
            "allow_cpu_fallback": allow_cpu_fallback,
            "warmup": True,
        }
        proc: BaseProcess = mp.Process(target=_worker_main, args=(child_conn, init_cfg), daemon=True)
        proc.start()
        child_conn.close()
        self._proc: BaseProcess | None = proc
        self._finalizer = weakref.finalize(self, _terminate_process, proc, parent_conn)

        try:
            if not parent_conn.poll(startup_timeout):
                raise TimeoutError(f"timed out after {startup_timeout:.0f}s")
            kind, info = parent_conn.recv()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"llama worker failed during startup: {exc}") from exc

        if kind == "oom":
            self.close()
            raise LlamaOOMError(str(info))
        if kind != "ready":
            self.close()
            raise RuntimeError(f"llama worker startup error: {info}")

        self.dim = int(info["dim"])
        self.device = str(info["device"])

    @property
    def closed(self) -> bool:
        return self._closed

    def is_alive(self) -> bool:
        return not self._closed and (self._proc is None or self._proc.is_alive())

    def _call(self, cmd: str, payload: Any) -> Any:
        with self._lock:
            if self._closed or self._conn is None:
                raise RuntimeError("llama worker is closed")
            try:
                self._conn.send((cmd, payload))
                if not self._conn.poll(self._call_timeout):
                    self.close()
                    raise RuntimeError(f"llama worker timed out during {cmd} after {self._call_timeout:.0f}s")
                kind, data = self._conn.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                proc = self._proc
                exitcode = None
                if proc is not None:
                    try:
                        proc.join(timeout=0.5)
                    except Exception:
                        pass
                    exitcode = proc.exitcode
                self.close()
                raise LlamaWorkerCrashError(f"llama worker crashed (exitcode={exitcode}): {exc}") from exc
            if kind == "ok":
                return data
            if kind == "oom":
                raise LlamaOOMError(str(data))
            raise RuntimeError(str(data))

    def ctx_size(self) -> int:
        return int(self._call("get_ctx_size", None))

    def encode_query(self, queries: Sequence[str], **_kwargs: Any) -> np.ndarray:
        texts = [f"{QUERY_PREFIX}{q}" for q in queries]
        return self._call("encode_texts", texts)

    def encode(self, inputs: Sequence[Any], **_kwargs: Any) -> np.ndarray:
        if all(isinstance(x, str) for x in inputs):
            return self._call("encode_texts", list(inputs))

        items: list[tuple[str, Any]] = []
        for item in inputs:
            if isinstance(item, str):
                items.append(("text", item))
            elif isinstance(item, dict) and "image" in item:
                items.append(("image", _prepare_image(item["image"], budget=280)))
            elif isinstance(item, dict) and "audio" in item:
                items.append(("audio", _prepare_audio(item["audio"])))
            elif isinstance(item, dict) and "video" in item:
                items.append(("video", _prepare_video(item["video"])))
            else:
                raise ValueError(f"unsupported model input: {type(item).__name__}")
        return self._call("encode_mixed", items)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        conn, self._conn = self._conn, None
        proc, self._proc = self._proc, None
        if conn is not None and proc is not None and proc.is_alive():
            try:
                conn.send(("shutdown", None))
                if conn.poll(timeout=1.0):
                    conn.recv()
            except Exception:
                pass
        _terminate_process(proc, conn)
        if hasattr(self, "_finalizer"):
            self._finalizer.detach()


def _terminate_process(proc: BaseProcess | None, conn: Connection | None) -> None:
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    if proc is not None:
        try:
            if proc.is_alive():
                proc.join(timeout=1.5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public Embedder wrapper
# ---------------------------------------------------------------------------

def _is_gpu_device(device: str | None) -> bool:
    if not device:
        return False
    return device.strip().lower() != "cpu"


def _parse_device_spec(device: str | None, *, has_vulkan_lib: bool) -> tuple[bool, int | None, str, bool, bool]:
    """Parse a device string into (use_gpu, gpu_index, normalized_label, explicit_index, allow_cpu_fallback)."""
    if device is None:
        if has_vulkan_lib:
            return (True, 0, "vulkan", False, True)
        return (False, None, "cpu", False, True)
    req = device.strip().lower()
    if req in ("", "auto"):
        if has_vulkan_lib:
            return (True, 0, "vulkan", False, True)
        return (False, None, "cpu", False, True)
    if req == "cpu":
        return (False, None, "cpu", False, True)
    if req in ("vulkan", "gpu", "cuda"):
        return (True, 0, "vulkan", False, False)
    for prefix in ("vulkan:", "gpu:", "cuda:"):
        if req.startswith(prefix):
            suffix = req[len(prefix):]
            if not suffix.isdigit():
                raise ValueError(f"invalid GPU device index in {device!r}")
            idx = int(suffix)
            return (True, idx, f"vulkan:{idx}", True, False)
    raise ValueError(f"unsupported device specification: {device!r}")


def _is_handle_alive(handle: Any) -> bool:
    if handle is None:
        return False
    if getattr(handle, "_closed", False) or getattr(handle, "closed", False):
        return False
    is_alive = getattr(handle, "is_alive", None)
    if callable(is_alive):
        return bool(is_alive())
    proc = getattr(handle, "_proc", None)
    if proc is not None:
        return bool(proc.is_alive())
    return True


def _cleanup_embedder_ref(ref: weakref.ref[Embedder]) -> None:
    inst = ref()
    if inst is not None:
        inst._atexit_cleanup()


class Embedder:
    """Lazily loaded, thread-safe model wrapper. A daemon can serve queries from one thread
    while another indexes: queries take priority over index batches (GpuLock), so a query waits
    for at most the batch already running.

    Queries are always text, so `embed_query` / `load_query` load a text-only worker
    (`embeddinggemma-2-Q8_0.gguf`, no mmproj) unless the full multimodal index worker
    (`_model`, with `mmproj-embeddinggemma-2-Q8_0.gguf`) is already loaded, in which case
    queries reuse it.

    Loading and unloading also hold the lock (GPU lock when on Vulkan/GPU, CPU query lock when
    `query_device == "cpu"`), so concurrent callers share a single load.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        model_name: str | None = None,
        device: str | None = None,
        query_device: str | None = None,
    ):
        self._cfg = cfg
        self.model_name = model_name or (cfg.model_name if cfg else DEFAULT_MODEL)
        self.model_id = self.model_name
        self.requested_device = device or (cfg.device if cfg else None)
        self.requested_query_device = query_device or (cfg.query_device if cfg else "auto")
        self.batch_sizes = dict(cfg.batch_sizes if cfg else DEFAULT_BATCH_SIZES)
        # Current batch size per modality: reduced after an OOM, regrown slowly.
        self._limits: dict[str, int] = {}
        self._ok_batches: dict[str, int] = {}
        self.dim = 768
        self.device: str | None = None
        self.query_device: str | None = None
        self.load_seconds = 0.0
        self.query_load_seconds = 0.0
        self.index_load_seconds = 0.0
        self.last_used = 0.0  # time.monotonic() of the last model call, load or unload
        self.query_last_used = 0.0
        self.index_last_used = 0.0
        self._loading = False
        self._query_loading = False
        self._index_loading = False
        self._loaded_once = False
        self._model: Any = None  # full multimodal index worker
        self._query_model: Any = None  # text-only query worker
        self._gpu = GpuLock()
        self._cpu_query_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = threading.Lock()
        self_ref = weakref.ref(self)
        atexit.register(_cleanup_embedder_ref, self_ref)

    def _atexit_cleanup(self) -> None:
        for handle in (self._model, self._query_model):
            if handle is not None and hasattr(handle, "close"):
                try:
                    handle.close()
                except Exception:
                    pass

    def worker_pids(self) -> set[int]:
        """PIDs of any currently running llama.cpp worker subprocesses."""
        pids: set[int] = set()
        for handle in (self._model, self._query_model):
            if handle is not None:
                proc = getattr(handle, "_proc", None)
                if proc is not None and proc.pid is not None:
                    pids.add(proc.pid)
        return pids

    def _drop_dead_index_locked(self) -> None:
        if self._model is not None and not _is_handle_alive(self._model):
            handle = self._model
            self._model = None
            if hasattr(handle, "close"):
                try:
                    handle.close()
                except Exception:
                    pass

    def _drop_dead_query_locked(self) -> None:
        if self._query_model is not None and not _is_handle_alive(self._query_model):
            handle = self._query_model
            self._query_model = None
            if hasattr(handle, "close"):
                try:
                    handle.close()
                except Exception:
                    pass

    # -- lifecycle --------------------------------------------------------------
    @property
    def index_model_loaded(self) -> bool:
        return _is_handle_alive(self._model)

    @property
    def query_model_loaded(self) -> bool:
        return _is_handle_alive(self._query_model)

    @property
    def loaded(self) -> bool:
        return _is_handle_alive(self._model) or _is_handle_alive(self._query_model)

    @property
    def query_ready(self) -> bool:
        return _is_handle_alive(self._model) or _is_handle_alive(self._query_model)

    @property
    def loading(self) -> bool:
        return self._loading or self._query_loading or self._index_loading

    def _lib_dir(self) -> Path:
        return resolve_llama_lib_dir(self._cfg)

    def _model_dir(self) -> Path:
        return resolve_model_dir(self._cfg)

    def _has_vulkan_lib(self) -> bool:
        lib_dir = self._lib_dir()
        found = (lib_dir / "libggml-vulkan.so").exists() or _vulkan_module(lib_dir) is not None
        return found and _system_has_vulkan()

    def _resolve_index_device_spec(self) -> tuple[bool, int | None, str, bool, bool]:
        return _parse_device_spec(self.requested_device, has_vulkan_lib=self._has_vulkan_lib())

    def _resolve_query_device_spec(self) -> tuple[bool, int | None, str, bool, bool]:
        req = (self.requested_query_device or "auto").strip().lower()
        if req in ("", "auto"):
            return self._resolve_index_device_spec()
        return _parse_device_spec(self.requested_query_device, has_vulkan_lib=self._has_vulkan_lib())

    def _resolve_index_device(self) -> str:
        return self._resolve_index_device_spec()[2]

    def _resolve_query_device(self) -> str:
        return self._resolve_query_device_spec()[2]

    @contextmanager
    def _query_lock(self, device: str | None = None) -> Iterator[None]:
        dev = device or self.query_device or self._resolve_query_device()
        if dev == "cpu":
            with self._cpu_query_lock:
                yield
        else:
            with self._gpu.query():
                yield

    def load(self) -> None:
        """Load the full multimodal index model."""
        with self._gpu.query():
            self._load_locked()

    def load_query(self) -> None:
        """Load the text-only query model (unless the full index model is already loaded)."""
        dev = self._resolve_query_device()
        if _is_handle_alive(self._query_model):
            return
        if dev != "cpu" and _is_handle_alive(self._model):
            return
        with self._query_lock(dev):
            self._load_query_locked(dev)

    def _load_locked(self) -> None:
        """Load the full model; the caller holds the GPU lock."""
        self._drop_dead_index_locked()
        if self._model is not None:
            return
        self._loading = True
        self._index_loading = True
        try:
            # Drop any GPU text-only query worker before loading the full worker so we don't
            # hold both in VRAM simultaneously.
            self._drop_dead_query_locked()
            if self._query_model is not None and _is_gpu_device(self.query_device):
                q_handle = self._query_model
                self._query_model = None
                if hasattr(q_handle, "close"):
                    q_handle.close()
            self._load_model()
        finally:
            self._loading = False
            self._index_loading = False
        now = time.monotonic()
        self.last_used = now
        self.index_last_used = now

    def _load_query_locked(self, device: str | None = None) -> None:
        """Load the text-only query model; the caller holds the appropriate query lock."""
        dev = device or self._resolve_query_device()
        self._drop_dead_query_locked()
        if dev != "cpu":
            self._drop_dead_index_locked()
            if self._model is not None:
                return
        if self._query_model is not None:
            return
        self._loading = True
        self._query_loading = True
        try:
            self._load_query_model(dev)
        finally:
            self._loading = False
            self._query_loading = False
        now = time.monotonic()
        self.last_used = now
        self.query_last_used = now

    def _spawn_worker(
        self,
        device_spec: tuple[bool, int | None, str, bool, bool],
        text_only: bool,
    ) -> _WorkerHandle:
        use_gpu, gpu_index, device_label, explicit_index, allow_cpu_fallback = device_spec
        model_dir = self._model_dir()
        text_path = ensure_model_file("text", model_dir)
        mmproj_path = None if text_only else ensure_model_file("mmproj", model_dir)
        worker = _WorkerHandle(
            lib_dir=self._lib_dir(),
            text_model_path=text_path,
            mmproj_path=mmproj_path,
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            explicit_index=explicit_index,
            device_label=device_label,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.dim = worker.dim
        self._loaded_once = True
        return worker

    def _load_model(self) -> None:
        started = time.monotonic()
        spec = self._resolve_index_device_spec()
        worker = self._spawn_worker(spec, text_only=False)
        self.device = worker.device
        self._model = worker
        self.index_load_seconds = time.monotonic() - started
        self.load_seconds = self.index_load_seconds
        log.info("loaded %s (full) on %s in %.1fs", self.model_name, self.device, self.index_load_seconds)

    def _load_query_model(self, device: str | None = None) -> None:
        started = time.monotonic()
        if device is None:
            spec = self._resolve_query_device_spec()
        else:
            spec = _parse_device_spec(device, has_vulkan_lib=self._has_vulkan_lib())
        worker = self._spawn_worker(spec, text_only=True)
        self.query_device = worker.device
        if self.device is None:
            self.device = worker.device
        self._query_model = worker
        self.query_load_seconds = time.monotonic() - started
        self.load_seconds = self.query_load_seconds
        log.info("loaded %s (text-only) on %s in %.1fs", self.model_name, self.query_device, self.query_load_seconds)

    def unload_index(self) -> bool:
        """Drop the full index worker and return its RAM/VRAM to the OS.
        Returns False if it was not loaded."""
        with self._gpu.query():
            if self._model is None:
                return False
            was_alive = _is_handle_alive(self._model)
            started = time.monotonic()
            handle = self._model
            self._model = None
            if hasattr(handle, "close"):
                handle.close()
            if not was_alive:
                return False
            now = time.monotonic()
            self.index_last_used = now
            self.last_used = now
            log.info("unloaded %s (full) in %.2fs", self.model_name, now - started)
            return True

    def unload_query(self) -> bool:
        """Drop the text-only query worker and free its RAM/VRAM.
        Returns False if it was not loaded."""
        dev = self.query_device or self._resolve_query_device()
        with self._query_lock(dev):
            if self._query_model is None:
                return False
            was_alive = _is_handle_alive(self._query_model)
            started = time.monotonic()
            handle = self._query_model
            self._query_model = None
            if hasattr(handle, "close"):
                handle.close()
            if not was_alive:
                return False
            now = time.monotonic()
            self.query_last_used = now
            self.last_used = now
            log.info("unloaded %s (text-only) in %.2fs", self.model_name, now - started)
            return True

    def unload(self) -> bool:
        """Drop any loaded workers (index and/or query) and return RAM/VRAM to the OS.
        Waits for a running batch or query; the next model call loads again.
        Returns False if nothing was loaded."""
        u_idx = self.unload_index()
        u_qry = self.unload_query()
        return u_idx or u_qry

    def release_cache(self) -> None:
        """No-op for llama.cpp workers (kept for API compatibility)."""
        with self._gpu.query():
            self._empty_cache()

    def _empty_cache(self, device: str | None = None) -> None:
        return None

    # -- encoding -------------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """Query vector (read-only array; repeated queries come from a small LRU cache)."""
        with self._query_cache_lock:
            hit = self._query_cache.get(query)
            if hit is not None:
                self._query_cache.move_to_end(query)
                return hit
        # The full index model is only touched under the GPU lock, a CPU query model only under
        # the CPU query lock (a GPU query model under the GPU lock). Models can be loaded or
        # dropped while we wait for a lock, so each attempt re-checks under it and retries.
        vec = None
        while vec is None:
            if not _is_handle_alive(self._query_model) and _is_handle_alive(self._model):
                with self._gpu.query():
                    self._drop_dead_index_locked()
                    if self._model is not None:
                        vec = self._run_query(self._model, query)
                continue
            dev = self.query_device or self._resolve_query_device()
            gpu_locked = dev != "cpu"
            with self._query_lock(dev):
                self._drop_dead_query_locked()
                if gpu_locked:
                    self._drop_dead_index_locked()
                if self._query_model is None and (self._model is None or not gpu_locked):
                    self._load_query_locked(dev)
                model = self._query_model
                if model is None and gpu_locked:
                    model = self._model
                if model is not None:
                    vec = self._run_query(model, query)
        out = _normalise(vec[0])
        out.flags.writeable = False
        with self._query_cache_lock:
            self._query_cache[query] = out
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return out

    def _run_query(self, model: Any, query: str) -> np.ndarray:
        """Encode one query; the caller holds the lock that guards `model`."""
        vec = model.encode_query([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        now = time.monotonic()
        self.last_used = now
        self.query_last_used = now
        return vec

    def embed_chunks(self, chunks: Sequence[Chunk], title: str | Sequence[str]) -> np.ndarray:
        return embed_chunks(self, chunks, title)

    def batch_limit(self, modality: str) -> int:
        return self._limits.get(modality, max(1, self.batch_sizes.get(modality, 8)))

    def embed_batch(self, modality: str, contents: Sequence[Any], titles: Sequence[str]) -> np.ndarray:
        """Embed single-modality contents in batches of the configured size.

        On an OOM the batch is halved and retried. The reduced size is remembered for
        the modality and only regrown after OOM_REGROW_AFTER clean batches.
        """
        inputs = [model_input(modality, c, t) for c, t in zip(contents, titles, strict=True)]
        parts: list[np.ndarray] = []
        i = 0
        while i < len(inputs):
            size = self.batch_limit(modality)
            batch = inputs[i : i + size]
            oom = False
            try:
                parts.append(self._encode(batch))
            except Exception as exc:
                if not _is_oom(exc) or len(batch) == 1:
                    raise
                oom = True
            if oom:
                del batch
                smaller = max(1, size // 2)
                self._limits[modality] = smaller
                self._ok_batches[modality] = 0
                log.warning("GPU out of memory on a %s batch of %d, continuing with %d", modality, size, smaller)
                self.release_cache()
                continue
            i += len(batch)
            self._regrow(modality, size)
        if not parts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _regrow(self, modality: str, size: int) -> None:
        if modality not in self._limits:
            return
        ok = self._ok_batches.get(modality, 0) + 1
        self._ok_batches[modality] = ok
        if ok >= OOM_REGROW_AFTER:
            configured = max(1, self.batch_sizes.get(modality, 8))
            grown = min(configured, size * 2)
            if grown >= configured:
                del self._limits[modality]
            else:
                self._limits[modality] = grown
            self._ok_batches[modality] = 0

    def _encode(self, inputs: list[Any]) -> np.ndarray:
        with self._gpu.batch():
            self._drop_dead_index_locked()
            self._load_locked()
            vecs = self._model.encode(
                inputs,
                prompt="",
                batch_size=len(inputs),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            now = time.monotonic()
            self.last_used = now
            self.index_last_used = now
        vecs = np.asarray(vecs, dtype=np.float32)
        if not np.isfinite(vecs).all():
            raise FloatingPointError("model produced non-finite embeddings")
        return _normalise(vecs)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, LlamaOOMError) or bool(getattr(exc, "is_oom", False))
