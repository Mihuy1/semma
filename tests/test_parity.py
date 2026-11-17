"""Parity, ABI, and worker-lifecycle tests for the llama.cpp Embedder backend."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semma.config import load_config
from semma.embedder import (
    DEFAULT_MAX_TOKENS,
    GGUF_SPECS,
    Embedder,
    LlamaWorkerCrashError,
    _is_oom,
    _LlamaContextParams,
    _LlamaEmbd,
    _LlamaModelParams,
    _MtmdContextParams,
    _MtmdDecoderPos,
    _MtmdInputText,
    _parse_device_spec,
    _prepare_audio,
    _prepare_video,
    ensure_model_file,
    fit_image_to_budget,
    resolve_llama_lib_dir,
    resolve_model_dir,
)

REFERENCE_NPZ = Path(__file__).parent / "data" / "reference_bf16.npz"

QUERIES = [
    "my CV",
    "how do I bake sourdough bread",
    "koulujen energiankulutus",
    "a red square",
]
DOCS_TEXT = [
    "The cat sat on the mat.",
    "Feed the sourdough starter, knead the dough and bake the loaf in a hot oven.",
    "Opinnäytetyössä tutkitaan koulujen energiankulutusta.",
]
DOCS_TITLES = ["none", "a.md", "c.md"]


def _synthetic_images() -> list[Image.Image]:
    img1 = Image.new("RGB", (64, 64), (255, 0, 0))

    xs2 = np.arange(300, dtype=np.int32)[None, :]
    ys2 = np.arange(200, dtype=np.int32)[:, None]
    arr2 = np.empty((200, 300, 3), dtype=np.uint8)
    arr2[:, :, 0] = (xs2 * 255 // 300).astype(np.uint8)
    arr2[:, :, 1] = (ys2 * 255 // 200).astype(np.uint8)
    arr2[:, :, 2] = 128
    img2 = Image.fromarray(arr2, mode="RGB")

    xs3 = np.arange(800, dtype=np.int32)[None, :]
    ys3 = np.arange(450, dtype=np.int32)[:, None]
    arr3 = np.empty((450, 800, 3), dtype=np.uint8)
    arr3[:, :, 0] = ((((xs3 // 32) + (ys3 // 32)) % 2) * 200 + 20).astype(np.uint8)
    arr3[:, :, 1] = (xs3 * 255 // 800).astype(np.uint8)
    arr3[:, :, 2] = (ys3 * 255 // 450).astype(np.uint8)
    img3 = Image.fromarray(arr3, mode="RGB")

    return [img1, img2, img3]


def _synthetic_audios() -> list[np.ndarray]:
    t1 = np.arange(16_000, dtype=np.float64) / 16_000.0
    a1 = (0.5 * np.sin(2.0 * np.pi * 440.0 * t1)).astype(np.float32)

    t5 = np.arange(5 * 16_000, dtype=np.float64) / 16_000.0
    a2 = (0.3 * np.sin(2.0 * np.pi * 440.0 * t5) + 0.2 * np.sin(2.0 * np.pi * 880.0 * t5)).astype(np.float32)
    return [a1, a2]


def _synthetic_video(n_frames: int = 8) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for k in range(n_frames):
        arr = np.zeros((120, 160, 3), dtype=np.uint8)
        arr[:, :, 2] = (k * 30) % 256
        y0 = 15 + 10 * (k % 8)
        x0 = 10 + 15 * (k % 8)
        arr[y0 : y0 + 25, x0 : x0 + 25, 0] = 255
        arr[y0 : y0 + 25, x0 : x0 + 25, 1] = 200
        frames.append(Image.fromarray(arr, mode="RGB"))
    return frames


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_ctypes_abi_struct_sizes() -> None:
    """Verified against /home/yahhas/Documents/llama.cpp build b12347 headers."""
    assert ctypes.sizeof(_LlamaModelParams) == 80
    assert _LlamaModelParams.load_mtp.offset == 77
    assert ctypes.sizeof(_LlamaContextParams) == 176
    assert _LlamaContextParams.embeddings.offset == 144
    assert _LlamaContextParams.kv_unified.offset == 149
    assert ctypes.sizeof(_LlamaEmbd) == 24
    assert ctypes.sizeof(_MtmdContextParams) == 96
    assert _MtmdContextParams.batch_max_tokens.offset == 72
    assert ctypes.sizeof(_MtmdInputText) == 24
    assert ctypes.sizeof(_MtmdDecoderPos) == 16


def test_fit_image_to_budget_matches_gemma4_rules() -> None:
    assert fit_image_to_budget(64, 64, budget=280) == (768, 768)
    assert fit_image_to_budget(800, 450, budget=280) == (1056, 576)
    assert fit_image_to_budget(160, 120, budget=140) == (624, 480)


def test_device_spec_parsing_and_media_validation() -> None:
    assert _parse_device_spec("cpu", has_vulkan_lib=True)[:3] == (False, None, "cpu")
    assert _parse_device_spec("vulkan", has_vulkan_lib=True)[:3] == (True, 0, "vulkan")
    assert _parse_device_spec("gpu", has_vulkan_lib=True)[:3] == (True, 0, "vulkan")
    assert _parse_device_spec("cuda", has_vulkan_lib=True)[:3] == (True, 0, "vulkan")
    assert _parse_device_spec(None, has_vulkan_lib=True)[:3] == (True, 0, "vulkan")
    assert _parse_device_spec(None, has_vulkan_lib=False)[:3] == (False, None, "cpu")
    assert _parse_device_spec("vulkan:0", has_vulkan_lib=True)[:3] == (True, 0, "vulkan:0")
    assert _parse_device_spec("vulkan:1", has_vulkan_lib=True)[:3] == (True, 1, "vulkan:1")
    assert _parse_device_spec("gpu:1", has_vulkan_lib=True)[:3] == (True, 1, "vulkan:1")
    assert _parse_device_spec("cuda:2", has_vulkan_lib=True)[:3] == (True, 2, "vulkan:2")
    with pytest.raises(ValueError):
        _parse_device_spec("vulkan:abc", has_vulkan_lib=True)
    with pytest.raises(ValueError):
        _parse_device_spec("tpu", has_vulkan_lib=True)

    with pytest.raises(ValueError, match="empty audio"):
        _prepare_audio(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="empty video"):
        _prepare_video([])


def test_worker_crash_error_retries_multibatch_and_fails_fast_on_single_item() -> None:
    assert _is_oom(LlamaWorkerCrashError("crashed"))

    class CrashOnMultiModel:
        def __init__(self) -> None:
            self._closed = False
            self.calls: list[int] = []

        def encode(self, inputs, **_kw):
            self.calls.append(len(inputs))
            if len(inputs) > 1:
                self._closed = True
                raise LlamaWorkerCrashError("llama worker crashed (exitcode=-6): EOFError")
            return np.ones((1, 768), dtype=np.float32)

        def close(self) -> None:
            self._closed = True

    emb = Embedder()
    emb.batch_sizes["text"] = 4
    models: list[CrashOnMultiModel] = []

    def fake_load_model() -> None:
        m = CrashOnMultiModel()
        models.append(m)
        emb._model = m

    emb._load_model = fake_load_model  # type: ignore[method-assign]
    out = emb.embed_batch("text", ["a", "b", "c", "d"], ["t1", "t2", "t3", "t4"])
    assert out.shape == (4, 768)
    # First model crashed on batch of 4, second model crashed on batch of 2, third succeeded on 1s.
    assert len(models) == 3
    assert emb.batch_limit("text") == 1

    # Single-item crash fails fast without retry.
    class AlwaysCrashModel:
        _closed = False

        def encode(self, inputs, **_kw):
            self._closed = True
            raise LlamaWorkerCrashError("llama worker crashed (exitcode=-6): EOFError")

    emb2 = Embedder()
    emb2._model = AlwaysCrashModel()
    with pytest.raises(LlamaWorkerCrashError):
        emb2.embed_batch("text", ["only"], ["t"])


def test_resolve_paths_and_ensure_model_file_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(path=Path("/nonexistent.toml"))
    cfg.model_dir = tmp_path / "cfg_models"
    cfg.llama_lib_dir = tmp_path / "cfg_libs"

    monkeypatch.delenv("SEMMA_MODEL_DIR", raising=False)
    monkeypatch.delenv("SEMMA_LLAMA_LIB_DIR", raising=False)
    assert resolve_model_dir(cfg) == tmp_path / "cfg_models"
    assert resolve_llama_lib_dir(cfg) == tmp_path / "cfg_libs"

    monkeypatch.setenv("SEMMA_MODEL_DIR", str(tmp_path / "env_models"))
    monkeypatch.setenv("SEMMA_LLAMA_LIB_DIR", str(tmp_path / "env_libs"))
    assert resolve_model_dir(cfg) == tmp_path / "env_models"
    assert resolve_llama_lib_dir(cfg) == tmp_path / "env_libs"

    filename, expected_size, _ = GGUF_SPECS["text"]
    fake_model_dir = tmp_path / "fake_models"
    fake_model_dir.mkdir()
    target = fake_model_dir / filename
    target.write_bytes(b"\0")
    os.truncate(target, expected_size)
    assert ensure_model_file("text", fake_model_dir) == target


@pytest.mark.gpu
def test_llama_cpp_parity_with_reference_bf16() -> None:
    ref = np.load(REFERENCE_NPZ)
    emb = Embedder(device="vulkan")
    try:
        # 1. Text-only query worker vs bf16 reference.
        emb.load_query()
        assert emb.query_model_loaded and not emb.index_model_loaded
        q_vecs = np.stack([emb.embed_query(q) for q in QUERIES], axis=0)
        for idx, q in enumerate(QUERIES):
            cos = float(q_vecs[idx] @ ref["queries"][idx])
            assert cos >= 0.999, f"query[{idx}] ({q!r}) cosine={cos:.6f}"

        # 2. Full multimodal worker vs bf16 reference.
        d_vecs = emb.embed_batch("text", DOCS_TEXT, DOCS_TITLES)
        assert emb.index_model_loaded and not emb.query_model_loaded
        for idx in range(len(DOCS_TEXT)):
            cos = float(d_vecs[idx] @ ref["docs_text"][idx])
            assert cos >= 0.999, f"docs_text[{idx}] cosine={cos:.6f}"

        # Full model answers queries identically to text-only worker.
        for idx, q in enumerate(QUERIES):
            emb._query_cache.clear()
            v_full = emb.embed_query(q)
            assert float(v_full @ q_vecs[idx]) >= 0.9999

        i_vecs = emb.embed_batch("image", _synthetic_images(), ["img1", "img2", "img3"])
        for idx in range(3):
            cos = float(i_vecs[idx] @ ref["images"][idx])
            assert cos >= 0.999, f"images[{idx}] cosine={cos:.6f}"

        a_vecs = emb.embed_batch("audio", _synthetic_audios(), ["a1", "a2"])
        assert float(a_vecs[0] @ ref["audios"][0]) >= 0.999
        # Upstream llama.cpp mtmd_audio_preprocessor_gemma4a clamps mel bins at 0.001 rather than
        # adding 0.001, introducing ~0.002 drift on sparse pure dual-tone sine waves (broadband
        # audio scores >= 0.9997).
        assert float(a_vecs[1] @ ref["audios"][1]) >= 0.997

        v_vec = emb.embed_batch("video", [_synthetic_video(8)], ["vid"])[0]
        assert float(v_vec @ ref["video"]) >= 0.999

        # 3. 32-frame video (>2048 tokens) expands context, then text calls shrink it back to 2048
        #    with identical (>2048-token and standard) text truncation & embeddings.
        long_text = "alpha beta gamma delta epsilon zeta eta theta " * 600  # > 2048 tokens
        before_vecs = emb.embed_batch("text", [long_text, DOCS_TEXT[0]], ["long", DOCS_TITLES[0]])
        assert emb._model is not None and emb._model.ctx_size() == DEFAULT_MAX_TOKENS

        v32 = emb.embed_batch("video", [_synthetic_video(32)], ["vid32"])[0]
        assert v32.shape == (768,) and v32.dtype == np.float32 and np.isfinite(v32).all()
        assert abs(float(np.linalg.norm(v32)) - 1.0) < 1e-5
        assert emb._model.ctx_size() >= 4608

        after_vecs = emb.embed_batch("text", [long_text, DOCS_TEXT[0]], ["long", DOCS_TITLES[0]])
        assert emb._model.ctx_size() == DEFAULT_MAX_TOKENS
        assert np.allclose(after_vecs, before_vecs, atol=1e-6)
        assert float(after_vecs[0] @ before_vecs[0]) >= 0.999999
        assert float(after_vecs[1] @ before_vecs[1]) >= 0.999999
    finally:
        emb.unload()


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["vulkan:0", "vulkan:1", "cpu"])
def test_multi_gpu_and_cpu_parity(device: str) -> None:
    ref = np.load(REFERENCE_NPZ)
    emb = Embedder(device=device)
    try:
        emb.load_query()
        assert emb.query_device == device
        assert emb.device == device
        assert emb.worker_pids()

        q_vecs = np.stack([emb.embed_query(q) for q in QUERIES], axis=0)
        for idx, q in enumerate(QUERIES):
            cos = float(q_vecs[idx] @ ref["queries"][idx])
            assert cos > 0.999, f"[{device}] query[{idx}] ({q!r}) cosine={cos:.6f}"

        d_vecs = emb.embed_batch("text", DOCS_TEXT, DOCS_TITLES)
        assert emb.device == device
        for idx in range(len(DOCS_TEXT)):
            cos = float(d_vecs[idx] @ ref["docs_text"][idx])
            assert cos > 0.999, f"[{device}] docs_text[{idx}] cosine={cos:.6f}"

        i_vecs = emb.embed_batch("image", _synthetic_images()[:1], ["img1"])
        cos_img = float(i_vecs[0] @ ref["images"][0])
        assert cos_img > 0.999, f"[{device}] image[0] cosine={cos_img:.6f}"
    finally:
        emb.unload()


@pytest.mark.gpu
def test_invalid_vulkan_index_raises() -> None:
    emb = Embedder(device="vulkan:99")
    try:
        with pytest.raises(RuntimeError):
            emb.load_query()
    finally:
        emb.unload()


@pytest.mark.gpu
def test_worker_crash_recovery() -> None:
    emb = Embedder(device="vulkan:0")
    try:
        emb.load_query()
        assert emb.query_model_loaded
        assert emb._query_model is not None and emb._query_model._proc is not None
        old_proc = emb._query_model._proc
        old_pid = old_proc.pid
        assert old_pid is not None and _pid_alive(old_pid)

        old_proc.kill()
        old_proc.join(timeout=5.0)
        assert not _pid_alive(old_pid)
        assert not emb.query_model_loaded
        assert not emb.loaded

        vec = emb.embed_query("another query after crash")
        assert vec.shape == (768,) and vec.dtype == np.float32 and np.isfinite(vec).all()
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
        assert emb.query_model_loaded
        assert emb._query_model is not None and emb._query_model._proc is not None
        assert emb._query_model._proc.pid != old_pid
    finally:
        emb.unload()


@pytest.mark.gpu
def test_worker_subprocess_cleanup_on_unload() -> None:
    emb = Embedder(device="vulkan")
    try:
        emb.load_query()
        assert emb._query_model is not None and emb._query_model._proc is not None
        q_pid = emb._query_model._proc.pid
        assert q_pid is not None and _pid_alive(q_pid)

        # Loading full GPU index worker drops the GPU query worker cleanly.
        emb.load()
        assert not _pid_alive(q_pid)
        assert emb._model is not None and emb._model._proc is not None
        i_pid = emb._model._proc.pid
        assert i_pid is not None and _pid_alive(i_pid)

        assert emb.unload() is True
        assert not emb.loaded
        assert not _pid_alive(i_pid)
    finally:
        emb.unload()
