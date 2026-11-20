# ADR 0001: Embedding Engine, llama.cpp ABI, and Model Lifecycle

## Status

Accepted

## Context

Semma embeds text, images, audio, and video locally using EmbeddingGemma 2 via `llama.cpp` / `libmtmd` (`ctypes`). Because it runs as a background desktop daemon alongside games and interactive desktop sessions, it must balance search responsiveness, VRAM residency, OOM recovery, and resumable model downloads.

## Decisions

### 1. llama.cpp / libmtmd ABI and Modular Backends
- `src/semma/embedder.py` binds against the `llama.cpp` / `libmtmd` C ABI verified against build `b12347`.
- Packaged builds compile `ggml` backends as dynamically loaded modules (`GGML_BACKEND_DL`), so there is no monolithic `libggml-cpu.so`; `ggml` selects the best CPU variant for the host automatically.
- Module backends are registered selectively: the Vulkan backend is loaded only for a GPU worker (a missing or broken `libvulkan` simply omits it), followed by the best CPU backend variant.
- Preprocessed image/audio bitmaps are freed immediately once `mtmd` chunks take ownership of the data.
- Context window expands dynamically for >2048-token inputs (such as 32-frame video) and shrinks back to 2048 for subsequent text calls so text truncation and embeddings remain identical.
- Upstream `llama.cpp` `mtmd_audio_preprocessor_gemma4a` clamps mel bins at `0.001` instead of adding `0.001`, introducing ~0.002 drift on sparse pure dual-tone sine waves (broadband audio parity remains >= 0.9997).

### 2. Dual Model Split (Full Index Model vs. Text-Only Query Worker)
- Query embedding uses a lightweight text-only worker (no multimodal projector loaded) when the full index model is not already resident; if the full index model is already loaded, queries reuse it instead of loading a second model.
- Before loading the full multimodal index worker onto the GPU, any GPU text-only query worker is unloaded first so both models never occupy VRAM simultaneously.
- The OCR engine is loaded per indexing run rather than kept resident for the daemon's lifetime.

### 3. Device Selection and Fallback
- Resolved device labels are not re-parsed as explicit settings: re-parsing `"auto"` (Vulkan if available, else CPU) into an explicit `"vulkan"` would disable fallback and break search on machines that have the Vulkan loader installed without a usable GPU.

### 4. Locking, GPU Yielding, and Idle Unload
- `GpuLock` prioritizes incoming search queries ahead of waiting indexing batches.
- The full index model is accessed exclusively under the GPU lock; a CPU query model is accessed exclusively under the CPU query lock (and a GPU query model under the GPU lock). Because models can be loaded or unloaded while waiting for a lock, callers re-check state after acquiring the lock and retry so concurrent queries and index batches never encode on the same model simultaneously and concurrent CPU query loads share a single load.
- The full index model unloads after `index_model_unload_seconds` once indexing is idle with no pending work. When indexing is paused (user pause or GPU busy), VRAM is released even with pending files and reloaded on the next slice.
- While another application (such as a game) keeps the GPU busy, VRAM is released quickly (`GPU_BUSY_UNLOAD_S`). Compositors and desktop shells that continuously use small amounts of GPU on KDE Wayland/X11 are ignored unless their SM utilization exceeds `DEFAULT_IGNORED_MAX_SM`.
- Toggling `pause_when_gpu_busy` off during a GPU pause resumes indexing immediately without a restart; toggling it back on makes the next sample take effect immediately regardless of `gpu_check_seconds`.

### 5. Per-Modality Batching, OOM Recovery, and Vector Normalization
- Default batch sizes were calibrated on an RTX 5070 Ti (~2,000-character text chunks): text batch 8 yields ~112 chunks/s at 2.7 GiB peak, whereas 32 drops to 71/s at 6.7 GiB due to padding/attention memory growth; images are preprocessing-bound (~21/s at any batch size), so batch 8 keeps memory low; video uses batch 1 (~0.7 s per 32-frame chunk) so queries never wait behind multiple video batches.
- On a CUDA/GPU OOM, the modality's batch size is halved and retried, remaining reduced until a streak of successful batches allows doubling back up to the configured limit.
- Query embeddings are cached because type-as-you-go search repeatedly sends the same prefixes during typing and backspacing.
- Output vectors are renormalized in `float32` because the model normalizes in `bfloat16` (~1% norm error).

### 6. Resumable Model Downloads
- Shutdown does not wait on a stuck model transfer; the `.part` file is preserved for resumption.
- If the server closes the connection early (reported by `urllib` as EOF), the partial file is kept and resumed on the next call.
- If a `.part` file is longer than the server's file (e.g., left over from another model revision) or fails byte integrity verification, it is discarded and restarted from scratch.
- Download readiness does not depend on the download lock: a complete file is ready with no active downloader, an interrupted download (part + lock file with no holder) is reported as not running, and probing readiness creates no files.
