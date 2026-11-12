"""Integration tests with the real model. Run with: GEMMA_GPU_TESTS=1 uv run pytest tests/test_gpu.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fakes import FakeExtractor
from test_index import make_cfg, scan, write_docs

from gemmasearch.embedder import Embedder
from gemmasearch.extract.base import Chunk
from gemmasearch.indexer import Indexer
from gemmasearch.search import Searcher
from gemmasearch.store import Store

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    e = Embedder()
    e.load()
    return e


def test_document_format_matches_builtin_prompt(embedder: Embedder) -> None:
    """Our "title: none | text: ..." formatting with prompt="" equals encode_document (no double prefix)."""
    text = "The cat sat on the mat."
    ours = embedder.embed_batch("text", [text], ["none"])[0]
    builtin = embedder._model.encode_document([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    assert float(ours @ builtin) > 0.99  # bf16 batching noise only
    assert ours.dtype == np.float32 and ours.shape == (768,)
    assert abs(float(np.linalg.norm(ours)) - 1) < 1e-2


def test_query_and_mixed_modalities(embedder: Embedder) -> None:
    from PIL import Image

    q = embedder.embed_query("a red square")
    chunks = [Chunk("text", "tomato soup recipe"), Chunk("image", Image.new("RGB", (64, 64), (255, 0, 0))),
              Chunk("audio", np.zeros(16_000, dtype=np.float32)), Chunk("text", "a red square")]
    vecs = embedder.embed_chunks(chunks, "file.txt")
    assert vecs.shape == (4, 768) and np.isfinite(vecs).all()
    assert float(vecs[3] @ q) > float(vecs[0] @ q)


def test_semantic_search_end_to_end(tmp_path: Path, embedder: Embedder) -> None:
    cfg = make_cfg(tmp_path)
    cfg.min_similarity, cfg.similarity_margin = 0.60, 0.10  # the shipped defaults
    write_docs(tmp_path, {
        "a.md": "Feed the sourdough starter, knead the dough and bake the loaf in a hot oven.",
        "b.md": "The Kubernetes cluster lost quorum when an etcd node ran out of disk space.",
        "c.md": "Opinnäytetyössä tutkitaan koulujen energiankulutusta.",
    })
    store = Store(cfg.db_path)
    scan(store, cfg)
    Indexer(store, cfg, embedder, extract_fn=FakeExtractor()).run()
    s = Searcher(store, cfg, embedder)
    assert Path(s.search("how do I bake bread")[0].path).name == "a.md"
    assert Path(s.search("server outage")[0].path).name == "b.md"
    assert Path(s.search("thesis about school energy use")[0].path).name == "c.md"
    assert s.search("quantum chromodynamics lecture notes") == []


def test_text_only_query_model_matches_full_and_drops_vision_audio(embedder: Embedder) -> None:
    q_emb = Embedder()
    q_emb.load_query()
    try:
        assert q_emb.query_model_loaded and not q_emb.index_model_loaded
        st_model = q_emb._query_model
        assert st_model is not None
        # Verify vision and audio encoders are absent from the text-only model.
        mod_names = [name for name, _ in st_model.named_modules()]
        assert not any("vision" in n or "audio" in n for n in mod_names), mod_names
        auto_model = st_model[0].auto_model
        assert getattr(auto_model.config, "vision_config", None) is None
        assert getattr(auto_model.config, "audio_config", None) is None

        for query in ("my CV", "how do I bake sourdough bread", "koulujen energiankulutus"):
            v_full = embedder.embed_query(query)
            v_text = q_emb.embed_query(query)
            cos = float(v_full @ v_text)
            assert cos > 0.9999, f"{query!r}: cosine={cos}"
    finally:
        q_emb.unload()
