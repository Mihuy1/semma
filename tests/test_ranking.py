"""Ranking: meaning z-scores + word coverage + name/type bonuses (search.py, DigUp's approach)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmbedder
from test_index import make_cfg
from test_review_fixes import names, put_background, put_vectors, unit, with_cos

from gemmasearch import filetypes as ft
from gemmasearch.config import load_config
from gemmasearch.search import STD_FLOOR, Searcher, meaning_scale, query_words
from gemmasearch.store import Store


def add_text(store: Store, path_like: str, text: str) -> None:
    """Give the (single) chunk of an indexed synthetic file some chunk text."""
    with store.write() as (db, _):
        cid = db.execute("SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path LIKE ?",
                         (f"%{path_like}",)).fetchone()[0]
        db.execute("INSERT INTO fts_chunks (rowid, text) VALUES (?, ?)", (cid, text))


@pytest.fixture(params=[False, True], ids=["sqlite-vec", "cache"])
def realistic(request, tmp_path: Path):
    """A small index whose bulk looks like real EmbeddingGemma cosines for a query: images
    around 0.55 and text around 0.59 (image cosines run lower), at the real z gate."""
    cfg = make_cfg(tmp_path)
    cfg.z_gate = 2.0
    cfg.duplicate_similarity = {}
    store = Store(cfg.db_path, vector_cache=request.param)
    q = unit(1)
    put_background(store, tmp_path, q, n=100, mean=0.55)
    put_background(store, tmp_path, q, n=100, mean=0.59, kind="text", modality="text", seed=5000)
    yield cfg, store, q, tmp_path
    store.close()


def test_best_meaning_match_beats_word_matches_with_lower_similarity(realistic) -> None:
    """The ViT screenshot bug: an image with no text whose vector is the best match must not
    rank below text files that merely contain the query's words (RRF put it 13th)."""
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Pictures/Screenshots/vit.png"), 30, with_cos(q, 0.76, 2))])
    put_vectors(store, [(str(root / "Documents/llava.md"), 40, with_cos(q, 0.66, 3)),
                        (str(root / "Documents/granite.md"), 40, with_cos(q, 0.65, 4))],
                kind="text", modality="text")
    add_text(store, "llava.md", "the vision transformer encoder of llava")
    add_text(store, "granite.md", "granite vision uses a transformer")
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "vision transformer", limit=10)
    assert names(res)[:3] == ["vit.png", "llava.md", "granite.md"]
    assert res[0].matched == ["vector"] and res[0].similarity == pytest.approx(0.76, abs=1e-3)
    assert "text" in res[1].matched
    # Raw cosine would favour text here; centring each modality on its own mean does not.
    assert res[0].score > res[1].score + 0.5


def test_file_name_match_ranks_first(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Documents/vision_transformer_slides.odp"), 40, with_cos(q, 0.62, 2))],
                kind="office", modality="text")
    put_vectors(store, [(str(root / "Documents/vit_notes.md"), 40, with_cos(q, 0.69, 3))],
                kind="text", modality="text")
    put_vectors(store, [(str(root / "Pictures/diagram.png"), 30, with_cos(q, 0.73, 4))])
    add_text(store, "vit_notes.md", "notes on the vision transformer")
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "vision transformer", limit=10)
    assert names(res)[0] == "vision_transformer_slides.odp"
    assert "name" in res[0].matched
    assert set(names(res)[1:3]) == {"vit_notes.md", "diagram.png"}


def test_type_word_boosts_the_kind_and_is_not_required_in_the_text(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Pictures/Screenshots/Screenshot_2026.png"), 30, with_cos(q, 0.68, 2)),
                        (str(root / "Pictures/beach.jpg"), 30, with_cos(q, 0.68, 3))])
    put_vectors(store, [(str(root / "Documents/kubernetes.md"), 40, with_cos(q, 0.66, 4))],
                kind="text", modality="text")
    add_text(store, "kubernetes.md", "kubernetes cluster setup")
    s = Searcher(store, cfg, FakeEmbedder())
    plain = s.search_vector(q, "kubernetes cluster", limit=10)
    assert names(plain)[0] == "kubernetes.md"
    # "screenshot" asks for an image, prefers real screenshots, and is not itself searched as a word:
    # the kubernetes notes still cover every content word.
    shot = s.search_vector(q, "screenshot of kubernetes cluster", limit=10)
    assert names(shot)[:3] == ["Screenshot_2026.png", "beach.jpg", "kubernetes.md"]
    assert next(r for r in shot if r.path.endswith("kubernetes.md")).score == pytest.approx(
        next(r for r in plain if r.path.endswith("kubernetes.md")).score)
    photo = s.search_vector(q, "photo of kubernetes cluster", limit=10)
    assert {names(photo)[0], names(photo)[1]} == {"Screenshot_2026.png", "beach.jpg"}


def test_words_keep_a_file_with_weak_meaning(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Documents/minutes.md"), 40, with_cos(q, 0.58, 2))], kind="text", modality="text")
    add_text(store, "minutes.md", "the quarterly budget review was postponed")
    s = Searcher(store, cfg, FakeEmbedder())
    # 2 of 3 content words ("quarterly", "budget"; not "approval"): coverage 0.67 >= 0.5.
    res = s.search_vector(q, "quarterly budget approval", limit=10)
    assert names(res) == ["minutes.md"] and res[0].matched == ["text"]
    # 1 of 3: too few, and its meaning doesn't stand out either.
    assert s.search_vector(q, "quarterly forecast approval", limit=10) == []


def test_nothing_stands_out_means_nothing_found(realistic) -> None:
    cfg, store, q, _ = realistic
    assert Searcher(store, cfg, FakeEmbedder()).search_vector(q, "zebra", limit=10) == []


def test_kinds_filter_candidates_but_not_the_scale(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Pictures/a.jpg"), 30, with_cos(q, 0.75, 2))])
    put_vectors(store, [(str(root / "Documents/a.md"), 40, with_cos(q, 0.80, 3))], kind="text", modality="text")
    s = Searcher(store, cfg, FakeEmbedder())
    both = s.search_vector(q, "", limit=10)
    only = s.search_vector(q, "", limit=10, kinds=[ft.IMAGE])
    assert names(only) == ["a.jpg"]
    assert only[0].score == pytest.approx(next(r for r in both if r.path.endswith("a.jpg")).score)


def test_meaning_scale_on_tiny_and_uniform_indexes() -> None:
    names_ = ["text", "image"]
    # Fewer than MODALITY_MIN vectors per modality: everything is centred on the overall mean.
    sims = np.array([0.5, 0.6, 0.7, 0.4], dtype=np.float32)
    sc = meaning_scale(sims, np.array([0, 0, 1, 1]), names_)
    assert sc.means == {} and sc.overall == pytest.approx(0.55)
    assert sc.z(0.7, "image") == pytest.approx((0.7 - 0.55) / float(np.std(sims - 0.55)), rel=1e-4)
    # All cosines equal: the std floor keeps z finite and small.
    flat = meaning_scale(np.full(30, 0.6, dtype=np.float32), np.zeros(30, dtype=np.int64), names_)
    assert flat.std == STD_FLOOR and flat.z(0.6, "text") == pytest.approx(0.0, abs=1e-5)
    # A modality with enough vectors is centred on its own mean.
    sims = np.concatenate([np.full(25, 0.6), np.full(25, 0.5)]).astype(np.float32)
    sc = meaning_scale(sims, np.array([0] * 25 + [1] * 25), names_)
    assert sc.means == pytest.approx({"text": 0.6, "image": 0.5})
    assert sc.z(0.6, "text") == pytest.approx(sc.z(0.5, "image"), abs=1e-4)
    assert meaning_scale(np.zeros(0, np.float32), np.zeros(0, np.int64), names_).std == 1.0


def test_query_words() -> None:
    w = query_words("screenshot of the Vision Transformer")
    assert w.terms == ["Vision", "Transformer"] and w.kinds == {ft.IMAGE} and w.screenshot
    w = query_words("kuvakaappaus opinnäytetyöstä")
    assert w.terms == ["opinnäytetyöstä"] and w.kinds == {ft.IMAGE} and w.screenshot
    w = query_words("paper about dogs")
    assert w.terms == ["dogs"] and w.kinds == {ft.PDF} and not w.screenshot
    # Only type words: they are the words to look for too.
    w = query_words("pdf")
    assert w.terms == ["pdf"] and w.kinds == {ft.PDF}
    w = query_words("the video")
    assert w.terms == ["video"] and w.kinds == {ft.VIDEO}
    assert query_words("the of").terms == ["the", "of"]
    assert query_words("a x 7").terms == ["7"]
    assert query_words("dog dog DOG").terms == ["dog"]


def test_removed_search_settings_warn_and_are_ignored(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[search]\nmin_similarity = 0.7\nsimilarity_margin = 0.1\nmin_similarity_short = 0.65\n"
                        "similarity_margin_short = 0.04\nrrf_k = 60\nz_gate = 2.5\n")
    cfg = load_config(path=cfg_path, home=tmp_path)
    err = capsys.readouterr().err
    for key in ("min_similarity", "similarity_margin", "min_similarity_short", "similarity_margin_short", "rrf_k"):
        assert f"search.{key} is no longer used" in err
        assert not hasattr(cfg, key)
    assert "z_gate" in err and "unknown key" not in err
    assert cfg.z_gate == 2.5


@pytest.mark.parametrize("value", ["high", -1, 11, True])
def test_invalid_z_gate_warns_and_keeps_the_default(tmp_path: Path, capsys, value) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"[search]\nz_gate = {value!r}\n".replace("'", '"').replace("True", "true"))
    cfg = load_config(path=cfg_path, home=tmp_path)
    assert "search.z_gate must be a number" in capsys.readouterr().err
    assert cfg.z_gate == 2.0
