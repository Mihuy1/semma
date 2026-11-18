"""Ranking: meaning z-scores + word coverage + name/type bonuses (search.py, DigUp's approach)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmbedder
from test_index import DOCS, make_cfg, run, scan, write_docs
from test_review_fixes import make_env, names, put_background, put_vectors, unit, with_cos

from semma import filetypes as ft
from semma import search as search_mod
from semma.config import load_config
from semma.query_filters import FILTERS, QueryFilter, parse_query, register, sql_filter_fn
from semma.search import STD_FLOOR, Searcher, is_lookup, meaning_scale, query_words
from semma.store import Store


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
    minutes = next(r for r in res if Path(r.path).name == "minutes.md")
    assert minutes.matched == ["text"]
    # 1 of 3: too few, and its meaning doesn't stand out either.
    assert "minutes.md" not in names(s.search_vector(q, "quarterly forecast approval", limit=10))


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


@pytest.mark.parametrize("value", [-1, 21])
def test_invalid_z_floor_warns_and_keeps_the_default(tmp_path: Path, capsys, value) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"[search]\nz_floor = {value!r}\n".replace("'", '"').replace("True", "true"))
    cfg = load_config(path=cfg_path, home=tmp_path)
    assert "search.z_floor must be a number" in capsys.readouterr().err
    assert cfg.z_floor == 4.0
    cfg_path.write_text("[search]\nz_floor = 0\n")
    assert load_config(path=cfg_path, home=tmp_path).z_floor == 0.0


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_invalid_location_prior_warns_and_keeps_the_default(tmp_path: Path, capsys, value) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"[search]\nlocation_prior = {value!r}\n".replace("'", '"').replace("True", "true"))
    cfg = load_config(path=cfg_path, home=tmp_path)
    assert "search.location_prior must be a number" in capsys.readouterr().err
    assert cfg.location_prior == 0.3


def test_cache_does_not_reload_on_concurrent_write_between_snapshot_and_meaning(
    tmp_path: Path, monkeypatch,
) -> None:
    cfg, store = make_env(tmp_path, DOCS)
    store.close()
    store = Store(cfg.db_path, vector_cache=True)
    _, emb = run(cfg, store)
    s = Searcher(store, cfg, emb)
    assert names(s.search("sourdough bread"))[0] == "bread.txt"
    cache = store.vectors
    assert cache is not None and cache.loads == 1

    # Commit a write after the read snapshot starts (during FTS) and before _meaning runs.
    orig_selective = s._selective
    wrote = False

    def selective_with_concurrent_write(db, terms):
        nonlocal wrote
        out = orig_selective(db, terms)
        if not wrote:
            wrote = True
            (tmp_path / "Documents/soup.md").write_text("tomato soup with basil")
            scan(store, cfg)
            run(cfg, store)
        return out

    monkeypatch.setattr(s, "_selective", selective_with_concurrent_write)
    res = s.search("sourdough bread")
    assert names(res)[0] == "bread.txt"
    assert cache.loads == 1 and cache.size == 4  # patched in flight, no reload from the older snapshot
    assert names(s.search("tomato soup basil"))[0] == "soup.md"
    assert cache.loads == 1
    store.close()


def test_name_bonus_not_capped_at_50_files(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.z_gate = 2.0
    cfg.duplicate_similarity = {}
    store = Store(cfg.db_path)
    q = unit(1)
    put_background(store, tmp_path, q, n=80, mean=0.55)
    reports = [
        (str(tmp_path / f"Pictures/report_{i:03d}.jpg"), 10, with_cos(q, 0.65 if i < 79 else 0.72, 200 + i))
        for i in range(80)
    ]
    put_vectors(store, reports + [(str(tmp_path / "Pictures/other.jpg"), 10, with_cos(q, 0.78, 999))])
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "report", limit=10)
    assert names(res)[0] == "report_079.jpg"
    assert "name" in res[0].matched
    other = next((r for r in res if Path(r.path).name == "other.jpg"), None)
    assert other is None or res[0].score > other.score
    store.close()


def test_per_word_prefixes_in_names_and_folders(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    write_docs(tmp_path, {
        "thesis_project.md": "unrelated body words",
        "other.md": "thesaurus unrelated words",
        "thesisProject/inside/notes.md": "unrelated body words",
    })
    store = Store(cfg.db_path)
    scan(store, cfg)
    _, emb = run(cfg, store)
    res = Searcher(store, cfg, emb).search("thes proj", limit=10)
    assert names(res)[0] == "thesis_project.md"
    assert "name" in res[0].matched
    other = [r for r in res if Path(r.path).name == "other.md"]
    assert not other or "text" not in other[0].matched  # chunk text does not prefix non-last "thes"
    store.close()


def test_folder_only_word_counts_half_in_coverage(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [
        (str(root / "Documents/thesisProject/in_folder.md"), 40, with_cos(q, 0.72, 2)),
        (str(root / "Documents/other/in_text.md"), 40, with_cos(q, 0.72, 3)),
    ], kind="text", modality="text")
    add_text(store, "in_folder.md", "completely unrelated body")
    add_text(store, "in_text.md", "notes about thesis work")
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "thesis", limit=10)
    assert names(res)[:2] == ["in_text.md", "in_folder.md"]
    by_name = {Path(r.path).name: r for r in res}
    assert by_name["in_text.md"].matched == ["vector", "text"]
    assert by_name["in_folder.md"].matched == ["vector", "folder"]
    # One word found outside the file name counts with half the word weight (SINGLE_TEXT_FACTOR).
    assert by_name["in_text.md"].score - by_name["in_folder.md"].score == pytest.approx(
        2.0 * 0.5 * (1.0 - 0.5**1.5), abs=1e-4)


def test_sqlite_vec_fallback_sample_handles_gaps_and_reuses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(search_mod, "SCALE_SAMPLE", 8)
    monkeypatch.setattr(search_mod, "MODALITY_SAMPLE_MIN", 4)
    cfg = make_cfg(tmp_path)
    cfg.z_gate = 2.0
    cfg.knn_k = 5
    cfg.duplicate_similarity = {}
    store = Store(cfg.db_path, vector_cache=False)
    q = unit(1)
    put_background(store, tmp_path, q, n=64, mean=0.55)
    # Delete every even chunk id so all remaining chunks have odd ids (id % step == 0 would return 0 rows).
    with store.write() as (db, _):
        db.execute("DELETE FROM files WHERE id IN (SELECT file_id FROM chunks WHERE id % 2 = 0)")
    put_vectors(store, [(str(tmp_path / "Pictures/target.jpg"), 30, with_cos(q, 0.82, 2))])

    s = Searcher(store, cfg, FakeEmbedder())
    res = s.search_vector(q, "", limit=5)
    assert names(res)[0] == "target.jpg"
    first_sample = s._sample
    assert first_sample is not None and len(first_sample.mat) == 8

    # Index changes within SCALE_SAMPLE_MAX_AGE: the full-size sample must be reused without refetching.
    orig_vectors = s._vectors
    vector_calls = 0

    def counted_vectors(db, chunk_ids):
        nonlocal vector_calls
        vector_calls += 1
        return orig_vectors(db, chunk_ids)

    monkeypatch.setattr(s, "_vectors", counted_vectors)
    put_vectors(store, [(str(tmp_path / "Pictures/extra1.jpg"), 10, with_cos(q, 0.54, 3))])
    assert names(s.search_vector(q, "", limit=5))[0] == "target.jpg"
    assert vector_calls == 0
    assert s._sample is first_sample

    # Once older than SCALE_SAMPLE_MAX_AGE and the index changed, the sample is refreshed.
    monkeypatch.setattr(search_mod, "SCALE_SAMPLE_MAX_AGE", -1.0)
    put_vectors(store, [(str(tmp_path / "Pictures/extra2.jpg"), 10, with_cos(q, 0.54, 4))])
    assert names(s.search_vector(q, "", limit=5))[0] == "target.jpg"
    assert vector_calls == 1
    assert s._sample is not first_sample
    store.close()


# -- lookups, single words, trimming, ambiguous type words, sqlite-vec fallback --------------------

def cos_at(store: Store, q: np.ndarray, z: float, modality: str) -> float:
    """The cosine that sits at this z for q on the store's current vectors (planting a few more
    vectors moves the scale only slightly, so tests keep gaps of >= 0.5 z)."""
    db = store.reader()
    rows = db.execute("SELECT embedding, modality FROM vec_chunks").fetchall()
    vecs = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    names_ = sorted({r[1] for r in rows})
    sc = meaning_scale(vecs @ q, np.array([names_.index(r[1]) for r in rows]), names_)
    return sc.offset(modality) + z * sc.std


def test_is_lookup() -> None:
    for q in ("b0bbce05d5", "0xff32:0x0074", "4815162342", "TRX50 K7Q2LM"):
        assert is_lookup(query_words(q).terms), q
    for q in ("123", "apollo 11", "ASTRO_A50_EXTRA_FEATURES", "vision transformer", "console.txt", ""):
        assert not is_lookup(query_words(q).terms), q


def test_lookup_ranks_the_file_containing_the_code_first(realistic) -> None:
    """A code means little to the embedding: the file containing it (meaning z ~1) must beat a
    vector-only match at z 4 (lookups weigh meaning 0.5 and words x2)."""
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Pictures/noise.jpg"), 10, with_cos(q, cos_at(store, q, 4.0, "image"), 2))])
    put_vectors(store, [(str(root / "Documents/console.txt"), 10, with_cos(q, cos_at(store, q, 1.0, "text"), 3))],
                kind="text", modality="text")
    add_text(store, "console.txt", "build id b0bbce05d5 finished")
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "b0bbce05d5", limit=10)
    assert names(res)[0] == "console.txt" and "text" in res[0].matched
    noise = next((r for r in res if r.path.endswith("noise.jpg")), None)
    assert noise is None or noise.score < res[0].score - 1.0


def test_single_word_mention_ranks_after_a_photo_that_stands_out(realistic) -> None:
    """The real "dog" case: the thesis mentions dogs (z ~3.6, the word in its text, location
    prior 0.9 * 0.3) and the dog photo has no text (z ~5.2). With the full word weight the thesis
    won (3.6 + 2 + 0.27 > 5.2); one word found only in text now adds half."""
    cfg, store, q, root = realistic
    photo, thesis = cos_at(store, q, 5.2, "image"), cos_at(store, q, 3.6, "text")
    put_vectors(store, [(str(root / "Pictures/random_image.jpg"), 10, with_cos(q, photo, 2))])
    put_vectors(store, [(str(root / "Documents/thesis.md"), 90, with_cos(q, thesis, 3))], kind="text", modality="text")
    add_text(store, "thesis.md", "figure 3 shows a dog next to the dataset samples")
    res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "dog", limit=10)
    assert names(res)[:2] == ["random_image.jpg", "thesis.md"]
    assert res[0].matched == ["vector"] and "text" in res[1].matched
    by = {Path(r.path).name: r for r in res}
    # Without the discount the thesis would have won by ~0.6.
    assert by["thesis.md"].score - 2.0 * 0.5 + 2.0 > by["random_image.jpg"].score


def test_nothing_stands_out_finds_nothing(realistic) -> None:
    """Gibberish: z is relative, so ~2% of any index passes the gate, but no match reaches
    z_floor and the query's words are in no file."""
    cfg, store, q, _ = realistic
    cfg.z_floor = 4.5
    s = Searcher(store, cfg, FakeEmbedder())
    assert s.search_vector(q, "asdfgh qwerty zxcv", limit=10) == []
    cfg.z_floor = 0.0
    assert s.search_vector(q, "asdfgh qwerty zxcv", limit=10)  # the tail the floor removes


def test_vector_only_tail_is_trimmed_relative_to_the_best_match(realistic) -> None:
    cfg, store, q, root = realistic
    cfg.z_floor = 4.5
    put_vectors(store, [(str(root / f"Pictures/{name}.jpg"), 10, with_cos(q, cos_at(store, q, z, "image"), seed))
                        for name, z, seed in (("best", 8.0, 2), ("close", 6.0, 3), ("far", 4.0, 4))])
    put_vectors(store, [(str(root / "Documents/maintenance.md"), 10, with_cos(q, cos_at(store, q, 0.5, "text"), 5))],
                kind="text", modality="text")
    add_text(store, "maintenance.md", "the car needs new tyres")
    s = Searcher(store, cfg, FakeEmbedder())
    # Several words: vector-only results within Z_MARGIN (3) of the best one.
    many = names(s.search_vector(q, "red sports racing", limit=10))
    assert many[:2] == ["best.jpg", "close.jpg"] and "far.jpg" not in many
    # One word: within SINGLE_WORD_MARGIN (1.5); files containing the word are always kept.
    one = s.search_vector(q, "car", limit=10)
    assert names(one) == ["best.jpg", "maintenance.md"]
    assert one[1].matched == ["text"]
    # `limit` still caps the list.
    assert len(s.search_vector(q, "red sports racing", limit=1)) == 1


def test_type_word_bonus_counts_towards_the_floor(realistic) -> None:
    """"photo of a car": images run lower in z, so the kind asked for counts towards z_floor."""
    cfg, store, q, root = realistic
    cfg.z_floor = 4.5
    put_vectors(store, [(str(root / "Pictures/audi.jpg"), 10, with_cos(q, cos_at(store, q, 4.0, "image"), 2))])
    s = Searcher(store, cfg, FakeEmbedder())
    assert names(s.search_vector(q, "photo of a car", limit=10)) == ["audi.jpg"]
    assert s.search_vector(q, "red car", limit=10) == []


def test_ambiguous_type_words_are_content_words_unless_used_as_hints() -> None:
    cases = {
        "image segmentation": (["image", "segmentation"], set()),
        "music theory": (["music", "theory"], set()),
        "voice notes": (["voice", "notes"], set()),
        "sound card driver": (["sound", "card", "driver"], set()),
        "image of a dog": (["dog"], {ft.IMAGE}),
        "notes about kubernetes": (["kubernetes"], {ft.TEXT, ft.OFFICE}),
        "attention in the paper": (["attention"], {ft.PDF}),
        "moon landing from a recording": (["moon", "landing"], {ft.AUDIO}),
        "screenshot firefox": (["firefox"], {ft.IMAGE}),
        "firefox photos": (["firefox"], {ft.IMAGE}),
        "moon landing video": (["moon", "landing"], {ft.VIDEO}),
        "kuvat lomalta": (["kuvat", "lomalta"], set()),
        "koira kuvassa": (["koira"], {ft.IMAGE}),
        "music": (["music"], set()),
    }
    for text, (terms, kinds) in cases.items():
        w = query_words(text)
        assert (w.terms, w.kinds) == (terms, kinds), text
    assert query_words("screenshot of x").screenshot and not query_words("image of x").screenshot


def test_ambiguous_type_word_gives_no_kind_bonus_or_partial_name_bonus(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [(str(root / "Pictures/Screenshots/Screenshot_1.png"), 30,
                         with_cos(q, cos_at(store, q, 3.0, "image"), 2))])
    put_vectors(store, [(str(root / "Documents/segmentation.md"), 30, with_cos(q, cos_at(store, q, 3.2, "text"), 3)),
                        (str(root / "Documents/string_theory.md"), 30, with_cos(q, cos_at(store, q, 0.5, "text"), 4))],
                kind="text", modality="text")
    add_text(store, "segmentation.md", "image segmentation with masks")
    s = Searcher(store, cfg, FakeEmbedder())
    res = s.search_vector(q, "image segmentation", limit=10)
    assert names(res)[0] == "segmentation.md"
    shot = next(r for r in res if r.path.endswith("Screenshot_1.png"))
    assert shot.score == pytest.approx(shot.z + 0.3 * 0.3, abs=1e-3)  # no +1 for being an image
    # "music theory": "theory" in a name is half the words, not the whole query: no NAME_BONUS.
    theory = next(r for r in s.search_vector(q, "music theory", limit=10) if r.path.endswith("string_theory.md"))
    assert theory.score < search_mod.NAME_BONUS


def test_sqlite_vec_fallback_scores_lexical_files_outside_the_knn(tmp_path: Path) -> None:
    """With a small knn_k a file holding the query's words but outside the KNN top-k used to
    count as z = 0 and lose to any vector-only match; its best text chunk is now scored."""
    results = {}
    for cache in (False, True):
        root = tmp_path / str(cache)
        cfg = make_cfg(root)
        cfg.z_gate, cfg.z_floor, cfg.knn_k = 2.0, 0.0, 3
        cfg.duplicate_similarity = {}
        store = Store(cfg.db_path, vector_cache=cache)
        q = unit(1)
        put_background(store, root, q, n=100, mean=0.55)
        put_background(store, root, q, n=100, mean=0.59, kind="text", modality="text", seed=5000)
        put_vectors(store, [(str(root / f"Pictures/top{i}.jpg"), 10, with_cos(q, cos_at(store, q, 6.0 - i * 0.2, "image"), 10 + i))
                            for i in range(3)])
        put_vectors(store, [(str(root / "Documents/notes.md"), 10, with_cos(q, cos_at(store, q, 3.5, "text"), 30))],
                    kind="text", modality="text")
        add_text(store, "notes.md", "kubernetes cluster upgrade notes")
        res = Searcher(store, cfg, FakeEmbedder()).search_vector(q, "kubernetes cluster", limit=10)
        results[cache] = res
        notes = next(r for r in res if r.path.endswith("notes.md"))
        # Outside the KNN top-3 (the sqlite-vec path) it used to count as z = 0 (score ~2).
        assert notes.z == pytest.approx(3.5, abs=0.4)
        assert notes.score > 5.0
        store.close()
    assert [r.z for r in results[False] if r.path.endswith("notes.md")] == pytest.approx(
        [r.z for r in results[True] if r.path.endswith("notes.md")], abs=0.3)


def test_sqlite_vec_sample_centres_a_rare_modality_on_its_own_mean(tmp_path: Path, monkeypatch) -> None:
    """A modality with too few vectors in an index-proportional sample (30 images among 2000
    text chunks: 1 in a 64-vector sample) is still sampled enough to get its own mean."""
    monkeypatch.setattr(search_mod, "SCALE_SAMPLE", 64)
    monkeypatch.setattr(search_mod, "MODALITY_SAMPLE_MIN", 24)
    cfg = make_cfg(tmp_path)
    store = Store(cfg.db_path, vector_cache=False)
    q = unit(1)
    put_background(store, tmp_path, q, n=2000, mean=0.60, kind="text", modality="text", seed=5000)
    put_background(store, tmp_path, q, n=30, mean=0.50)
    s = Searcher(store, cfg, FakeEmbedder())
    sc = s._sample_scale(store.reader(), q)
    assert set(sc.means) == {"text", "image"}
    assert sc.means["image"] == pytest.approx(0.50, abs=0.02)
    assert sc.means["text"] == pytest.approx(0.60, abs=0.02)
    exact = meaning_scale(*(lambda rows: (
        np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows]) @ q,
        np.array([0 if r[1] == "image" else 1 for r in rows]), ["image", "text"]))(
        store.reader().execute("SELECT embedding, modality FROM vec_chunks").fetchall()))
    assert sc.std == pytest.approx(exact.std, rel=0.2) and sc.center == pytest.approx(exact.center, abs=0.01)
    store.close()


def test_meaning_scale_matches_a_direct_computation_and_weights_act_as_copies() -> None:
    rng = np.random.default_rng(3)
    sims = rng.normal(0.55, 0.05, 500).astype(np.float32)
    mods = (rng.random(500) < 0.3).astype(np.int64)
    sc = meaning_scale(sims, mods, ["text", "image"])
    means = np.array([sims[mods == 0].mean(), sims[mods == 1].mean()])
    centred = sims - means[mods]
    assert sc.center == pytest.approx(float(centred.mean()), abs=1e-6)
    assert sc.std == pytest.approx(float(centred.std()), rel=1e-4)
    # A weight of 2 is the same as the vector appearing twice.
    w = np.where(np.arange(500) < 100, 2.0, 1.0)
    dup = meaning_scale(np.concatenate([sims, sims[:100]]), np.concatenate([mods, mods[:100]]), ["text", "image"])
    weighted = meaning_scale(sims, mods, ["text", "image"], weights=w)
    assert weighted.std == pytest.approx(dup.std, rel=1e-5) and weighted.means == pytest.approx(dup.means)
    # `files` decides MODALITY_MIN: 5 sampled images standing for 50 distinct files still get their own mean.
    few = meaning_scale(sims[:25], np.array([1] * 5 + [0] * 20), ["text", "image"], files=[20, 50])
    assert set(few.means) == {"text", "image"}


def test_sqlite_vec_sample_spreads_over_the_whole_index(tmp_path: Path, monkeypatch) -> None:
    """Older chunks (low ids) can differ from newer ones: the sample must span every id, not
    take the first SCALE_SAMPLE (it did whenever the index had < 2 * SCALE_SAMPLE chunks)."""
    monkeypatch.setattr(search_mod, "SCALE_SAMPLE", 64)
    cfg = make_cfg(tmp_path)
    store = Store(cfg.db_path, vector_cache=False)
    q = unit(1)
    put_background(store, tmp_path, q, n=60, mean=0.50, std=0.01, kind="text", modality="text", seed=5000)
    put_background(store, tmp_path / "new", q, n=60, mean=0.60, std=0.01, kind="text", modality="text", seed=7000)
    sc = Searcher(store, cfg, FakeEmbedder())._sample_scale(store.reader(), q)
    assert sc.means["text"] == pytest.approx(0.55, abs=0.01)
    store.close()


def test_parse_query_prefixes_and_extensibility() -> None:
    for raw in ("image: cats", "image:cats", "  IMAGE:   cats  "):
        p = parse_query(raw)
        assert p.text == "cats" and p.flt is not None and p.flt.name == "image"
    p_vid = parse_query("video: rocket\nlaunch")
    assert p_vid.text == "rocket\nlaunch" and p_vid.flt is not None and p_vid.flt.name == "video"
    p_shot = parse_query("screenshot: error")
    assert p_shot.text == "error" and p_shot.flt is not None and p_shot.flt.name == "screenshot"
    assert parse_query("image:").text == "" and parse_query("image:").flt is not None
    assert parse_query("   screenshot:   ").text == "" and parse_query("   screenshot:   ").flt is not None
    for literal in ("foo: bar", "http://example.com", "0xff32:0x0074", "plain query"):
        p_lit = parse_query(literal)
        assert p_lit.flt is None and p_lit.text == literal

    # Registering a new filter in the future works immediately:
    register(QueryFilter("note", frozenset({ft.TEXT, ft.OFFICE})), "notes")
    try:
        assert parse_query("note: budget").flt == FILTERS["note"]
        assert parse_query("NOTES: budget").flt == FILTERS["note"]
    finally:
        FILTERS.pop("note", None)
        FILTERS.pop("notes", None)


def test_sql_filter_fn_mixed_case_name() -> None:
    register(QueryFilter("Note", frozenset({ft.TEXT}), lambda _p: False), "Notes")
    try:
        assert sql_filter_fn("Note", "/x/a.txt") == 0
    finally:
        FILTERS.pop("note", None)
        FILTERS.pop("notes", None)


def test_prefix_filters_restrict_candidates_and_strip_prefix(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [
        (str(root / "Pictures/Screenshots/err_panel.png"), 30, with_cos(q, 0.72, 2)),
        (str(root / "Pictures/Screenshot_2026-01-01.png"), 30, with_cos(q, 0.70, 3)),
        (str(root / "Pictures/vacation_photo.jpg"), 30, with_cos(q, 0.78, 4)),
        (str(root / "Pictures/connection_refused_photo.png"), 30, with_cos(q, 0.65, 7)),
    ])
    put_vectors(store, [
        (str(root / "Videos/demo_clip.mp4"), 30, with_cos(q, 0.74, 5)),
    ], kind=ft.VIDEO, modality="video")
    put_vectors(store, [
        (str(root / "Documents/error_notes.md"), 40, with_cos(q, 0.82, 6)),
    ], kind=ft.TEXT, modality="text")
    add_text(store, "err_panel.png", "connection refused error")
    add_text(store, "connection_refused_photo.png", "connection refused error")
    add_text(store, "error_notes.md", "connection refused error in vacation_photo")

    emb = FakeEmbedder()
    s = Searcher(store, cfg, emb)

    # Empty query after prefix returns [] without calling embed_query.
    assert s.search("image:") == []
    assert s.search("screenshot:   ") == []
    assert s.search_vector(q, "image:", limit=10) == []

    # image: keeps all images (photos + screenshots), drops text and video even when text has higher z.
    imgs = s.search_vector(q, "image: connection refused", limit=10)
    assert set(names(imgs)) == {
        "err_panel.png", "Screenshot_2026-01-01.png", "vacation_photo.jpg", "connection_refused_photo.png",
    }

    # video: keeps only video files.
    vids = s.search_vector(q, "video: connection refused", limit=10)
    assert names(vids) == ["demo_clip.mp4"]

    # screenshot: keeps only files matching ocr.is_screenshot (Screenshots folder or Screenshot prefix),
    # dropping vacation_photo.jpg (higher cosine 0.78), connection_refused_photo.png (exact lexical match
    # in both name and chunk text, exercising gs_query_filter in SQL), and error_notes.md (cosine 0.82).
    shots = s.search_vector(q, "screenshot: connection refused", limit=10)
    assert set(names(shots)) == {"err_panel.png", "Screenshot_2026-01-01.png"}

    # Explicit kinds argument intersects with prefix filter; disjoint intersection returns [].
    assert s.search_vector(q, "image: connection refused", limit=10, kinds=[ft.VIDEO]) == []
    assert set(names(s.search_vector(q, "image: connection refused", limit=10, kinds=[ft.IMAGE, ft.TEXT]))) == {
        "err_panel.png", "Screenshot_2026-01-01.png", "vacation_photo.jpg", "connection_refused_photo.png",
    }


def test_prefix_filter_sql_predicate_excludes_lexical_non_screenshot(realistic) -> None:
    cfg, store, q, root = realistic
    put_vectors(store, [
        (str(root / "Pictures/Screenshots/err_panel.png"), 30, with_cos(q, 0.72, 2)),
        (str(root / "Pictures/holiday/beach.png"), 30, with_cos(q, 0.40, 3)),
    ])
    add_text(store, "err_panel.png", "sunset beach")
    add_text(store, "beach.png", "sunset beach")

    s = Searcher(store, cfg, FakeEmbedder())
    imgs = s.search_vector(q, "image: sunset beach", limit=10)
    assert {"err_panel.png", "beach.png"} <= set(names(imgs))

    shots = s.search_vector(q, "screenshot: sunset beach", limit=10)
    assert names(shots) == ["err_panel.png"]
    assert "beach.png" not in names(shots)


def test_meaning_scale_rare_long_file_uses_overall_mean(tmp_path: Path) -> None:
    """One 21-chunk video among many text files has 21 chunks (>= MODALITY_MIN) but only 1 file
    (< MODALITY_MIN): it must use the overall mean rather than its own high mean."""
    rng = np.random.default_rng(7)
    n_text, n_vid = 1000, 21
    sims = np.concatenate([
        np.clip(rng.normal(0.56, 0.02, n_text), -0.99, 0.99),
        np.full(n_vid, 0.72),
    ]).astype(np.float32)
    mods = np.array([0] * n_text + [1] * n_vid, dtype=np.int64)
    cfg = make_cfg(tmp_path)
    sc = meaning_scale(sims, mods, ["text", "video"], files=[n_text, 1])
    assert "video" not in sc.means
    assert sc.z(0.744, "video") > cfg.z_floor + 0.5
    # With chunk count as files (the old bug), video gets its own high mean and drops below z_floor (and default z_gate 2.0).
    sc_chunks = meaning_scale(sims, mods, ["text", "video"], files=[n_text, n_vid])
    assert "video" in sc_chunks.means
    assert sc_chunks.z(0.744, "video") < 2.0

    # When image is present in >= MODALITY_MIN files and video is in < MODALITY_MIN files,
    # video falls back to the image mean (both visual SigLIP) rather than overall.
    n_img = 200
    img_sims = np.concatenate([
        np.clip(rng.normal(0.60, 0.02, n_text), -0.99, 0.99),
        np.clip(rng.normal(0.52, 0.02, n_img), -0.99, 0.99),
        np.full(n_vid, 0.72),
    ]).astype(np.float32)
    img_mods = np.array([0] * n_text + [1] * n_img + [2] * n_vid, dtype=np.int64)
    sc_img = meaning_scale(img_sims, img_mods, ["text", "image", "video"], files=[n_text, n_img, 1])
    assert "video" not in sc_img.means
    assert "image" in sc_img.means
    assert sc_img.offset("video") == pytest.approx(sc_img.offset("image"))
    assert sc_img.offsets(["text", "image", "video"])[2] == pytest.approx(sc_img.offset("image"))

    # If image itself has < MODALITY_MIN files, video falls back to overall.
    sc_few_img = meaning_scale(img_sims, img_mods, ["text", "image", "video"], files=[n_text, 5, 1])
    assert "image" not in sc_few_img.means
    assert sc_few_img.offset("video") == pytest.approx(sc_few_img.overall + sc_few_img.center)


def _put_multi_chunk(store: Store, path: str, kind: str, modality: str, vecs: list[np.ndarray]) -> int:
    """Insert one indexed file with multiple chunks into SQLite (updates delta for live cache)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"x")
    with store.write() as (db, delta):
        fid = db.execute(
            "INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
            "VALUES (?, ?, 1, 1.0, 0, 30, 'indexed', 1)",
            (path, kind),
        ).lastrowid
        assert fid is not None
        for seq, vec in enumerate(vecs):
            cid = db.execute("INSERT INTO chunks (file_id, seq, modality) VALUES (?, ?, ?)",
                             (fid, seq, modality)).lastrowid
            assert cid is not None
            v = np.asarray(vec, dtype=np.float32)
            unit_vec = v / np.linalg.norm(v)
            db.execute("INSERT INTO vec_chunks VALUES (?, ?, ?, ?)",
                       (cid, unit_vec, modality, kind))
            delta.added.append((cid, int(fid), unit_vec, modality, kind))
        return int(fid)


@pytest.mark.parametrize("cache", [False, True], ids=["sqlite-vec", "cache"])
def test_single_multi_chunk_video_is_found_with_and_without_filter(tmp_path: Path, cache: bool) -> None:
    """A single 21-chunk video with a realistic tight cosine range (0.70-0.74) among background
    files must use the overall mean (files=1 < MODALITY_MIN), reach z >= 5.0, and rank #1 across
    plain, prefix-filtered, and vector-only type-hint queries for both sqlite-vec and the cache."""
    cfg = make_cfg(tmp_path)
    cfg.z_gate = 2.0
    cfg.duplicate_similarity = {}
    store = Store(cfg.db_path, vector_cache=cache)
    q = unit(1)
    put_background(store, tmp_path, q, n=500, mean=0.55, std=0.025)
    put_background(store, tmp_path, q, n=500, mean=0.59, std=0.025, kind="text", modality="text", seed=5000)
    vpath = str(tmp_path / "Videos/porsche_clip.mp4")
    _put_multi_chunk(
        store,
        vpath,
        ft.VIDEO,
        "video",
        [with_cos(q, 0.74 if seq == 0 else 0.71, 900 + seq) for seq in range(21)],
    )

    s = Searcher(store, cfg, FakeEmbedder())
    for query in ("porsche", "video: porsche", "video about car"):
        res = s.search_vector(q, query, limit=10)
        assert names(res)[0] == "porsche_clip.mp4", query
        assert res[0].z is not None and res[0].z >= 5.0, query
        assert res[0].z >= cfg.z_floor, query
    store.close()


def test_vector_cache_file_stats_invalidate_on_remove_and_replace(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    store = Store(cfg.db_path, vector_cache=True)
    q = unit(1)
    put_background(store, tmp_path, q, n=30, mean=0.55, kind="text", modality="text", seed=100)
    vpath = str(tmp_path / "Videos/long.mp4")
    _put_multi_chunk(
        store,
        vpath,
        ft.VIDEO,
        "video",
        [with_cos(q, 0.72, 200 + seq) for seq in range(21)],
    )

    cache = store.vectors
    assert cache is not None
    with cache.lock:
        cache.ensure_fresh(store.reader())
        sc1 = cache.scores(q)
        vid_code = cache._codes["video"]
        txt_code = cache._codes["text"]
        assert sc1.files[vid_code] == 1
        assert sc1.files[txt_code] == 30

    # Replace with a 4-chunk video at the same path (local write patches the cache).
    with store.write() as (db, delta):
        old_id = db.execute("SELECT id FROM files WHERE path = ?", (vpath,)).fetchone()[0]
        db.execute("DELETE FROM files WHERE id = ?", (old_id,))
        delta.removed_files.add(old_id)
        new_id = db.execute(
            "INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
            "VALUES (?, ?, 1, 2.0, 0, 30, 'indexed', 1)",
            (vpath, ft.VIDEO),
        ).lastrowid
        assert new_id is not None
        for seq in range(4):
            cid = db.execute("INSERT INTO chunks (file_id, seq, modality) VALUES (?, ?, ?)",
                             (new_id, seq, "video")).lastrowid
            assert cid is not None
            v = np.asarray(with_cos(q, 0.72, 300 + seq), dtype=np.float32)
            vec = v / np.linalg.norm(v)
            db.execute("INSERT INTO vec_chunks VALUES (?, ?, ?, ?)", (cid, vec, "video", ft.VIDEO))
            delta.added.append((cid, new_id, vec, "video", ft.VIDEO))

    with cache.lock:
        cache.ensure_fresh(store.reader())
        sc2 = cache.scores(q)
        assert cache.loads == 1
        assert sc2.valid is not None
        assert sc2.files[vid_code] == 1
        assert sc2.files[txt_code] == 30
        assert (sc2.valid & (sc2.modality == vid_code)).sum() == 4

    # Add a second video file (distinct video file count becomes 2).
    vpath2 = str(tmp_path / "Videos/second.mp4")
    _put_multi_chunk(
        store,
        vpath2,
        ft.VIDEO,
        "video",
        [with_cos(q, 0.72, 400 + seq) for seq in range(3)],
    )

    with cache.lock:
        cache.ensure_fresh(store.reader())
        sc3 = cache.scores(q)
        assert cache.loads == 1
        assert sc3.valid is not None
        assert sc3.files[vid_code] == 2
        assert sc3.files[txt_code] == 30
        assert (sc3.valid & (sc3.modality == vid_code)).sum() == 7

    # Remove the first video (distinct video file count drops back to 1).
    with store.write() as (db, delta):
        old_id = db.execute("SELECT id FROM files WHERE path = ?", (vpath,)).fetchone()[0]
        db.execute("DELETE FROM files WHERE id = ?", (old_id,))
        delta.removed_files.add(old_id)

    with cache.lock:
        cache.ensure_fresh(store.reader())
        sc4 = cache.scores(q)
        assert cache.loads == 1
        assert sc4.valid is not None
        assert sc4.files[vid_code] == 1
        assert (sc4.valid & (sc4.modality == vid_code)).sum() == 3

    # Remove the second video (distinct video file count drops to 0).
    with store.write() as (db, delta):
        old_id = db.execute("SELECT id FROM files WHERE path = ?", (vpath2,)).fetchone()[0]
        db.execute("DELETE FROM files WHERE id = ?", (old_id,))
        delta.removed_files.add(old_id)

    with cache.lock:
        cache.ensure_fresh(store.reader())
        sc5 = cache.scores(q)
        assert cache.loads == 1
        assert sc5.valid is not None
        assert sc5.files[vid_code] == 0
        assert sc5.files[txt_code] == 30
        assert (sc5.valid & (sc5.modality == vid_code)).sum() == 0
    store.close()

