"""Screenshots with OCR: the extra text chunk, and keeping the index in line with the setting.

The recogniser is faked (these run without Tesseract) except in the one test marked
needs_tesseract.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmbedder, bag_of_words
from PIL import Image
from test_ocr import needs_tesseract, text_image

from semma import ocr
from semma.config import Config, load_config
from semma.extract import extract
from semma.extract import image as image_mod
from semma.extract.text import MAX_CHARS
from semma.indexer import Indexer
from semma.scanner import Scanner
from semma.search import Searcher
from semma.store import Store

TEXTS = {
    "Screenshot From 2026-09-25 14-20-31.png": "Select Interpreter\nPython 3.14.7 64-bit /usr/bin/python",
    "kuva.png": "Taulukko 1: Hakumenetelmien vertailussa käytettävät suorituskyky- ja resurssimittarit",
}


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """OCR effective with English; the recogniser returns TEXTS[name]. Returns the names OCR'd."""
    calls: list[str] = []

    def ocr_text(path: Path, languages: list[str]) -> str:
        calls.append(path.name)
        assert languages == ["eng"]
        return TEXTS.get(path.name, "")

    monkeypatch.setattr(ocr, "effective_languages", lambda cfg: ["eng"] if cfg.ocr_enabled else [])
    monkeypatch.setattr(image_mod, "ocr_text", ocr_text)
    return calls


def png(path: Path, size: tuple[int, int] = (320, 200), color: str = "white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def ocr_cfg(tmp_path: Path, enabled: bool = True) -> Config:
    cfg = Config(home=tmp_path, db_path=tmp_path / "index.db", roots=[])
    cfg.ocr_enabled = enabled
    return cfg


# ---------------------------------------------------------------------------------- extraction


def test_screenshot_yields_its_text_after_the_image(tmp_path: Path, fake_ocr: list[str]) -> None:
    shot = png(tmp_path / "Pictures" / "Screenshot From 2026-09-25 14-20-31.png")
    chunks = list(extract(shot, "image", ocr_cfg(tmp_path)))
    assert [c.modality for c in chunks] == ["image", "text"]
    text = chunks[1]
    assert text.content == TEXTS[shot.name]
    assert text.snippet == "Select Interpreter Python 3.14.7 64-bit /usr/bin/python"
    assert text.locator == "" and text.position is None


def test_screenshot_folder_counts_and_other_images_are_not_read(tmp_path: Path, fake_ocr: list[str]) -> None:
    in_folder = png(tmp_path / "Kuvat" / "Kuvakaappaukset" / "kuva.png")
    photo = png(tmp_path / "Pictures" / "holiday.png")
    assert [c.modality for c in extract(in_folder, "image", ocr_cfg(tmp_path))] == ["image", "text"]
    assert [c.modality for c in extract(photo, "image", ocr_cfg(tmp_path))] == ["image"]
    assert fake_ocr == ["kuva.png"]


def test_ocr_off_or_no_language_reads_nothing(tmp_path: Path, fake_ocr: list[str],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    shot = png(tmp_path / "Screenshots" / "Screenshot From 2026-09-25 14-20-31.png")
    assert [c.modality for c in extract(shot, "image", ocr_cfg(tmp_path, enabled=False))] == ["image"]
    monkeypatch.setattr(ocr, "effective_languages", lambda cfg: [])  # e.g. no library installed
    assert [c.modality for c in extract(shot, "image", ocr_cfg(tmp_path))] == ["image"]
    assert fake_ocr == []


def test_long_text_is_chunked_like_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    long = "\n".join(f"Line {i} of a long terminal session with plenty of words in it" for i in range(200))
    monkeypatch.setattr(ocr, "effective_languages", lambda cfg: ["eng"])
    monkeypatch.setattr(image_mod, "ocr_text", lambda path, langs: long)
    chunks = list(extract(png(tmp_path / "Screenshots" / "term.png"), "image", ocr_cfg(tmp_path)))
    texts = [c for c in chunks if c.modality == "text"]
    assert chunks[0].modality == "image" and len(texts) >= 4
    assert all(len(c.content) <= MAX_CHARS for c in texts)
    assert texts[0].content.startswith("Line 0 ") and "Line 199 " in texts[-1].content


def test_ocr_failure_keeps_the_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(path: Path, langs: list[str]) -> str:
        raise MemoryError()

    monkeypatch.setattr(ocr, "effective_languages", lambda cfg: ["eng"])
    monkeypatch.setattr(image_mod, "ocr_text", broken)
    chunks = list(extract(png(tmp_path / "Screenshots" / "a.png"), "image", ocr_cfg(tmp_path)))
    assert [c.modality for c in chunks] == ["image"]


def test_recognizer_failure_inside_ocr_text_keeps_the_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(image: Image.Image, languages: list[str]) -> str:
        raise OSError("libtesseract went away")

    monkeypatch.setattr(ocr, "effective_languages", lambda cfg: ["eng"])
    monkeypatch.setattr(ocr, "recognize", broken)
    chunks = list(extract(png(tmp_path / "Screenshots" / "a.png"), "image", ocr_cfg(tmp_path)))
    assert [c.modality for c in chunks] == ["image"]


@needs_tesseract
def test_real_tesseract_reads_a_screenshot(tmp_path: Path) -> None:
    shot = tmp_path / "Pictures" / "Screenshots" / "dialog.png"
    shot.parent.mkdir(parents=True)
    text_image(["Select Interpreter", "Create Virtual Environment"]).save(shot)
    cfg = ocr_cfg(tmp_path)
    cfg.ocr_languages = ("eng",)
    try:
        chunks = list(extract(shot, "image", cfg))
    finally:
        ocr.release_thread()
    assert [c.modality for c in chunks] == ["image", "text"]
    assert chunks[1].content == "Select Interpreter\nCreate Virtual Environment"


# ---------------------------------------------------------------------------------- the index

FILES = {
    "Pictures/Screenshots/Screenshot From 2026-09-25 14-20-31.png": "white",
    "Kuvat/Kuvakaappaukset/kuva.png": "gray",
    "Pictures/holiday.png": "blue",
}


@pytest.fixture
def env(tmp_path: Path, fake_ocr: list[str]):
    cfg = load_config(path=tmp_path / "nonexistent.toml", home=tmp_path)
    cfg.include_mounts = False
    cfg.db_path = tmp_path / "db" / "index.db"
    cfg.roots = [tmp_path]
    cfg.ocr_enabled = True
    for rel, color in FILES.items():
        png(tmp_path / rel, color=color)
    store = Store(cfg.db_path)
    scanner = Scanner(cfg)
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(), gen)
    store.finish_scan(gen, scanner.stats.roots)
    ix = Indexer(store, cfg, FakeEmbedder(), prefetch=False)
    stats = ix.run()
    assert stats.indexed == 3
    fake_ocr.clear()
    yield cfg, store, ix, fake_ocr
    store.close()


def modalities(store: Store) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path, modality in store.db.execute("SELECT f.path, c.modality FROM chunks c JOIN files f ON f.id = c.file_id "
                                           "ORDER BY f.path, c.seq"):
        out.setdefault(Path(path).name, []).append(modality)
    return out


def pending(store: Store) -> set[str]:
    return {Path(r[0]).name for r in store.db.execute("SELECT path FROM files WHERE status = 'pending'")}


def fts(store: Store, word: str) -> set[str]:
    return {Path(r[0]).name for r in store.db.execute(
        "SELECT f.path FROM fts_chunks JOIN chunks c ON c.id = fts_chunks.rowid JOIN files f ON f.id = c.file_id "
        "WHERE fts_chunks MATCH ?", (word,))}


SHOT = "Screenshot From 2026-09-25 14-20-31.png"


def test_screenshots_are_indexed_with_their_text(env) -> None:
    cfg, store, _, _ = env
    assert modalities(store) == {SHOT: ["image", "text"], "kuva.png": ["image", "text"], "holiday.png": ["image"]}
    assert store.get_meta("ocr_languages") == "eng"
    assert fts(store, "interpreter") == {SHOT}
    assert fts(store, "hakumenetelmien") == {"kuva.png"}
    # Search finds it by a word only in the screenshot, and shows the text, even when the
    # picture is the closer match in meaning (its chunk has no snippet to show).
    image_chunk = store.db.execute("SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id "
                                   "WHERE f.path LIKE ? AND c.modality = 'image'", (f"%{SHOT}",)).fetchone()[0]
    store.db.execute("UPDATE vec_chunks SET embedding = ? WHERE chunk_id = ?",
                     (bag_of_words("select interpreter").tobytes(), image_chunk))
    store.db.commit()
    cfg.z_gate, cfg.z_floor = 0.5, 0.0  # 5 chunks: no z can exceed 2
    res = Searcher(store, cfg, FakeEmbedder()).search("select interpreter")
    assert res[0].path.endswith(SHOT) and res[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert res[0].modality == "text" and res[0].snippet.startswith("Select Interpreter")


def test_turning_ocr_off_deletes_the_text_without_reindexing(env) -> None:
    cfg, store, ix, calls = env
    cache = store.warm_vector_cache()
    cfg.ocr_enabled = False
    assert ix.reconcile_ocr() == 0
    assert pending(store) == set()
    assert modalities(store) == {SHOT: ["image"], "kuva.png": ["image"], "holiday.png": ["image"]}
    assert fts(store, "interpreter") == set()
    assert store.get_meta("ocr_languages") == ""
    n_chunks = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert store.db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == n_chunks == 3
    alive = cache.alive[: cache.n]
    assert int(alive.sum()) == 3
    assert set(cache.chunk_ids[: cache.n][alive].tolist()) == {r[0] for r in store.db.execute("SELECT id FROM chunks")}
    row = cache.row_of[int(cache.chunk_ids[: cache.n][alive][0])]
    stored = store.db.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?",
                              (int(cache.chunk_ids[row]),)).fetchone()[0]
    assert np.allclose(cache.mat[row], np.frombuffer(stored, dtype=np.float32), atol=1e-6)
    assert ix.reconcile_ocr() == 0 and calls == []  # recorded: nothing more to do

    # On again: only the screenshots are read again.
    cfg.ocr_enabled = True
    assert ix.reconcile_ocr() == 2
    assert pending(store) == {SHOT, "kuva.png"}
    ix.run()
    assert sorted(calls) == sorted([SHOT, "kuva.png"])
    assert modalities(store)[SHOT] == ["image", "text"] and store.get_meta("ocr_languages") == "eng"


def test_other_languages_reread_the_screenshots(env, monkeypatch: pytest.MonkeyPatch) -> None:
    _, store, ix, _ = env
    monkeypatch.setattr(ocr, "effective_languages", lambda cfg: ["fin", "eng"])
    assert ix.reconcile_ocr() == 2
    assert pending(store) == {SHOT, "kuva.png"}
    assert modalities(store)[SHOT] == ["image", "text"]  # searchable until replaced
    assert store.get_meta("ocr_languages") == "fin+eng"
    assert ix.reconcile_ocr() == 0


def test_index_from_before_ocr_counts_as_off(env) -> None:
    cfg, store, ix, _ = env
    cfg.ocr_enabled = False
    ix.reconcile_ocr()  # no OCR chunks left, like an index from before OCR existed
    store.db.execute("DELETE FROM meta WHERE key = 'ocr_languages'")
    store.db.commit()
    assert ix.reconcile_ocr() == 0 and store.get_meta("ocr_languages") == ""
    store.db.execute("DELETE FROM meta WHERE key = 'ocr_languages'")
    store.db.commit()
    cfg.ocr_enabled = True
    assert ix.reconcile_ocr() == 2


def test_the_engine_is_freed_on_the_extraction_thread(tmp_path: Path, fake_ocr: list[str],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[str] = []
    monkeypatch.setattr(ocr, "release_thread", lambda: freed.append(threading.current_thread().name) or True)
    cfg = load_config(path=tmp_path / "nonexistent.toml", home=tmp_path)
    cfg.include_mounts, cfg.roots, cfg.db_path, cfg.ocr_enabled = False, [tmp_path], tmp_path / "index.db", True
    png(tmp_path / "Screenshots" / "kuva.png")
    store = Store(cfg.db_path)
    try:
        scanner = Scanner(cfg)
        gen = store.begin_scan()
        store.upsert_scanned(scanner.scan(), gen)
        store.finish_scan(gen, scanner.stats.roots)
        assert Indexer(store, cfg, FakeEmbedder(), prefetch=True).run().indexed == 1
        assert freed == ["semma-extract"]
    finally:
        store.close()
