"""OCR (semma.ocr): languages, screenshot detection, the word filter and the Tesseract binding.

Everything except the last section runs without Tesseract; the binding tests skip when the
library or English data is missing (as on CI).
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from semma import ocr


@pytest.mark.parametrize(("env", "code"), [
    ({"LANG": "fi_FI.UTF-8"}, "fin"),
    ({"LANG": "de_DE.UTF-8@euro"}, "deu"),
    ({"LANG": "zh_CN.UTF-8"}, "chi_sim"),
    ({"LANG": "zh_TW.UTF-8"}, "chi_tra"),
    ({"LANG": "zh_HK"}, "chi_tra"),
    ({"LANG": "sr_RS.UTF-8@latin"}, "srp_latn"),
    ({"LANG": "sr_RS.UTF-8"}, "srp"),
    ({"LANG": "nb_NO.UTF-8"}, "nor"),
    ({"LANG": "fil_PH"}, "fil"),
    ({"LANG": "C.UTF-8"}, None),
    ({"LANG": "POSIX"}, None),
    ({}, None),
    ({"LANG": "xx_YY"}, None),
    ({"LANG": "en_US.UTF-8", "LC_MESSAGES": "sv_SE.UTF-8"}, "swe"),
    ({"LANG": "en_US.UTF-8", "LC_MESSAGES": "sv_SE.UTF-8", "LC_ALL": "fi_FI.UTF-8"}, "fin"),
    ({"LANG": "fi_FI.UTF-8", "LC_ALL": ""}, "fin"),
])
def test_locale_language(env: dict[str, str], code: str | None) -> None:
    assert ocr.locale_language(env) == code


def test_language_names() -> None:
    assert ocr.language_name("fin") == "Finnish"
    assert ocr.language_name("chi_sim") == "Chinese (Simplified)"
    assert ocr.language_name("xyz") == "xyz"
    assert len(ocr.LANGUAGE_NAMES) > 110
    assert all(ocr.LANGUAGE_CODE.fullmatch(c) for c in ocr.LANGUAGE_NAMES)
    assert set(ocr.ISO639_1.values()) <= set(ocr.LANGUAGE_NAMES)


def test_default_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANG", "fi_FI.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    assert ocr.default_languages(["eng", "fin", "swe"]) == ["fin", "eng"]
    assert ocr.default_languages(["eng"]) == ["eng"]
    assert ocr.default_languages([]) == []
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    assert ocr.default_languages(["eng", "fin"]) == ["eng"]
    monkeypatch.setenv("LANG", "C")
    assert ocr.default_languages(["eng", "fin"]) == ["eng"]


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch):
    """Pretend the library is there with these languages installed."""
    langs = ["deu", "eng", "fin", "fra", "swe"]
    monkeypatch.setattr(ocr, "library_available", lambda: True)
    monkeypatch.setattr(ocr, "installed_languages", lambda: list(langs))
    monkeypatch.setenv("LANG", "fi_FI.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    return langs


def cfg(enabled: bool = True, languages: tuple[str, ...] | None = None) -> SimpleNamespace:
    return SimpleNamespace(ocr_enabled=enabled, ocr_languages=languages)


def test_effective_languages(installed: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert ocr.effective_languages(cfg(enabled=False)) == []
    assert ocr.effective_languages(cfg()) == ["fin", "eng"]
    assert ocr.effective_languages(cfg(languages=("swe", "eng"))) == ["swe", "eng"]
    assert ocr.effective_languages(cfg(languages=("jpn", "fra"))) == ["fra"]
    assert ocr.effective_languages(cfg(languages=("deu", "deu", "eng"))) == ["deu", "eng"]
    assert ocr.effective_languages(cfg(languages=("deu", "eng", "fin", "fra"))) == ["deu", "eng", "fin"]
    assert ocr.effective_languages(cfg(languages=("jpn",))) == []
    monkeypatch.setattr(ocr, "library_available", lambda: False)
    assert ocr.effective_languages(cfg()) == []


def test_installed_languages_lists_language_data_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("eng", "fin", "chi_sim", "osd", "equ", "Latin"):
        (tmp_path / f"{name}.traineddata").write_bytes(b"x")
    (tmp_path / "configs").mkdir()
    (tmp_path / "eng.user-words").write_text("x")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    monkeypatch.setattr(ocr, "library_available", lambda: True)
    assert ocr.tessdata_dir() == tmp_path
    assert ocr.installed_languages() == ["chi_sim", "eng", "fin"]
    parent = tmp_path / "prefix"
    (parent / "tessdata").mkdir(parents=True)
    (parent / "tessdata" / "swe.traineddata").write_bytes(b"x")
    monkeypatch.setenv("TESSDATA_PREFIX", str(parent))
    assert ocr.installed_languages() == ["swe"]
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "missing"))
    assert ocr.installed_languages() == []
    monkeypatch.setattr(ocr, "library_available", lambda: False)
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert ocr.installed_languages() == []


def test_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr.os.path, "exists", lambda p: False)
    fedora = {"ID": "fedora", "VERSION_ID": "45"}
    assert ocr.install_hint(None, fedora) == "sudo dnf install tesseract-libs tesseract-langpack-eng"
    assert ocr.install_hint("chi_sim", fedora) == "sudo dnf install tesseract-langpack-chi_sim"
    assert ocr.install_hint("fin", {"ID": "almalinux", "ID_LIKE": "rhel centos fedora"}) == \
        "sudo dnf install tesseract-langpack-fin"
    ubuntu = {"ID": "ubuntu", "ID_LIKE": "debian"}
    assert ocr.install_hint(None, ubuntu) == "sudo apt install libtesseract5 tesseract-ocr-eng"
    assert ocr.install_hint("chi_sim", ubuntu) == "sudo apt install tesseract-ocr-chi-sim"
    mint = {"ID": "linuxmint", "ID_LIKE": "ubuntu debian"}
    assert ocr.install_hint("fin", mint) == "sudo apt install tesseract-ocr-fin"
    assert ocr.install_hint("fin", {"ID": "arch"}) == ""
    assert ocr.install_hint(None, {}) == ""
    monkeypatch.setattr(ocr.os.path, "exists", lambda p: p == "/run/ostree-booted")
    assert ocr.install_hint("fin", fedora) == "rpm-ostree install tesseract-langpack-fin"


@pytest.mark.parametrize(("path", "expected"), [
    ("/home/u/Pictures/Screenshots/Screenshot From 2026-09-25 14-20-31.png", True),
    ("/home/u/Pictures/Screenshots/2026/holiday.jpg", True),
    ("/home/u/Kuvat/Näyttökuva 2026-01-01.png", True),
    ("/home/u/Images/Capture d’écran du 2026-01-01.png", True),
    ("/home/u/Images/Capture_d'écran_2026.png", True),
    ("/home/u/SCREENSHOTS/a.png", True),
    ("/home/u/Pictures/holiday.jpg", False),
    ("/home/u/Pictures/Screenshots/notes.txt", False),
    ("/home/u/Documents/my screenshot.png", False),
])
def test_is_screenshot(path: str, expected: bool) -> None:
    assert ocr.is_screenshot(Path(path)) is expected


def words(text: str, conf: float = 95.0) -> list[tuple[str, float, bool]]:
    """Triples for every word of every line of `text`, all with confidence `conf`."""
    out = []
    for line in text.splitlines():
        ws = line.split()
        out += [(w, conf, i == len(ws) - 1) for i, w in enumerate(ws)]
    return out


def test_filter_keeps_confident_words_and_lines() -> None:
    triples = words("Select Interpreter\nPython 3.14.7 64-bit /usr/bin/python")
    assert ocr.filter_words(triples) == "Select Interpreter\nPython 3.14.7 64-bit /usr/bin/python"


def test_filter_drops_junk() -> None:
    line = [("o¢eee", 25), ("Firefox", 57), ("a", 35), ("even", 61), ("@", 81), ("HD", 59), ("=", 26),
            ("@", 64), (":", 7)]
    rows = [(w, float(c), i == len(line) - 1) for i, (w, c) in enumerate(line)]
    rows += words("Tab: Rust GUI Frameworks and ZapFast - Google Gemini")
    rows += [("|", 92.0, False), ("©", 96.0, False), ("2", 96.0, False), ("4", 30.0, True)]
    assert ocr.filter_words(rows) == "even\nTab: Rust GUI Frameworks and ZapFast Google Gemini\n2"


def test_filter_drops_too_little_text() -> None:
    assert ocr.filter_words(words("MiniLM-mallin")) == ""
    assert ocr.filter_words(words("OK Go 1 2 3 4 5 6 7 8 9 10 11")) == ""
    assert ocr.filter_words(words("Save the file")) == ""
    assert ocr.filter_words(words("ORS Leth pokes eee les alte", conf=30)) == ""
    assert ocr.filter_words([]) == ""
    assert ocr.filter_words(words("Save the changes before closing")) == "Save the changes before closing"


def test_prepare_image_caps_pixels_and_makes_grey() -> None:
    big = Image.new("RGB", (7680, 2160), "white")
    out = ocr.prepare_image(big)
    assert out.mode == "L" and out.width * out.height <= ocr.MAX_OCR_PIXELS
    assert abs(out.width / out.height - 7680 / 2160) < 0.01
    small = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
    out = ocr.prepare_image(small)
    assert out.mode == "L" and out.size == (800, 600) and out.getpixel((0, 0)) == 255
    assert ocr.prepare_image(Image.new("P", (40, 40))).mode == "L"


def test_library_candidates_skip_old_majors(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"/usr/lib64": ["/usr/lib64/libtesseract.so.5.4", "/usr/lib64/libtesseract.so.5.5",
                            "/usr/lib64/libtesseract.so.5.5.4", "/usr/lib64/libtesseract.so.4"]}
    monkeypatch.setattr(ocr.glob, "glob", lambda pattern: files.get(pattern.rsplit("/", 1)[0], []))
    monkeypatch.setattr(ocr.ctypes.util, "find_library", lambda name: None)
    assert ocr._lib_candidates() == ["libtesseract.so.5", "/usr/lib64/libtesseract.so.5.5.4",
                                     "/usr/lib64/libtesseract.so.5.5", "/usr/lib64/libtesseract.so.5.4"]


def test_no_library_means_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "_library", lambda: None)
    assert ocr.recognize(Image.new("RGB", (200, 100), "white"), ["eng"]) == ""
    assert ocr.recognize(Image.new("RGB", (200, 100), "white"), []) == ""
    assert ocr.release_thread() is False


needs_tesseract = pytest.mark.skipif(not (ocr.library_available() and "eng" in ocr.installed_languages()),
                                     reason="needs libtesseract 5 with English data")


def text_image(lines: list[str], size: int = 28) -> Image.Image:
    font = ImageFont.load_default(size=size)
    img = Image.new("RGB", (900, 70 + 50 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((30, 30 + 50 * i), line, fill="black", font=font)
    return img


@needs_tesseract
def test_recognize_reads_rendered_text() -> None:
    img = text_image(["Select Interpreter", "The quick brown fox jumps over the lazy dog"])
    try:
        text = ocr.recognize(img, ["eng"])
    finally:
        assert ocr.release_thread() is True
    assert text.splitlines() == ["Select Interpreter", "The quick brown fox jumps over the lazy dog"]
    assert ocr.release_thread() is False


@needs_tesseract
def test_engine_per_thread_and_reinit_for_new_languages() -> None:
    img = text_image(["Hello world from the screenshot test"])
    seen: dict[str, str] = {}

    def work(name: str) -> None:
        try:
            seen[name] = ocr.recognize(img, ["eng"])
        finally:
            ocr.release_thread()

    threads = [threading.Thread(target=work, args=(f"t{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(seen.values()) == {"Hello world from the screenshot test"}
    try:
        assert ocr.recognize(img, ["eng"]) == "Hello world from the screenshot test"
        first = ocr._local.engine
        ocr.recognize(img, ["eng", "eng"])
        assert ocr._local.engine is not first and first.handle is None
    finally:
        ocr.release_thread()
    gc.collect()
    assert not ocr._engines


@needs_tesseract
def test_missing_language_is_skipped_quietly() -> None:
    img = text_image(["Hello world from the screenshot test"])
    try:
        assert ocr.recognize(img, ["zzz"]) == ""
        assert ocr.recognize(img, ["zzz"]) == ""
    finally:
        ocr.release_thread()
