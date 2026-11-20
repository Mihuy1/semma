"""Optional OCR for screenshots, through the system's Tesseract library.

A screenshot's pixels say little to the image embedding about the words in it; the words
themselves (a dialog, a table, an error message) are what people search for. When OCR is on,
`extract.image` adds the recognised text of each screenshot as a text chunk (FTS and a text
embedding).

Light by design: libtesseract (5.x) is called through ctypes, the way `embedder` binds
llama.cpp; nothing is bundled and there is no new Python dependency. Without the library or
its language data, OCR is quietly unavailable and everything else works as before.

Measured on this machine (libtesseract 5.5, `eng` from tessdata_fast): loading one language
costs ~27 MB of RAM (two: ~50 MB, so at most MAX_LANGUAGES); a typical screenshot takes
0.02-0.3 s, a 2536x1259 one full of text 0.8 s (1.7 s with fin+eng). 2x upscaling before OCR
was mixed (better on some screenshots, worse on others), 25 % slower and +40 MB: not done.

Threads: a TessBaseAPI handle is not thread-safe, so every thread gets its own engine
(in practice only the indexer's extraction thread, `semma-extract`, has one). The engine is
created on the first screenshot and freed by `release_thread()` when that thread's indexing run
ends, which also hands Tesseract's cached dictionaries and the freed heap back to the system.
Tesseract uses OpenMP like llama.cpp in the same process: never set OMP_THREAD_LIMIT or other
process-wide OpenMP variables here, they would slow llama.cpp's CPU path (Tesseract measured
the same with 1 thread as with the default).

Public, cheap and never raising (safe to call from a GUI thread; none of them loads a
language model): MAX_LANGUAGES, library_available, installed_languages, language_name,
locale_language, default_languages, effective_languages, install_hint, is_screenshot.
`recognize` does the work.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import glob
import logging
import os
import re
import threading
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from .config import Config

log = logging.getLogger(__name__)

MAX_LANGUAGES = 3

LANGUAGE_CODE = re.compile(r"[a-z]{3}(?:_[a-z]+)*")
NOT_LANGUAGES = frozenset({"osd", "equ"})

MAX_OCR_PIXELS = 3840 * 2160
SOURCE_DPI = 96

MIN_WORD_CONFIDENCE = 60.0
MIN_SHORT_WORD_CONFIDENCE = 80.0
MIN_WORDS = 3
MIN_ALNUM = 20


LANGUAGE_NAMES: dict[str, str] = {
    "afr": "Afrikaans", "amh": "Amharic", "ara": "Arabic", "asm": "Assamese", "aze": "Azerbaijani",
    "aze_cyrl": "Azerbaijani (Cyrillic)", "bel": "Belarusian", "ben": "Bengali", "bod": "Tibetan",
    "bos": "Bosnian", "bre": "Breton", "bul": "Bulgarian", "cat": "Catalan", "ceb": "Cebuano",
    "ces": "Czech", "chi_sim": "Chinese (Simplified)", "chi_sim_vert": "Chinese (Simplified, vertical)",
    "chi_tra": "Chinese (Traditional)", "chi_tra_vert": "Chinese (Traditional, vertical)",
    "chr": "Cherokee", "cos": "Corsican", "cym": "Welsh", "dan": "Danish", "dan_frak": "Danish (Fraktur)",
    "deu": "German", "deu_frak": "German (Fraktur)", "deu_latf": "German (Fraktur, Latin)",
    "div": "Dhivehi", "dzo": "Dzongkha", "ell": "Greek", "eng": "English", "enm": "Middle English",
    "epo": "Esperanto", "est": "Estonian", "eus": "Basque", "fao": "Faroese", "fas": "Persian",
    "fil": "Filipino", "fin": "Finnish", "fra": "French", "frk": "German (Fraktur)", "frm": "Middle French",
    "fry": "Western Frisian", "gla": "Scottish Gaelic", "gle": "Irish", "glg": "Galician",
    "grc": "Ancient Greek", "guj": "Gujarati", "hat": "Haitian Creole", "heb": "Hebrew", "hin": "Hindi",
    "hrv": "Croatian", "hun": "Hungarian", "hye": "Armenian", "iku": "Inuktitut", "ind": "Indonesian",
    "isl": "Icelandic", "ita": "Italian", "ita_old": "Italian (Old)", "jav": "Javanese", "jpn": "Japanese",
    "jpn_vert": "Japanese (vertical)", "kan": "Kannada", "kat": "Georgian", "kat_old": "Georgian (Old)",
    "kaz": "Kazakh", "khm": "Khmer", "kir": "Kyrgyz", "kmr": "Kurdish (Kurmanji)", "kor": "Korean",
    "kor_vert": "Korean (vertical)", "kur": "Kurdish (Arabic)", "lao": "Lao", "lat": "Latin",
    "lav": "Latvian", "lit": "Lithuanian", "ltz": "Luxembourgish", "mal": "Malayalam", "mar": "Marathi",
    "mkd": "Macedonian", "mlt": "Maltese", "mon": "Mongolian", "mri": "Maori", "msa": "Malay",
    "mya": "Burmese", "nep": "Nepali", "nld": "Dutch", "nor": "Norwegian", "oci": "Occitan",
    "ori": "Odia", "pan": "Punjabi", "pol": "Polish", "por": "Portuguese", "pus": "Pashto",
    "que": "Quechua", "ron": "Romanian", "rus": "Russian", "san": "Sanskrit", "sin": "Sinhala",
    "slk": "Slovak", "slk_frak": "Slovak (Fraktur)", "slv": "Slovenian", "snd": "Sindhi",
    "spa": "Spanish", "spa_old": "Spanish (Old)", "sqi": "Albanian", "srp": "Serbian",
    "srp_latn": "Serbian (Latin)", "sun": "Sundanese", "swa": "Swahili", "swe": "Swedish",
    "syr": "Syriac", "tam": "Tamil", "tat": "Tatar", "tel": "Telugu", "tgk": "Tajik", "tgl": "Tagalog",
    "tha": "Thai", "tir": "Tigrinya", "ton": "Tongan", "tur": "Turkish", "uig": "Uyghur",
    "ukr": "Ukrainian", "urd": "Urdu", "uzb": "Uzbek", "uzb_cyrl": "Uzbek (Cyrillic)",
    "vie": "Vietnamese", "yid": "Yiddish", "yor": "Yoruba",
}

ISO639_1: dict[str, str] = {
    "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "az": "aze", "be": "bel", "bg": "bul",
    "bn": "ben", "bo": "bod", "br": "bre", "bs": "bos", "ca": "cat", "co": "cos", "cs": "ces",
    "cy": "cym", "da": "dan", "de": "deu", "dv": "div", "dz": "dzo", "el": "ell", "en": "eng",
    "eo": "epo", "es": "spa", "et": "est", "eu": "eus", "fa": "fas", "fi": "fin", "fo": "fao",
    "fr": "fra", "fy": "fry", "ga": "gle", "gd": "gla", "gl": "glg", "gu": "guj", "he": "heb",
    "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun", "hy": "hye", "id": "ind", "is": "isl",
    "it": "ita", "iu": "iku", "iw": "heb", "ja": "jpn", "jv": "jav", "ka": "kat", "kk": "kaz",
    "km": "khm", "kn": "kan", "ko": "kor", "ku": "kmr", "ky": "kir", "la": "lat", "lb": "ltz",
    "lo": "lao", "lt": "lit", "lv": "lav", "mi": "mri", "mk": "mkd", "ml": "mal", "mn": "mon",
    "mr": "mar", "ms": "msa", "mt": "mlt", "my": "mya", "nb": "nor", "ne": "nep", "nl": "nld",
    "nn": "nor", "no": "nor", "oc": "oci", "or": "ori", "pa": "pan", "pl": "pol", "ps": "pus",
    "pt": "por", "qu": "que", "ro": "ron", "ru": "rus", "sa": "san", "sd": "snd", "si": "sin",
    "sk": "slk", "sl": "slv", "sq": "sqi", "sr": "srp", "su": "sun", "sv": "swe", "sw": "swa",
    "ta": "tam", "te": "tel", "tg": "tgk", "th": "tha", "ti": "tir", "tl": "tgl", "to": "ton",
    "tr": "tur", "tt": "tat", "ug": "uig", "uk": "ukr", "ur": "urd", "uz": "uzb", "vi": "vie",
    "yi": "yid", "yo": "yor", "zh": "chi_sim",
}
_TRADITIONAL_CHINESE = frozenset({"TW", "HK", "MO"})


def language_name(code: str) -> str:
    """English display name of a Tesseract language code ("fin" -> "Finnish"), else the code."""
    return LANGUAGE_NAMES.get(code, code)


def locale_language(environ: dict[str, str] | None = None) -> str | None:
    """The Tesseract code of the system locale's language (LC_ALL, else LC_MESSAGES, else
    LANG): "fi_FI.UTF-8" -> "fin", "zh_TW" -> "chi_tra". None for C/POSIX/unset/unknown."""
    env = os.environ if environ is None else environ
    value = next((env[k] for k in ("LC_ALL", "LC_MESSAGES", "LANG") if env.get(k)), "")
    name, _, modifier = value.partition("@")
    name = name.split(".", 1)[0]
    lang, _, territory = name.partition("_")
    lang = lang.lower()
    if lang in ("", "c", "posix"):
        return None
    if lang == "zh":
        return "chi_tra" if territory.upper() in _TRADITIONAL_CHINESE else "chi_sim"
    if lang == "sr" and modifier.lower() == "latin":
        return "srp_latn"
    if lang in LANGUAGE_NAMES:
        return lang
    return ISO639_1.get(lang)


def default_languages(installed: Sequence[str] | None = None) -> list[str]:
    """The automatic choice: the locale's language, then English, of those installed."""
    have = set(installed_languages() if installed is None else installed)
    return [c for c in dict.fromkeys((locale_language(), "eng")) if c and c in have]


def effective_languages(cfg: Config) -> list[str]:
    """The languages OCR uses with these settings, primary first ([]: OCR is off or cannot run).
    The configured list (or the automatic one), restricted to installed languages, at most
    MAX_LANGUAGES. The order matters: the first is Tesseract's primary language."""
    if not getattr(cfg, "ocr_enabled", False) or not library_available():
        return []
    installed = installed_languages()
    wanted = cfg.ocr_languages if cfg.ocr_languages is not None else default_languages(installed)
    have = set(installed)
    return [c for c in dict.fromkeys(wanted) if c in have][:MAX_LANGUAGES]


def _os_release() -> dict[str, str]:
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        out = {}
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                out[key.strip()] = value.strip().strip("\"'")
        return out
    return {}


def install_hint(code: str | None = None, os_release: dict[str, str] | None = None) -> str:
    """The command that installs the library with English (code None) or one language's data
    on this distribution; "" when it is neither Fedora-like nor Debian-like."""
    info = _os_release() if os_release is None else os_release
    family = {info.get("ID", "").lower(), *info.get("ID_LIKE", "").lower().split()}
    if family & {"fedora", "rhel", "centos"}:
        pkgs = [f"tesseract-langpack-{code}"] if code else ["tesseract-libs", "tesseract-langpack-eng"]
        tool = "rpm-ostree install" if os.path.exists("/run/ostree-booted") else "sudo dnf install"
        return f"{tool} {' '.join(pkgs)}"
    if family & {"debian", "ubuntu"}:
        pkgs = [f"tesseract-ocr-{code.replace('_', '-')}"] if code else ["libtesseract5", "tesseract-ocr-eng"]
        return f"sudo apt install {' '.join(pkgs)}"
    return ""


SCREENSHOT_DIRS = frozenset({
    "screenshots", "screenshot", "kuvakaappaukset", "näyttökuvat", "bildschirmfotos",
    "captures d'écran", "capturas de pantalla", "capturas de ecrã", "capturas de tela", "schermate",
    "schermafbeeldingen", "schermafdrukken", "skärmbilder", "skärmdumpar", "skærmbilleder",
    "skjermbilder", "zrzuty ekranu", "snímky obrazovky", "képernyőképek", "ekran görüntüleri",
    "снимки экрана", "знімки екрана", "スクリーンショット", "截图", "屏幕截图", "螢幕擷圖", "스크린샷",
})
SCREENSHOT_PREFIXES = (
    "screenshot", "screen shot", "näyttökuva", "kuvakaappaus", "bildschirmfoto", "capture d'écran",
    "captura de pantalla", "captura de ecrã", "captura de tela", "schermata", "schermafbeelding",
    "schermafdruk", "skärmbild", "skärmavbild", "skærmbillede", "skjermbilde", "zrzut ekranu",
    "snímek obrazovky", "képernyőkép", "ekran görüntüsü", "снимок экрана", "знімок екрана",
    "スクリーンショット", "截图", "屏幕截图", "螢幕擷圖", "스크린샷",
)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold().replace("’", "'").replace("_", " ")


def is_screenshot(path: Path | str) -> bool:
    """An image file in a screenshots folder (at any depth) or named like a screenshot."""
    from . import filetypes as ft

    p = Path(path)
    if ft.classify(p.name) != ft.IMAGE:
        return False
    if _norm(p.name).startswith(SCREENSHOT_PREFIXES):
        return True
    return any(_norm(part) in SCREENSHOT_DIRS for part in p.parent.parts)


_LIB_DIRS = ("/usr/lib64", "/usr/lib", "/usr/lib/*-linux-gnu", "/usr/local/lib64", "/usr/local/lib")
_SONAME = re.compile(r"libtesseract\.so\.(\d+)((?:\.\d+)*)$")
MIN_MAJOR = 5
_TESSDATA_DIRS = ("/usr/share/tesseract/tessdata", "/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tessdata",
                  "/usr/local/share/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata")

_vp, _cp, _int = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
_API: tuple[tuple[str, Any, list[Any]], ...] = (
    ("TessVersion", _cp, []),
    ("TessBaseAPICreate", _vp, []),
    ("TessBaseAPIDelete", None, [_vp]),
    ("TessBaseAPIEnd", None, [_vp]),
    ("TessBaseAPIClear", None, [_vp]),
    ("TessBaseAPIInit3", _int, [_vp, _cp, _cp]),
    ("TessBaseAPISetVariable", _int, [_vp, _cp, _cp]),
    ("TessBaseAPISetImage", None, [_vp, _cp, _int, _int, _int, _int]),
    ("TessBaseAPISetSourceResolution", None, [_vp, _int]),
    ("TessBaseAPIRecognize", _int, [_vp, _vp]),
    ("TessBaseAPIGetIterator", _vp, [_vp]),
    ("TessBaseAPIClearPersistentCache", None, [_vp]),
    ("TessResultIteratorDelete", None, [_vp]),
    ("TessResultIteratorNext", _int, [_vp, _int]),
    ("TessResultIteratorGetPageIterator", _vp, [_vp]),
    ("TessResultIteratorGetUTF8Text", _vp, [_vp, _int]),
    ("TessResultIteratorConfidence", ctypes.c_float, [_vp, _int]),
    ("TessPageIteratorIsAtFinalElement", _int, [_vp, _int, _int]),
    ("TessDeleteText", None, [_vp]),
)
RIL_PARA, RIL_TEXTLINE, RIL_WORD = 1, 2, 3

_lib_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_lib_tried = False


def _major(name: str) -> int | None:
    m = _SONAME.search(name)
    return int(m.group(1)) if m else None


def _version_key(path: str) -> tuple[int, ...]:
    m = _SONAME.search(path)
    return tuple(int(x) for x in (m.group(1) + m.group(2)).split(".")) if m else ()


def _lib_candidates() -> list[str]:
    """Names to dlopen, best first: the Debian soname (found through the loader's cache), the
    newest versioned file in the usual directories, and what ctypes.util finds (it runs
    ldconfig/gcc, which a minimal system may not have, so it comes last)."""
    found: list[str] = []
    for d in _LIB_DIRS:
        found += glob.glob(os.path.join(d, "libtesseract.so.*"))
    files = sorted((p for p in set(found) if (_major(p) or 0) >= MIN_MAJOR), key=_version_key, reverse=True)
    out = ["libtesseract.so.5", *files]
    try:
        name = ctypes.util.find_library("tesseract")
    except Exception:
        name = None
    if name and (_major(name) or 0) >= MIN_MAJOR:
        out.append(name)
    return list(dict.fromkeys(out))


def _bind(lib: ctypes.CDLL) -> bool:
    for name, restype, argtypes in _API:
        try:
            fn = getattr(lib, name)
        except AttributeError:
            log.debug("libtesseract has no %s", name)
            return False
        fn.restype, fn.argtypes = restype, argtypes
    version = (lib.TessVersion() or b"").decode("ascii", "replace")
    try:
        return int(version.split(".")[0]) >= MIN_MAJOR
    except ValueError:
        return False


def _library() -> ctypes.CDLL | None:
    """The bound libtesseract, loaded once per process (None: not installed or unusable)."""
    global _lib, _lib_tried
    if _lib_tried:
        return _lib
    with _lib_lock:
        if not _lib_tried:
            for name in _lib_candidates():
                try:
                    lib = ctypes.CDLL(name)
                except OSError:
                    continue
                if _bind(lib):
                    _lib = lib
                    break
            _lib_tried = True
    return _lib


def library_available() -> bool:
    """libtesseract 5 (or later, with the C API used here) can be loaded."""
    return _library() is not None


def tessdata_dir() -> Path | None:
    """The language data directory: $TESSDATA_PREFIX when set (as Tesseract reads it), else the
    first of the distributions' directories that has data. OCR passes it to Tesseract
    explicitly, so `installed_languages` always lists what an engine will load."""
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        d = Path(env)
        if not any(d.glob("*.traineddata")) and any((d / "tessdata").glob("*.traineddata")):
            d = d / "tessdata"
        return d if d.is_dir() else None
    for name in _TESSDATA_DIRS:
        d = Path(name)
        try:
            if any(d.glob("*.traineddata")):
                return d
        except OSError:
            continue
    return None


def installed_languages() -> list[str]:
    """Sorted codes of the languages with data installed ([]: no library or no data)."""
    if not library_available():
        return []
    d = tessdata_dir()
    if d is None:
        return []
    try:
        names = os.listdir(d)
    except OSError:
        return []
    codes = {n.removesuffix(".traineddata") for n in names if n.endswith(".traineddata")}
    return sorted(c for c in codes if LANGUAGE_CODE.fullmatch(c) and c not in NOT_LANGUAGES)


def filter_words(words: Iterable[tuple[str, float, bool]]) -> str:
    """Text from Tesseract's (word, confidence 0-100, ends a line) triples: unreliable words
    dropped (see MIN_WORD_CONFIDENCE), one output line per text line; "" when what is left is
    not enough real text (MIN_WORDS, MIN_ALNUM)."""
    lines: list[str] = []
    line: list[str] = []
    long_words = alnum = 0
    for word, conf, line_end in words:
        word = word.strip()
        n = sum(ch.isalnum() for ch in word)
        if n and conf >= (MIN_WORD_CONFIDENCE if n >= 3 else MIN_SHORT_WORD_CONFIDENCE):
            line.append(word)
            alnum += n
            long_words += n >= 3
        if line_end and line:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    if long_words < MIN_WORDS or alnum < MIN_ALNUM:
        return ""
    return "\n".join(lines)


def prepare_image(image: PILImage.Image) -> PILImage.Image:
    """8-bit grey at full resolution, at most MAX_OCR_PIXELS (transparency composited on white)."""
    from PIL import Image

    from .extract.image import to_rgb

    if image.mode != "L":
        image = (image if image.mode == "RGB" else to_rgb(image)).convert("L")
    w, h = image.size
    if w * h > MAX_OCR_PIXELS:
        scale = (MAX_OCR_PIXELS / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS,
                             reducing_gap=3.0)
    return image


class _Engine:
    """One initialised TessBaseAPI handle for `languages`; used by one thread at a time."""

    def __init__(self, lib: ctypes.CDLL, datapath: Path, languages: tuple[str, ...]):
        self.lib = lib
        self.languages = languages
        self.lock = threading.Lock()
        self.handle: int | None = lib.TessBaseAPICreate()
        if not self.handle:
            raise OSError("TessBaseAPICreate failed")
        lib.TessBaseAPISetVariable(self.handle, b"debug_file", b"/dev/null")
        if lib.TessBaseAPIInit3(self.handle, os.fsencode(str(datapath)), "+".join(languages).encode()) != 0:
            lib.TessBaseAPIDelete(self.handle)
            self.handle = None
            raise OSError(f"Tesseract cannot load {'+'.join(languages)} from {datapath}")

    def words(self, image: PILImage.Image) -> list[tuple[str, float, bool]]:
        """(word, confidence, ends a line) for an "L" image."""
        lib, h = self.lib, self.handle
        assert h is not None and image.mode == "L"
        data = image.tobytes()
        lib.TessBaseAPISetImage(h, data, image.width, image.height, 1, image.width)
        lib.TessBaseAPISetSourceResolution(h, SOURCE_DPI)
        out: list[tuple[str, float, bool]] = []
        try:
            if lib.TessBaseAPIRecognize(h, None) != 0:
                return out
            it = lib.TessBaseAPIGetIterator(h)
            if not it:
                return out
            try:
                page = lib.TessResultIteratorGetPageIterator(it)
                while True:
                    p = lib.TessResultIteratorGetUTF8Text(it, RIL_WORD)
                    if p:
                        try:
                            word = ctypes.string_at(p).decode("utf-8", "replace")
                        finally:
                            lib.TessDeleteText(p)
                        end = bool(lib.TessPageIteratorIsAtFinalElement(page, RIL_TEXTLINE, RIL_WORD))
                        out.append((word, float(lib.TessResultIteratorConfidence(it, RIL_WORD)), end))
                    if not lib.TessResultIteratorNext(it, RIL_WORD):
                        break
            finally:
                lib.TessResultIteratorDelete(it)
        finally:
            lib.TessBaseAPIClear(h)
        return out

    def close(self) -> None:
        h, self.handle = self.handle, None
        if h:
            self.lib.TessBaseAPIEnd(h)
            self.lib.TessBaseAPIClearPersistentCache(h)
            self.lib.TessBaseAPIDelete(h)


_local = threading.local()
_engines: set[_Engine] = set()
_engines_lock = threading.Lock()
_failed: set[tuple[str, tuple[str, ...]]] = set()
_atexit_registered = False


def _engine(languages: tuple[str, ...]) -> _Engine | None:
    """This thread's engine for `languages`, created (or re-created for another list) on demand."""
    global _atexit_registered
    current: _Engine | None = getattr(_local, "engine", None)
    if current is not None and current.languages == languages and current.handle:
        return current
    if current is not None:
        _free(current)
        _local.engine = None
    lib = _library()
    datapath = tessdata_dir()
    if lib is None or datapath is None:
        return None
    key = (str(datapath), languages)
    if key in _failed:
        return None
    try:
        engine = _Engine(lib, datapath, languages)
    except OSError as exc:
        _failed.add(key)
        log.warning("OCR unavailable: %s", exc)
        return None
    with _engines_lock:
        _engines.add(engine)
        if not _atexit_registered:
            atexit.register(_release_all)
            _atexit_registered = True
    _local.engine = engine
    log.debug("OCR engine loaded (%s)", "+".join(languages))
    return engine


def _free(engine: _Engine) -> None:
    with _engines_lock:
        _engines.discard(engine)
    with engine.lock:
        engine.close()


def _trim_heap() -> None:
    """Give freed heap back to the system: after End/Delete glibc keeps Tesseract's ~25-50 MB
    in its arenas (RSS unchanged) until trimmed."""
    try:
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except OSError:
        pass


def release_thread() -> bool:
    """Free the calling thread's engine, if it has one (the end of an indexing run). Returns
    True if one was freed."""
    engine: _Engine | None = getattr(_local, "engine", None)
    if engine is None:
        return False
    _local.engine = None
    _free(engine)
    _trim_heap()
    log.debug("OCR engine freed")
    return True


def _release_all() -> None:
    """At exit: end the engines not in use, so libtesseract's cache does not print leak warnings."""
    with _engines_lock:
        engines = list(_engines)
        _engines.clear()
    for engine in engines:
        if engine.lock.acquire(blocking=False):
            try:
                engine.close()
            finally:
                engine.lock.release()


def recognize(image: PILImage.Image, languages: Sequence[str]) -> str:
    """The recognised and filtered text of `image` ("" when there is nothing usable, or OCR
    cannot run). Runs on the calling thread's engine; call `release_thread` when done."""
    langs = tuple(languages)[:MAX_LANGUAGES]
    if not langs:
        return ""
    engine = _engine(langs)
    if engine is None:
        return ""
    gray = prepare_image(image)
    with engine.lock:
        if not engine.handle:
            return ""
        words = engine.words(gray)
    return filter_words(words)
