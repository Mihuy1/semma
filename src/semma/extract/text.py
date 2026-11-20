"""Plain-text files: decoding, binary detection and the reusable text chunker.

Chunking targets ~2,000 characters (~500-700 tokens) per chunk, never more than 2,500, with
~250 characters (~12 %) of overlap between consecutive chunks. Cut points are chosen by
preference: a heading (no overlap needed, the heading starts the next chunk), a paragraph
break, a sentence end, a line break, any whitespace, and only as a last resort a hard cut.
"""

from __future__ import annotations

import bisect
import codecs
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Chunk, ExtractionError

if TYPE_CHECKING:
    from ..config import Config

TARGET_CHARS = 2000
MAX_CHARS = 2500
MIN_CHARS = 1000
MIN_HEADING_CUT = 300
OVERLAP_CHARS = 250
SNIPPET_CHARS = 160
MAX_CHUNKS = 1000
MAX_HEADING_IN_LOCATOR = 48

_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE = re.compile(r"[.!?…][\"'”’»)\]]*\s+")
_NEWLINE = re.compile(r"\n\s*")
_SPACE = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class TextSpan:
    """One chunk of a longer text: its character range, stripped text and nearest heading."""

    start: int
    end: int
    text: str
    heading: str

    @property
    def starts_with_heading(self) -> bool:
        return bool(self.heading) and self.text.startswith(self.heading[:20])

    def content(self) -> str:
        """Chunk text for the embedder: prefixed with its section heading when that is cut off."""
        if self.heading and not self.starts_with_heading:
            return f"{self.heading}\n\n{self.text}"
        return self.text


def _best(pattern: re.Pattern[str], text: str, lo: int, hi: int, target: int) -> int | None:
    """End offset of the match of `pattern` in text[lo:hi] closest to `target`."""
    best = None
    for m in pattern.finditer(text, lo, hi):
        if best is None or abs(m.end() - target) < abs(best - target):
            best = m.end()
    return best if best is not None and best <= hi else None


def _overlap_start(text: str, s: int, end: int, overlap: int) -> int:
    """Where the next chunk starts: ~`overlap` chars before `end`, at a sentence/word start."""
    if overlap <= 0:
        return end
    lo = max(s + 1, end - int(overlap * 1.4))
    hi = max(lo, end - overlap // 2)
    m = _SENTENCE.search(text, lo, hi)
    if m:
        return m.end()
    lo = max(s + 1, end - overlap)
    m = _SPACE.search(text, lo, end)
    if m and m.end() < end:
        return m.end()
    return max(s + 1, end - overlap)


def chunk_text(
    text: str,
    headings: Sequence[tuple[int, str]] = (),
    *,
    target: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> Iterator[TextSpan]:
    """Split `text` into overlapping chunks. Lazy; never yields empty/whitespace-only spans.

    `headings` is a sorted sequence of (offset of the heading line, heading title); chunks
    preferably start at a heading and each span reports the heading it falls under.
    """
    n = len(text)
    h_offsets = [o for o, _ in headings]
    s = _skip_space(text, 0)
    while s < n:
        if n - s <= max_chars:
            end = nxt = n
        else:
            hi = s + max_chars
            i = bisect.bisect_right(h_offsets, hi) - 1
            if i >= 0 and h_offsets[i] >= s + MIN_HEADING_CUT:
                end = nxt = h_offsets[i]
            else:
                lo = s + min_chars
                tgt = s + target
                end = (
                    _best(_PARAGRAPH, text, lo, hi, tgt)
                    or _best(_SENTENCE, text, lo, hi, tgt)
                    or _best(_NEWLINE, text, lo, hi, tgt)
                    or _best(_SPACE, text, lo, hi, tgt)
                    or hi
                )
                j = bisect.bisect_left(h_offsets, end - 2)
                at_heading = j < len(h_offsets) and h_offsets[j] <= end + 2
                nxt = end if at_heading else _overlap_start(text, s, end, overlap)
        body = text[s:end].strip()
        if body:
            k = bisect.bisect_right(h_offsets, s) - 1
            heading = headings[k][1] if k >= 0 else ""
            yield TextSpan(s, end, body, heading)
        s = _skip_space(text, max(nxt, s + 1))


def _skip_space(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The first ~`limit` characters, whitespace-collapsed, cut at a word boundary."""
    flat = " ".join(text[: limit * 3].split())
    if len(flat) <= limit:
        return flat
    cut = flat.rfind(" ", int(limit * 0.6), limit)
    return flat[: cut if cut > 0 else limit].rstrip(",;:") + "…"


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\x0c", "\n\n").replace("\x00", "").replace("\u2028", "\n").replace("\u2029", "\n\n")


def _short(s: str, limit: int = MAX_HEADING_IN_LOCATOR) -> str:
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def spans_to_chunks(
    spans: Iterable[TextSpan],
    locate,
    *,
    single_locator: tuple[str, float | None] = ("", None),
) -> Iterator[Chunk]:
    """Turn spans into text Chunks; a file that yields a single span gets `single_locator`."""
    it = iter(spans)
    first = next(it, None)
    if first is None:
        return
    second = next(it, None)
    if second is None:
        loc, pos = single_locator
        yield Chunk("text", first.content(), loc, make_snippet(first.text), pos)
        return
    for i, span in enumerate(_chain(first, second, it)):
        if i >= MAX_CHUNKS:
            return
        loc, pos = locate(span, i)
        yield Chunk("text", span.content(), loc, make_snippet(span.text), pos)


def _chain(a, b, rest):
    yield a
    yield b
    yield from rest


_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_ALLOWED_CONTROLS = frozenset("\t\n\r\x0c\x0b\x1b")


def _control_ratio(text: str) -> float:
    sample = text[:65536]
    if not sample:
        return 0.0
    bad = sum(1 for c in sample if c not in _ALLOWED_CONTROLS and unicodedata.category(c) == "Cc")
    return bad / len(sample)


def _utf16_without_bom(sample: bytes) -> str | None:
    if len(sample) < 4:
        return None
    even, odd = sample[0::2], sample[1::2]
    if odd.count(0) > 0.3 * len(odd) and even.count(0) < 0.05 * len(even):
        return "utf-16-le"
    if even.count(0) > 0.3 * len(even) and odd.count(0) < 0.05 * len(odd):
        return "utf-16-be"
    return None


_WESTERN_EXTRA = frozenset("–—‘’‚“”„…•€°§«»±²³µ¶·¹º¼½¾¿¡¢£¥©®™×÷  ")


def _western_legacy(data: bytes) -> str | None:
    """cp1252 (a superset of Latin-1 for text), if the result looks like Western European text.

    charset-normalizer is unreliable on short Finnish/German/... texts (it guesses e.g. mac_latin2),
    so the overwhelmingly common Windows-1252 case is checked first: the decoding must be valid,
    non-ASCII characters must be Latin-1 letters or common punctuation, and only a minority of
    letters may be non-ASCII (that rejects Cyrillic/Greek/... single-byte encodings).
    """
    try:
        text = data.decode("cp1252")
    except UnicodeDecodeError:
        return None
    sample = text[:200_000]
    letters = non_ascii_letters = implausible = 0
    for c in sample:
        if c.isascii():
            letters += c.isalpha()
            continue
        if c.isalpha():
            letters += 1
            non_ascii_letters += 1
            if not ("À" <= c <= "ÿ" or c in "ŠšŽžŒœŸƒ"):
                implausible += 1
        elif c not in _WESTERN_EXTRA:
            implausible += 1
    if letters == 0 or non_ascii_letters > 0.35 * letters:
        return None
    if implausible > 0.05 * max(1, non_ascii_letters):
        return None
    return text


def decode_text(data: bytes) -> str:
    """Decode bytes of a text file; raise ExtractionError if it looks binary."""
    if not data:
        return ""
    text: str | None = None
    for bom, enc in _BOMS:
        if data.startswith(bom):
            text = data[len(bom) :].decode(enc, errors="replace")
            break
    if text is None:
        sample = data[:8192]
        if b"\x00" in sample:
            enc = _utf16_without_bom(sample[: len(sample) // 2 * 2])
            if enc is None:
                raise ExtractionError("binary content (NUL bytes), not a text file")
            text = data.decode(enc, errors="replace")
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = _western_legacy(data)
            if text is None:
                from charset_normalizer import from_bytes

                best = from_bytes(data).best()
                if best is None:
                    raise ExtractionError("binary or undecodable content, not a text file") from None
                text = str(best)
    if _control_ratio(text) > 0.01:
        raise ExtractionError("binary content (control characters), not a text file")
    return normalize_newlines(text)


MAX_HEADING_LINE = 300

_MD_ATX = re.compile(r" {0,3}(#{1,6})[ \t]+(\S.*)")
_MD_SETEXT_EXCLUDE = re.compile(r" {0,3}[-=#>*+|]")
_MD_SETEXT_UNDER = re.compile(r" {0,3}(?:=+|-{3,})[ \t]*")
_MD_FENCE = re.compile(r" {0,3}(?:```|~~~)")
_ORG = re.compile(r"(\*+)[ \t]+(\S.*)")
_ADOC = re.compile(r"(={1,6})[ \t]+(\S.*)")
_RST_UNDER = re.compile(r"([=\-~^\"'`#*+:.])\1{2,}[ \t]*")
_TEX = re.compile(r"^[ \t]*\\(?:part|chapter|section|subsection|subsubsection)\*?\{([^}\n]+)\}", re.M)


def _lines(text: str) -> list[tuple[int, str]]:
    """(offset, line) for every line of `text` (without the newline)."""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((pos, line))
        pos += len(line) + 1
    return out


def _short_line(line: str) -> bool:
    return len(line) <= MAX_HEADING_LINE


def _markdown_headings(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    lines = _lines(text)
    in_fence = False
    skip_next = False
    for i, (off, line) in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if _MD_FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not _short_line(line):
            continue
        if m := _MD_ATX.match(line):
            found.append((off, m.group(2).rstrip(" \t#")))
            continue
        if i + 1 < len(lines) and not _MD_SETEXT_EXCLUDE.match(line):
            lead = len(line) - len(line.lstrip(" "))
            title = line.rstrip(" \t")
            if lead <= 3 and title[lead : lead + 1].strip():
                if _MD_SETEXT_UNDER.fullmatch(lines[i + 1][1]):
                    found.append((off, title[lead:]))
                    skip_next = True
    return found


def _prefix_headings(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """Single-line headings like org `* Title` and AsciiDoc `= Title`."""
    found = []
    for off, line in _lines(text):
        if line[:1] in ("*", "=") and _short_line(line) and (m := pattern.match(line)):
            found.append((off, m.group(2).rstrip(" \t")))
    return found


def _rst_headings(text: str) -> list[tuple[int, str]]:
    found = []
    lines = _lines(text)
    i = 0
    while i + 1 < len(lines):
        off, line = lines[i]
        under = lines[i + 1][1]
        if line[:1].strip() and _short_line(line) and _RST_UNDER.fullmatch(under):
            title = line.rstrip(" \t")
            if len(title) <= len(line) - len(title) + 1 + len(under) + 2:
                found.append((off, title))
                i += 2
                continue
        i += 1
    return found


def find_headings(text: str, suffix: str) -> list[tuple[int, str]]:
    """(offset, title) of section headings for markup formats; [] for plain text. Linear time."""
    suffix = suffix.lower().lstrip(".")
    found: list[tuple[int, str]] = []
    if suffix in ("md", "markdown"):
        found = _markdown_headings(text)
    elif suffix == "org":
        found = _prefix_headings(text, _ORG)
    elif suffix in ("adoc", "asciidoc"):
        found = _prefix_headings(text, _ADOC)
    elif suffix == "rst":
        found = _rst_headings(text)
    elif suffix == "tex":
        found = [(m.start(), m.group(1)) for m in _TEX.finditer(text)]
    found = [(o, " ".join(t.split())) for o, t in found if t.strip()]
    return sorted(set(found))


def extract_text(path: Path, cfg: Config | None = None) -> Iterator[Chunk]:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ExtractionError(f"cannot read file: {e.strerror or e}") from e
    text = decode_text(data)
    del data
    headings = find_headings(text, path.suffix)

    line = 1
    last = 0

    def locate(span: TextSpan, _i: int) -> tuple[str, float | None]:
        nonlocal line, last
        line += text.count("\n", last, span.start)
        last = span.start
        loc = f"line {line}"
        if span.heading:
            loc += f" · {_short(span.heading)}"
        return loc, float(line)

    yield from spans_to_chunks(chunk_text(text, headings), locate)
