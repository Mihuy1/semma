"""Office documents without third-party libraries: ODF (odt/ott/odp), DOCX, PPTX and RTF.

ZIP members are read with an uncompressed-size cap (zip bombs) and any XML with a DTD is
rejected by the parser itself, whatever the encoding (entity-expansion attacks; real office XML
never has one).
"""

from __future__ import annotations

import codecs
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Chunk, ExtractionError
from .text import (
    MAX_CHUNKS,
    TextSpan,
    _short,
    chunk_text,
    make_snippet,
    normalize_newlines,
    spans_to_chunks,
)

if TYPE_CHECKING:
    from ..config import Config

MAX_MEMBER_BYTES = 32 * 2**20
MAX_TOTAL_BYTES = 256 * 2**20
MAX_SLIDES = 1000
MIN_SLIDE_CHARS = 3

NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _q(tag: str) -> str:
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


class _Zip:
    """A zip archive with bounded, DTD-free XML reads."""

    def __init__(self, path: Path) -> None:
        try:
            self.zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile as e:
            raise ExtractionError("not a valid office file (bad zip archive)") from e
        except OSError as e:
            raise ExtractionError(f"cannot read file: {e.strerror or e}") from e
        self.total = 0

    def __enter__(self) -> _Zip:
        return self

    def __exit__(self, *exc) -> None:
        self.zf.close()

    def names(self) -> list[str]:
        return self.zf.namelist()

    def has(self, name: str) -> bool:
        try:
            self.zf.getinfo(name)
            return True
        except KeyError:
            return False

    def read(self, name: str) -> bytes:
        try:
            info = self.zf.getinfo(name)
        except KeyError:
            raise ExtractionError(f"missing part {name}") from None
        if info.flag_bits & 0x1:
            raise ExtractionError("document is encrypted")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ExtractionError(f"{name} too large ({info.file_size >> 20} MiB uncompressed)")
        try:
            with self.zf.open(info) as fh:
                data = fh.read(MAX_MEMBER_BYTES + 1)
        except (zipfile.BadZipFile, OSError, EOFError, NotImplementedError) as e:
            raise ExtractionError(f"cannot read {name}: {e}") from e
        if len(data) > MAX_MEMBER_BYTES:
            raise ExtractionError(f"{name} too large")
        self.total += len(data)
        if self.total > MAX_TOTAL_BYTES:
            raise ExtractionError("archive expands to too much data")
        return data

    def xml(self, name: str) -> ET.Element:
        return parse_xml(self.read(name), name)


class _DTDFound(Exception):
    pass


class _NoDTDBuilder(ET.TreeBuilder):
    """A TreeBuilder that aborts parsing at the start of any <!DOCTYPE ...>.

    expat reports the doctype before reading its internal subset, so no entity is ever
    declared. This works for every encoding expat accepts (a byte search for b"<!DOCTYPE"
    misses UTF-16 parts).
    """

    def doctype(self, name, pubid, system):
        raise _DTDFound


def _reject_dtd(*_args) -> None:
    raise _DTDFound


def parse_xml(data: bytes, name: str = "XML") -> ET.Element:
    parser = ET.XMLParser(target=_NoDTDBuilder())
    expat = getattr(parser, "parser", None)
    if expat is not None:
        expat.StartDoctypeDeclHandler = _reject_dtd
        expat.EntityDeclHandler = _reject_dtd
    try:
        parser.feed(data)
        return parser.close()
    except _DTDFound:
        raise ExtractionError(f"{name} contains a DTD; refusing to parse") from None
    except (ET.ParseError, ValueError) as e:
        raise ExtractionError(f"{name} is not valid XML ({e}); encrypted or corrupt?") from e


class _Doc:
    """Accumulates paragraphs into one text plus heading offsets for the chunker."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.headings: list[tuple[int, str]] = []

    def para(self, text: str, heading: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        if heading:
            self.headings.append((self.length, " ".join(text.split())))
        self.parts.append(text)
        self.parts.append("\n\n")
        self.length += len(text) + 2

    def text(self) -> str:
        return "".join(self.parts)


def _doc_chunks(doc: _Doc) -> Iterator[Chunk]:
    text = doc.text()

    def locate(span: TextSpan, i: int) -> tuple[str, float | None]:
        return (f"§ {_short(span.heading)}" if span.heading else f"part {i + 1}"), None

    yield from spans_to_chunks(chunk_text(text, doc.headings), locate)


def _slide_chunks(slides: Iterator[tuple[int, str]]) -> Iterator[Chunk]:
    count = 0
    for number, text in slides:
        text = normalize_newlines(text).strip()
        if sum(not c.isspace() for c in text) < MIN_SLIDE_CHARS:
            continue
        for span in chunk_text(text):
            if count >= MAX_CHUNKS:
                return
            count += 1
            yield Chunk("text", span.text, f"slide {number}", make_snippet(span.text), float(number))


_ODF_SKIP = {
    _q(t)
    for t in (
        "text:tracked-changes", "office:annotation", "office:annotation-end", "text:note-citation",
        "text:table-of-content", "text:illustration-index", "text:table-index",
        "text:object-index", "text:user-index", "text:alphabetical-index",
        "text:sequence-decls", "text:variable-decls", "text:user-field-decls",
        "office:forms", "svg:title", "svg:desc", "office:event-listeners",
    )
}
_ODF_PARA = {_q("text:p"), _q("text:h")}
_ODF_S, _ODF_TAB, _ODF_BR = _q("text:s"), _q("text:tab"), _q("text:line-break")
_ODF_NOTE_BODY = _q("text:note-body")
_ODF_ROW, _ODF_CELL = _q("table:table-row"), _q("table:table-cell")
_ODF_FRAME = _q("draw:frame")


def _odf_inline(el: ET.Element, out: list[str], notes: list[ET.Element]) -> None:
    """Text of a paragraph element (inline content only; nested paragraphs are deferred)."""
    if el.text:
        out.append(el.text)
    for child in el:
        tag = child.tag
        if tag == _ODF_S:
            out.append(" " * int(child.get(_q("text:c"), "1") or 1))
        elif tag == _ODF_TAB:
            out.append("\t")
        elif tag == _ODF_BR:
            out.append("\n")
        elif tag == _ODF_NOTE_BODY or tag == _ODF_FRAME:
            notes.append(child)
        elif tag not in _ODF_SKIP:
            _odf_inline(child, out, notes)
        if child.tail:
            out.append(child.tail)


def _odf_paragraph_text(el: ET.Element, doc: _Doc | None, heading: bool) -> str:
    out: list[str] = []
    notes: list[ET.Element] = []
    _odf_inline(el, out, notes)
    text = "".join(out)
    if doc is not None:
        doc.para(text, heading)
        for note in notes:
            _odf_blocks(note, doc)
    return text


def _odf_blocks(el: ET.Element, doc: _Doc) -> None:
    for child in el:
        tag = child.tag
        if tag in _ODF_SKIP:
            continue
        if tag in _ODF_PARA:
            _odf_paragraph_text(child, doc, heading=tag == _q("text:h"))
        elif tag == _ODF_ROW:
            cells = []
            for cell in child:
                if cell.tag not in (_ODF_CELL, _q("table:covered-table-cell")):
                    continue
                sub = _Doc()
                _odf_blocks(cell, sub)
                cells.append(" ".join(sub.text().split()))
            doc.para(" | ".join(c for c in cells if c))
        else:
            _odf_blocks(child, doc)


def _odf_text(root: ET.Element) -> _Doc:
    doc = _Doc()
    body = root.find("office:body", NS)
    if body is None:
        raise ExtractionError("content.xml has no office:body")
    _odf_blocks(body, doc)
    return doc


def _odp_slides(root: ET.Element) -> Iterator[tuple[int, str]]:
    pres = root.find("office:body/office:presentation", NS)
    if pres is None:
        return
    for number, page in enumerate(pres.iterfind("draw:page", NS), 1):
        if number > MAX_SLIDES:
            break
        doc = _Doc()
        _odf_blocks(page, doc)
        yield number, doc.text()


def extract_odf(path: Path) -> Iterator[Chunk]:
    with _Zip(path) as z:
        root = z.xml("content.xml")
    if root.find("office:body/office:presentation", NS) is not None:
        yield from _slide_chunks(_odp_slides(root))
    else:
        yield from _doc_chunks(_odf_text(root))


_W_P, _W_TBL, _W_TR, _W_TC = _q("w:p"), _q("w:tbl"), _q("w:tr"), _q("w:tc")
_W_T, _W_TAB, _W_BR, _W_CR = _q("w:t"), _q("w:tab"), _q("w:br"), _q("w:cr")
_W_NBH, _W_SH = _q("w:noBreakHyphen"), _q("w:softHyphen")
_W_TXBX = _q("w:txbxContent")
_W_PPR, _W_RPR = _q("w:pPr"), _q("w:rPr")
_W_VAL = _q("w:val")
_MC_FALLBACK = _q("mc:Fallback")
_W_SKIP_INLINE = {_q("w:del"), _q("w:delText"), _q("w:instrText"), _MC_FALLBACK, _W_PPR, _W_RPR}


def _docx_heading_styles(z: _Zip) -> set[str]:
    if not z.has("word/styles.xml"):
        return set()
    try:
        root = z.xml("word/styles.xml")
    except ExtractionError:
        return set()
    ids: set[str] = set()
    for style in root.iterfind("w:style", NS):
        sid = style.get(_q("w:styleId"), "")
        name_el = style.find("w:name", NS)
        name = (name_el.get(_W_VAL, "") if name_el is not None else "").lower()
        if re.match(r"(heading \d|title)$", name) or style.find("w:pPr/w:outlineLvl", NS) is not None:
            ids.add(sid)
    return ids


def _docx_inline(el: ET.Element, out: list[str], boxes: list[ET.Element]) -> None:
    for child in el:
        tag = child.tag
        if tag == _W_T:
            out.append(child.text or "")
        elif tag == _W_TAB:
            out.append("\t")
        elif tag in (_W_BR, _W_CR):
            out.append("\n")
        elif tag == _W_NBH:
            out.append("-")
        elif tag == _W_TXBX:
            boxes.append(child)
        elif tag not in _W_SKIP_INLINE and tag != _W_SH:
            _docx_inline(child, out, boxes)


def _docx_blocks(el: ET.Element, doc: _Doc, heading_styles: set[str]) -> None:
    for child in el:
        tag = child.tag
        if tag == _W_P:
            out: list[str] = []
            boxes: list[ET.Element] = []
            _docx_inline(child, out, boxes)
            ppr = child.find("w:pPr", NS)
            heading = False
            if ppr is not None:
                style = ppr.find("w:pStyle", NS)
                heading = (style is not None and style.get(_W_VAL) in heading_styles) or (
                    ppr.find("w:outlineLvl", NS) is not None
                )
            doc.para("".join(out), heading)
            for box in boxes:
                _docx_blocks(box, doc, heading_styles)
        elif tag == _W_TBL:
            for row in child.iterfind("w:tr", NS):
                cells = []
                for cell in row.iterfind("w:tc", NS):
                    sub = _Doc()
                    _docx_blocks(cell, sub, heading_styles)
                    cells.append(" ".join(sub.text().split()))
                doc.para(" | ".join(c for c in cells if c))
        elif tag not in (_MC_FALLBACK, _q("w:sectPr"), _q("w:del")):
            _docx_blocks(child, doc, heading_styles)


def extract_docx(path: Path) -> Iterator[Chunk]:
    with _Zip(path) as z:
        root = z.xml("word/document.xml")
        heading_styles = _docx_heading_styles(z)
        doc = _Doc()
        body = root.find("w:body", NS)
        if body is None:
            raise ExtractionError("word/document.xml has no w:body")
        _docx_blocks(body, doc, heading_styles)
        for part in ("word/footnotes.xml", "word/endnotes.xml"):
            if z.has(part):
                _docx_blocks(z.xml(part), doc, heading_styles)
    yield from _doc_chunks(doc)


_A_P, _A_T, _A_BR = _q("a:p"), _q("a:t"), _q("a:br")
_SLIDE_RE = re.compile(r"ppt/slides/slide(\d+)\.xml$")


def _pptx_text(root: ET.Element, only_body_placeholders: bool = False) -> str:
    paras: list[str] = []
    shapes = root.iter(_q("p:sp")) if only_body_placeholders else [root]
    for shape in shapes:
        if only_body_placeholders:
            ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            if ph is None or ph.get("type") != "body":
                continue
        for p in shape.iter(_A_P):
            out: list[str] = []
            for el in p.iter():
                if el.tag == _A_T:
                    out.append(el.text or "")
                elif el.tag == _A_BR:
                    out.append("\n")
            line = "".join(out).strip()
            if line:
                paras.append(line)
    return "\n".join(paras)


def _pptx_notes_part(z: _Zip, slide_name: str) -> str | None:
    rels = slide_name.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels"
    if not z.has(rels):
        return None
    try:
        root = z.xml(rels)
    except ExtractionError:
        return None
    for rel in root.iterfind("rel:Relationship", NS):
        if rel.get("Type", "").endswith("/notesSlide"):
            target = rel.get("Target", "")
            name = "ppt/" + target.removeprefix("../") if target.startswith("../") else target
            return name.lstrip("/")
    return None


def _pptx_slide_order(z: _Zip) -> list[str]:
    """Slide part names in presentation order (p:sldIdLst), or [] if that cannot be read."""
    rels_name = "ppt/_rels/presentation.xml.rels"
    if not (z.has("ppt/presentation.xml") and z.has(rels_name)):
        return []
    try:
        pres = z.xml("ppt/presentation.xml")
        rels = z.xml(rels_name)
    except ExtractionError:
        return []
    targets = {}
    for rel in rels.iterfind("rel:Relationship", NS):
        target = rel.get("Target", "")
        if rel.get("TargetMode") == "External" or not target:
            continue
        path = target.lstrip("/") if target.startswith("/") else posixpath.join("ppt", target)
        targets[rel.get("Id")] = posixpath.normpath(path)
    order: list[str] = []
    for sld in pres.iterfind("p:sldIdLst/p:sldId", NS):
        name = targets.get(sld.get(_q("r:id")))
        if name and name not in order and z.has(name):
            order.append(name)
    return order


def extract_pptx(path: Path) -> Iterator[Chunk]:
    with _Zip(path) as z:
        order = _pptx_slide_order(z)
        if order:
            slides = list(enumerate(order, 1))[:MAX_SLIDES]
        else:
            slides = sorted(
                ((int(m.group(1)), name) for name in z.names() if (m := _SLIDE_RE.match(name))),
            )[:MAX_SLIDES]
        if not slides:
            raise ExtractionError("no slides found (not a PPTX presentation?)")

        def gen() -> Iterator[tuple[int, str]]:
            for number, name in slides:
                text = _pptx_text(z.xml(name))
                notes = _pptx_notes_part(z, name)
                if notes and z.has(notes):
                    note_text = _pptx_text(z.xml(notes), only_body_placeholders=True)
                    if note_text:
                        text = f"{text}\n\n{note_text}"
                yield number, text

        items = list(gen())
    yield from _slide_chunks(iter(items))


_RTF_TOKEN = re.compile(
    rb"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|[\r\n]+|([^\\{}\r\n]+)",
    re.S,
)
_RTF_SKIP_DEST = frozenset(
    b"""fonttbl colortbl stylesheet info pict object objdata fldinst header footer headerl
    headerr headerf footerl footerr footerf themedata colorschememapping datastore latentstyles
    listtable listoverridetable rsidtbl generator xmlnstbl mmathPr revtbl filetbl bkmkstart
    bkmkend nonshppict private pgdsctbl listtext pntext pntxta pntxtb xe tc factoidname
    userprops docvar template operator author title subject keywords comment doccomm
    shppict blipuid wgrffmtfilter fchars lchars background""".split()
)
_RTF_CHARS = {
    b"par": "\n\n", b"pard": "", b"line": "\n", b"tab": "\t", b"cell": " | ", b"nestcell": " | ",
    b"row": "\n", b"nestrow": "\n", b"sect": "\n\n", b"page": "\n\n", b"column": "\n",
    b"emdash": "—", b"endash": "–", b"emspace": " ", b"enspace": " ", b"qmspace": " ",
    b"bullet": "•", b"lquote": "‘", b"rquote": "’", b"ldblquote": "“", b"rdblquote": "”",
    b"zwj": "\u200d", b"zwnj": "\u200c",
}
_RTF_SYMBOLS = {b"~": "\u00a0", b"-": "", b"_": "\u2011", b"\\": "\\", b"{": "{", b"}": "}"}


def rtf_to_text(data: bytes) -> str:
    """A small, forgiving RTF-to-plain-text converter."""
    if not data.lstrip().startswith(b"{\\rtf"):
        raise ExtractionError("not an RTF document")
    out: list[str] = []
    pending = bytearray()
    encoding = "cp1252"
    skip = False
    uc = 1
    stack: list[tuple[bool, int]] = []
    to_skip = 0
    first_in_group = False
    star = False

    def flush() -> None:
        if pending:
            out.append(pending.decode(encoding, errors="replace"))
            pending.clear()

    pos, n = 0, len(data)
    while pos < n:
        m = _RTF_TOKEN.match(data, pos)
        if m is None:
            pos += 1
            continue
        pos = m.end()
        word, arg, hexbyte, symbol, brace, text = m.groups()
        if hexbyte is None:
            flush()
        if brace == b"{":
            stack.append((skip, uc))
            first_in_group, star = True, False
            continue
        if brace == b"}":
            if stack:
                skip, uc = stack.pop()
            to_skip = 0
            first_in_group = star = False
            continue
        was_first, first_in_group = first_in_group, False
        if symbol == b"*":
            star = was_first
            continue
        if word is not None:
            if star or word in _RTF_SKIP_DEST:
                skip = True
            star = False
            if word == b"bin" and arg:
                pos += max(0, int(arg))
                continue
            if word == b"ansicpg" and arg:
                try:
                    encoding = codecs.lookup(f"cp{int(arg)}").name
                except LookupError:
                    pass
                continue
            if word == b"uc" and arg is not None:
                uc = max(0, int(arg))
                continue
            if word == b"u" and arg is not None:
                if not skip:
                    code = int(arg)
                    if -32768 <= code < 0:
                        code += 65536
                    out.append(chr(code) if 0 < code <= 0xFFFF else "\ufffd")
                to_skip = uc
                continue
            if skip:
                continue
            if word in _RTF_CHARS:
                to_skip = 0
                out.append(_RTF_CHARS[word])
            continue
        if hexbyte is not None:
            if to_skip:
                to_skip -= 1
                continue
            if not skip:
                pending.append(int(hexbyte, 16))
            continue
        if symbol is not None:
            if to_skip:
                to_skip -= 1
                continue
            if not skip and symbol in _RTF_SYMBOLS:
                out.append(_RTF_SYMBOLS[symbol])
            continue
        if text is not None:
            if to_skip:
                drop = min(to_skip, len(text))
                text = text[drop:]
                to_skip -= drop
            if not skip and text:
                out.append(text.decode(encoding, errors="replace"))
    flush()
    text = "".join(out).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def extract_rtf(path: Path) -> Iterator[Chunk]:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ExtractionError(f"cannot read file: {e.strerror or e}") from e
    doc = _Doc()
    for para in normalize_newlines(rtf_to_text(data)).split("\n\n"):
        doc.para(para)
    yield from _doc_chunks(doc)


_BY_SUFFIX = {
    ".odt": extract_odf, ".ott": extract_odf, ".odp": extract_odf, ".otp": extract_odf,
    ".docx": extract_docx, ".docm": extract_docx, ".dotx": extract_docx,
    ".pptx": extract_pptx, ".pptm": extract_pptx,
    ".rtf": extract_rtf,
}


def extract_office(path: Path, cfg: Config | None = None) -> Iterator[Chunk]:
    fn = _BY_SUFFIX.get(path.suffix.lower())
    if fn is None:
        raise ExtractionError(f"unsupported office format {path.suffix!r}")
    yield from fn(path)
