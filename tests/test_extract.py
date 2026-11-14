"""Extractors for text, PDF, office documents and images (fixtures are generated at runtime)."""

from __future__ import annotations

import io
import struct
import time
import zipfile
import zlib
from pathlib import Path

import pytest
from PIL import Image

from semma.config import Config
from semma.extract import ExtractionError, extract
from semma.extract.office import rtf_to_text
from semma.extract.text import (
    MAX_CHARS,
    MAX_CHUNKS,
    chunk_text,
    decode_text,
    find_headings,
    make_snippet,
)

FINNISH = "Hyvää päivää! Öljyä, äyriäisiä ja Åland – tämä on söpö testi."


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(home=tmp_path, db_path=tmp_path / "index.db", roots=[])


def run(path: Path, kind: str, cfg: Config):
    return list(extract(path, kind, cfg))


# ---------------------------------------------------------------------------------- text


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252", "iso-8859-15"]
)
def test_text_encodings(tmp_path, cfg, encoding):
    text = FINNISH if encoding == "cp1252" else FINNISH.replace("–", "-")
    body = "\n".join([text] * 3) + "\nKöyhä mökkiläinen söi lohikäärmeen pääsiäisenä.\n"
    p = tmp_path / "note.txt"
    p.write_bytes(body.encode(encoding))
    chunks = run(p, "text", cfg)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.modality == "text" and c.locator == "" and c.position is None
    assert c.content.startswith("Hyvää päivää! Öljyä") and "lohikäärmeen" in c.content
    assert c.snippet.startswith("Hyvää päivää! Öljyä")


def test_decode_non_western_legacy_encodings():
    ru = "Привет, как дела? Это тестовый текст на русском языке, довольно короткий."
    assert decode_text(ru.encode("cp1251")) == ru
    assert decode_text(FINNISH.encode("cp1252")) == FINNISH


def test_newlines_normalised(tmp_path, cfg):
    p = tmp_path / "crlf.txt"
    p.write_bytes(b"one\r\ntwo\rthree\n")
    (c,) = run(p, "text", cfg)
    assert c.content == "one\ntwo\nthree"


def test_binary_txt_rejected(tmp_path, cfg):
    p = tmp_path / "fake.txt"
    p.write_bytes(bytes(range(256)) * 50)
    with pytest.raises(ExtractionError, match="binary"):
        run(p, "text", cfg)


def test_empty_and_whitespace_files_yield_nothing(tmp_path, cfg):
    (tmp_path / "e.txt").write_bytes(b"")
    (tmp_path / "w.txt").write_text(" \n\n\t \n")
    assert run(tmp_path / "e.txt", "text", cfg) == []
    assert run(tmp_path / "w.txt", "text", cfg) == []


def test_missing_file_is_extraction_error(tmp_path, cfg):
    with pytest.raises(ExtractionError):
        run(tmp_path / "nope.txt", "text", cfg)


def _long_markdown() -> str:
    para = (
        "Semanttinen haku etsii merkityksen perusteella. Se ei vaadi täsmällisiä avainsanoja. "
        "Upotusmalli muuntaa tekstin vektoreiksi, joita verrataan kosinietäisyydellä. "
    )
    parts = ["# Title\n\nIntro paragraph.\n"]
    for i in range(1, 9):
        parts.append(f"## Section {i}\n")
        for j in range(6):
            parts.append(f"Paragraph {i}.{j}. " + para * 2 + "\n")
        parts.append("```\n# not a heading inside code\n```\n")
    return "\n".join(parts)


def test_markdown_chunking(tmp_path, cfg):
    text = _long_markdown()
    p = tmp_path / "doc.md"
    p.write_text(text, encoding="utf-8")
    chunks = run(p, "text", cfg)
    assert len(chunks) > 5
    for c in chunks:
        assert c.content.strip()
        assert len(c.content) <= MAX_CHARS + 40  # + a prefixed heading
        assert c.locator.startswith("line ")
        assert c.position == int(c.locator.split()[1].strip("·"))
        assert len(c.snippet) <= 161
    # headings found, code-fence comment is not a heading
    titles = [t for _, t in find_headings(text, ".md")]
    assert titles[:2] == ["Title", "Section 1"] and "not a heading inside code" not in titles
    # line numbers are correct
    lines = text.split("\n")
    for c in chunks:
        line = int(c.position)
        assert lines[line - 1].strip() and lines[line - 1].strip()[:15] in c.content
    # chunks after the first section boundary carry their section in locator/content
    assert any("Section" in c.locator for c in chunks)


def test_headings_per_format():
    md = (
        "# ATX title ##  \n\nSetext title  \n=====\n\n```\n# fenced\nfenced setext\n---\n```\n"
        "## After fence\n    indented\n---\n" + "#  " + "x" * 400 + "\n"
    )
    assert [t for _, t in find_headings(md, ".md")] == ["ATX title", "Setext title", "After fence"]
    assert find_headings("* Top  \n** Sub\n*bold* text\n", ".org") == [(0, "Top"), (8, "Sub")]
    assert [t for _, t in find_headings("= Doc\n\n== Part one \ntext\n", ".adoc")] == ["Doc", "Part one"]
    rst = "Title\n=====\n\ntext\n\nSub  \n---\n\nToo long a title\n~~~\n"
    assert [t for _, t in find_headings(rst, ".rst")] == ["Title", "Sub"]
    assert find_headings("intro\n\\section{Intro}\n", ".tex") == [(6, "Intro")]


@pytest.mark.parametrize("suffix", ["md", "org", "adoc", "rst", "tex"])
def test_heading_detection_linear_on_pathological_input(tmp_path, cfg, suffix):
    """Heading-like lines with huge runs of spaces, and many fences + headings, stay fast."""
    space_line = {"md": "# x", "org": "* x", "adoc": "= x", "rst": "x", "tex": "\\section{x"}[suffix]
    under = "\n====" if suffix == "rst" else ""
    body = (space_line + " " * 200_000 + "y" + under + "\n") * 5
    body += (space_line + " " * 290 + "y" + under + "\n") * 2_000  # just under the line limit
    body += "```\n# h\n" * 80_000 if suffix == "md" else "* h\n= h\nh\n===\n" * 80_000
    p = tmp_path / f"evil.{suffix}"
    p.write_text(body)
    assert p.stat().st_size >= 1_900_000
    t = time.perf_counter()
    chunks = run(p, "text", cfg)
    assert time.perf_counter() - t < 2.0
    assert 0 < len(chunks) <= MAX_CHUNKS


def test_chunks_per_file_capped(tmp_path, cfg):
    p = tmp_path / "huge.log"
    line = "2026-01-01 12:00:00 INFO service started, handling request number {} for user\n"
    p.write_text("".join(line.format(i) for i in range(50_000)))  # ~4 MB: ~2,000 chunks uncapped
    chunks = run(p, "text", cfg)
    assert len(chunks) == MAX_CHUNKS == 1000
    assert chunks[0].locator == "line 1"


def test_chunker_sizes_and_overlap():
    sentences = [f"Lause numero {i} kertoo jotakin tärkeää aiheesta {i % 7}." for i in range(800)]
    text = " ".join(sentences)
    spans = list(chunk_text(text))
    assert len(spans) > 10
    for a, b in zip(spans, spans[1:]):
        assert 1000 <= len(a.text) <= MAX_CHARS
        overlap = a.end - b.start
        assert 0.08 * len(a.text) <= overlap <= 0.2 * len(a.text), overlap
        assert b.text[0].isupper()  # overlap starts at a sentence boundary
    # all text covered, in order
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(a.start < b.start for a, b in zip(spans, spans[1:]))


def test_chunker_unbreakable_text_and_paragraphs():
    blob = "x" * 6000
    spans = list(chunk_text(blob))
    assert all(0 < len(s.text) <= MAX_CHARS for s in spans)
    assert "".join(s.text for s in spans).count("x") >= 6000
    paras = "\n\n".join("Para %d. " % i + "word " * 60 for i in range(30))
    for s in chunk_text(paras):
        assert s.text.startswith("Para") or s.start > 0


def test_snippet():
    assert make_snippet("  a\n\n b\tc ") == "a b c"
    long = "word " * 100
    s = make_snippet(long)
    assert len(s) <= 161 and s.endswith("…")


# ---------------------------------------------------------------------------------- PDF


def _pdf_escape(s: str) -> bytes:
    return s.encode("cp1252").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def make_text_pdf(path: Path, pages: list[list[str]]) -> None:
    """A minimal valid PDF with Helvetica text, one list of lines per page."""
    objs: list[bytes] = []
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % i for i in page_ids), n))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        stream = b"BT /F1 11 Tf 13 TL 50 780 Td " + b" ".join(b"(%s) Tj T*" % _pdf_escape(l) for l in lines) + b" ET"
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >>"
            b" /Contents %d 0 R >>" % (page_ids[i] + 1)
        )
        objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for num, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (num, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref))
    path.write_bytes(out.getvalue())


def test_pdf_text_pages(tmp_path, cfg):
    p = tmp_path / "doc.pdf"
    long_page = [f"Rivi {i}: tämä on pitkä sivu, jossa on paljon tekstiä (osa {i})." for i in range(60)]
    make_text_pdf(p, [[FINNISH, "Toinen rivi sivulla yksi."], long_page, []])
    chunks = run(p, "pdf", cfg)
    first = chunks[0]
    assert first.modality == "text" and first.locator == "p. 1" and first.position == 1
    assert "Hyvää päivää" in first.content and "(" not in first.snippet[:5]
    page2 = [c for c in chunks if c.locator == "p. 2"]
    assert len(page2) >= 2 and all(c.modality == "text" and c.position == 2 for c in page2)
    assert "Rivi 59" in page2[-1].content
    # the empty third page becomes an image chunk
    page3 = [c for c in chunks if c.locator == "p. 3"]
    assert len(page3) == 1 and page3[0].modality == "image"
    img = page3[0].content
    assert img.mode == "RGB" and max(img.size) == 1600


def test_pdf_scanned_pages_rendered(tmp_path, cfg):
    p = tmp_path / "scan.pdf"
    a = Image.new("RGB", (850, 1100), "white")
    b = Image.new("RGB", (1100, 850), "lightblue")
    a.save(p, save_all=True, append_images=[b], resolution=100)
    chunks = run(p, "pdf", cfg)
    assert [c.modality for c in chunks] == ["image", "image"]
    assert [c.locator for c in chunks] == ["p. 1", "p. 2"]
    assert max(chunks[0].content.size) == 1600 and chunks[1].content.size[0] > chunks[1].content.size[1]


def test_pdf_bad_page_skipped(tmp_path, cfg, monkeypatch):
    import pypdfium2

    from semma.extract import pdf as pdf_mod

    p = tmp_path / "doc.pdf"
    make_text_pdf(p, [[FINNISH, "Sivu yksi."], [FINNISH, "Sivu kaksi."], []])
    real_text, real_render = pdf_mod._page_text, pdf_mod._render
    calls = {"n": 0}

    def flaky_text(page):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("broken content stream")
        return real_text(page)

    def broken_render(page):
        raise pypdfium2.PdfiumError("render failed")

    monkeypatch.setattr(pdf_mod, "_page_text", flaky_text)
    monkeypatch.setattr(pdf_mod, "_render", broken_render)
    chunks = run(p, "pdf", cfg)
    assert [c.locator for c in chunks] == ["p. 1"]  # page 2 text and page 3 render failed
    monkeypatch.setattr(pdf_mod, "_page_text", lambda page: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ExtractionError, match="no page could be read"):
        run(p, "pdf", cfg)
    monkeypatch.setattr(pdf_mod, "_page_text", real_text)
    monkeypatch.setattr(pdf_mod, "_render", real_render)
    assert len(run(p, "pdf", cfg)) == 3


def test_pdf_broken(tmp_path, cfg):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not really a pdf\n")
    with pytest.raises(ExtractionError):
        run(p, "pdf", cfg)


def test_pdf_generator_close_early(tmp_path, cfg):
    p = tmp_path / "doc.pdf"
    make_text_pdf(p, [[FINNISH]] * 5)
    gen = extract(p, "pdf", cfg)
    assert next(gen).locator == "p. 1"
    gen.close()


# --------------------------------------------------------------------------------- office


def write_zip(path: Path, members: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)


ODF_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
)


def test_odt(tmp_path, cfg):
    body = "".join(
        f'<text:h text:outline-level="1">Luku {i}</text:h>'
        f"<text:p>Kappale {i}: {FINNISH} " + "Lisää tekstiä. " * 120 + "</text:p>"
        for i in range(1, 4)
    )
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content {ODF_NS}><office:body><office:text>
<text:table-of-content><text:index-body><text:p>Sisällys 1</text:p></text:index-body></text:table-of-content>
<text:h text:outline-level="1">Johdanto</text:h>
<text:p>A<text:s text:c="3"/>B<text:tab/>C<text:line-break/>D <text:span>span</text:span>
 tail<text:note><text:note-citation>1</text:note-citation><text:note-body><text:p>Alaviite.</text:p></text:note-body></text:note></text:p>
<text:list><text:list-item><text:p>Listakohta yksi</text:p></text:list-item></text:list>
<table:table><table:table-row><table:table-cell><text:p>Solu 1</text:p></table:table-cell>
<table:table-cell><text:p>Solu 2</text:p></table:table-cell></table:table-row></table:table>
{body}
</office:text></office:body></office:document-content>"""
    p = tmp_path / "thesis.odt"
    write_zip(p, {"mimetype": "application/vnd.oasis.opendocument.text", "content.xml": content})
    chunks = run(p, "office", cfg)
    assert len(chunks) >= 3
    first = chunks[0].content
    assert "Sisällys" not in first  # table of contents skipped
    assert first.startswith("Johdanto\n\nA   B\tC\nD span\n tail")
    assert "Alaviite." in first and "1Alaviite" not in first
    assert "Listakohta yksi" in first and "Solu 1 | Solu 2" in first
    assert chunks[0].locator == "§ Johdanto" and chunks[0].position is None
    assert any(c.locator == "§ Luku 2" for c in chunks)
    # a chunk inside a section that doesn't start with its heading gets it as context
    for c in chunks:
        if c.locator.startswith("§ Luku"):
            assert c.content.startswith(c.locator[2:])


def test_odp(tmp_path, cfg):
    pages = "".join(
        f'<draw:page draw:name="s{i}"><draw:frame><draw:text-box><text:p>Dia {i} otsikko</text:p>'
        f"<text:p>Sisältö {i}</text:p></draw:text-box></draw:frame></draw:page>"
        for i in (1, 2)
    ) + '<draw:page draw:name="empty"/>'
    content = f'<office:document-content {ODF_NS}><office:body><office:presentation>{pages}</office:presentation></office:body></office:document-content>'
    p = tmp_path / "slides.odp"
    write_zip(p, {"content.xml": content})
    chunks = run(p, "office", cfg)
    assert [c.locator for c in chunks] == ["slide 1", "slide 2"]
    assert chunks[1].content == "Dia 2 otsikko\n\nSisältö 2" and chunks[1].position == 2


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def test_docx(tmp_path, cfg):
    doc = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document {W}><w:body>
<w:p><w:pPr><w:pStyle w:val="Otsikko1"/></w:pPr><w:r><w:t>Johdanto</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Hyvää </w:t></w:r><w:r><w:t>päivää</w:t><w:tab/><w:t>sarake</w:t><w:br/><w:t>rivi</w:t></w:r>
<w:del><w:r><w:delText>poistettu</w:delText></w:r></w:del></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>A1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:sectPr/></w:body></w:document>"""
    styles = f'<w:styles {W}><w:style w:styleId="Otsikko1"><w:name w:val="heading 1"/></w:style></w:styles>'
    p = tmp_path / "doc.docx"
    write_zip(p, {"word/document.xml": doc, "word/styles.xml": styles})
    (c,) = run(p, "office", cfg)
    assert c.content == "Johdanto\n\nHyvää päivää\tsarake\nrivi\n\nA1 | B1"
    assert c.locator == "" and c.snippet.startswith("Johdanto Hyvää")


def test_docx_nested_table_not_duplicated(tmp_path, cfg):
    cell = '<w:tc><w:p><w:r><w:t>{}</w:t></w:r></w:p></w:tc>'
    inner = f'<w:tbl><w:tr>{cell.format("sisä1")}{cell.format("sisä2")}</w:tr></w:tbl>'
    doc = (f"<w:document {W}><w:body><w:tbl><w:tr>{cell.format('A')}"
           f"<w:tc>{inner}<w:p/></w:tc></w:tr></w:tbl></w:body></w:document>")
    p = tmp_path / "nested.docx"
    write_zip(p, {"word/document.xml": doc})
    (c,) = run(p, "office", cfg)
    assert c.content.count("sisä1") == 1 and c.content.startswith("A | sisä1 | sisä2")


def test_pptx_numeric_slide_order_and_notes(tmp_path, cfg):
    A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'

    def slide(text):
        return f'<p:sld {A}><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r><a:br/><a:r><a:t>toinen</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>'

    notes = (
        f'<p:notes {A}><p:cSld><p:spTree>'
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>2</a:t></a:r></a:p></p:txBody></p:sp>'
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Puhujan muistiinpano</a:t></a:r></a:p></p:txBody></p:sp>'
        "</p:spTree></p:cSld></p:notes>"
    )
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide7.xml"/>'
        "</Relationships>"
    )
    p = tmp_path / "deck.pptx"
    write_zip(p, {
        "ppt/slides/slide10.xml": slide("Kymmenes"),
        "ppt/slides/slide2.xml": slide("Toinen dia"),
        "ppt/slides/slide1.xml": slide("Ensimmäinen"),
        "ppt/slides/_rels/slide2.xml.rels": rels,
        "ppt/notesSlides/notesSlide7.xml": notes,
    })
    chunks = run(p, "office", cfg)
    assert [c.locator for c in chunks] == ["slide 1", "slide 2", "slide 10"]
    assert chunks[0].content == "Ensimmäinen\ntoinen"
    assert chunks[1].content == "Toinen dia\ntoinen\n\nPuhujan muistiinpano"


def test_pptx_presentation_order(tmp_path, cfg):
    A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
    R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'

    def slide(text):
        return f'<p:sld {A}><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>'

    pres = (f'<p:presentation {A} {R}><p:sldIdLst><p:sldId id="256" r:id="rId9"/>'
            '<p:sldId id="257" r:id="rId2"/><p:sldId id="258" r:id="rId3"/></p:sldIdLst></p:presentation>')
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId2" Type="x/slide" Target="slides/slide1.xml"/>'
        '<Relationship Id="rId3" Type="x/slide" Target="/ppt/slides/slide2.xml"/>'
        '<Relationship Id="rId9" Type="x/slide" Target="slides/slide3.xml"/>'
        '<Relationship Id="rId1" Type="x/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        "</Relationships>"
    )
    p = tmp_path / "reordered.pptx"
    write_zip(p, {
        "ppt/presentation.xml": pres,
        "ppt/_rels/presentation.xml.rels": rels,
        "ppt/slides/slide1.xml": slide("Toinen esityksessä"),
        "ppt/slides/slide2.xml": slide("Kolmas esityksessä"),
        "ppt/slides/slide3.xml": slide("Ensimmäinen esityksessä"),
        "ppt/slides/slide4.xml": slide("Ei esityksessä"),  # orphan part: not in sldIdLst
    })
    chunks = run(p, "office", cfg)
    assert [(c.locator, c.content) for c in chunks] == [
        ("slide 1", "Ensimmäinen esityksessä"),
        ("slide 2", "Toinen esityksessä"),
        ("slide 3", "Kolmas esityksessä"),
    ]


RTF = (
    rb"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\froman Times New Roman;}}"
    rb"{\colortbl;\red0\green0\blue0;}{\stylesheet{\s0 Normal;}}{\info{\title Salainen otsikko}}"
    rb"{\*\generator Riched20;}\viewkind4\uc1\pard\f0\fs24 Hyv\'e4\'e4 p\'e4iv\'e4\'e4!\par"
    b"\r\n"
    rb"Euro \u8364? ja \u-3913? merkki{\*\fldinst HYPERLINK \"x\"}{\fldrslt linkki}\par "
    rb"Sarake\tab arvo\line rivi \{aaltosulkeet\}{\pict\pngblip 89504e47}\par}"
)


def test_rtf_to_text():
    text = rtf_to_text(RTF)
    assert "Hyvää päivää!" in text
    assert "Euro € ja \uf0b7 merkkilinkki" in text
    assert "Sarake\tarvo\nrivi {aaltosulkeet}" in text
    for junk in ("Times", "Salainen", "Riched", "HYPERLINK", "89504e47", "Normal", "?"):
        assert junk not in text


def test_rtf_file(tmp_path, cfg):
    p = tmp_path / "doc.rtf"
    p.write_bytes(RTF)
    (c,) = run(p, "office", cfg)
    assert c.content.startswith("Hyvää päivää!\n\nEuro €")


def test_rtf_unicode_escapes():
    text = rtf_to_text(rb"{\rtf1\ansi Party \u-10179?\u-8518? time \u55357?x \u99999999?y \u8364?}")
    assert text == "Party \U0001f6ba time \ufffdx \ufffdy €"
    text.encode("utf-8")


def test_rtf_emoji_file_storable(tmp_path, cfg):
    p = tmp_path / "emoji.rtf"
    p.write_bytes(rb"{\rtf1\ansi Party \u-10179?\u-8518? time, \u-10179? lone, \u0? nul}")
    (c,) = run(p, "office", cfg)
    assert "\U0001f6ba" in c.content and "\x00" not in c.content
    for s in (c.content, c.snippet, c.locator):
        s.encode("utf-8")


def test_dispatcher_sanitises_text_fields(tmp_path, cfg, monkeypatch):
    from semma import extract as extract_mod
    from semma.extract import Chunk, clean_str

    def dirty(path, cfg):
        yield Chunk("text", "a\x00b \ud83d\udeba c\udc00", "line 1 \ud800", "snip\x00\udfff", 1.0)
        yield Chunk("image", Image.new("RGB", (40, 40)), "p. 2\x00", "", 2.0)

    monkeypatch.setitem(extract_mod.EXTRACTORS, "text", dirty)
    a, b = run(tmp_path / "x.txt", "text", cfg)
    assert (a.content, a.locator, a.snippet) == ("ab \U0001f6ba c\ufffd", "line 1 \ufffd", "snip\ufffd")
    assert b.locator == "p. 2" and isinstance(b.content, Image.Image)
    assert clean_str("plain") == "plain"


@pytest.mark.parametrize("name", ["bad.docx", "bad.odt", "bad.pptx", "bad.rtf"])
def test_office_corrupt(tmp_path, cfg, name):
    p = tmp_path / name
    p.write_bytes(b"PK\x03\x04 garbage" if name != "bad.rtf" else b"not rtf")
    with pytest.raises(ExtractionError):
        run(p, "office", cfg)


def test_office_rejects_dtd_and_missing_parts(tmp_path, cfg):
    p = tmp_path / "evil.docx"
    write_zip(p, {"word/document.xml": '<!DOCTYPE x [<!ENTITY a "aaaa">]><w:document %s/>' % W})
    with pytest.raises(ExtractionError, match="DTD"):
        run(p, "office", cfg)
    q = tmp_path / "empty.odt"
    write_zip(q, {"mimetype": "x"})
    with pytest.raises(ExtractionError, match="content.xml"):
        run(q, "office", cfg)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_office_rejects_dtd_in_any_encoding(tmp_path, cfg, encoding):
    import codecs

    from semma.extract.office import MAX_MEMBER_BYTES, parse_xml

    decl = "UTF-8" if encoding == "utf-8" else "UTF-16"
    xml = (f'<?xml version="1.0" encoding="{decl}"?>'
           f'<!DOCTYPE d [<!ENTITY a "{"A" * 1000}">]><w:document {W}>&a;&a;</w:document>')
    bom = {"utf-16-le": codecs.BOM_UTF16_LE, "utf-16-be": codecs.BOM_UTF16_BE}.get(encoding, b"")
    data = bom + xml.encode(encoding)
    assert (b"<!DOCTYPE" in data) == (encoding == "utf-8")
    with pytest.raises(ExtractionError, match="DTD"):
        parse_xml(data)
    p = tmp_path / "evil.docx"
    write_zip(p, {"word/document.xml": data})
    with pytest.raises(ExtractionError, match="DTD"):
        run(p, "office", cfg)
    # a DTD-free UTF-16 part still parses
    ok = f'<?xml version="1.0" encoding="UTF-16"?><d>{FINNISH}</d>'.encode("utf-16")
    assert parse_xml(ok).text == FINNISH
    assert MAX_MEMBER_BYTES <= 32 * 2**20


def test_office_zip_bomb_guard(tmp_path, cfg, monkeypatch):
    from semma.extract import office

    monkeypatch.setattr(office, "MAX_MEMBER_BYTES", 10_000)
    p = tmp_path / "bomb.docx"
    write_zip(p, {"word/document.xml": f"<w:document {W}><w:body>" + " " * 50_000 + "</w:body></w:document>"})
    with pytest.raises(ExtractionError, match="too large"):
        run(p, "office", cfg)


# --------------------------------------------------------------------------------- images


def test_png_alpha_composited_on_white(tmp_path, cfg):
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    img.paste((0, 0, 255, 255), (0, 0, 32, 64))
    p = tmp_path / "a.png"
    img.save(p)
    (c,) = run(p, "image", cfg)
    assert c.modality == "image" and c.locator == ""
    out = c.content
    assert out.mode == "RGB" and out.getpixel((10, 10)) == (0, 0, 255) and out.getpixel((50, 10)) == (255, 255, 255)


def test_jpeg_exif_rotation_and_downscale(tmp_path, cfg):
    img = Image.new("RGB", (4000, 2000), "green")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    p = tmp_path / "r.jpg"
    img.save(p, exif=exif)
    (c,) = run(p, "image", cfg)
    w, h = c.content.size
    assert h > w and max(w, h) <= 1536 and c.content.mode == "RGB"


def test_tiny_image_skipped(tmp_path, cfg):
    p = tmp_path / "icon.png"
    Image.new("RGB", (16, 16)).save(p)
    assert run(p, "image", cfg) == []


def test_animated_gif_first_frame(tmp_path, cfg):
    frames = [Image.new("RGB", (80, 60), c) for c in ("red", "green", "blue")]
    p = tmp_path / "anim.gif"
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=100, loop=0)
    (c,) = run(p, "image", cfg)
    r, g, b = c.content.getpixel((5, 5))
    assert r > 200 and g < 50 and b < 50


def test_sixteen_bit_and_cmyk(tmp_path, cfg):
    import numpy as np

    p16 = tmp_path / "depth.png"
    Image.fromarray((np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 10)).save(p16)
    pc = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (64, 64), (0, 255, 255, 0)).save(pc)
    for p in (p16, pc):
        (c,) = run(p, "image", cfg)
        assert c.content.mode == "RGB" and c.content.size == (64, 64)


def test_heic(tmp_path, cfg):
    pillow_heif = pytest.importorskip("pillow_heif")
    p = tmp_path / "photo.heic"
    try:
        pillow_heif.from_pillow(Image.new("RGB", (128, 96), "orange")).save(p, quality=80)
    except Exception as e:  # pragma: no cover - libheif built without an encoder
        pytest.skip(f"no HEIC encoder: {e}")
    (c,) = run(p, "image", cfg)
    assert c.content.size == (128, 96)


def test_decompression_bomb(tmp_path, cfg, monkeypatch):
    from semma.extract import image as image_mod

    p = tmp_path / "big.png"
    Image.new("L", (3000, 3000)).save(p)
    image_mod._init()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    with pytest.raises(ExtractionError, match="too large"):
        run(p, "image", cfg)


def _png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b"")


def test_image_pixel_limit_about_100_megapixels(tmp_path, cfg):
    from semma.extract import image as image_mod

    image_mod._init()
    assert Image.MAX_IMAGE_PIXELS * 2 <= 100_000_000
    p = tmp_path / "huge.png"
    p.write_bytes(_png_header(11_000, 10_000))  # 110 MP, rejected from the header alone
    with pytest.raises(ExtractionError, match="too large"):
        run(p, "image", cfg)


def test_jpeg2000_reduced_while_decoding(tmp_path, cfg):
    p = tmp_path / "big.jp2"
    try:
        Image.new("RGB", (4000, 3000), "red").save(p)
    except (OSError, KeyError) as e:  # pragma: no cover - Pillow without OpenJPEG
        pytest.skip(f"no JPEG 2000 support: {e}")
    (c,) = run(p, "image", cfg)
    assert c.content.size == (1536, 1152)


def test_corrupt_image(tmp_path, cfg):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0 definitely not a jpeg" * 10)
    with pytest.raises(ExtractionError):
        run(p, "image", cfg)


def test_unknown_kind(tmp_path, cfg):
    with pytest.raises(ExtractionError):
        run(tmp_path / "x", "spreadsheet", cfg)


def test_tiny_text_is_not_embedded(tmp_path):
    from semma.config import load_config
    from semma.extract import extract

    cfg = load_config(path=tmp_path / "none.toml", home=tmp_path)
    tiny = tmp_path / "CLAUDE.md"
    tiny.write_text("@AGENTS.md\n")
    short_note = tmp_path / "note.txt"
    short_note.write_text("Call the dentist about the appointment on Friday")
    assert list(extract(tiny, "text", cfg)) == []
    assert len(list(extract(short_note, "text", cfg))) == 1
