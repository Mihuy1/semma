"""Map file names to the content kinds Semma knows how to index."""

from __future__ import annotations

TEXT = "text"
PDF = "pdf"
OFFICE = "office"
IMAGE = "image"
AUDIO = "audio"
VIDEO = "video"

KINDS: tuple[str, ...] = (TEXT, PDF, OFFICE, IMAGE, AUDIO, VIDEO)

EXTENSIONS: dict[str, str] = {
    **dict.fromkeys(("txt", "text", "md", "markdown", "rst", "org", "tex", "adoc", "asciidoc"), TEXT),
    "pdf": PDF,
    **dict.fromkeys(("odt", "ott", "odp", "docx", "pptx", "rtf"), OFFICE),
    **dict.fromkeys(
        ("jpg", "jpeg", "png", "webp", "heic", "heif", "avif", "gif", "bmp", "tif", "tiff"), IMAGE
    ),
    **dict.fromkeys(("mp3", "flac", "ogg", "oga", "opus", "m4a", "wav", "aac", "wma"), AUDIO),
    **dict.fromkeys(("mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "mpg", "mpeg"), VIDEO),
}


_EXCLUDED_STEMS = frozenset({
    "cmakelists", "cmakecache", "robots", "constraints", "copying", "notice", "notices",
    "third_party_notices", "thirdpartynotices",
})
_EXCLUDED_STEM_PREFIXES = ("requirements", "license", "licence")


def classify(name: str) -> str | None:
    """Return the kind for a file name, or None if it should not be indexed."""
    dot = name.rfind(".")
    if dot <= 0:
        return None
    kind = EXTENSIONS.get(name[dot + 1 :].lower())
    if kind == TEXT:
        stem = name[:dot].casefold()
        if stem in _EXCLUDED_STEMS or stem.startswith(_EXCLUDED_STEM_PREFIXES):
            return None
    return kind
