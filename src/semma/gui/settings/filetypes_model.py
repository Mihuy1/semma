"""The File types section's settings (scan.kinds, scan.exclude_extensions) as plain values.
No Qt here."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ... import filetypes as ft
from ..model import KIND_NAMES
from .context import Changes, Problem, SettingsContext

SECTION = "scan"

KIND_EXTENSIONS: dict[str, str] = {
    kind: ", ".join(ext for ext, k in ft.EXTENSIONS.items() if k == kind)
    for kind in ft.KINDS
}


def normalise_kinds(kinds: Iterable[str]) -> list[str]:
    """Known kinds in the canonical `filetypes.KINDS` order, de-duplicated."""
    present = set(kinds)
    return [k for k in ft.KINDS if k in present]


def parse_extensions(raw: str | Iterable[str]) -> list[str]:
    """Normalise user-entered extensions: lowercase, no leading dot, de-duplicated, sorted."""
    tokens = re.split(r"[\s,]+", raw) if isinstance(raw, str) else [
        part for item in raw for part in re.split(r"[\s,]+", str(item))
    ]
    cleaned = {t.strip().lstrip(".").lower() for t in tokens if t.strip().lstrip(".")}
    return sorted(cleaned)


def format_extensions(extensions: Iterable[str]) -> str:
    return ", ".join(parse_extensions(extensions))


@dataclass
class FiletypesValues:
    kinds: list[str] = field(default_factory=lambda: list(ft.KINDS))
    exclude_extensions: list[str] = field(default_factory=list)

    def copy(self) -> FiletypesValues:
        return FiletypesValues(
            kinds=list(self.kinds),
            exclude_extensions=list(self.exclude_extensions),
        )


DEFAULTS = FiletypesValues()


def load(ctx: SettingsContext) -> FiletypesValues:
    cfg = ctx.config
    return FiletypesValues(
        kinds=normalise_kinds(cfg.kinds),
        exclude_extensions=parse_extensions(cfg.exclude_extensions),
    )


def changes(saved: FiletypesValues, edited: FiletypesValues) -> Changes:
    """Only the keys the user changed; a key changed back to its default is removed (None)."""
    out: dict[str, object] = {}
    saved_kinds = normalise_kinds(saved.kinds)
    edited_kinds = normalise_kinds(edited.kinds)
    if edited_kinds != saved_kinds:
        out["kinds"] = None if edited_kinds == DEFAULTS.kinds else edited_kinds

    saved_exts = parse_extensions(saved.exclude_extensions)
    edited_exts = parse_extensions(edited.exclude_extensions)
    if edited_exts != saved_exts:
        out["exclude_extensions"] = None if edited_exts == DEFAULTS.exclude_extensions else edited_exts

    return {SECTION: out} if out else {}


def problems(values: FiletypesValues, ctx: SettingsContext | None = None) -> list[Problem]:
    out: list[Problem] = []
    if not normalise_kinds(values.kinds):
        out.append(Problem("Nothing would be indexed: turn on at least one file type.", blocking=True))
    for ext in parse_extensions(values.exclude_extensions):
        if ext not in ft.EXTENSIONS:
            out.append(Problem(f"Semma does not index .{ext} files anyway."))
    return out


__all__ = [
    "DEFAULTS",
    "KIND_EXTENSIONS",
    "KIND_NAMES",
    "SECTION",
    "FiletypesValues",
    "changes",
    "format_extensions",
    "load",
    "normalise_kinds",
    "parse_extensions",
    "problems",
]
