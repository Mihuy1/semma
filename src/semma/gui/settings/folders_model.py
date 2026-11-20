"""The Folders section's settings (scan.roots, scan.exclude_paths, scan.include_mounts,
scan.skip_hidden) as plain values. No Qt here."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .context import Changes, Problem, SettingsContext

SECTION = "scan"
MOUNT_PARENTS = ("/mnt", "/media", "/run/media")


@dataclass
class FoldersValues:
    """Folder lists hold paths as written to config.toml ("~/Documents")."""
    roots: list[str] = field(default_factory=lambda: ["~"])
    exclude_paths: list[str] = field(default_factory=list)
    include_mounts: bool = True
    skip_hidden: bool = True

    def copy(self) -> FoldersValues:
        return replace(self, roots=list(self.roots), exclude_paths=list(self.exclude_paths))


DEFAULTS = FoldersValues()
KEYS = ("roots", "exclude_paths", "include_mounts", "skip_hidden")


def load(ctx: SettingsContext) -> FoldersValues:
    cfg = ctx.config
    return FoldersValues(
        roots=[ctx.display_path(p) for p in cfg.roots],
        exclude_paths=[ctx.display_path(p) for p in cfg.exclude_paths],
        include_mounts=cfg.include_mounts,
        skip_hidden=cfg.skip_hidden,
    )


def changes(saved: FoldersValues, edited: FoldersValues) -> Changes:
    """Only the keys the user changed; a key changed back to its default is removed (None)."""
    out = {}
    for key in KEYS:
        new = getattr(edited, key)
        if new != getattr(saved, key):
            out[key] = None if new == getattr(DEFAULTS, key) else new
    return {SECTION: out} if out else {}


def add_folder(folders: list[str], path: Path | str, ctx: SettingsContext) -> tuple[list[str], str | None]:
    """`folders` with `path` added (as "~/..."), or unchanged and the reason when it is there
    already."""
    resolved = ctx.resolve(str(path))
    text = ctx.display_path(resolved)
    if any(ctx.resolve(f) == resolved for f in folders):
        return folders, f"{text} is in the list already."
    return [*folders, text], None


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


NOTHING_INDEXED = "nothing_indexed"


def problems(values: FoldersValues, ctx: SettingsContext) -> list[Problem]:
    out: list[Problem] = []
    roots = [ctx.resolve(r) for r in values.roots]
    excludes = [ctx.resolve(e) for e in values.exclude_paths]
    for label, items, resolved in (("indexed", values.roots, roots), ("skipped", values.exclude_paths, excludes)):
        seen: set[Path] = set()
        for text, p in zip(items, resolved, strict=True):
            if p in seen:
                out.append(Problem(f"{text} is in the {label} folders twice."))
            seen.add(p)
    if not roots and not values.include_mounts:
        out.append(Problem("Nothing is indexed: add a folder, or index external drives.", code=NOTHING_INDEXED))
    mounts = [Path(m) for m in MOUNT_PARENTS] if values.include_mounts else []
    for text, p in zip(values.exclude_paths, excludes, strict=True):
        if p in roots:
            out.append(Problem(f"{text} is both indexed and skipped."))
        elif not any(_inside(p, r) for r in roots + mounts):
            out.append(Problem(f"{text} is not inside an indexed folder, so skipping it changes nothing."))
    return out
