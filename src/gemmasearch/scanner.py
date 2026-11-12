"""Phase 1: walk the filesystem (no model involved) and score every indexable file."""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import filetypes as ft
from .config import Config
from .priority import Score, Scorer


@dataclass(slots=True)
class ScannedFile:
    path: str
    kind: str
    size: int
    mtime: float
    in_project: bool
    score: Score


@dataclass
class ScanStats:
    roots: list[Path] = field(default_factory=list)
    dirs: int = 0
    files_seen: int = 0
    matched: int = 0
    too_large: int = 0
    errors: int = 0
    bad_names: int = 0  # names that are not valid UTF-8 (cannot be stored or shown)
    pruned: Counter[str] = field(default_factory=Counter)
    # Directories (and roots) that could not be listed. Their previously indexed files must
    # not be forgotten just because this scan could not see them.
    failed_dirs: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _is_app_install(names: list[str]) -> bool:
    """An installed Windows app or game: at least one .exe next to at least two .dll files.

    (A lone installer plus one DLL in Downloads is not enough.)
    """
    exe = dll = 0
    for n in names:
        low = n.lower()
        if low.endswith(".exe"):
            exe += 1
        elif low.endswith(".dll"):
            dll += 1
    return exe >= 1 and dll >= 2


def _utf8_ok(path: str) -> bool:
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:  # surrogate-escaped bytes from a non-UTF-8 file name
        return False
    return True


class Scanner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scorer = Scorer(cfg)
        self.stats = ScanStats()
        self._exclude_paths = {str(p) for p in cfg.exclude_paths}
        # Location anchors (~, ~/Documents, ~/Downloads, ...) are never treated as projects or
        # app installs: a dotfiles .git in ~ or an .exe+.dll in Downloads must not demote/prune them.
        self._anchors = {str(p) for p in cfg.locations}

    def _prune_reason(self, name: str, full: str, in_project: bool, at_root: bool) -> str | None:
        cfg = self.cfg
        if cfg.skip_hidden and name.startswith("."):
            return "hidden"
        folded = name.casefold()
        if folded in cfg.exclude_dir_names:
            return name
        if at_root and folded in cfg.root_system_dir_names:
            return name
        if in_project and folded in cfg.project_build_dir_names:
            return f"project/{name}"
        if full in self._exclude_paths:
            return "exclude_paths"
        if full in self._roots:
            return "other-root"  # walked separately as its own root
        if os.path.ismount(full):
            return "mount"  # other filesystems (network shares, USB) are only scanned as roots
        return None

    def scan(self, roots: list[Path] | None = None, now: float | None = None) -> Iterator[ScannedFile]:
        cfg = self.cfg
        now = time.time() if now is None else now
        roots = cfg.effective_roots() if roots is None else roots
        self.stats = stats = ScanStats(roots=list(roots))
        self._roots = {str(r) for r in roots}
        started = time.monotonic()
        markers = cfg.project_markers

        for root in roots:
            root_str = str(root)
            stack: list[tuple[str, bool]] = [(root_str, False)]
            while stack:
                dirpath, in_project = stack.pop()
                try:
                    with os.scandir(dirpath) as it:
                        entries = list(it)
                except OSError:
                    stats.errors += 1
                    stats.failed_dirs.append(dirpath)
                    continue
                stats.dirs += 1
                if dirpath == root_str and not entries:
                    # An empty root is most likely an unmounted mount point: treat it as
                    # unavailable rather than as "everything was deleted".
                    stats.failed_dirs.append(dirpath)
                    continue
                names = [e.name for e in entries]
                protected = dirpath == root_str or dirpath in self._anchors
                if not protected:
                    if "pyvenv.cfg" in names:
                        stats.pruned["venv"] += 1
                        continue
                    if cfg.skip_app_installs and _is_app_install(names):
                        stats.pruned["app-install"] += 1
                        continue
                    if not in_project and any(n in markers for n in names):
                        in_project = True

                for entry in entries:
                    name = entry.name
                    if not _utf8_ok(entry.path):
                        stats.bad_names += 1
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            reason = self._prune_reason(name, entry.path, in_project, dirpath == root_str)
                            if reason:
                                stats.pruned[reason] += 1
                            else:
                                stack.append((entry.path, in_project))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        stats.files_seen += 1
                        if cfg.skip_hidden and name.startswith("."):
                            continue
                        kind = ft.classify(name)
                        if kind is None:
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        stats.errors += 1
                        continue
                    if st.st_size > cfg.max_bytes.get(kind, 1 << 62):
                        stats.too_large += 1
                        continue
                    stats.matched += 1
                    score = self.scorer.score(Path(entry.path), kind, st.st_size, st.st_mtime,
                                              in_project, root, now)
                    yield ScannedFile(entry.path, kind, st.st_size, st.st_mtime, in_project, score)

        stats.seconds = time.monotonic() - started
