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
    pruned: Counter[str] = field(default_factory=Counter)
    seconds: float = 0.0


def _is_app_install(names: list[str]) -> bool:
    """A folder holding both .exe and .dll files is an installed Windows app or game."""
    lowered = [n.lower() for n in names]
    return any(n.endswith(".exe") for n in lowered) and any(n.endswith(".dll") for n in lowered)


class Scanner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scorer = Scorer(cfg)
        self.stats = ScanStats()
        self._exclude_paths = {str(p) for p in cfg.exclude_paths}

    def _prune_reason(self, name: str, full: str, in_project: bool) -> str | None:
        cfg = self.cfg
        if cfg.skip_hidden and name.startswith("."):
            return "hidden"
        if name in cfg.exclude_dir_names:
            return name
        if in_project and name in cfg.project_build_dir_names:
            return f"project/{name}"
        if full in self._exclude_paths:
            return "exclude_paths"
        return None

    def scan(self, roots: list[Path] | None = None, now: float | None = None) -> Iterator[ScannedFile]:
        cfg = self.cfg
        now = time.time() if now is None else now
        roots = cfg.effective_roots() if roots is None else roots
        self.stats = stats = ScanStats(roots=list(roots))
        started = time.monotonic()
        markers = cfg.project_markers

        for root in roots:
            stack: list[tuple[str, bool]] = [(str(root), False)]
            while stack:
                dirpath, in_project = stack.pop()
                try:
                    with os.scandir(dirpath) as it:
                        entries = list(it)
                except OSError:
                    stats.errors += 1
                    continue
                stats.dirs += 1
                names = [e.name for e in entries]
                if dirpath != str(root) and cfg.skip_app_installs and _is_app_install(names):
                    stats.pruned["app-install"] += 1
                    continue
                if not in_project and any(n in markers for n in names):
                    in_project = True

                for entry in entries:
                    name = entry.name
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            reason = self._prune_reason(name, entry.path, in_project)
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
