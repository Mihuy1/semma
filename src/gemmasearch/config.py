"""Configuration: built-in defaults, optionally overridden by ~/.config/gemmasearch/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import filetypes as ft

# Directory names that are never descended into, wherever they appear.
DEFAULT_EXCLUDE_DIR_NAMES: tuple[str, ...] = (
    # dependency / tooling caches
    "node_modules", "__pycache__", "site-packages", "venv", "npm-cache", "bower_components",
    # Windows system folders (NTFS data drives)
    "$RECYCLE.BIN", "System Volume Information", "Config.Msi", "WUDownloadCache",
    "msdownld.tmp", "Program Files", "Program Files (x86)", "ProgramData", "Windows",
    "WindowsApps", "WpSystem", "found.000", "lost+found",
    # game libraries
    "SteamLibrary", "steamapps", "XboxGames", "Epic Games", "GOG Games",
    # extracted AppImages
    "squashfs-root",
)

# Directory names skipped only inside a software project (outside one they may be real folders).
DEFAULT_PROJECT_BUILD_DIR_NAMES: tuple[str, ...] = (
    "build", "dist", "target", "out", "vendor", "env", "coverage", "bin", "obj",
)

# A directory containing any of these is the root of a software project.
DEFAULT_PROJECT_MARKERS: tuple[str, ...] = (
    ".git", ".hg", ".svn", "pyproject.toml", "setup.py", "package.json", "Cargo.toml",
    "go.mod", "CMakeLists.txt", "meson.build", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json", "mix.exs", "Package.swift", "pubspec.yaml", "deno.json",
)

DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    ft.PDF: 30, ft.OFFICE: 30, ft.TEXT: 25, ft.IMAGE: 20, ft.AUDIO: 15, ft.VIDEO: 10,
}

# Files below this size get the tiny-file penalty (icons, empty notes, stubs).
DEFAULT_TINY_BYTES: dict[str, int] = {
    ft.TEXT: 200, ft.PDF: 2_000, ft.OFFICE: 2_000, ft.IMAGE: 20_000,
    ft.AUDIO: 50_000, ft.VIDEO: 200_000,
}

# Files above this size are not indexed at all.
DEFAULT_MAX_BYTES: dict[str, int] = {
    ft.TEXT: 20 * 2**20, ft.PDF: 500 * 2**20, ft.OFFICE: 200 * 2**20,
    ft.IMAGE: 200 * 2**20, ft.AUDIO: 4 * 2**30, ft.VIDEO: 100 * 2**30,
}

# (max age in days, bonus points); the first matching row wins.
DEFAULT_RECENCY: tuple[tuple[float, float], ...] = ((30, 15), (365, 10), (3 * 365, 5))

DEFAULT_MODEL = "google/embeddinggemma-2"

# Chunks per model call, per modality (a CUDA OOM halves the batch and retries). Measured on an
# RTX 5070 Ti with ~2,000-character text chunks: text 8 -> 112 chunks/s at 2.7 GiB peak, but
# 32 -> 71/s at 6.7 GiB (padding and attention memory grow faster than the GPU gains);
# images run ~21/s at any batch size (preprocessing-bound), so 8 keeps memory low.
DEFAULT_BATCH_SIZES: dict[str, int] = {"text": 8, "image": 8, "audio": 8, "video": 2}


def _env_path(var: str, fallback: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value and os.path.isabs(value) else fallback


def xdg_config_home(home: Path) -> Path:
    return _env_path("XDG_CONFIG_HOME", home / ".config")


def xdg_data_home(home: Path) -> Path:
    return _env_path("XDG_DATA_HOME", home / ".local" / "share")


def read_user_dirs(home: Path, config_home: Path) -> dict[str, Path]:
    """Parse user-dirs.dirs (XDG_DOCUMENTS_DIR="$HOME/Documents" ...) into {"DOCUMENTS": path}."""
    dirs: dict[str, Path] = {}
    try:
        lines = (config_home / "user-dirs.dirs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return dirs
    for line in lines:
        line = line.strip()
        if not line.startswith("XDG_") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').replace("$HOME", str(home))
        name = key.removeprefix("XDG_").removesuffix("_DIR")
        if value and Path(value) != home:
            dirs[name] = Path(value)
    return dirs


def default_locations(home: Path, user_dirs: dict[str, Path]) -> dict[Path, float]:
    def d(key: str, fallback: str) -> Path:
        return user_dirs.get(key, home / fallback)

    return {
        d("DOCUMENTS", "Documents"): 40,
        d("DESKTOP", "Desktop"): 35,
        d("PICTURES", "Pictures"): 30,
        d("MUSIC", "Music"): 30,
        d("VIDEOS", "Videos"): 30,
        d("DOWNLOAD", "Downloads"): 15,
        home: 20,
    }


def detect_mounts() -> list[Path]:
    """Mounted data drives under /mnt, /media and /run/media/$USER."""
    user = os.environ.get("USER", "")
    parents = [Path("/mnt"), Path("/media"), Path("/run/media") / user] if user else [Path("/mnt"), Path("/media")]
    mounts: list[Path] = []
    for parent in parents:
        try:
            children = sorted(parent.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if child.is_dir() and os.path.ismount(child) and os.access(child, os.R_OK | os.X_OK):
                    mounts.append(child)
            except OSError:
                continue
    return mounts


@dataclass
class Config:
    home: Path
    db_path: Path
    roots: list[Path]
    include_mounts: bool = True
    skip_hidden: bool = True
    skip_app_installs: bool = True  # prune folders containing both .exe and .dll files
    exclude_dir_names: frozenset[str] = frozenset(DEFAULT_EXCLUDE_DIR_NAMES)
    project_build_dir_names: frozenset[str] = frozenset(DEFAULT_PROJECT_BUILD_DIR_NAMES)
    project_markers: frozenset[str] = frozenset(DEFAULT_PROJECT_MARKERS)
    exclude_paths: list[Path] = field(default_factory=list)

    locations: dict[Path, float] = field(default_factory=dict)
    other_location: float = 10
    type_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS))
    project_penalty: float = -20
    depth_free_levels: int = 3
    depth_penalty_per_level: float = -2
    depth_penalty_max: float = -10
    recency: tuple[tuple[float, float], ...] = DEFAULT_RECENCY
    tiny_bytes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TINY_BYTES))
    tiny_penalty: float = -15
    huge_video_bytes: int = 4 * 2**30
    huge_video_penalty: float = -10
    max_bytes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_BYTES))

    # Media: only the first N seconds of an audio or video file are indexed.
    max_media_seconds: float = 30 * 60
    media_segment_seconds: float = 60

    # Embedding model and indexing.
    model_name: str = DEFAULT_MODEL
    device: str | None = None  # None: CUDA (bfloat16) when available, else CPU (float32)
    batch_sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_BATCH_SIZES))
    prefetch_chunks: int = 8  # extracted chunks buffered ahead of the GPU (bounds memory)

    # Search. EmbeddingGemma similarities are compressed: on a calibration set unrelated
    # query/file pairs scored 0.43-0.69 (median 0.53, p90 0.60), relevant ones 0.62-0.81.
    # A vector-only match must clear `min_similarity` AND lie within `similarity_margin` of the
    # best match of the query (short queries inflate every similarity, which the margin absorbs).
    min_similarity: float = 0.60
    similarity_margin: float = 0.10
    knn_k: int = 200
    rrf_k: int = 60

    def effective_roots(self) -> list[Path]:
        """Configured roots plus detected mounts, de-duplicated and with nested roots removed."""
        roots = list(self.roots)
        if self.include_mounts:
            roots += detect_mounts()
        resolved: list[Path] = []
        for root in roots:
            root = root.expanduser()
            if root.is_dir() and root not in resolved:
                resolved.append(root)
        return [r for r in resolved if not any(r != o and r.is_relative_to(o) for o in resolved)]


def _expand(value: str, home: Path) -> Path:
    """Expand ~ and $HOME against the configured home (not the process's)."""
    value = value.replace("$HOME", str(home))
    if value == "~":
        return home
    if value.startswith("~/"):
        return home / value[2:]
    return Path(value)


def default_config_path(home: Path | None = None) -> Path:
    home = home or Path.home()
    return xdg_config_home(home) / "gemmasearch" / "config.toml"


def load_config(path: Path | None = None, home: Path | None = None) -> Config:
    """Build the default config, then apply overrides from the TOML file if it exists."""
    home = home or Path.home()
    config_home = xdg_config_home(home)
    cfg = Config(
        home=home,
        db_path=xdg_data_home(home) / "gemmasearch" / "index.db",
        roots=[home],
        locations=default_locations(home, read_user_dirs(home, config_home)),
    )
    path = path or default_config_path(home)
    if path.is_file():
        with path.open("rb") as fh:
            apply_overrides(cfg, tomllib.load(fh))
    return cfg


def apply_overrides(cfg: Config, data: dict[str, Any]) -> None:
    home = cfg.home
    scan = data.get("scan", {})
    prio = data.get("priority", {})
    media = data.get("media", {})
    model = data.get("model", {})
    index = data.get("index", {})
    search = data.get("search", {})

    if "db_path" in data:
        cfg.db_path = _expand(data["db_path"], home)
    if "roots" in scan:
        cfg.roots = [_expand(p, home) for p in scan["roots"]]
    for key in ("include_mounts", "skip_hidden", "skip_app_installs"):
        if key in scan:
            setattr(cfg, key, bool(scan[key]))
    for key in ("exclude_dir_names", "project_build_dir_names", "project_markers"):
        if key in scan:
            setattr(cfg, key, frozenset(scan[key]))
    if "extra_exclude_dir_names" in scan:
        cfg.exclude_dir_names = cfg.exclude_dir_names | frozenset(scan["extra_exclude_dir_names"])
    if "exclude_paths" in scan:
        cfg.exclude_paths = [_expand(p, home) for p in scan["exclude_paths"]]

    if "locations" in prio:
        # Merge: a configured path replaces the default weight for that path, others are kept.
        for p, pts in prio["locations"].items():
            cfg.locations[_expand(p, home)] = float(pts)
    if "type_weights" in prio:
        cfg.type_weights.update({k: float(v) for k, v in prio["type_weights"].items()})
    for key in ("other_location", "project_penalty", "depth_penalty_per_level", "depth_penalty_max",
                "tiny_penalty", "huge_video_penalty"):
        if key in prio:
            setattr(cfg, key, float(prio[key]))
    if "depth_free_levels" in prio:
        cfg.depth_free_levels = int(prio["depth_free_levels"])
    if "recency" in prio:
        cfg.recency = tuple((float(a), float(b)) for a, b in prio["recency"])

    if "max_minutes" in media:
        cfg.max_media_seconds = float(media["max_minutes"]) * 60
    if "segment_seconds" in media:
        cfg.media_segment_seconds = float(media["segment_seconds"])

    if "name" in model:
        cfg.model_name = str(model["name"])
    if "device" in model:
        cfg.device = str(model["device"]) or None
    if "batch_sizes" in index:
        cfg.batch_sizes.update({k: max(1, int(v)) for k, v in index["batch_sizes"].items()})
    if "prefetch_chunks" in index:
        cfg.prefetch_chunks = max(1, int(index["prefetch_chunks"]))
    for key in ("min_similarity", "similarity_margin"):
        if key in search:
            setattr(cfg, key, float(search[key]))
    for key in ("knn_k", "rrf_k"):
        if key in search:
            setattr(cfg, key, int(search[key]))
