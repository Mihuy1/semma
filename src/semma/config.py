"""Configuration: built-in defaults, optionally overridden by ~/.config/semma/config.toml."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import filetypes as ft

# Directory names that are never descended into, wherever they appear (compared case-insensitively).
DEFAULT_EXCLUDE_DIR_NAMES: tuple[str, ...] = (
    # dependency / tooling caches
    "node_modules", "__pycache__", "site-packages", "npm-cache", "bower_components",
    # game libraries
    "SteamLibrary", "steamapps", "XboxGames", "Epic Games", "GOG Games",
    # extracted AppImages
    "squashfs-root",
)

# Operating-system folders pruned only directly below a scan root (e.g. an NTFS data drive),
# so that a user's own "~/Documents/Windows" notes folder is still indexed.
DEFAULT_ROOT_SYSTEM_DIR_NAMES: tuple[str, ...] = (
    "$RECYCLE.BIN", "System Volume Information", "Config.Msi", "WUDownloadCache",
    "msdownld.tmp", "Program Files", "Program Files (x86)", "ProgramData", "Windows",
    "WindowsApps", "WpSystem", "found.000", "lost+found", "Recovery", "PerfLogs",
)

# Directory names skipped only inside a software project (outside one they may be real folders).
DEFAULT_PROJECT_BUILD_DIR_NAMES: tuple[str, ...] = (
    "build", "dist", "target", "out", "vendor", "env", "venv", "coverage", "bin", "obj",
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
# Video is 1: a 32-frame chunk takes ~0.7 s, and a query never waits for more than one batch.
DEFAULT_BATCH_SIZES: dict[str, int] = {"text": 8, "image": 8, "audio": 8, "video": 1}

# sqlite-vec refuses KNN queries with k above 4096.
MAX_KNN_K = 4096

# Near-duplicate collapsing in search results: two files whose best chunks for the query have at
# least this cosine similarity are shown as one result. Calibrated on the real index
# (1,190 files): copies of one COCO photo in different folders (re-encoded/resized) score
# 0.88-1.00 (median 0.98, p5 0.955) while the closest pair of *different* COCO photos scores
# 0.95, so images use 0.96. Text chunks are much more compressed: unchanged sections of two
# thesis versions score >= 0.998, the same thesis as .odt vs .md (different chunk boundaries)
# 0.97-0.98, and a separate notes file about the thesis up to 0.987, so text uses 0.99.
DEFAULT_DUPLICATE_SIMILARITY: dict[str, float] = {"text": 0.99, "image": 0.96, "audio": 0.98, "video": 0.97}


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
    skip_app_installs: bool = True  # prune installed Windows apps/games (see scanner._is_app_install)
    exclude_dir_names: frozenset[str] = frozenset(DEFAULT_EXCLUDE_DIR_NAMES)
    root_system_dir_names: frozenset[str] = frozenset(DEFAULT_ROOT_SYSTEM_DIR_NAMES)
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
    device: str | None = None  # None: GPU (Vulkan / CUDA) when available, else CPU
    query_device: str = "auto"  # "auto" (same as indexing), "cpu", "vulkan", "gpu" or "cuda"
    llama_lib_dir: Path | None = None
    model_dir: Path | None = None
    batch_sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_BATCH_SIZES))
    prefetch_chunks: int = 8  # extracted chunks buffered ahead of the GPU (bounds memory)

    # Search. A chunk's meaning score is a z-score: how many standard deviations its cosine
    # similarity to the query stands out from every stored vector's (after centring each
    # modality on its own mean, since image cosines run lower than text ones). EmbeddingGemma
    # cosines are compressed (unrelated pairs score ~0.45-0.70), so a fixed cosine threshold
    # does not work across queries; z does. A file whose best chunk reaches `z_gate` is a
    # meaning match; below it a file needs half the query's words (or its name) to be shown.
    # 2.0 keeps roughly the top 2-3% of chunks for a query (DigUp's value).
    z_gate: float = 2.0
    # A file without the query's words is shown only if the query's best meaning match reaches
    # `z_floor` (the kind bonus of a type word counts: "photo of ..."); else nothing stands out
    # and the query finds only files containing its words, or nothing. Gibberish tops out at
    # z 2.8-4.5 (median 3.9, 35 strings, all text chunks) on a 2.9k-chunk index; real one-concept
    # queries at 4.4-7 ("dog" 5.2, "firefox" 5.2, "zebra" 6.9), so 4.5 trades the weakest
    # (bare "car": its photo at z 4.4) for "nothing found" on noise. Noise tops grow slowly
    # with the index (the maximum of more draws): a much larger index may want ~5. 0 disables.
    z_floor: float = 4.5
    knn_k: int = 200  # only used without the vector cache (sqlite-vec KNN)
    # Score bonus in z units for priority 100 (scaled linearly; 0 disables). Only meant to break
    # near-ties in favour of files in the user's own folders.
    location_prior: float = 0.3
    duplicate_similarity: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DUPLICATE_SIMILARITY))
    # A long-running process keeps all vectors in RAM (768 floats = 3 KiB per chunk) up to this
    # many chunks; beyond it, searches use sqlite-vec directly.
    vector_cache_max_chunks: int = 1_000_000

    # Daemon (semmad). The query model is unloaded (VRAM freed) after this long with no
    # queries; the next query reloads it (~1.5 s). The full index model is unloaded sooner once
    # indexing finishes.
    model_idle_unload_minutes: float = 15
    index_model_unload_seconds: float = 60
    rescan_hours: float = 24  # full rescan interval (inotify triggers partial rescans in between)
    watch_debounce_seconds: float = 5
    # Indexing pauses while processes other than ours use more than this % of the GPU's SMs
    # (sampled every gpu_check_seconds while indexing; every 60 s while paused).
    gpu_busy_threshold: float = 30
    gpu_check_seconds: float = 30
    krunner_max_results: int = 10
    query_debounce_ms: float = 100  # search-as-you-type: only the last keystroke is searched

    def effective_roots(self) -> list[Path]:
        """Configured roots plus detected mounts, canonicalised and de-duplicated.

        Nested roots are kept: the scanner never descends into another root (or into any mount
        point), so each root is walked exactly once, by itself.
        """
        roots = list(self.roots)
        if self.include_mounts:
            roots += detect_mounts()
        resolved: list[Path] = []
        for root in roots:
            root = canonical(root.expanduser())
            if root.is_dir() and root not in resolved:
                resolved.append(root)
        return resolved

    def canonicalize(self) -> None:
        """Resolve symlinks in every configured path so paths compare equal everywhere
        (e.g. Fedora Atomic, where /home is a symlink to /var/home)."""
        self.home = canonical(self.home)
        self.roots = [canonical(p) for p in self.roots]
        self.exclude_paths = [canonical(p) for p in self.exclude_paths]
        self.locations = {canonical(p): pts for p, pts in self.locations.items()}
        if self.llama_lib_dir is not None:
            self.llama_lib_dir = canonical(self.llama_lib_dir)
        if self.model_dir is not None:
            self.model_dir = canonical(self.model_dir)
        self.exclude_dir_names = frozenset(n.casefold() for n in self.exclude_dir_names)
        self.root_system_dir_names = frozenset(n.casefold() for n in self.root_system_dir_names)
        self.project_build_dir_names = frozenset(n.casefold() for n in self.project_build_dir_names)


def canonical(path: Path) -> Path:
    """Absolute path with symlinks resolved (non-strict: missing paths are fine)."""
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError):
        return path.expanduser().absolute()


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
    return xdg_config_home(home) / "semma" / "config.toml"


def load_config(path: Path | None = None, home: Path | None = None) -> Config:
    """Build the default config, then apply overrides from the TOML file if it exists."""
    home = home or Path.home()
    config_home = xdg_config_home(home)
    cfg = Config(
        home=home,
        db_path=xdg_data_home(home) / "semma" / "index.db",
        roots=[home],
        locations=default_locations(home, read_user_dirs(home, config_home)),
    )
    path = path or default_config_path(home)
    if path.is_file():
        with path.open("rb") as fh:
            apply_overrides(cfg, tomllib.load(fh))
    cfg.canonicalize()
    return cfg


KNOWN_KEYS: dict[str, frozenset[str]] = {
    "": frozenset({"db_path", "scan", "priority", "media", "model", "index", "search", "daemon"}),
    "scan": frozenset({"roots", "include_mounts", "skip_hidden", "skip_app_installs", "exclude_dir_names",
                       "root_system_dir_names", "project_build_dir_names", "project_markers",
                       "extra_exclude_dir_names", "exclude_paths"}),
    "priority": frozenset({"locations", "type_weights", "other_location", "project_penalty",
                           "depth_penalty_per_level", "depth_penalty_max", "tiny_penalty", "huge_video_penalty",
                           "depth_free_levels", "recency"}),
    "media": frozenset({"max_minutes", "segment_seconds"}),
    "model": frozenset({"name", "device", "query_device", "llama_lib_dir", "model_dir"}),
    "index": frozenset({"batch_sizes", "prefetch_chunks"}),
    "search": frozenset({"z_gate", "z_floor", "knn_k", "location_prior", "duplicate_similarity", "vector_cache_max_chunks"}),
    "daemon": frozenset({"model_idle_unload_minutes", "index_model_unload_seconds", "rescan_hours",
                         "watch_debounce_seconds", "gpu_busy_threshold", "gpu_check_seconds", "krunner_max_results",
                         "query_debounce_ms", "query_device"}),
}

# Settings of earlier versions that are accepted with a warning and ignored.
REMOVED_KEYS: dict[str, frozenset[str]] = {
    "search": frozenset({"min_similarity", "similarity_margin", "min_similarity_short", "similarity_margin_short",
                         "rrf_k"}),
}

VALID_QUERY_DEVICES = frozenset({"auto", "cpu", "cuda", "vulkan", "gpu"})


def _warn(msg: str) -> None:
    print(f"semma: config: {msg}", file=sys.stderr)


def check_unknown_keys(data: dict[str, Any]) -> list[str]:
    """Names of keys this version does not know (typos would otherwise be silently ignored)."""
    unknown = [k for k in data if k not in KNOWN_KEYS[""]]
    for section, keys in KNOWN_KEYS.items():
        value = data.get(section) if section else None
        if isinstance(value, dict):
            removed = REMOVED_KEYS.get(section, frozenset())
            unknown += [f"{section}.{k}" for k in value if k not in keys and k not in removed]
    return unknown


def _is_valid_device_spec(val: str) -> bool:
    if val in VALID_QUERY_DEVICES:
        return True
    for prefix in ("cuda:", "vulkan:", "gpu:"):
        if val.startswith(prefix) and val[len(prefix):].isdigit():
            return True
    return False


def _bounded_float(raw: Any, key_name: str, lo: float, hi: float, default: float) -> float:
    val = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
    if val is None or not lo <= val <= hi:
        _warn(f"{key_name} must be a number between {lo:g} and {hi:g} (got {raw!r}); using {default}")
        return default
    return val


def _parse_query_device(raw: Any, key_name: str) -> str:
    if isinstance(raw, str):
        val = raw.strip().lower()
        if _is_valid_device_spec(val):
            return val
    _warn(f"{key_name} must be 'auto', 'cpu', 'vulkan', 'gpu' or 'cuda' (got {raw!r}); using 'auto'")
    return "auto"


def _parse_device(raw: Any, key_name: str) -> str | None:
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in ("", "auto"):
            return None
        if _is_valid_device_spec(val):
            return val
    _warn(f"{key_name} must be 'auto', 'cpu', 'vulkan', 'gpu' or 'cuda' (got {raw!r}); using 'auto'")
    return None


def apply_overrides(cfg: Config, data: dict[str, Any]) -> None:
    home = cfg.home
    for key in check_unknown_keys(data):
        _warn(f"unknown key {key!r} ignored")
    for section, keys in REMOVED_KEYS.items():
        value = data.get(section)
        if isinstance(value, dict):
            for key in sorted(keys & value.keys()):
                _warn(f"{section}.{key} is no longer used and is ignored (search ranks by how far a match "
                      f"stands out from the whole index; see search.z_gate)")
    scan = data.get("scan", {})
    prio = data.get("priority", {})
    media = data.get("media", {})
    model = data.get("model", {})
    index = data.get("index", {})
    search = data.get("search", {})
    daemon = data.get("daemon", {})

    if "db_path" in data:
        cfg.db_path = _expand(data["db_path"], home)
    if "roots" in scan:
        cfg.roots = [_expand(p, home) for p in scan["roots"]]
    for key in ("include_mounts", "skip_hidden", "skip_app_installs"):
        if key in scan:
            setattr(cfg, key, bool(scan[key]))
    for key in ("exclude_dir_names", "root_system_dir_names", "project_build_dir_names", "project_markers"):
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
        cfg.device = _parse_device(model["device"], "model.device")
    if "query_device" in model:
        cfg.query_device = _parse_query_device(model["query_device"], "model.query_device")
    if "llama_lib_dir" in model:
        cfg.llama_lib_dir = _expand(str(model["llama_lib_dir"]), home) if model["llama_lib_dir"] else None
    if "model_dir" in model:
        cfg.model_dir = _expand(str(model["model_dir"]), home) if model["model_dir"] else None
    if "batch_sizes" in index:
        cfg.batch_sizes.update({k: max(1, int(v)) for k, v in index["batch_sizes"].items()})
    if "prefetch_chunks" in index:
        cfg.prefetch_chunks = max(1, int(index["prefetch_chunks"]))
    if "location_prior" in search:
        cfg.location_prior = _bounded_float(search["location_prior"], "search.location_prior", 0.0, 2.0,
                                            cfg.location_prior)
    if "z_gate" in search:
        cfg.z_gate = _bounded_float(search["z_gate"], "search.z_gate", 0.0, 10.0, cfg.z_gate)
    if "z_floor" in search:
        cfg.z_floor = _bounded_float(search["z_floor"], "search.z_floor", 0.0, 20.0, cfg.z_floor)
    if "knn_k" in search:
        k = int(search["knn_k"])
        if not 1 <= k <= MAX_KNN_K:
            _warn(f"search.knn_k must be between 1 and {MAX_KNN_K}; using {min(max(k, 1), MAX_KNN_K)}")
        cfg.knn_k = min(max(k, 1), MAX_KNN_K)
    if "duplicate_similarity" in search:
        cfg.duplicate_similarity.update({k: float(v) for k, v in search["duplicate_similarity"].items()})
    if "vector_cache_max_chunks" in search:
        cfg.vector_cache_max_chunks = max(0, int(search["vector_cache_max_chunks"]))
    for key in ("model_idle_unload_minutes", "index_model_unload_seconds", "rescan_hours", "watch_debounce_seconds",
                "gpu_busy_threshold", "gpu_check_seconds", "query_debounce_ms"):
        if key in daemon:
            setattr(cfg, key, max(0.0, float(daemon[key])))
    if "query_device" in daemon:
        cfg.query_device = _parse_query_device(daemon["query_device"], "daemon.query_device")
    if "krunner_max_results" in daemon:
        cfg.krunner_max_results = max(1, int(daemon["krunner_max_results"]))
