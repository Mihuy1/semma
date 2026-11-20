"""Priority score (0-100) deciding the order files are indexed in."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import filetypes as ft
from .config import Config

DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class Score:
    total: float
    parts: dict[str, float]

    def explain(self) -> str:
        return " ".join(f"{k}{v:+g}" for k, v in self.parts.items() if v)


class Scorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._locations = sorted(cfg.locations.items(), key=lambda kv: len(kv[0].parts), reverse=True)

    def location(self, path: Path) -> tuple[Path | None, float]:
        for loc, points in self._locations:
            if path.is_relative_to(loc):
                return loc, points
        return None, self.cfg.other_location

    def score(self, path: Path, kind: str, size: int, mtime: float, in_project: bool,
              root: Path, now: float) -> Score:
        cfg = self.cfg
        anchor, loc_points = self.location(path)
        parts: dict[str, float] = {"loc": loc_points, "type": cfg.type_weights.get(kind, 0.0)}

        if in_project:
            parts["project"] = cfg.project_penalty

        base = anchor if anchor is not None else root
        depth = len(path.parent.relative_to(base).parts) if path.is_relative_to(base) else 0
        extra = depth - cfg.depth_free_levels
        if extra > 0:
            parts["depth"] = max(cfg.depth_penalty_max, extra * cfg.depth_penalty_per_level)

        age_days = (now - mtime) / DAY
        for max_days, bonus in cfg.recency:
            if age_days <= max_days:
                parts["recent"] = bonus
                break

        if size < cfg.tiny_bytes.get(kind, 0):
            parts["tiny"] = cfg.tiny_penalty
        if kind == ft.VIDEO and size > cfg.huge_video_bytes:
            parts["huge"] = cfg.huge_video_penalty

        total = min(100.0, max(0.0, sum(parts.values())))
        return Score(round(total, 1), parts)
