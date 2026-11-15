"""The Priorities section's settings (priority.locations, priority.other_location) as plain
values. No Qt here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...config import canonical, default_locations, read_user_dirs, xdg_config_home
from .context import Changes, Problem, SettingsContext

SECTION = "priority"

DEFAULT_LOCATIONS: dict[str, int] = {
    "~/Documents": 40,
    "~/Desktop": 35,
    "~/Pictures": 30,
    "~/Music": 30,
    "~/Videos": 30,
    "~/Downloads": 15,
    "~": 20,
}
DEFAULT_OTHER_LOCATION = 10
DEFAULT_ADDED_SCORE = 50
MIN_SCORE = 0
MAX_SCORE = 100


def clamp_score(val: float | int) -> int:
    return max(MIN_SCORE, min(MAX_SCORE, round(float(val))))


def default_locations_for(ctx: SettingsContext) -> dict[str, int]:
    """The built-in folder scores for `ctx`, with display paths ('~/Documents', '~')."""
    raw_home = ctx.config.source_home or ctx.home
    raw_defaults = default_locations(raw_home, read_user_dirs(raw_home, xdg_config_home(raw_home)))
    return {ctx.display_path(canonical(p)): clamp_score(pts) for p, pts in raw_defaults.items()}


@dataclass
class PrioritiesValues:
    """`locations` maps display paths ('~/Documents', '~') to integer scores 0-100."""

    locations: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LOCATIONS))
    other_location: int = DEFAULT_OTHER_LOCATION
    defaults: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LOCATIONS), compare=False, repr=False
    )

    def copy(self) -> PrioritiesValues:
        return PrioritiesValues(
            locations=dict(self.locations),
            other_location=int(self.other_location),
            defaults=dict(self.defaults),
        )

    def is_default_folder(self, folder: str) -> bool:
        return folder in self.defaults


DEFAULTS = PrioritiesValues()


def load(ctx: SettingsContext) -> PrioritiesValues:
    cfg = ctx.config
    defaults = default_locations_for(ctx)
    locations = dict(defaults)
    for path, pts in cfg.locations.items():
        locations[ctx.display_path(path)] = clamp_score(pts)
    return PrioritiesValues(
        locations=locations,
        other_location=clamp_score(cfg.other_location),
        defaults=defaults,
    )


def location_overrides(locations: dict[str, int], defaults: dict[str, int]) -> dict[str, int]:
    """Only entries that differ from `defaults` (modified default folders or user-added folders)."""
    return {
        folder: round(float(score))
        for folder, score in locations.items()
        if defaults.get(folder) != round(float(score))
    }


def changes(saved: PrioritiesValues, edited: PrioritiesValues) -> Changes:
    """Only keys the user changed; returning to defaults yields None so the key is removed."""
    out: dict[str, object] = {}
    defaults = edited.defaults or saved.defaults or DEFAULT_LOCATIONS

    norm_saved = {k: round(float(v)) for k, v in saved.locations.items()}
    norm_edited = {k: round(float(v)) for k, v in edited.locations.items()}
    if norm_edited != norm_saved:
        overrides = location_overrides(norm_edited, defaults)
        out["locations"] = overrides if overrides else None

    saved_other = round(float(saved.other_location))
    edited_other = round(float(edited.other_location))
    if edited_other != saved_other:
        out["other_location"] = None if edited_other == DEFAULT_OTHER_LOCATION else edited_other

    return {SECTION: out} if out else {}


def add_location(
    locations: dict[str, int],
    path: Path | str,
    ctx: SettingsContext,
    score: int = DEFAULT_ADDED_SCORE,
) -> tuple[dict[str, int], str | None]:
    """`locations` with `path` added (as '~/...'), or unchanged and the reason if already present."""
    resolved = ctx.resolve(str(path))
    text = ctx.display_path(resolved)
    if any(ctx.resolve(f) == resolved for f in locations):
        return locations, f"{text} is in the list already."
    return {**locations, text: clamp_score(score)}, None


def remove_location(
    locations: dict[str, int],
    folder: str,
    defaults: dict[str, int] | None = None,
) -> dict[str, int]:
    """Remove a user-added folder; default folders cannot be removed."""
    defs = DEFAULT_LOCATIONS if defaults is None else defaults
    if folder in defs:
        return dict(locations)
    return {k: v for k, v in locations.items() if k != folder}


def reset_locations(values: PrioritiesValues) -> PrioritiesValues:
    """Return a copy of `values` with `locations` restored to `values.defaults`."""
    return PrioritiesValues(
        locations=dict(values.defaults),
        other_location=values.other_location,
        defaults=dict(values.defaults),
    )


def problems(values: PrioritiesValues, ctx: SettingsContext | None = None) -> list[Problem]:
    out: list[Problem] = []
    for folder, score in values.locations.items():
        if not MIN_SCORE <= score <= MAX_SCORE:
            out.append(Problem(f"The score for {folder} must be between {MIN_SCORE} and {MAX_SCORE}.", blocking=True))
    if not MIN_SCORE <= values.other_location <= MAX_SCORE:
        out.append(
            Problem(f"The score for other folders must be between {MIN_SCORE} and {MAX_SCORE}.", blocking=True)
        )
    return out
