"""The Media section's settings (media.max_minutes) as plain values. No Qt here."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .context import Changes, Problem, SettingsContext

SECTION = "media"
DEFAULT_MAX_MINUTES = 30
MIN_MAX_MINUTES = 1
MAX_MAX_MINUTES = 720


@dataclass
class MediaValues:
    max_minutes: int = DEFAULT_MAX_MINUTES

    def copy(self) -> MediaValues:
        return replace(self)


DEFAULTS = MediaValues()


def load(ctx: SettingsContext) -> MediaValues:
    minutes = max(MIN_MAX_MINUTES, round(ctx.config.max_media_seconds / 60))
    return MediaValues(max_minutes=minutes)


def changes(saved: MediaValues, edited: MediaValues) -> Changes:
    """Only keys the user changed; returning to the default yields None so the key is removed."""
    new_val = round(float(edited.max_minutes))
    old_val = round(float(saved.max_minutes))
    if new_val == old_val:
        return {}
    return {SECTION: {"max_minutes": None if new_val == DEFAULT_MAX_MINUTES else new_val}}


def problems(values: MediaValues, ctx: SettingsContext | None = None) -> list[Problem]:
    if values.max_minutes < MIN_MAX_MINUTES:
        return [Problem(f"The duration limit must be at least {MIN_MAX_MINUTES} minute.", blocking=True)]
    return []
