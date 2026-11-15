"""The Media section's settings (media.max_minutes, scan.max_file_mb for audio and video)
as plain values. No Qt here."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ... import filetypes as ft
from ...config import DEFAULT_MAX_BYTES
from .context import Changes, Problem, SettingsContext

SECTION = "media"
SCAN_SECTION = "scan"
DEFAULT_MAX_MINUTES = 30
MIN_MAX_MINUTES = 1
MAX_MAX_MINUTES = 720

DEFAULT_AUDIO_GB = DEFAULT_MAX_BYTES[ft.AUDIO] / 2**30  # 4.0
DEFAULT_VIDEO_GB = DEFAULT_MAX_BYTES[ft.VIDEO] / 2**30  # 100.0
MIN_MAX_FILE_GB = 0.1
MAX_MAX_FILE_GB = 1000.0


def gb_to_mb(gb: float) -> int:
    return round(round(float(gb), 1) * 1024)


def _other_overrides(ctx: SettingsContext) -> dict[str, int | float]:
    """Existing `scan.max_file_mb` overrides for kinds other than audio and video."""
    scan = ctx.raw.get(SCAN_SECTION)
    raw_mb = scan.get("max_file_mb") if isinstance(scan, dict) else None
    if not isinstance(raw_mb, dict):
        return {}
    out: dict[str, int | float] = {}
    for kind, val in raw_mb.items():
        if kind not in ft.KINDS or kind in (ft.AUDIO, ft.VIDEO):
            continue
        if not isinstance(val, int | float) or isinstance(val, bool) or not (val > 0 and math.isfinite(val)):
            continue
        if round(float(val) * 2**20) != DEFAULT_MAX_BYTES[kind]:
            out[kind] = val
    return out


@dataclass
class MediaValues:
    max_minutes: int = DEFAULT_MAX_MINUTES
    max_audio_gb: float = DEFAULT_AUDIO_GB
    max_video_gb: float = DEFAULT_VIDEO_GB
    other_max_file_mb: dict[str, int | float] = field(
        default_factory=dict, compare=False, repr=False
    )

    def copy(self) -> MediaValues:
        return MediaValues(
            max_minutes=int(self.max_minutes),
            max_audio_gb=float(self.max_audio_gb),
            max_video_gb=float(self.max_video_gb),
            other_max_file_mb=dict(self.other_max_file_mb),
        )


DEFAULTS = MediaValues()


def load(ctx: SettingsContext) -> MediaValues:
    cfg = ctx.config
    minutes = max(MIN_MAX_MINUTES, round(cfg.max_media_seconds / 60))
    audio_gb = max(MIN_MAX_FILE_GB, min(MAX_MAX_FILE_GB, round(cfg.max_bytes[ft.AUDIO] / 2**30, 1)))
    video_gb = max(MIN_MAX_FILE_GB, min(MAX_MAX_FILE_GB, round(cfg.max_bytes[ft.VIDEO] / 2**30, 1)))
    return MediaValues(
        max_minutes=minutes,
        max_audio_gb=audio_gb,
        max_video_gb=video_gb,
        other_max_file_mb=_other_overrides(ctx),
    )


def changes(saved: MediaValues, edited: MediaValues) -> Changes:
    """Only keys the user changed; returning to the default yields None so the key is removed."""
    out: Changes = {}

    new_min = round(float(edited.max_minutes))
    old_min = round(float(saved.max_minutes))
    if new_min != old_min:
        out[SECTION] = {"max_minutes": None if new_min == DEFAULT_MAX_MINUTES else new_min}

    new_audio = round(float(edited.max_audio_gb), 1)
    old_audio = round(float(saved.max_audio_gb), 1)
    new_video = round(float(edited.max_video_gb), 1)
    old_video = round(float(saved.max_video_gb), 1)
    if new_audio != old_audio or new_video != old_video:
        mb_table: dict[str, int | float] = dict(edited.other_max_file_mb or saved.other_max_file_mb)
        if new_audio != DEFAULT_AUDIO_GB:
            mb_table[ft.AUDIO] = gb_to_mb(new_audio)
        if new_video != DEFAULT_VIDEO_GB:
            mb_table[ft.VIDEO] = gb_to_mb(new_video)
        out[SCAN_SECTION] = {"max_file_mb": mb_table if mb_table else None}

    return out


def problems(values: MediaValues, ctx: SettingsContext | None = None) -> list[Problem]:
    out: list[Problem] = []
    if values.max_minutes < MIN_MAX_MINUTES:
        out.append(Problem(f"The duration limit must be at least {MIN_MAX_MINUTES} minute.", blocking=True))
    if not MIN_MAX_FILE_GB <= values.max_audio_gb <= MAX_MAX_FILE_GB:
        out.append(
            Problem(
                f"The audio size limit must be between {MIN_MAX_FILE_GB:g} GB and {MAX_MAX_FILE_GB:g} GB.",
                blocking=True,
            )
        )
    if not MIN_MAX_FILE_GB <= values.max_video_gb <= MAX_MAX_FILE_GB:
        out.append(
            Problem(
                f"The video size limit must be between {MIN_MAX_FILE_GB:g} GB and {MAX_MAX_FILE_GB:g} GB.",
                blocking=True,
            )
        )
    return out
