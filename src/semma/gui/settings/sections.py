"""The sections of the Settings page, in the order they are listed."""

from __future__ import annotations

from .base import SettingsSection
from .filetypes import FiletypesSection
from .folders import FoldersSection
from .media import MediaSection
from .performance import PerformanceSection
from .priorities import PrioritiesSection
from .screenshots import ScreenshotsSection

SECTIONS: list[type[SettingsSection]] = [
    FoldersSection,
    FiletypesSection,
    PrioritiesSection,
    MediaSection,
    ScreenshotsSection,
    PerformanceSection,
]
