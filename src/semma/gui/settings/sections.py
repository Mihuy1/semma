"""The sections of the Settings page, in the order they are listed."""

from __future__ import annotations

from .base import SettingsSection, placeholder
from .filetypes import FiletypesSection
from .folders import FoldersSection
from .media import MediaSection
from .priorities import PrioritiesSection

SECTIONS: list[type[SettingsSection]] = [
    FoldersSection,
    FiletypesSection,
    PrioritiesSection,
    MediaSection,
    placeholder("performance", "Performance", "speedometer"),
]
