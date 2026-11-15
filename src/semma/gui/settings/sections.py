"""The sections of the Settings page, in the order they are listed."""

from __future__ import annotations

from .base import SettingsSection, placeholder
from .filetypes import FiletypesSection
from .folders import FoldersSection

SECTIONS: list[type[SettingsSection]] = [
    FoldersSection,
    FiletypesSection,
    placeholder("priorities", "Priorities", "view-sort-descending"),
    placeholder("media", "Media", "applications-multimedia"),
    placeholder("performance", "Performance", "speedometer"),
]
