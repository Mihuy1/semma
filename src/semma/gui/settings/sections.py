"""The sections of the Settings page, in the order they are listed."""

from __future__ import annotations

from .base import SettingsSection, placeholder
from .folders import FoldersSection

SECTIONS: list[type[SettingsSection]] = [
    FoldersSection,
    placeholder("filetypes", "File types", "document-multiple"),
    placeholder("priorities", "Priorities", "view-sort-descending"),
    placeholder("media", "Media", "applications-multimedia"),
    placeholder("performance", "Performance", "speedometer"),
]
