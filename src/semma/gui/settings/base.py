"""The contract of a settings section.

A section is one entry in the list on the left of the Settings page and the form on the right.
Subclass `SettingsSection`, set `key`, `title` and `icon`, build the form in `__init__` (add
widgets to `self.body`), and implement:

- `load(ctx)`: fill the widgets from the saved settings (`ctx.config` effective values,
  `ctx.raw` what config.toml sets). Called at start, after Apply and after Revert.
- `changes()`: what the user changed, as `config_write.update_config` takes it:
  {"section": {"key": value}}; only keys whose value differs from the saved one, and None for
  a key set back to its default (it is removed from config.toml). {} when nothing changed.
- `problems()` (optional): notes shown under the form; a `Problem(..., blocking=True)` keeps
  Apply disabled.

Emit `self.edited` whenever a widget changes; the page then asks every section for
`changes()` / `problems()` to enable Apply and Revert. Keep the logic that turns values into
changes in a Qt-free `<name>_model.py` next to the section so it is tested without a display
(see `folders_model.py` and `folders.py`), and register the class in `sections.SECTIONS`.
"""

from __future__ import annotations

from typing import ClassVar

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..widgets import scaled_font
from .context import Changes, Problem, SettingsContext


class SettingsSection(QWidget):
    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    icon: ClassVar[str] = "preferences-other"
    description: ClassVar[str] = ""

    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(self.title, self)
        heading.setFont(scaled_font(self, 1.25, bold=True))
        outer.addWidget(heading)
        if self.description:
            desc = QLabel(self.description, self)
            desc.setWordWrap(True)
            outer.addWidget(desc)
        self.body = QVBoxLayout()
        outer.addLayout(self.body)
        outer.addStretch(1)
        self.ctx: SettingsContext | None = None

    def load(self, ctx: SettingsContext) -> None:
        self.ctx = ctx

    def changes(self) -> Changes:
        return {}

    def is_modified(self) -> bool:
        return bool(self.changes())

    def problems(self) -> list[Problem]:
        return []


def placeholder(key: str, title: str, icon: str) -> type[SettingsSection]:
    """A registered section that has no settings yet."""

    class Placeholder(SettingsSection):
        description = "Coming soon. Until then these settings can be set in config.toml."

    Placeholder.key, Placeholder.title, Placeholder.icon = key, title, icon
    Placeholder.__name__ = Placeholder.__qualname__ = f"{key.capitalize()}Placeholder"
    return Placeholder
