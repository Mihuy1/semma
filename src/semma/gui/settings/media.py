"""Settings › Media: how much of each audio and video file is indexed (media.max_minutes)."""

from __future__ import annotations

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import media_model as mm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext


class MediaSection(SettingsSection):
    key = "media"
    title = "Media"
    icon = "applications-multimedia"
    description = "How much of each audio and video file Semma indexes."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = mm.MediaValues()

        box = QGroupBox("Audio and video", self)
        bl = QVBoxLayout(box)

        row = QHBoxLayout()
        prefix = QLabel("Index the first", box)
        self.max_minutes = QSpinBox(box)
        self.max_minutes.setRange(mm.MIN_MAX_MINUTES, mm.MAX_MAX_MINUTES)
        self.max_minutes.setSuffix(" minutes")
        self.max_minutes.valueChanged.connect(self._edited)
        prefix.setBuddy(self.max_minutes)
        suffix = QLabel("of each audio and video file", box)
        row.addWidget(prefix)
        row.addWidget(self.max_minutes)
        row.addWidget(suffix)
        row.addStretch(1)
        bl.addLayout(row)

        note = QLabel(
            "Longer limits make indexing slower. Files already indexed are not indexed again.",
            box,
        )
        note.setWordWrap(True)
        note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        bl.addWidget(note)

        self.body.addWidget(box)

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = mm.load(ctx)
        self.max_minutes.blockSignals(True)
        self.max_minutes.setValue(self._saved.max_minutes)
        self.max_minutes.blockSignals(False)

    def values(self) -> mm.MediaValues:
        return mm.MediaValues(max_minutes=self.max_minutes.value())

    def changes(self) -> Changes:
        return mm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return mm.problems(self.values(), self.ctx) if self.ctx is not None else []

    def _edited(self, *_args: object) -> None:
        self.edited.emit()
