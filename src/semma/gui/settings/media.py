"""Settings › Media: how much of each audio and video file is indexed (media.max_minutes,
scan.max_file_mb)."""

from __future__ import annotations

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QDoubleSpinBox,
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
    icon_fallbacks = (
        "applications-multimedia-symbolic", "folder-music-symbolic", "audio-x-generic-symbolic", "video-x-generic",
    )
    description = "How much of each audio and video file Semma indexes."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = mm.MediaValues()

        box = QGroupBox("Audio and video", self)
        bl = QVBoxLayout(box)
        bl.setSpacing(8)

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
            "Longer limits make indexing slower. A longer limit re-indexes the files that were cut off; "
            "a shorter one removes what lies past it.",
            box,
        )
        note.setWordWrap(True)
        note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        bl.addWidget(note)

        bl.addSpacing(4)

        audio_row = QHBoxLayout()
        audio_label = QLabel("Skip audio files larger than", box)
        self.max_audio_gb = QDoubleSpinBox(box)
        self.max_audio_gb.setDecimals(1)
        self.max_audio_gb.setRange(mm.MIN_MAX_FILE_GB, mm.MAX_MAX_FILE_GB)
        self.max_audio_gb.setSingleStep(0.5)
        self.max_audio_gb.setSuffix(" GB")
        self.max_audio_gb.valueChanged.connect(self._edited)
        audio_label.setBuddy(self.max_audio_gb)
        audio_row.addWidget(audio_label)
        audio_row.addWidget(self.max_audio_gb)
        audio_row.addStretch(1)
        bl.addLayout(audio_row)

        video_row = QHBoxLayout()
        video_label = QLabel("Skip video files larger than", box)
        self.max_video_gb = QDoubleSpinBox(box)
        self.max_video_gb.setDecimals(1)
        self.max_video_gb.setRange(mm.MIN_MAX_FILE_GB, mm.MAX_MAX_FILE_GB)
        self.max_video_gb.setSingleStep(1.0)
        self.max_video_gb.setSuffix(" GB")
        self.max_video_gb.valueChanged.connect(self._edited)
        video_label.setBuddy(self.max_video_gb)
        video_row.addWidget(video_label)
        video_row.addWidget(self.max_video_gb)
        video_row.addStretch(1)
        bl.addLayout(video_row)

        label_w = max(audio_label.sizeHint().width(), video_label.sizeHint().width())
        audio_label.setMinimumWidth(label_w)
        video_label.setMinimumWidth(label_w)

        size_note = QLabel("Larger files are left out of the index.", box)
        size_note.setWordWrap(True)
        size_note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        bl.addWidget(size_note)

        self.body.addWidget(box)

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = mm.load(ctx)
        for spin, val in (
            (self.max_minutes, self._saved.max_minutes),
            (self.max_audio_gb, self._saved.max_audio_gb),
            (self.max_video_gb, self._saved.max_video_gb),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def values(self) -> mm.MediaValues:
        return mm.MediaValues(
            max_minutes=self.max_minutes.value(),
            max_audio_gb=round(self.max_audio_gb.value(), 1),
            max_video_gb=round(self.max_video_gb.value(), 1),
            other_max_file_mb=dict(self._saved.other_max_file_mb),
        )

    def changes(self) -> Changes:
        return mm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return mm.problems(self.values(), self.ctx) if self.ctx is not None else []

    def _edited(self, *_args: object) -> None:
        self.edited.emit()
