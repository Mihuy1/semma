"""Settings › Screenshots: reading text from screenshots with Tesseract (ocr.enabled, ocr.languages)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from . import screenshots_model as sm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext


class ScreenshotsSection(SettingsSection):
    key = "screenshots"
    title = "Screenshots"
    icon = "accessories-screenshot-tool"
    icon_fallbacks = (
        "camera-photo-symbolic",
        "image-x-generic-symbolic",
        "camera-photo",
        "image-x-generic",
    )
    description = "Find screenshots by the words in them."

    def __init__(self, parent: QWidget | None = None, *, facts: sm.OcrFacts | None = None) -> None:
        super().__init__(parent)
        self._injected_facts = facts
        self._facts: sm.OcrFacts | None = facts
        self._saved = sm.ScreenshotsValues()

        text_box = QGroupBox("Screenshot text", self)
        tl = QVBoxLayout(text_box)
        tl.setSpacing(8)

        self.enabled_check = QCheckBox("Read the text in screenshots", text_box)
        self.enabled_check.setChecked(sm.DEFAULT_ENABLED)
        self.enabled_check.toggled.connect(self._on_enabled_toggled)
        tl.addWidget(self.enabled_check)

        note = QLabel(
            "Makes screenshots findable by the words in them. A screenshot is a picture in a "
            "Screenshots folder or with a name that starts with Screenshot. Turning this on reads "
            "the screenshots that are already indexed again; turning it off removes their text.",
            text_box,
        )
        note.setWordWrap(True)
        note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        tl.addWidget(note)

        self.missing_note = QLabel(text_box)
        self.missing_note.setWordWrap(True)
        self.missing_note.hide()
        tl.addWidget(self.missing_note)

        self.missing_command = QLabel(text_box)
        self.missing_command.setWordWrap(True)
        self.missing_command.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.missing_command.setContentsMargins(16, 0, 0, 0)
        self.missing_command.hide()
        tl.addWidget(self.missing_command)

        lang_box = QGroupBox("Languages", self)
        ll = QVBoxLayout(lang_box)
        ll.setSpacing(8)

        self.language_labels: list[QLabel] = []
        self.language_combos: list[QComboBox] = []
        row_titles = ("First language", "Second language", "Third language")
        for title in row_titles[:sm.MAX_LANGUAGES]:
            row = QHBoxLayout()
            lbl = QLabel(title, lang_box)
            combo = QComboBox(lang_box)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.addItem("None", None)
            combo.currentIndexChanged.connect(self._edited)
            lbl.setBuddy(combo)
            row.addWidget(lbl)
            row.addWidget(combo)
            row.addStretch(1)
            ll.addLayout(row)
            self.language_labels.append(lbl)
            self.language_combos.append(combo)

        label_w = max(lbl.sizeHint().width() for lbl in self.language_labels)
        for lbl in self.language_labels:
            lbl.setMinimumWidth(label_w)

        limit_note = QLabel(
            "You can choose up to three languages. Each language uses about 25 MB more memory "
            "while screenshots are being read and makes reading slower, so choose only the "
            "languages your screenshots are in. Put the one you use most first.",
            lang_box,
        )
        limit_note.setWordWrap(True)
        ll.addWidget(limit_note)

        self.more_languages_note = QLabel(lang_box)
        self.more_languages_note.setWordWrap(True)
        self.more_languages_note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.more_languages_note.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ll.addWidget(self.more_languages_note)

        self.body.addWidget(text_box)
        self.body.addWidget(lang_box)
        self._update_language_rows_enabled()

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        facts = self._injected_facts if self._injected_facts is not None else sm.OcrFacts.detect()
        self._facts = facts
        self._saved = sm.load(ctx, facts)

        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(self._saved.enabled)
        self.enabled_check.blockSignals(False)

        can_enable = facts.library_available and bool(facts.installed_languages)
        self.enabled_check.setEnabled(can_enable or self._saved.enabled)
        self._update_missing_note(facts)
        self._update_more_languages_note(facts)

        choices = sm.language_choices(self._saved.languages, facts)
        for i, combo in enumerate(self.language_combos):
            combo.blockSignals(True)
            combo.clear()
            target = self._saved.languages[i] if i < len(self._saved.languages) else None
            selected_idx = 0
            for idx, (label, code) in enumerate(choices):
                combo.addItem(label, code)
                if code == target:
                    selected_idx = idx
            combo.setCurrentIndex(selected_idx)
            combo.blockSignals(False)

        self._update_language_rows_enabled()

    def values(self) -> sm.ScreenshotsValues:
        codes = [combo.currentData() for combo in self.language_combos]
        return sm.ScreenshotsValues(
            enabled=self.enabled_check.isChecked(),
            languages=sm.normalise_languages(codes),
            default_languages=(
                list(self._saved.default_languages)
                if self._saved.default_languages is not None
                else None
            ),
        )

    def changes(self) -> Changes:
        return sm.changes(self._saved, self.values(), self._facts)

    def problems(self) -> list[Problem]:
        return sm.problems(self.values(), self.ctx, self._facts) if self.ctx is not None else []

    def _update_language_rows_enabled(self) -> None:
        enabled = self.enabled_check.isChecked()
        for lbl, combo in zip(self.language_labels, self.language_combos):
            lbl.setEnabled(enabled)
            combo.setEnabled(enabled)

    def _update_missing_note(self, facts: sm.OcrFacts) -> None:
        if not facts.library_available:
            hint = facts.install_hint(None)
            if hint:
                self.missing_note.setText(
                    "Reading text in screenshots needs Tesseract. Install it with:"
                )
                self.missing_command.setText(hint)
                self.missing_command.show()
            else:
                self.missing_note.setText("Reading text in screenshots needs Tesseract.")
                self.missing_command.setText("")
                self.missing_command.hide()
            self.missing_note.show()
            return

        if not facts.installed_languages:
            hint = facts.install_hint(None)
            if hint:
                self.missing_note.setText(
                    "No Tesseract language packs are installed. Install English with:"
                )
                self.missing_command.setText(hint)
                self.missing_command.show()
            else:
                self.missing_note.setText("No Tesseract language packs are installed.")
                self.missing_command.setText("")
                self.missing_command.hide()
            self.missing_note.show()
            return

        self.missing_note.setText("")
        self.missing_note.hide()
        self.missing_command.setText("")
        self.missing_command.hide()

    def _update_more_languages_note(self, facts: sm.OcrFacts) -> None:
        hint = facts.install_hint(sm.EXAMPLE_LANGUAGE)
        if hint:
            self.more_languages_note.setText(
                f"More languages can be added by installing their Tesseract language pack, for example: {hint}"
            )
        else:
            self.more_languages_note.setText(
                "More languages can be added by installing their Tesseract language pack."
            )

    def _on_enabled_toggled(self, _checked: bool) -> None:
        self._update_language_rows_enabled()
        self._edited()

    def _edited(self, *_args: object) -> None:
        self.edited.emit()
