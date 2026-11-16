"""Settings › File types: which file kinds are indexed and which extensions are skipped."""

from __future__ import annotations

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ... import filetypes as ft
from . import filetypes_model as ftm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext


class FiletypesSection(SettingsSection):
    key = "filetypes"
    title = "File types"
    icon = "document-multiple"
    icon_fallbacks = ("x-office-document-symbolic", "folder-documents-symbolic", "x-office-document", "text-x-generic")
    description = (
        "Choose which kinds of files Semma indexes. "
        "Turning a file type off removes its files from the index."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = ftm.FiletypesValues()

        kinds_box = QGroupBox("Indexed file types", self)
        kl = QVBoxLayout(kinds_box)
        kl.setSpacing(5)
        self.kind_checks: dict[str, QCheckBox] = {}
        for kind in ft.KINDS:
            item = QVBoxLayout()
            item.setSpacing(1)
            cb = QCheckBox(ftm.KIND_NAMES[kind], kinds_box)
            cb.toggled.connect(self._edited)
            self.kind_checks[kind] = cb
            exts = QLabel(ftm.KIND_EXTENSIONS[kind], kinds_box)
            exts.setWordWrap(True)
            exts.setForegroundRole(QPalette.ColorRole.PlaceholderText)
            exts.setContentsMargins(24, 0, 0, 0)
            item.addWidget(cb)
            item.addWidget(exts)
            kl.addLayout(item)

        skip_box = QGroupBox("Skipped extensions", self)
        sl = QVBoxLayout(skip_box)
        sl.setSpacing(8)
        hint = QLabel(
            "Files with these extensions are not indexed (separated by commas or spaces).",
            skip_box,
        )
        hint.setWordWrap(True)
        hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        sl.addWidget(hint)
        self.exclude_extensions = QLineEdit(skip_box)
        self.exclude_extensions.setPlaceholderText("e.g. md, tif, mov")
        self.exclude_extensions.textChanged.connect(self._edited)
        self.skipped_extensions = self.exclude_extensions
        sl.addWidget(self.exclude_extensions)

        self.body.addWidget(kinds_box)
        self.body.addWidget(skip_box)

    def _edited(self, *_args: object) -> None:
        self.edited.emit()

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = ftm.load(ctx)
        saved_kinds = set(self._saved.kinds)
        for kind, cb in self.kind_checks.items():
            cb.blockSignals(True)
            cb.setChecked(kind in saved_kinds)
            cb.blockSignals(False)
        self.exclude_extensions.blockSignals(True)
        self.exclude_extensions.setText(ftm.format_extensions(self._saved.exclude_extensions))
        self.exclude_extensions.blockSignals(False)

    def values(self) -> ftm.FiletypesValues:
        kinds = [kind for kind, cb in self.kind_checks.items() if cb.isChecked()]
        exts = ftm.parse_extensions(self.exclude_extensions.text())
        return ftm.FiletypesValues(kinds=kinds, exclude_extensions=exts)

    def changes(self) -> Changes:
        return ftm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return ftm.problems(self.values(), self.ctx) if self.ctx is not None else []
