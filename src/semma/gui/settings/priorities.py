"""Settings › Priorities: folder priority scores (priority.locations, priority.other_location)."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets import theme_icon
from . import priorities_model as pm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext
from .folders import HOME_LABEL


class PrioritiesSection(SettingsSection):
    key = "priorities"
    title = "Priorities"
    icon = "view-sort-descending"
    icon_fallbacks = ("view-sort-descending-symbolic", "starred-symbolic", "starred")
    description = "Folders with a higher score are indexed first and win ties in search results."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = pm.PrioritiesValues()
        self._defaults: dict[str, int] = dict(pm.DEFAULT_LOCATIONS)
        self._folders: list[str] = []
        self.score_spins: dict[str, QSpinBox] = {}
        # Replaced in tests; returns the chosen folder or "".
        self.choose_folder: Callable[[str], str] = self._choose_folder

        folders_box = QGroupBox("Folder scores", self)
        fl = QVBoxLayout(folders_box)
        fl.setSpacing(10)

        table_row = QHBoxLayout()
        table_row.setSpacing(10)
        self.table = QTableWidget(0, 2, folders_box)
        self.table.setHorizontalHeaderLabels(["Folder", "Score (0–100)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        spin_probe = QSpinBox(self.table)
        spin_probe.hide()
        spin_probe.setRange(pm.MIN_SCORE, pm.MAX_SCORE)
        spin_probe.ensurePolished()
        spin_sh = spin_probe.sizeHint()
        spin_probe.setParent(None)
        spin_probe.deleteLater()
        self.table.setColumnWidth(1, max(120, spin_sh.width() + 24, header.sectionSizeHint(1) + 16))
        self._row_h = max(self.table.fontMetrics().height() + 14, spin_sh.height() + 8, 34)
        self.table.verticalHeader().setDefaultSectionSize(self._row_h)
        self.table.itemSelectionChanged.connect(self._update_buttons)

        buttons = QVBoxLayout()
        buttons.setSpacing(6)
        self.add_button = QPushButton("Add…", folders_box)
        self.remove_button = QPushButton("Remove", folders_box)
        self.reset_button = QPushButton("Reset to defaults", folders_box)
        self._apply_icons()
        btn_w = max(
            self.add_button.sizeHint().width(),
            self.remove_button.sizeHint().width(),
            self.reset_button.sizeHint().width(),
        )
        for btn in (self.add_button, self.remove_button, self.reset_button):
            btn.setMinimumWidth(btn_w)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        self.reset_button.clicked.connect(self.reset_to_defaults)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)

        table_row.addWidget(self.table, 1)
        table_row.addLayout(buttons)
        fl.addLayout(table_row)
        fl.addSpacing(6)

        other_row = QHBoxLayout()
        other_label = QLabel("Other folders:", folders_box)
        self.other_location = QSpinBox(folders_box)
        self.other_location.setRange(pm.MIN_SCORE, pm.MAX_SCORE)
        self.other_location.valueChanged.connect(self._edited)
        other_label.setBuddy(self.other_location)
        other_row.addWidget(other_label)
        other_row.addWidget(self.other_location)
        other_row.addStretch(1)
        fl.addLayout(other_row)

        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.hide()

        self.body.addWidget(folders_box)
        self.body.addWidget(self.message)
        self._fit_table_height()
        self._update_buttons()

    def _apply_icons(self) -> None:
        self.add_button.setIcon(theme_icon("list-add", widget=self))
        self.remove_button.setIcon(theme_icon("list-remove", widget=self))
        self.reset_button.setIcon(theme_icon("edit-undo", widget=self))
        folder_icon = theme_icon("folder", "folder-symbolic", widget=self)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None:
                item.setIcon(folder_icon)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def _fit_table_height(self) -> None:
        header_h = max(self.table.horizontalHeader().sizeHint().height(), 28)
        visible_rows = min(max(len(self._folders), 1), 10)
        h = visible_rows * self._row_h + header_h + 2 * self.table.frameWidth() + 2
        self.table.setMinimumHeight(h)
        if len(self._folders) <= 10:
            self.table.setMaximumHeight(h)
        else:
            self.table.setMaximumHeight(10 * self._row_h + header_h + 2 * self.table.frameWidth() + 2)

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = pm.load(ctx)
        self._defaults = dict(self._saved.defaults)
        self._set_locations(self._saved.locations)
        self.other_location.blockSignals(True)
        self.other_location.setValue(self._saved.other_location)
        self.other_location.blockSignals(False)
        self._show_message("")
        self._update_buttons()

    def _set_locations(self, locations: dict[str, int]) -> None:
        self._folders = list(locations.keys())
        self.score_spins.clear()
        self.table.setRowCount(0)
        self.table.setRowCount(len(self._folders))
        icon = theme_icon("folder", "folder-symbolic", widget=self)
        for row, (folder, score) in enumerate(locations.items()):
            label = HOME_LABEL if folder == "~" else folder
            item = QTableWidgetItem(icon, label)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(str(self.ctx.resolve(folder)) if self.ctx is not None else folder)
            self.table.setItem(row, 0, item)

            cell = QWidget(self.table)
            cell.setObjectName("prioritySpinCell")
            cell.setAutoFillBackground(False)
            cell.setStyleSheet("#prioritySpinCell { background: transparent; }")
            spin = QSpinBox(cell)
            spin.setRange(pm.MIN_SCORE, pm.MAX_SCORE)
            spin.setValue(pm.clamp_score(score))
            spin.valueChanged.connect(self._on_score_changed)
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(6, 3, 6, 3)
            cl.addWidget(spin)
            self.table.setCellWidget(row, 1, cell)
            self.score_spins[folder] = spin
        self._fit_table_height()
        self._update_buttons()

    def locations(self) -> dict[str, int]:
        return {folder: self.score_spins[folder].value() for folder in self._folders}

    def values(self) -> pm.PrioritiesValues:
        return pm.PrioritiesValues(
            locations=self.locations(),
            other_location=self.other_location.value(),
            defaults=dict(self._defaults),
        )

    def changes(self) -> Changes:
        return pm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return pm.problems(self.values(), self.ctx) if self.ctx is not None else []

    def add_folder(self, path: str, score: int = pm.DEFAULT_ADDED_SCORE) -> None:
        assert self.ctx is not None
        updated, msg = pm.add_location(self.locations(), path, self.ctx, score=score)
        if msg:
            self._show_message(msg)
            return
        self._set_locations(updated)
        self.table.selectRow(len(updated) - 1)
        self._edited()

    def add(self, path: str, score: int = pm.DEFAULT_ADDED_SCORE) -> None:
        self.add_folder(path, score=score)

    def remove_folder(self, folder: str) -> None:
        if folder in self._defaults:
            return
        updated = pm.remove_location(self.locations(), folder, self._defaults)
        if updated != self.locations():
            self._set_locations(updated)
            self._edited()

    def reset_to_defaults(self) -> None:
        self._set_locations(dict(self._defaults))
        self._edited()

    def _choose_folder(self, start: str) -> str:
        return QFileDialog.getExistingDirectory(self, "Choose a folder", start)

    def _add(self) -> None:
        start = str(self.ctx.home) if self.ctx is not None else ""
        path = self.choose_folder(start)
        if path:
            self.add_folder(path)

    def _selected_folder(self) -> str | None:
        row = self.table.currentRow()
        if 0 <= row < len(self._folders):
            selected = self.table.selectionModel().selectedRows()
            if selected:
                return self._folders[selected[0].row()]
        return None

    def _remove(self) -> None:
        folder = self._selected_folder()
        if folder is not None and folder not in self._defaults:
            self.remove_folder(folder)

    def _update_buttons(self) -> None:
        folder = self._selected_folder()
        can_remove = folder is not None and folder not in self._defaults
        self.remove_button.setEnabled(can_remove)
        if folder is not None and folder in self._defaults:
            self.remove_button.setToolTip("Built-in folders cannot be removed; change their score or reset to defaults.")
        else:
            self.remove_button.setToolTip("")
        has_custom = self.locations() != self._defaults if self.score_spins else False
        self.reset_button.setEnabled(has_custom)

    def _on_score_changed(self, *_args: object) -> None:
        self._update_buttons()
        self._edited()

    def _show_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def _edited(self, *_args: object) -> None:
        self._show_message("")
        self._update_buttons()
        self.edited.emit()
