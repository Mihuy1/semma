"""Settings › Folders: which folders are indexed and which are skipped. The reference section:
widgets only, the values and changes live in `folders_model`."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import theme_icon
from . import folders_model as fm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext

HOME_LABEL = "Home folder (~)"


class FolderList(QWidget):
    """A list of folders with Add (the desktop's folder chooser) and Remove."""

    changed = Signal()

    def __init__(self, add_text: str, chooser_title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._folders: list[str] = []
        self._chooser_title = chooser_title
        self.ctx: SettingsContext | None = None
        self.on_message: Callable[[str], None] = lambda text: None
        # Replaced in tests; returns the chosen folder or "".
        self.choose_folder: Callable[[str], str] = self._choose_folder

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget(self)
        self.list.setUniformItemSizes(True)
        # About five rows: the form stays without a scroll bar at the default window size.
        row = max(self.list.fontMetrics().height(), self.list.iconSize().height(), 16) + 6
        self.list.setMinimumHeight(3 * row)
        self.list.setMaximumHeight(5 * row + 2 * self.list.frameWidth())
        self.list.itemSelectionChanged.connect(self._update_buttons)
        buttons = QVBoxLayout()
        self.add_button = QPushButton(theme_icon("list-add", widget=self), add_text, self)
        self.remove_button = QPushButton(theme_icon("list-remove", widget=self), "Remove", self)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        lay.addWidget(self.list, 1)
        lay.addLayout(buttons)
        self._update_buttons()

    def folders(self) -> list[str]:
        return list(self._folders)

    def set_folders(self, folders: list[str]) -> None:
        self._folders = list(folders)
        self.list.clear()
        icon = theme_icon("folder", widget=self)
        for f in self._folders:
            item = QListWidgetItem(icon, HOME_LABEL if f == "~" else f, self.list)
            item.setToolTip(str(self.ctx.resolve(f)) if self.ctx is not None else f)
        self._update_buttons()

    def add(self, path: str) -> None:
        assert self.ctx is not None
        folders, message = fm.add_folder(self._folders, path, self.ctx)
        if message:
            self.on_message(message)
            return
        self.set_folders(folders)
        self.list.setCurrentRow(len(folders) - 1)
        self.changed.emit()

    def _choose_folder(self, start: str) -> str:
        return QFileDialog.getExistingDirectory(self, self._chooser_title, start)

    def _add(self) -> None:
        start = str(self.ctx.home) if self.ctx is not None else ""
        path = self.choose_folder(start)
        if path:
            self.add(path)

    def _remove(self) -> None:
        rows = sorted((self.list.row(i) for i in self.list.selectedItems()), reverse=True)
        if not rows:
            return
        for r in rows:
            del self._folders[r]
        self.set_folders(self._folders)
        self.changed.emit()

    def _update_buttons(self) -> None:
        self.remove_button.setEnabled(bool(self.list.selectedItems()))


class FoldersSection(SettingsSection):
    key = "folders"
    title = "Folders"
    icon = "folder"
    description = "Semma indexes the files in these folders and keeps watching them for changes."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = fm.FoldersValues()

        roots_box = QGroupBox("Indexed folders", self)
        rl = QVBoxLayout(roots_box)
        self.roots = FolderList("Add…", "Choose a folder to index", roots_box)
        rl.addWidget(self.roots)
        self.include_mounts = QCheckBox("Also index external drives (mounted under /mnt, /media and /run/media)",
                                        roots_box)
        rl.addWidget(self.include_mounts)

        skip_box = QGroupBox("Skipped folders", self)
        sl = QVBoxLayout(skip_box)
        hint = QLabel("Files in these folders and the folders inside them are not indexed.", skip_box)
        hint.setWordWrap(True)
        sl.addWidget(hint)
        self.excludes = FolderList("Add…", "Choose a folder to skip", skip_box)
        sl.addWidget(self.excludes)
        self.skip_hidden = QCheckBox("Skip hidden files and folders (names starting with a dot)", skip_box)
        sl.addWidget(self.skip_hidden)

        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.hide()
        self.body.addWidget(roots_box)
        self.body.addWidget(skip_box)
        self.body.addWidget(self.message)

        for lst in (self.roots, self.excludes):
            lst.changed.connect(self._edited)
            lst.on_message = self._show_message
        self.include_mounts.toggled.connect(self._edited)
        self.skip_hidden.toggled.connect(self._edited)

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = fm.load(ctx)
        self.roots.ctx = self.excludes.ctx = ctx
        widgets = (self.include_mounts, self.skip_hidden)
        for w in widgets:
            w.blockSignals(True)
        self.roots.set_folders(self._saved.roots)
        self.excludes.set_folders(self._saved.exclude_paths)
        self.include_mounts.setChecked(self._saved.include_mounts)
        self.skip_hidden.setChecked(self._saved.skip_hidden)
        for w in widgets:
            w.blockSignals(False)
        self._show_message("")

    def values(self) -> fm.FoldersValues:
        return fm.FoldersValues(roots=self.roots.folders(), exclude_paths=self.excludes.folders(),
                                include_mounts=self.include_mounts.isChecked(),
                                skip_hidden=self.skip_hidden.isChecked())

    def changes(self) -> Changes:
        return fm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return fm.problems(self.values(), self.ctx) if self.ctx is not None else []

    def _show_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def _edited(self) -> None:
        self._show_message("")
        self.edited.emit()
