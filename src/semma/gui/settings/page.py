"""The Settings page: the list of sections, the selected section's form, Apply and Revert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QPalette, QShowEvent
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...config_write import ConfigWriteError, update_config
from ...daemon import DaemonNotRunning
from ..client import describe_error
from ..qtasync import Relay
from ..widgets import MessageBar, theme_icon
from .base import SettingsSection
from .context import Problem, SettingsContext, merge_changes
from .sections import SECTIONS

RESTART_NOTICE = "Restart the service to apply these changes."


class SettingsPage(QWidget):
    reload_finished = Signal(bool)  # True when applied cleanly (or daemon stopped); False if restart needed or error
    setup_requested = Signal()  # Run setup again: the flag is already false and the daemon was told

    def __init__(self, client: Any, config_path: Path | None = None, home: Path | None = None,
                 sections: list[type[SettingsSection]] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.config_path = config_path
        self.home = home
        self.ctx: SettingsContext | None = None
        self._relay = Relay(self)

        self.nav = QListWidget(self)
        self.nav.setObjectName("settingsNav")
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setIconSize(QSize(22, 22))
        self.nav.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.stack = QStackedWidget(self)
        self._section_classes: list[type[SettingsSection]] = list(sections or SECTIONS)
        self.sections: list[SettingsSection] = []
        for cls in self._section_classes:
            section = cls()
            section.edited.connect(self._update)
            self.sections.append(section)
            scroll = QScrollArea(self.stack)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setBackgroundRole(QPalette.ColorRole.Window)
            scroll.viewport().setBackgroundRole(QPalette.ColorRole.Window)
            scroll.viewport().setAutoFillBackground(False)
            holder = QWidget()
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(10, 2, 10, 8)
            hl.addWidget(section)
            scroll.setWidget(holder)
            holder.setAutoFillBackground(False)
            self.stack.addWidget(scroll)
            QListWidgetItem(theme_icon(cls.icon, *cls.icon_fallbacks, "preferences-other", widget=self),
                            cls.title, self.nav)
        self.nav.setFixedWidth(max(self.nav.sizeHintForColumn(0) + 32, 175))
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.problems_label = QLabel(self)
        self.problems_label.setWordWrap(True)
        self.problems_label.hide()
        self.error = MessageBar(self)     # the config cannot be read or written
        self.notice = MessageBar(self)    # saved; restart needed, or the daemon was not told

        self.buttons = QDialogButtonBox(self)
        self.apply_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Apply)
        self.revert_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        self.revert_button.setText("Revert")
        self.apply_button.clicked.connect(self.apply)
        self.revert_button.clicked.connect(self.revert)
        self.setup_button = QPushButton("Run setup again", self)
        self.setup_button.setToolTip("Pause indexing and choose folders, file types and the processor again. "
                                     "Files already indexed are kept.")
        self.setup_button.clicked.connect(self.run_setup_again)
        self._apply_icons()

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self.nav)
        top.addWidget(self.stack, 1)
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        actions = QHBoxLayout()
        actions.addWidget(self.setup_button)
        actions.addStretch(1)
        actions.addWidget(self.buttons)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        lay.addLayout(top, 1)
        lay.addWidget(self.problems_label)
        lay.addWidget(self.error)
        lay.addWidget(self.notice)
        lay.addWidget(sep)
        lay.addLayout(actions)
        self.reload()

    def _apply_icons(self) -> None:
        for i, cls in enumerate(self._section_classes):
            item = self.nav.item(i)
            if item is not None:
                item.setIcon(theme_icon(cls.icon, *cls.icon_fallbacks, "preferences-other", widget=self))
        self.revert_button.setIcon(theme_icon("edit-undo", "document-revert", widget=self))
        self.apply_button.setIcon(
            theme_icon("dialog-ok-apply", "object-select-symbolic", "document-save", widget=self)
        )

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    # -- loading and state ----------------------------------------------------------------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Pick up a config.toml edited by hand meanwhile, unless the user is editing here."""
        super().showEvent(event)
        if not self.is_modified():
            self.reload()

    def reload(self) -> bool:
        """Read the saved settings again and load every section from them."""
        try:
            self.ctx = SettingsContext.load(self.config_path, self.home)
        except ConfigWriteError as exc:
            self.ctx = None
            self.error.show_message("error", f"The settings cannot be read: {exc}")
            self._update()
            return False
        self.error.clear()
        for section in self.sections:
            section.load(self.ctx)
        self._update()
        return True

    def is_modified(self) -> bool:
        return self.ctx is not None and any(s.is_modified() for s in self.sections)

    def modified_sections(self) -> list[SettingsSection]:
        return [s for s in self.sections if s.is_modified()] if self.ctx is not None else []

    def problems(self) -> list[Problem]:
        return [p for s in self.sections for p in s.problems()]

    def can_apply(self) -> bool:
        return self.is_modified() and not any(p.blocking for p in self.problems())

    def _update(self) -> None:
        modified = self.is_modified()
        problems = self.problems() if self.ctx is not None else []
        blocking = any(p.blocking for p in problems)
        self.apply_button.setEnabled(modified and not blocking)
        self.revert_button.setEnabled(modified or self.ctx is None)
        self.setup_button.setEnabled(self.ctx is not None)
        for i, s in enumerate(self.sections):
            item = self.nav.item(i)
            font = item.font()
            font.setBold(s.is_modified() if self.ctx is not None else False)
            item.setFont(font)
        self.problems_label.setText("\n".join(p.text for p in problems))
        self.problems_label.setVisible(bool(problems))

    # -- actions --------------------------------------------------------------------------
    def revert(self) -> None:
        self.notice.clear()
        self.reload()

    def apply(self) -> bool:
        if self.ctx is None or any(p.blocking for p in self.problems()):
            return False
        changes = merge_changes(*(s.changes() for s in self.sections))
        if not changes:
            return False
        try:
            update_config(self.ctx.config_path, changes)
        except ConfigWriteError as exc:
            self.error.show_message("error", f"The settings were not saved: {exc}")
            return False
        self.notice.clear()
        if not self.reload():
            return False
        self.apply_button.setEnabled(False)
        self._relay.when_done(self.client.reload_config(), self._reloaded, self._reload_failed)
        return True

    def _reloaded(self, needs_restart: list[str] | None) -> None:
        if needs_restart is None or needs_restart:
            text = RESTART_NOTICE
            if needs_restart:
                text += f" Waiting for a restart: {', '.join(needs_restart)}."
            self.notice.show_message("warning", text, "Restart service", self.restart_service)
            ok_to_close = False
        else:
            self.notice.show_message("info", "Saved and applied.")
            ok_to_close = True
        self._update()
        self.reload_finished.emit(ok_to_close)

    def _reload_failed(self, exc: BaseException) -> None:
        if isinstance(exc, DaemonNotRunning):
            self.notice.show_message("info", "Saved. The service is not running; it uses these settings when it starts.")
            ok_to_close = True
        else:
            self.notice.show_message("warning", f"Saved, but the service could not be told: {describe_error(exc)} "
                                                f"{RESTART_NOTICE}", "Restart service", self.restart_service)
            ok_to_close = False
        self._update()
        self.reload_finished.emit(ok_to_close)

    def restart_service(self) -> None:
        self.notice.action_button().setEnabled(False)
        self._relay.when_done(self.client.systemctl("restart"), self._restarted, self._restart_failed)

    def _restarted(self, result: tuple[int, str]) -> None:
        code, output = result
        self.notice.action_button().setEnabled(True)
        if code == 0:
            self.notice.show_message("info", "The service was restarted with the new settings.")
        else:
            self._restart_failed(RuntimeError(output or f"systemctl exited with status {code}"))

    def _restart_failed(self, exc: BaseException) -> None:
        self.notice.action_button().setEnabled(True)
        self.notice.show_message("error", f"The service could not be restarted: {describe_error(exc)}",
                                 "Restart service", self.restart_service)

    def run_setup_again(self) -> None:
        """Set onboarding_complete = false, tell the daemon, and ask the window to open setup.

        The index is not deleted. Indexing stays paused until setup finishes and sets the flag true.
        """
        if self.ctx is None or not self._confirm_setup_again():
            return
        try:
            update_config(self.ctx.config_path, {"": {"onboarding_complete": False}})
        except ConfigWriteError as exc:
            self.error.show_message("error", f"Setup was not started: {exc}")
            return
        self.notice.clear()
        if not self.reload():
            return
        self._relay.when_done(self.client.reload_config(), self._setup_paused, self._setup_pause_failed)

    def _confirm_setup_again(self) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Run setup again")
        box.setText("Indexing pauses until you finish setup.")
        extra = ("Unapplied edits on this page are not included until you apply them.\n\n"
                 if self.is_modified() else "")
        box.setInformativeText(f"{extra}Files already indexed are kept.")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        yes = box.button(QMessageBox.StandardButton.Yes)
        if yes is not None:
            yes.setText("Run setup again")
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _setup_paused(self, needs_restart: list[str] | None) -> None:
        if needs_restart is None:
            self.notice.show_message(
                "warning", "Restart the service so it pauses indexing. Setup will open now.")
        self.setup_requested.emit()

    def _setup_pause_failed(self, exc: BaseException) -> None:
        if not isinstance(exc, DaemonNotRunning):
            self.notice.show_message(
                "warning", f"Saved, but the service could not be told: {describe_error(exc)}")
        self.setup_requested.emit()

    def select(self, key: str) -> SettingsSection:
        for i, s in enumerate(self.sections):
            if s.key == key:
                self.nav.setCurrentRow(i)
                return s
        raise KeyError(key)

