"""The Settings page: the list of sections, the selected section's form, Apply and Revert."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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

    def __init__(self, client: Any, config_path: Path | None = None, home: Path | None = None,
                 sections: list[type[SettingsSection]] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.config_path = config_path
        self.home = home
        self.ctx: SettingsContext | None = None
        self._relay = Relay(self)

        self.nav = QListWidget(self)
        self.nav.setIconSize(QSize(22, 22))
        self.nav.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.stack = QStackedWidget(self)
        self.sections: list[SettingsSection] = []
        for cls in sections or SECTIONS:
            section = cls()
            section.edited.connect(self._update)
            self.sections.append(section)
            scroll = QScrollArea(self.stack)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            holder = QWidget()
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(12, 0, 4, 0)
            hl.addWidget(section)
            scroll.setWidget(holder)
            self.stack.addWidget(scroll)
            QListWidgetItem(theme_icon(cls.icon, "preferences-other", widget=self), cls.title, self.nav)
        self.nav.setFixedWidth(max(self.nav.sizeHintForColumn(0) + 2 * self.nav.frameWidth() + 16, 150))
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

        top = QHBoxLayout()
        top.addWidget(self.nav)
        top.addWidget(self.stack, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top, 1)
        lay.addWidget(self.problems_label)
        lay.addWidget(self.error)
        lay.addWidget(self.notice)
        lay.addWidget(self.buttons)
        self.reload()

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

    def select(self, key: str) -> SettingsSection:
        for i, s in enumerate(self.sections):
            if s.key == key:
                self.nav.setCurrentRow(i)
                return s
        raise KeyError(key)

