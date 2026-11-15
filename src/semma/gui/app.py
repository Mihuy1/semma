"""The application and its main window."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget, QWidget

from .. import dbus_names
from ..config import load_config
from .client import DaemonClient
from .qtasync import Relay
from .settings.page import SettingsPage
from .status_page import StatusPage
from .widgets import theme_icon

APP_ID = dbus_names.GNOME_DESKTOP_ID.removesuffix(".desktop")  # io.github.Semma
DEFAULT_SIZE = (780, 600)


class MainWindow(QMainWindow):
    activate_requested = Signal(str)

    def __init__(self, client: Any, config_path: Path | None = None, home: Path | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.setWindowTitle("Semma")
        self.setWindowIcon(app_icon())
        if home is None:
            try:
                home = load_config(config_path).home
            except Exception:
                home = Path.home()
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.status_page = StatusPage(client, home, self.tabs)
        self.settings_page = SettingsPage(client, config_path, home, parent=self.tabs)
        self.tabs.addTab(self.status_page, theme_icon("dialog-information", widget=self), "Status")
        self.tabs.addTab(self.settings_page, theme_icon("configure", "preferences-system", "preferences-other",
                                                        widget=self), "Settings")
        self.setCentralWidget(self.tabs)
        self.resize(*DEFAULT_SIZE)
        self.activate_requested.connect(self.bring_forward)
        self._closing_after_apply = False
        self._in_close_event = False
        self._apply_close_result: bool | None = None
        self.settings_page.reload_finished.connect(self._on_reload_finished)

    def bring_forward(self, token: str = "") -> None:
        if token:
            os.environ["XDG_ACTIVATION_TOKEN"] = token  # used by Qt's Wayland activation
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()

    def _show_settings_section(self) -> None:
        self.tabs.setCurrentWidget(self.settings_page)
        modified = self.settings_page.modified_sections()
        if not modified:
            return
        current = self.settings_page.nav.currentRow()
        if not (0 <= current < len(self.settings_page.sections)
                and self.settings_page.sections[current].is_modified()):
            self.settings_page.select(modified[0].key)

    def _confirm_unapplied_dialog(self) -> QMessageBox:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unapplied changes")
        box.setText("You have unapplied settings changes.")
        names = ", ".join(s.title for s in self.settings_page.modified_sections())
        prefix = f"Changed sections: {names}.\n\n" if names else ""
        if self.settings_page.can_apply():
            box.setInformativeText(f"{prefix}Apply your changes before closing, or discard them?")
            box.setStandardButtons(
                QMessageBox.StandardButton.Apply
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
        else:
            box.setInformativeText(
                f"{prefix}Fix the problems in Settings before applying, or discard your changes."
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.setEscapeButton(QMessageBox.StandardButton.Cancel)
        return box

    def _try_apply_and_close(self, event: QCloseEvent) -> None:
        self._closing_after_apply = True
        self._apply_close_result = None
        if not self.settings_page.apply():
            self._closing_after_apply = False
            event.ignore()
            self._show_settings_section()
            return
        if self._apply_close_result is not None:
            can_close = self._apply_close_result
            self._closing_after_apply = False
            self._apply_close_result = None
            if can_close:
                super().closeEvent(event)
            else:
                event.ignore()
                self._show_settings_section()
            return
        event.ignore()

    def _on_reload_finished(self, can_close: bool) -> None:
        if not self._closing_after_apply:
            return
        if self._in_close_event:
            self._apply_close_result = can_close
            return
        self._closing_after_apply = False
        if can_close:
            self.close()
        else:
            self._show_settings_section()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._closing_after_apply:
            event.ignore()
            return
        if not self.settings_page.is_modified():
            super().closeEvent(event)
            return
        self._in_close_event = True
        try:
            choice = self._confirm_unapplied_dialog().exec()
            if choice == QMessageBox.StandardButton.Discard:
                super().closeEvent(event)
                return
            if choice == QMessageBox.StandardButton.Apply and self.settings_page.can_apply():
                self._try_apply_and_close(event)
                return
            event.ignore()
            self._show_settings_section()
        finally:
            self._in_close_event = False


def app_icon() -> QIcon:
    return theme_icon(APP_ID, "system-search", "edit-find")


def make_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if app is None:
        QApplication.setApplicationName("semma")
        QApplication.setApplicationDisplayName("Semma")
        QApplication.setDesktopFileName(APP_ID)  # Wayland app_id -> io.github.Semma.desktop
        app = QApplication(argv if argv is not None else sys.argv[:1])
    assert isinstance(app, QApplication)
    return app


def run(config_path: Path | None = None) -> int:
    app = make_app()
    client = DaemonClient()
    relay = Relay()
    window: MainWindow | None = None

    def on_activate(token: str) -> None:  # on the client's thread
        relay.post(lambda: window.activate_requested.emit(token) if window is not None else None)

    from .single import claim

    try:
        owned = client.submit(claim(on_activate)).result(timeout=5)
    except Exception:
        owned = None
    if owned is False:  # another window came forward
        client.close()
        return 0
    window = MainWindow(client, config_path)
    window.setWindowIcon(app_icon())
    app.setWindowIcon(app_icon())
    window.show()
    try:
        return app.exec()
    finally:
        client.close()
