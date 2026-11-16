"""The application and its main window."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QCloseEvent, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget, QWidget

from .. import dbus_names
from ..config import load_config
from ..onboarding import onboarding_done
from .client import DaemonClient
from .onboarding import OnboardingWindow
from .qtasync import Relay
from .settings.page import SettingsPage
from .status_page import StatusPage
from .theme import apply_desktop_theme, prepare_platform_env
from .widgets import theme_icon

APP_ID = dbus_names.GNOME_DESKTOP_ID.removesuffix(".desktop")  # io.github.Semma
DEFAULT_SIZE = (800, 640)


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
        self.tabs.setBackgroundRole(QPalette.ColorRole.Window)
        self.status_page = StatusPage(client, home, self.tabs)
        self.settings_page = SettingsPage(client, config_path, home, parent=self.tabs)
        self.tabs.addTab(self.status_page, QIcon(), "Status")
        self.tabs.addTab(self.settings_page, QIcon(), "Settings")
        self._apply_icons()
        self.setCentralWidget(self.tabs)
        self.resize(*DEFAULT_SIZE)
        self.activate_requested.connect(self.bring_forward)
        self._wizard: OnboardingWindow | None = None
        self.settings_page.setup_requested.connect(self.open_setup)
        self._closing_after_apply = False
        self._in_close_event = False
        self._apply_close_result: bool | None = None
        self.settings_page.reload_finished.connect(self._on_reload_finished)

    def _apply_icons(self) -> None:
        self.tabs.setTabIcon(0, theme_icon("dialog-information-symbolic", "dialog-information", widget=self))
        self.tabs.setTabIcon(1, theme_icon("configure", "preferences-system-symbolic", "preferences-system",
                                            "preferences-other", widget=self))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def bring_forward(self, token: str = "") -> None:
        wizard = self._wizard
        if wizard is not None and wizard.isVisible():
            wizard.bring_forward(token)
            return
        if token:
            os.environ["XDG_ACTIVATION_TOKEN"] = token  # used by Qt's Wayland activation
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()

    def open_setup(self) -> None:
        """Open the wizard over this window. The caller has already cleared the completion flag."""
        if self._wizard is not None and self._wizard.isVisible():
            self._wizard.raise_()
            self.hide()
            return
        if self._wizard is not None:
            self._wizard.deleteLater()  # the closed one from an earlier run
        wizard = OnboardingWindow(self.client, self.settings_page.config_path, self.settings_page.home,
                                  rerun=True)
        self._wizard = wizard
        wizard.finished_ok.connect(self._setup_finished)
        wizard.closed_early.connect(self._setup_closed)
        wizard.show()
        self.hide()

    def _setup_finished(self) -> None:
        self.settings_page.reload()
        self.show()
        self.raise_()
        wizard = self._wizard
        if wizard is not None:
            wizard.close()

    def _setup_closed(self) -> None:
        self.tabs.setCurrentWidget(self.status_page)  # it says that setup is not finished
        if not self.isVisible():
            self.show()
            self.raise_()

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
    return theme_icon(APP_ID, "system-search", "edit-find", monochrome=False)


def make_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if app is None:
        prepare_platform_env()
        QApplication.setApplicationName("semma")
        QApplication.setApplicationDisplayName("Semma")
        QApplication.setDesktopFileName(APP_ID)  # Wayland app_id -> io.github.Semma.desktop
        app = QApplication(argv if argv is not None else sys.argv[:1])
        apply_desktop_theme(app)
    assert isinstance(app, QApplication)
    return app


def _setup_required(config_path: Path | None) -> bool:
    """True when the user still has to finish the wizard. A broken config opens the main window."""
    try:
        cfg = load_config(config_path)
    except Exception:
        return False
    return not onboarding_done(cfg)


def run(config_path: Path | None = None) -> int:
    app = make_app()
    client = DaemonClient()
    relay = Relay()
    current: dict[str, MainWindow | OnboardingWindow | None] = {"window": None}

    def on_activate(token: str) -> None:  # on the client's thread
        def raise_current() -> None:
            window = current["window"]
            if window is not None:
                window.bring_forward(token)  # the wizard or the main window
        relay.post(raise_current)

    from .single import claim

    try:
        owned = client.submit(claim(on_activate)).result(timeout=5)
    except Exception:
        owned = None
    if owned is False:  # another window came forward
        client.close()
        return 0

    def show_main() -> MainWindow:
        window = MainWindow(client, config_path)
        window.setWindowIcon(app_icon())
        current["window"] = window
        window.show()
        return window

    if _setup_required(config_path):
        wizard = OnboardingWindow(client, config_path)
        wizard.setWindowIcon(app_icon())
        current["window"] = wizard

        def started() -> None:
            show_main()
            wizard.close()

        wizard.finished_ok.connect(started)
        wizard.show()
    else:
        show_main()
    app.setWindowIcon(app_icon())
    try:
        return app.exec()
    finally:
        client.close()
