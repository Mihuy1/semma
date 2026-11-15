"""The application and its main window."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget

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

    def bring_forward(self, token: str = "") -> None:
        if token:
            os.environ["XDG_ACTIVATION_TOKEN"] = token  # used by Qt's Wayland activation
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()


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
