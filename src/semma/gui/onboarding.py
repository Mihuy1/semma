"""First-run setup: a few pages, then indexing is allowed to start.

The pages edit the same config.toml keys as Settings. Closing the window early asks first: go
back to setup, finish later (edits are discarded and indexing stays off), or skip the pages and
start indexing with the saved settings (the defaults on a first run). The model download is
shown on every page; it is started by the daemon, not by this window.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QPalette, QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import dbus_names
from ..config_write import ConfigWriteError, update_config
from ..daemon import DaemonNotRunning
from . import model
from .client import describe_error
from .qtasync import Relay
from .settings.context import SettingsContext, merge_changes
from .settings.filetypes import FiletypesSection
from .settings.folders import FoldersSection
from .settings.folders_model import NOTHING_INDEXED
from .settings.media import MediaSection
from .settings.page import RESTART_NOTICE
from .settings.performance import GraphicsCardGroup
from .settings.performance_model import PerformanceValues
from .settings.performance_model import changes as device_changes
from .settings.performance_model import load as load_performance
from .settings.performance_model import problems as device_problems
from .settings.screenshots import ScreenshotsSection
from .widgets import scaled_font, theme_icon

APP_ID = dbus_names.GNOME_DESKTOP_ID.removesuffix(".desktop")
POLL_MS = 1000


class _WelcomePage(QWidget):
    key = "welcome"
    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(14)
        self._icon_label = QLabel(self)
        self._icon_label.setAutoFillBackground(False)
        self._icon_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._icon_label.setStyleSheet("background: transparent;")
        self._apply_icons()
        header.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
        title = QLabel("Welcome to Semma", self)
        title.setFont(scaled_font(self, 1.45, bold=True))
        header.addWidget(title, 1, Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(header)

        body = QLabel(
            "Semma searches your files by meaning, from KRunner and the GNOME overview.\n\n"
            "The model downloads in the background. Indexing starts when you finish these steps. "
            "You can change any of this later in Settings.",
            self,
        )
        body.setWordWrap(True)
        lay.addWidget(body)
        lay.addStretch(1)

    def _apply_icons(self) -> None:
        icon_pm = theme_icon(APP_ID, "system-search", "edit-find", widget=self).pixmap(56, 56)
        self._icon_label.setPixmap(icon_pm)
        self._icon_label.setVisible(not icon_pm.isNull())

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def load(self, _ctx: SettingsContext) -> None:
        return None

    def changes(self) -> dict[str, dict[str, Any]]:
        return {}

    def problems(self) -> list[Any]:
        return []


class _DevicePage(QWidget):
    """Model device and GPU-busy pause settings. The rest of Performance stays on Settings."""

    key = "device"
    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx: SettingsContext | None = None
        self._saved_values: PerformanceValues | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        title = QLabel("Performance", self)
        title.setFont(scaled_font(self, 1.25, bold=True))
        desc = QLabel(
            "Where the model runs and whether indexing pauses while other programs use the graphics card. "
            "These are the same settings as Settings → Performance.",
            self,
        )
        desc.setWordWrap(True)
        desc.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.box = GraphicsCardGroup(
            "Applied before the model is loaded. If it is already loaded, the service needs a restart.",
            self,
        )
        self.device_combo = self.box.device_combo
        self.pause_when_gpu_busy = self.box.pause_when_gpu_busy
        self.gpu_busy_label = self.box.gpu_busy_label
        self.gpu_busy_threshold = self.box.gpu_busy_threshold
        self.box.edited.connect(self.edited)
        lay.addWidget(title)
        lay.addWidget(desc)
        lay.addWidget(self.box)
        lay.addStretch(1)

    def load(self, ctx: SettingsContext) -> None:
        self.ctx = ctx
        saved = load_performance(ctx)
        self._saved_values = saved
        self.box.load(saved)

    def _edited_values(self) -> PerformanceValues:
        assert self._saved_values is not None
        return self.box.apply_to(self._saved_values)

    def changes(self) -> dict[str, dict[str, Any]]:
        if self._saved_values is None:
            return {}
        return device_changes(self._saved_values, self._edited_values())

    def problems(self) -> list[Any]:
        if self._saved_values is None or self.ctx is None:
            return []
        return device_problems(self._edited_values(), self.ctx)


class OnboardingWindow(QMainWindow):
    finished_ok = Signal()
    closed_early = Signal()

    def __init__(self, client: Any, config_path: Path | None = None, home: Path | None = None,
                 parent: QWidget | None = None, *, rerun: bool = False) -> None:
        super().__init__(parent)
        self.client = client
        self.config_path = config_path
        self.home = home
        self.rerun = rerun
        self._saved = False
        self._relay = Relay(self)
        self._completed = False
        self._saving = False
        self._awaiting_restart = False
        self._index = 0
        self._in_flight = False
        self._load_error = ""
        self.ctx: SettingsContext | None = None

        self.setWindowTitle("Semma setup")
        self.setWindowIcon(theme_icon(APP_ID, "system-search", "edit-find", monochrome=False))
        self.resize(800, 640)

        self.step_label = QLabel(self)
        self.step_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)

        self.welcome = _WelcomePage()
        self.folders = FoldersSection()
        self.filetypes = FiletypesSection()
        self.media = MediaSection()
        self.screenshots = ScreenshotsSection()
        self.device = _DevicePage()
        self._forms: list[Any] = [
            self.welcome, self.folders, self.filetypes, self.media, self.screenshots, self.device,
        ]

        self.stack = QStackedWidget(self)
        for form in self._forms:
            if form is not self.welcome:
                form.edited.connect(self._update_buttons)
            scroll = QScrollArea(self.stack)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setBackgroundRole(QPalette.ColorRole.Window)
            scroll.viewport().setBackgroundRole(QPalette.ColorRole.Window)
            scroll.viewport().setAutoFillBackground(False)
            scroll.setWidget(form)
            form.setAutoFillBackground(False)
            self.stack.addWidget(scroll)

        self.problems = QLabel(self)
        self.problems.setWordWrap(True)
        self.notice = QLabel(self)
        self.notice.setWordWrap(True)
        self.notice.hide()
        self.restart_button = QPushButton("Restart service", self)
        self.restart_button.hide()
        self.restart_button.clicked.connect(self._restart_service)

        self.footer_label = QLabel(self)
        self.footer_label.setWordWrap(True)
        self.footer_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.footer_bar = QProgressBar(self)
        self.footer_bar.setTextVisible(False)
        self.start_button = QPushButton("Start service", self)
        self.start_button.clicked.connect(self._start_service)
        self.start_button.hide()
        footer = QVBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(self.footer_label)
        bar_row = QHBoxLayout()
        bar_row.addWidget(self.footer_bar, 1)
        bar_row.addWidget(self.start_button)
        footer.addLayout(bar_row)

        self.back_button = QPushButton("Back", self)
        self.next_button = QPushButton("Next", self)
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)
        lay.addWidget(self.step_label)
        lay.addWidget(self.stack, 1)
        lay.addWidget(self.problems)
        lay.addWidget(self.notice)
        lay.addWidget(self.restart_button, 0, Qt.AlignmentFlag.AlignRight)
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(line)
        lay.addLayout(footer)
        lay.addLayout(buttons)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

        self._load()
        self._show_page(0)

    def bring_forward(self, token: str = "") -> None:
        if token:
            os.environ["XDG_ACTIVATION_TOKEN"] = token
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._timer.start()
        self._poll()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._completed:
            if self._saving:
                event.ignore()
                return
            if self._saved:
                self._complete()
            elif self.ctx is None:
                self.closed_early.emit()
            else:
                choice = self.confirm_close()
                if choice == "skip":
                    event.ignore()
                    QTimer.singleShot(0, lambda: self._save(include_pages=False))
                    return
                if choice != "later":
                    event.ignore()
                    return
                self.closed_early.emit()
        self._timer.stop()
        super().closeEvent(event)

    def confirm_close(self) -> str:
        """Ask what closing early should do: "stay", "later", or "skip" (start indexing now)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Setup is not finished")
        box.setText("Indexing and desktop search stay off until setup is finished.")
        if self.rerun:
            box.setInformativeText("You can finish later from Settings, or keep the settings you "
                                   "already had and start indexing again now. Changes made in "
                                   "these steps are discarded either way.")
            skip_text = "Keep current settings"
        else:
            box.setInformativeText("You can finish later by opening Semma again, or start indexing "
                                   "now with the default settings and change them later in Settings.")
            skip_text = "Use defaults"
        stay = box.addButton("Keep setting up", QMessageBox.ButtonRole.RejectRole)
        later = box.addButton("Finish later", QMessageBox.ButtonRole.DestructiveRole)
        skip = box.addButton(skip_text, QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(stay)
        box.setEscapeButton(stay)
        box.exec()
        clicked = box.clickedButton()
        if clicked is skip:
            return "skip"
        if clicked is later:
            return "later"
        return "stay"

    def _load(self) -> None:
        try:
            self.ctx = SettingsContext.load(self.config_path, self.home)
        except ConfigWriteError as exc:
            self.ctx = None
            self._load_error = f"The settings cannot be read: {exc}"
            self.problems.setText(self._load_error)
            self.problems.show()
            return
        self._load_error = ""
        for form in self._forms:
            form.load(self.ctx)

    def _form(self) -> Any:
        return self._forms[self._index]

    def _page_problems(self) -> list[Any]:
        if self.ctx is None:
            return []
        found = list(self._form().problems())
        if getattr(self._form(), "key", "") != "folders":
            return found
        return [replace(p, blocking=True) if p.code == NOTHING_INDEXED else p for p in found]

    def _blocked(self) -> bool:
        return any(getattr(p, "blocking", False) for p in self._page_problems())

    def _show_page(self, index: int) -> None:
        self._index = index
        self.stack.setCurrentIndex(index)
        self.step_label.setText(f"Step {index + 1} of {len(self._forms)}")
        self._update_buttons()

    def _apply_icons(self) -> None:
        self.start_button.setIcon(theme_icon("media-playback-start", widget=self))
        self.back_button.setIcon(theme_icon("go-previous", widget=self))
        last = self._index == len(self._forms) - 1
        if self._awaiting_restart:
            self.next_button.setIcon(theme_icon("window-close", widget=self))
        elif last:
            self.next_button.setIcon(theme_icon("media-playback-start", widget=self))
        else:
            self.next_button.setIcon(theme_icon("go-next", widget=self))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def _update_buttons(self) -> None:
        last = self._index == len(self._forms) - 1
        self.back_button.setEnabled(self._index > 0 and not self._saving)
        self.next_button.setEnabled(self.ctx is not None and not self._blocked() and not self._saving)
        if self._awaiting_restart:
            self.next_button.setText("Close")
            self.next_button.setEnabled(not self._saving)
        elif last:
            self.next_button.setText("Start indexing")
        else:
            self.next_button.setText("Next")
        self._apply_icons()
        if self._load_error:
            self.problems.setText(self._load_error)
            self.problems.setVisible(True)
        else:
            texts = [p.text for p in self._page_problems()]
            self.problems.setText("\n".join(texts))
            self.problems.setVisible(bool(texts))

    def _back(self) -> None:
        if self._index > 0 and not self._saving:
            self._awaiting_restart = False
            self._saved = False
            self.notice.hide()
            self.restart_button.hide()
            self._show_page(self._index - 1)

    def _next(self) -> None:
        if self._awaiting_restart:
            self._complete()
            return
        if self._blocked() or self.ctx is None or self._saving:
            return
        if self._index < len(self._forms) - 1:
            self._show_page(self._index + 1)
            return
        self._save()

    def _save(self, include_pages: bool = True) -> None:
        """Write the completion flag and tell the service. Without `include_pages` the edits
        on the pages are left out: the saved settings (or the defaults) are used as they are."""
        if self.ctx is None:
            return
        changes = merge_changes(*(form.changes() for form in self._forms)) if include_pages else {}
        changes.setdefault("", {})["onboarding_complete"] = True
        try:
            update_config(self.ctx.config_path, changes)
        except ConfigWriteError as exc:
            self.problems.setText(f"The settings were not saved: {exc}")
            self.problems.show()
            return
        self._saved = True
        self._saving = True
        self._update_buttons()
        self._relay.when_done(self.client.reload_config(), self._reloaded, self._reload_failed)

    def _reloaded(self, needs_restart: list[str] | None) -> None:
        self._saving = False
        if needs_restart is None:
            self.notice.setText("Saved. The running service is an older version; restart it to finish setup.")
            self.notice.show()
            self.restart_button.show()
            self._awaiting_restart = True
            self._update_buttons()
            return
        if needs_restart:
            self.notice.setText(f"{RESTART_NOTICE} Waiting for a restart: {', '.join(needs_restart)}.")
            self.notice.show()
            self.restart_button.show()
            self._awaiting_restart = True
            self._update_buttons()
            return
        self._complete()

    def _reload_failed(self, exc: BaseException) -> None:
        self._saving = False
        if isinstance(exc, DaemonNotRunning):
            self._complete()
            return
        self.notice.setText(f"Saved, but the service could not be told: {describe_error(exc)} {RESTART_NOTICE}")
        self.notice.show()
        self.restart_button.show()
        self._awaiting_restart = True
        self._update_buttons()

    def _restart_service(self) -> None:
        self.restart_button.setEnabled(False)
        self._relay.when_done(self.client.systemctl("restart"), self._restarted, self._restart_failed)

    def _restarted(self, result: tuple[int, str]) -> None:
        code, output = result
        if code == 0:
            self._complete()
            return
        self.restart_button.setEnabled(True)
        self.notice.setText(output or f"The service could not be restarted (status {code}).")
        self.notice.show()

    def _restart_failed(self, exc: BaseException) -> None:
        self.restart_button.setEnabled(True)
        self.notice.setText(f"The service could not be restarted: {describe_error(exc)}")
        self.notice.show()

    def _complete(self) -> None:
        self._completed = True
        self._timer.stop()
        self.finished_ok.emit()

    def _poll(self) -> None:
        if self._in_flight:
            return
        self._in_flight = True
        self._relay.when_done(self.client.status(), self._got_status, self._status_failed)

    def _got_status(self, st: dict[str, Any] | None) -> None:
        self._in_flight = False
        self._show_footer(model.download_footer(st))

    def _status_failed(self, exc: BaseException) -> None:
        self._in_flight = False
        self.footer_label.setText(f"Could not get the download status: {describe_error(exc)}")
        self.footer_bar.hide()
        self.start_button.hide()

    def _show_footer(self, view: model.DownloadFooter) -> None:
        self.footer_label.setText(view.text)
        self.start_button.setVisible(view.offer_start)
        if view.progress is not None:
            done, total = view.progress
            scale = max(1, (max(total, 1) + 2**30) // 2**30)
            self.footer_bar.setRange(0, max(total // scale, 1))
            self.footer_bar.setValue(min(done, total) // scale)
            self.footer_bar.show()
        elif view.busy:
            self.footer_bar.setRange(0, 0)
            self.footer_bar.show()
        else:
            self.footer_bar.hide()

    def _start_service(self) -> None:
        self.start_button.setEnabled(False)
        self._relay.when_done(self.client.systemctl("start"), self._service_started, self._service_start_failed)

    def _service_started(self, result: tuple[int, str]) -> None:
        code, output = result
        self.start_button.setEnabled(True)
        if code != 0:
            self.footer_label.setText(output or f"The service did not start (status {code}).")
        self._poll()

    def _service_start_failed(self, exc: BaseException) -> None:
        self.start_button.setEnabled(True)
        self.footer_label.setText(f"The service did not start: {describe_error(exc)}")
