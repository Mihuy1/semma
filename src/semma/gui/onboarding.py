"""First-run setup: a few pages, then indexing is allowed to start.

The pages edit the same config.toml keys as Settings. Closing the window discards edits and
leaves the completion flag unchanged (unless Run setup again already set it false). The model
download is shown on every page; it is started by the daemon, not by this window.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
from .settings.context import Problem, SettingsContext, merge_changes
from .settings.filetypes import FiletypesSection
from .settings.folders import FoldersSection
from .settings.media import MediaSection
from .settings.page import RESTART_NOTICE
from .settings.performance_model import changes as device_changes
from .settings.performance_model import device_choices, load as load_performance, normalise_device
from .settings.performance_model import problems as device_problems
from .widgets import scaled_font, theme_icon

APP_ID = dbus_names.GNOME_DESKTOP_ID.removesuffix(".desktop")
POLL_MS = 1000


class _WelcomePage(QWidget):
    key = "welcome"
    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        title = QLabel("Welcome", self)
        title.setFont(scaled_font(self, 1.25, bold=True))
        body = QLabel(
            "Semma searches your files by meaning, from KRunner and the GNOME overview.\n\n"
            "The model downloads in the background. Indexing starts when you finish these steps. "
            "You can change any of this later in Settings.",
            self,
        )
        body.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(body)
        lay.addStretch(1)

    def load(self, _ctx: SettingsContext) -> None:
        return None

    def changes(self) -> dict[str, dict[str, Any]]:
        return {}

    def problems(self) -> list[Any]:
        return []


class _DevicePage(QWidget):
    """Only the model device. The rest of Performance stays on the Settings page."""

    key = "device"
    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx: SettingsContext | None = None
        self._saved_values = None
        lay = QVBoxLayout(self)
        title = QLabel("Device", self)
        title.setFont(scaled_font(self, 1.25, bold=True))
        desc = QLabel(
            "Where the model runs. Automatic uses the graphics card when one is available, "
            "and the processor otherwise. This is the same setting as Settings → Performance.",
            self,
        )
        desc.setWordWrap(True)
        box = QGroupBox("Graphics card and processor", self)
        bl = QVBoxLayout(box)
        row = QHBoxLayout()
        label = QLabel("Run the model on:", box)
        self.device_combo = QComboBox(box)
        self.device_combo.currentIndexChanged.connect(self.edited.emit)
        label.setBuddy(self.device_combo)
        row.addWidget(label)
        row.addWidget(self.device_combo)
        row.addStretch(1)
        bl.addLayout(row)
        note = QLabel(
            "Applied before the model is loaded. If it is already loaded, the service needs a restart.",
            box,
        )
        note.setWordWrap(True)
        bl.addWidget(note)
        lay.addWidget(title)
        lay.addWidget(desc)
        lay.addWidget(box)
        lay.addStretch(1)

    def load(self, ctx: SettingsContext) -> None:
        self.ctx = ctx
        saved = load_performance(ctx)
        self._saved_values = saved
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        norm = normalise_device(saved.device)
        selected = 0
        for idx, (label, value) in enumerate(device_choices(saved.device)):
            self.device_combo.addItem(label, value)
            if value == norm:
                selected = idx
        self.device_combo.setCurrentIndex(selected)
        self.device_combo.blockSignals(False)

    def _device(self) -> str | None:
        return normalise_device(self.device_combo.currentData())

    def changes(self) -> dict[str, dict[str, Any]]:
        if self._saved_values is None:
            return {}
        edited = self._saved_values.copy()
        edited.device = self._device()
        return device_changes(self._saved_values, edited)

    def problems(self) -> list[Any]:
        if self._saved_values is None or self.ctx is None:
            return []
        edited = self._saved_values.copy()
        edited.device = self._device()
        return device_problems(edited, self.ctx)


class OnboardingWindow(QMainWindow):
    finished_ok = Signal()
    closed_early = Signal()

    def __init__(self, client: Any, config_path: Path | None = None, home: Path | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.config_path = config_path
        self.home = home
        self._relay = Relay(self)
        self._completed = False
        self._saving = False
        self._awaiting_restart = False
        self._index = 0
        self._in_flight = False
        self.ctx: SettingsContext | None = None

        self.setWindowTitle("Semma setup")
        self.setWindowIcon(theme_icon(APP_ID, "system-search", "edit-find"))
        self.resize(780, 600)

        self.welcome = _WelcomePage()
        self.folders = FoldersSection()
        self.filetypes = FiletypesSection()
        self.media = MediaSection()
        self.device = _DevicePage()
        self._forms: list[Any] = [self.welcome, self.folders, self.filetypes, self.media, self.device]

        self.stack = QStackedWidget(self)
        for form in self._forms:
            if form is not self.welcome:
                form.edited.connect(self._update_buttons)
            scroll = QScrollArea(self.stack)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(form)
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
        self.footer_bar = QProgressBar(self)
        self.footer_bar.setTextVisible(False)
        self.start_button = QPushButton(theme_icon("media-playback-start", widget=self), "Start service", self)
        self.start_button.clicked.connect(self._start_service)
        self.start_button.hide()
        footer = QVBoxLayout()
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
        buttons.addStretch(1)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)

        central = QWidget(self)
        lay = QVBoxLayout(central)
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
        self._timer.stop()
        if not self._completed:
            self.closed_early.emit()
        super().closeEvent(event)

    # -- pages -----------------------------------------------------------------------------
    def _load(self) -> None:
        try:
            self.ctx = SettingsContext.load(self.config_path, self.home)
        except ConfigWriteError as exc:
            self.ctx = None
            self.problems.setText(f"The settings cannot be read: {exc}")
            return
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
        # The settings page only warns when nothing would be indexed. Setup refuses to continue.
        return [Problem(p.text, blocking=True) if "Nothing is indexed" in p.text else p for p in found]

    def _blocked(self) -> bool:
        return any(getattr(p, "blocking", False) for p in self._page_problems())

    def _show_page(self, index: int) -> None:
        self._index = index
        self.stack.setCurrentIndex(index)
        self._update_buttons()

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
        texts = [p.text for p in self._page_problems()]
        self.problems.setText("\n".join(texts))
        self.problems.setVisible(bool(texts))

    def _back(self) -> None:
        if self._index > 0 and not self._saving:
            self._awaiting_restart = False
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

    # -- saving ----------------------------------------------------------------------------
    def _save(self) -> None:
        if self.ctx is None:
            return
        changes = merge_changes(*(form.changes() for form in self._forms))
        changes.setdefault("", {})["onboarding_complete"] = True
        try:
            update_config(self.ctx.config_path, changes)
        except ConfigWriteError as exc:
            self.problems.setText(f"The settings were not saved: {exc}")
            self.problems.show()
            return
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

    # -- download footer -------------------------------------------------------------------
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
