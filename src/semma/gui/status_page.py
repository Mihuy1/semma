"""The Status page: what the daemon is doing now, polled about once a second while visible."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QHideEvent, QPalette, QShowEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import model
from .client import describe_error
from .qtasync import Relay
from .widgets import ElidedLabel, MessageBar, scaled_font, theme_icon

POLL_MS = 1000
FAILED_LIST_LIMIT = 1000


class FailedDialog(QDialog):
    """The failed files with their errors, loaded when the dialog opens."""

    def __init__(self, client: Any, home: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Files that could not be indexed")
        self.resize(900, 440)
        self.client, self.home = client, home
        self._relay = Relay(self)
        lay = QVBoxLayout(self)
        self.summary = QLabel("Loading…", self)
        self.summary.setWordWrap(True)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["File", "Error"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.header().setStretchLastSection(True)
        self.tree.setColumnWidth(0, 420)
        self.message = MessageBar(self)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.retry_button = self.buttons.addButton("Retry failed", QDialogButtonBox.ButtonRole.ActionRole)
        self._apply_icons()
        self.retry_button.setEnabled(False)
        self.retry_button.clicked.connect(self.retry)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.summary)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.message)
        lay.addWidget(self.buttons)
        self.load()

    def _apply_icons(self) -> None:
        self.retry_button.setIcon(theme_icon("view-refresh", widget=self))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def load(self) -> None:
        self.summary.setText("Loading…")
        self._relay.when_done(self.client.failed(FAILED_LIST_LIMIT), self._loaded, self._failed)

    def _loaded(self, rows: list[tuple[str, str]] | None) -> None:
        self.tree.clear()
        if rows is None:
            self.summary.setText("The Semma service is not running.")
            return
        for path, error in rows:
            item = QTreeWidgetItem([model.short_path(path, self.home), error.splitlines()[0] if error else ""])
            item.setToolTip(0, path)
            item.setToolTip(1, error)
            self.tree.addTopLevelItem(item)
        if not rows:
            self.summary.setText("No files failed.")
        elif len(rows) >= FAILED_LIST_LIMIT:
            self.summary.setText(f"The first {model.format_count(len(rows))} files that could not be indexed, "
                                 f"the most important first.")
        else:
            self.summary.setText(f"{model.plural(len(rows), 'file')} could not be indexed.")
        self.retry_button.setEnabled(bool(rows))

    def _failed(self, exc: BaseException) -> None:
        self.summary.setText("")
        self.message.show_message("error", describe_error(exc))

    def retry(self) -> None:
        self.retry_button.setEnabled(False)
        self._relay.when_done(self.client.call("RetryFailed"), self._retried, self._failed)

    def _retried(self, n: Any) -> None:
        self.message.show_message("info", f"{model.plural(int(n or 0), 'file')} queued to be indexed again.")
        self.load()


class StatusPage(QWidget):
    def __init__(self, client: Any, home: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.home = home
        self._relay = Relay(self)
        self._in_flight = False
        self._poll_error = False
        self.status: dict[str, Any] | None = None
        self.view: model.StatusView | None = None
        self.running: bool | None = None  # None until the first answer

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.poll)

        self.pages = QStackedWidget(self)
        self.pages.addWidget(self._build_waiting())
        self.pages.addWidget(self._build_running())
        self.pages.addWidget(self._build_not_running())
        self._apply_icons()
        self.error = MessageBar(self)
        lay = QVBoxLayout(self)
        lay.addWidget(self.error)
        lay.addWidget(self.pages, 1)

    # -- building ---------------------------------------------------------------------------
    def _build_waiting(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        label = QLabel("Connecting to the Semma service…", w)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)
        return w

    def _build_running(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        summary_card = QFrame(w)
        summary_card.setObjectName("statusSummaryCard")
        sl = QVBoxLayout(summary_card)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(8)

        self.headline = QLabel(summary_card)
        self.headline.setFont(scaled_font(self, 1.35, bold=True))
        self.headline.setWordWrap(True)
        self.headline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.progress = QProgressBar(summary_card)
        self.progress.setTextVisible(False)
        self.detail = ElidedLabel("", summary_card)
        self.secondary = QLabel(summary_card)
        for label in (self.detail, self.secondary):
            label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        sl.addWidget(self.headline)
        sl.addWidget(self.progress)
        sl.addWidget(self.detail)
        sl.addWidget(self.secondary)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.pause_button = QPushButton(summary_card)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.rescan_button = QPushButton(theme_icon("view-refresh", widget=self), "Rescan now", summary_card)
        self.rescan_button.setToolTip("Look for new, changed and deleted files in every indexed folder now")
        self.rescan_button.clicked.connect(self.rescan)
        actions.addWidget(self.pause_button)
        actions.addWidget(self.rescan_button)
        actions.addStretch(1)
        sl.addSpacing(4)
        sl.addLayout(actions)
        lay.addWidget(summary_card)

        files = QGroupBox("Files", w)
        fl = QVBoxLayout(files)
        fl.setSpacing(10)
        self.kinds_grid = QGridLayout()
        self.kinds_grid.setHorizontalSpacing(28)
        self.kinds_grid.setVerticalSpacing(6)
        self._kind_rows: list[list[QLabel]] = []
        self._kind_keys: list[str] = []
        self._kind_separator: QFrame | None = None
        heads = ["", "Indexed", "Waiting", "Failed"]
        self._kind_heads: list[QLabel] = []
        for col, text in enumerate(heads):
            label = QLabel(text, files)
            self._kind_heads.append(label)
            label.setFont(scaled_font(self, 1.0, bold=True))
            label.setAlignment(Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignRight)
            self.kinds_grid.addWidget(label, 0, col)
        self.kinds_grid.setColumnStretch(4, 1)
        fl.addLayout(self.kinds_grid)
        self.no_files = QLabel("No files found yet.", files)
        self.no_files.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        fl.addWidget(self.no_files)

        line = QHBoxLayout()
        line.setSpacing(8)
        self.failed_label = QLabel(files)
        self.show_failed_button = QPushButton("Show failed files…", files)
        self.show_failed_button.clicked.connect(self.show_failed)
        self.retry_button = QPushButton(theme_icon("view-refresh", widget=self), "Retry failed", files)
        self.retry_button.clicked.connect(self.retry_failed)
        line.addWidget(self.failed_label, 1)
        line.addWidget(self.show_failed_button)
        line.addWidget(self.retry_button)
        fl.addSpacing(4)
        fl.addLayout(line)
        lay.addWidget(files)

        self.last_scan = QLabel(w)
        self.last_scan.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        lay.addWidget(self.last_scan)
        lay.addStretch(1)
        return w

    def _build_not_running(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)
        lay.addStretch(1)
        self._not_running_icon = QLabel(w)
        size = self.style().pixelMetric(QStyle.PixelMetric.PM_LargeIconSize) * 2
        self._not_running_icon.setPixmap(theme_icon("dialog-warning", widget=self).pixmap(size, size))
        self._not_running_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("The Semma service is not running", w)
        title.setFont(scaled_font(self, 1.35, bold=True))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QLabel("New and changed files are not indexed and searches find nothing until it runs.", w)
        text.setWordWrap(True)
        text.setMaximumWidth(420)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.start_button = QPushButton(theme_icon("media-playback-start", widget=self), "Start service", w)
        self.start_button.clicked.connect(self.start_service)
        self.start_message = QLabel(w)
        self.start_message.setWordWrap(True)
        self.start_message.setMaximumWidth(460)
        self.start_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.start_button)
        row.addStretch(1)

        def centered_label(label: QLabel) -> QHBoxLayout:
            box = QHBoxLayout()
            box.addStretch(1)
            box.addWidget(label, 100)
            box.addStretch(1)
            return box

        lay.addWidget(self._not_running_icon)
        lay.addWidget(title)
        lay.addLayout(centered_label(text))
        lay.addSpacing(8)
        lay.addLayout(row)
        lay.addLayout(centered_label(self.start_message))
        lay.addStretch(2)
        return w

    def _apply_icons(self) -> None:
        self.rescan_button.setIcon(theme_icon("view-refresh", widget=self))
        self.retry_button.setIcon(theme_icon("view-refresh", widget=self))
        self.start_button.setIcon(theme_icon("media-playback-start", widget=self))
        paused = self.view.paused if self.view is not None else False
        self.pause_button.setIcon(
            theme_icon("media-playback-start" if paused else "media-playback-pause", widget=self)
        )
        size = self.style().pixelMetric(QStyle.PixelMetric.PM_LargeIconSize) * 2
        self._not_running_icon.setPixmap(theme_icon("dialog-warning", widget=self).pixmap(size, size))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    # -- polling ----------------------------------------------------------------------------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.timer.start()
        self.poll()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        self.timer.stop()

    def _should_poll(self) -> bool:
        win = self.window()
        return self.isVisible() and not win.isMinimized()

    def poll(self) -> None:
        """Ask for Status unless a call is still on its way (one may wait for its timeout)."""
        if self._in_flight or not self._should_poll():
            return
        self._in_flight = True
        self._relay.when_done(self.client.status(), self._got_status, self._status_failed)

    def _got_status(self, st: dict[str, Any] | None) -> None:
        self._in_flight = False
        if self._poll_error:
            self._poll_error = False
            self.error.clear()
        self.set_status(st)

    def _status_failed(self, exc: BaseException) -> None:
        self._in_flight = False
        self.error.show_message("error", f"Could not get the status: {describe_error(exc)}")
        self._poll_error = True  # cleared by the next answer

    # -- showing ------------------------------------------------------------------------------
    def set_status(self, st: dict[str, Any] | None, now: float | None = None) -> None:
        self.status = st
        if st is None:
            if self.running is not False:
                self.start_message.setText("")
            self.running = False
            self.pages.setCurrentIndex(2)
            return
        self.running = True
        self.start_button.setEnabled(True)
        self.pages.setCurrentIndex(1)
        view = self.view = model.status_view(st, self.home, time.time() if now is None else now)
        h = view.headline
        self.headline.setText(h.text)
        if h.progress is not None:
            done, total = h.progress
            # QProgressBar takes ints; byte counts of a model file can pass 2**31.
            scale = max(1, (max(total, 1) + 2**30) // 2**30)
            self.progress.setRange(0, max(total // scale, 1))
            self.progress.setValue(min(done, total) // scale)
            self.progress.show()
        elif h.busy:
            self.progress.setRange(0, 0)
            self.progress.show()
        else:
            self.progress.hide()
        self.detail.setText(h.detail)
        self.detail.setVisible(bool(h.detail))
        self.secondary.setText(h.secondary)
        self.secondary.setVisible(bool(h.secondary))

        self.pause_button.setText("Resume" if view.paused else "Pause")
        self.pause_button.setIcon(theme_icon("media-playback-start" if view.paused else "media-playback-pause",
                                             widget=self))
        self._set_kinds(view)
        if view.failed:
            self.failed_label.setText(f"{model.plural(view.failed, 'file')} could not be indexed.")
        else:
            self.failed_label.setText("No files failed.")
        self.show_failed_button.setVisible(bool(view.failed))
        self.retry_button.setVisible(bool(view.failed))
        self.last_scan.setText(f"Last scan: {view.last_scan[0].lower() + view.last_scan[1:]}"
                               if view.last_scan != "Not yet" else "Last scan: not yet")

    def _set_kinds(self, view: model.StatusView) -> None:
        keys = [r.kind for r in view.kinds] + (["total"] if len(view.kinds) > 1 else [])
        if keys != self._kind_keys:
            for labels in self._kind_rows:
                for label in labels:
                    self.kinds_grid.removeWidget(label)
                    label.deleteLater()
            if self._kind_separator is not None:
                self.kinds_grid.removeWidget(self._kind_separator)
                self._kind_separator.deleteLater()
                self._kind_separator = None
            self._kind_rows = []
            parent = self.no_files.parentWidget()
            grid_row = 1
            for key in keys:
                if key == "total":
                    sep = QFrame(parent)
                    sep.setFrameShape(QFrame.Shape.HLine)
                    sep.setFrameShadow(QFrame.Shadow.Sunken)
                    self.kinds_grid.addWidget(sep, grid_row, 0, 1, 4)
                    self._kind_separator = sep
                    grid_row += 1
                labels = [QLabel(parent) for _ in range(4)]
                for col, label in enumerate(labels):
                    label.setAlignment(Qt.AlignmentFlag.AlignLeft if col == 0 else Qt.AlignmentFlag.AlignRight)
                    if key == "total":
                        label.setFont(scaled_font(self, 1.0, bold=True))
                    self.kinds_grid.addWidget(label, grid_row, col)
                self._kind_rows.append(labels)
                grid_row += 1
            self._kind_keys = keys
        rows = [(r.name, r.indexed, r.pending, r.failed) for r in view.kinds]
        if len(view.kinds) > 1:
            rows.append(("Total", view.indexed, view.pending, view.failed))
        for labels, values in zip(self._kind_rows, rows, strict=True):
            labels[0].setText(values[0])
            for label, v in zip(labels[1:], values[1:], strict=True):
                label.setText(model.format_count(v))
        self.no_files.setVisible(not rows)
        for label in self._kind_heads:
            label.setVisible(bool(rows))

    # -- actions ----------------------------------------------------------------------------
    def _call(self, method: str, button: QPushButton | None = None) -> None:
        if button is not None:
            button.setEnabled(False)

        def done(_: Any) -> None:
            if button is not None:
                button.setEnabled(True)
            self.poll()

        def failed(exc: BaseException) -> None:
            if button is not None:
                button.setEnabled(True)
            self.error.show_message("error", f"{method} failed: {describe_error(exc)}")

        self._relay.when_done(self.client.call(method), done, failed)

    def toggle_pause(self) -> None:
        self._call("Resume" if self.view is not None and self.view.paused else "Pause", self.pause_button)

    def rescan(self) -> None:
        self._call("Rescan", self.rescan_button)

    def retry_failed(self) -> None:
        self._call("RetryFailed", self.retry_button)

    def show_failed(self) -> FailedDialog:
        dialog = FailedDialog(self.client, self.home, self)
        dialog.finished.connect(lambda _: self.poll())
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()
        return dialog

    def start_service(self) -> None:
        self.start_button.setEnabled(False)
        self.start_message.setText("Starting…")
        self._relay.when_done(self.client.systemctl("start"), self._started, self._start_failed)

    def _started(self, result: tuple[int, str]) -> None:
        code, output = result
        if code != 0:
            self._start_failed(RuntimeError(output or f"systemctl exited with status {code}"))
            return
        self.start_message.setText("Started. Waiting for it to answer…")
        self.poll()

    def _start_failed(self, exc: BaseException) -> None:
        self.start_button.setEnabled(True)
        self.start_message.setText(f"The service could not be started: {describe_error(exc)}")

