"""System tray icon (`semma tray`): a separate process that polls the daemon and shows its state."""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QObject, QProcess, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .app import make_app
from .client import DaemonClient, describe_error
from .qtasync import Relay
from .widgets import tinted_icon

log = logging.getLogger(__name__)

TRAY_BUS_NAME = "io.github.Semma.Tray"
BUSY_POLL_MS = 2_000
IDLE_POLL_MS = 5_000
TRAY_WAIT_INTERVAL_MS = 5_000
TRAY_WAIT_TIMEOUT_S = 120.0

ICONS_DIR = Path(__file__).parent / "icons"
TrayKind = Literal["offline", "setup", "idle", "busy", "paused"]

_KIND_TO_ICON: dict[str, str] = {
    "idle": "idle",
    "busy": "busy",
    "paused": "paused",
    "offline": "offline",
    "setup": "offline",
}


@dataclass(frozen=True)
class TrayView:
    """Display state derived from the daemon's `State()` / `Status()` dict."""

    kind: TrayKind
    tooltip: str
    status_line: str
    can_pause: bool
    can_resume: bool


def _int(state: dict[str, Any], key: str) -> int:
    try:
        v = float(state.get(key) or 0)
    except (TypeError, ValueError):
        return 0
    return int(v) if math.isfinite(v) else 0


def tray_view(state: dict[str, Any] | None) -> TrayView:
    """Compute the tray's icon kind, tooltip, status line and action state from `state`."""
    if state is None:
        return TrayView(
            kind="offline",
            tooltip="Semma service is not running",
            status_line="Semma service is not running",
            can_pause=False,
            can_resume=False,
        )

    if state.get("setup_pending"):
        return TrayView(
            kind="setup",
            tooltip="Semma — Setup not finished",
            status_line="Setup not finished",
            can_pause=False,
            can_resume=False,
        )

    st = str(state.get("state") or "")
    paused = bool(state.get("paused")) or st == "paused"

    if st == "downloading":
        dl_file = str(state.get("download_file") or "")
        status_line = "Downloading model…"
        tooltip = f"Semma — Downloading model\n{dl_file}" if dl_file else "Semma — Downloading model"
        return TrayView(
            kind="busy",
            tooltip=tooltip,
            status_line=status_line,
            can_pause=not paused,
            can_resume=paused,
        )

    if st == "scanning":
        return TrayView(
            kind="busy",
            tooltip="Semma — Scanning folders",
            status_line="Scanning folders…",
            can_pause=not paused,
            can_resume=paused,
        )

    if st == "loading":
        return TrayView(
            kind="busy",
            tooltip="Semma — Loading model",
            status_line="Loading model…",
            can_pause=not paused,
            can_resume=paused,
        )

    if paused:
        reason = str(state.get("pause_reason") or "").strip()
        status_line = f"Paused: {reason}" if reason else "Paused"
        return TrayView(
            kind="paused",
            tooltip=f"Semma — {status_line}",
            status_line=status_line,
            can_pause=False,
            can_resume=True,
        )

    if st == "indexing":
        done = _int(state, "run_done")
        total = _int(state, "run_total")
        status_line = f"Indexing {done:,} / {total:,} files" if total > 0 else "Indexing files…"
        current = str(state.get("current_file") or "")
        basename = Path(current).name if current else ""
        tooltip = f"Semma — {status_line}\n{basename}" if basename else f"Semma — {status_line}"
        return TrayView(
            kind="busy",
            tooltip=tooltip,
            status_line=status_line,
            can_pause=True,
            can_resume=False,
        )

    dl_error = str(state.get("download_error") or "").strip()
    if dl_error:
        status_line = f"Model download failed: {dl_error}"
        return TrayView(
            kind="idle",
            tooltip=f"Semma — {status_line}",
            status_line=status_line,
            can_pause=True,
            can_resume=False,
        )

    return TrayView(
        kind="idle",
        tooltip="Semma — Idle",
        status_line="Idle",
        can_pause=True,
        can_resume=False,
    )


def icon_kind_for(view_kind: str) -> str:
    """Map a `TrayView.kind` to one of the four SVG icon names."""
    return _KIND_TO_ICON.get(view_kind, "offline")


def icon_path(icon_kind: str) -> Path:
    """Path to the SVG icon for `icon_kind` (idle, busy, paused, offline)."""
    return ICONS_DIR / f"semma-tray-{icon_kind}.svg"


def load_tray_icon(icon_kind: str) -> QIcon:
    """Load the monochrome SVG for `icon_kind` and tint it for the active desktop palette."""
    raw = QIcon(str(icon_path(icon_kind)))
    if raw.isNull():
        return raw
    tinted = tinted_icon(raw, monochrome=True)
    return tinted if not tinted.isNull() else raw


class Tray(QObject):
    """Owns the `QSystemTrayIcon`, its context menu, and non-blocking daemon polling."""

    def __init__(self, client: Any, config_path: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.client = client
        self.config_path = config_path
        self._relay = Relay(self)
        self._in_flight = False
        self._icons: dict[str, QIcon] = {}
        self.icon_kind: str = ""
        self.view: TrayView = tray_view(None)
        self.last_error: str = ""

        self.tray = QSystemTrayIcon(self)
        self.menu = QMenu()
        self.status_action = self.menu.addAction("")
        self.status_action.setEnabled(False)
        self.menu.addSeparator()
        self.open_action = self.menu.addAction("Open Semma")
        self.open_action.triggered.connect(self.open_semma)
        self.pause_action = self.menu.addAction("Pause indexing")
        self.pause_action.triggered.connect(self.toggle_pause)
        self.rescan_action = self.menu.addAction("Rescan now")
        self.rescan_action.triggered.connect(self.rescan)
        self.menu.addSeparator()
        self.quit_action = self.menu.addAction("Quit")
        self.quit_action.triggered.connect(self.quit)

        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)

        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.paletteChanged.connect(self._refresh_icons)

        self.timer = QTimer(self)
        self.timer.setInterval(IDLE_POLL_MS)
        self.timer.timeout.connect(self.poll)

        self.apply_view(self.view)

    def _icon(self, icon_kind: str) -> QIcon:
        icon = self._icons.get(icon_kind)
        if icon is None:
            icon = load_tray_icon(icon_kind)
            self._icons[icon_kind] = icon
        return icon

    def _refresh_icons(self) -> None:
        self._icons.clear()
        if self.icon_kind:
            self.tray.setIcon(self._icon(self.icon_kind))

    def _set_poll_interval(self, interval_ms: int) -> None:
        if self.timer.interval() != interval_ms:
            self.timer.setInterval(interval_ms)

    def start(self) -> None:
        """Show the tray icon and start polling the daemon."""
        self.tray.show()
        self.timer.start()
        self.poll()

    def stop(self) -> None:
        """Stop polling and hide the tray icon."""
        self.timer.stop()
        self.tray.hide()

    def poll(self) -> None:
        """Poll `client.state()` unless a request is already in flight."""
        if self._in_flight:
            return
        self._in_flight = True
        self._relay.when_done(self.client.state(), self._got_state, self._state_failed)

    def _got_state(self, state: dict[str, Any] | None) -> None:
        self._in_flight = False
        self.last_error = ""
        self.set_state(state)

    def _state_failed(self, exc: BaseException) -> None:
        self._in_flight = False
        msg = describe_error(exc)
        self.last_error = msg
        self.status_action.setText(msg)
        self.tray.setToolTip(f"Semma — {msg}")
        self._set_poll_interval(IDLE_POLL_MS)

    def set_state(self, state: dict[str, Any] | None) -> TrayView:
        """Apply a daemon state dict and return the resulting `TrayView`."""
        view = tray_view(state)
        self.apply_view(view)
        return view

    def apply_view(self, view: TrayView) -> None:
        """Update the icon, tooltip, menu actions, and poll interval from `view`."""
        self.view = view
        self.status_action.setText(view.status_line)
        self.tray.setToolTip(view.tooltip)

        next_icon_kind = icon_kind_for(view.kind)
        if next_icon_kind != self.icon_kind:
            self.icon_kind = next_icon_kind
            self.tray.setIcon(self._icon(next_icon_kind))

        if view.can_resume:
            self.pause_action.setText("Resume indexing")
            self.pause_action.setEnabled(True)
        elif view.can_pause:
            self.pause_action.setText("Pause indexing")
            self.pause_action.setEnabled(True)
        else:
            self.pause_action.setText("Pause indexing")
            self.pause_action.setEnabled(False)

        self.rescan_action.setEnabled(view.kind not in ("offline", "setup"))
        self._set_poll_interval(BUSY_POLL_MS if view.kind == "busy" else IDLE_POLL_MS)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_semma()

    def gui_command(self) -> tuple[str, list[str]]:
        """Command and arguments used to launch or raise the Semma window."""
        args = ["-I", "-m", "semma"] if sys.flags.isolated else ["-m", "semma"]  # the package runs -I
        if self.config_path is not None:
            args.extend(["--config", str(self.config_path)])
        args.append("gui")
        return sys.executable, args

    def open_semma(self) -> bool:
        """Start `semma gui` detached (its single-instance logic raises an existing window)."""
        program, args = self.gui_command()
        result = QProcess.startDetached(program, args)
        return result[0] if isinstance(result, tuple) else bool(result)

    def toggle_pause(self) -> None:
        """Pause or resume indexing according to the current view."""
        if self.view.can_resume:
            self._call("Resume")
        elif self.view.can_pause:
            self._call("Pause")

    def rescan(self) -> None:
        """Ask the daemon to start a rescan now."""
        self._call("Rescan")

    def _call(self, method: str) -> None:
        self._relay.when_done(self.client.call(method), self._call_done, self._call_failed)

    def _call_done(self, _: Any) -> None:
        self.poll()

    def _call_failed(self, exc: BaseException) -> None:
        msg = describe_error(exc)
        self.last_error = msg
        self.status_action.setText(msg)
        self.tray.showMessage("Semma", msg, QSystemTrayIcon.MessageIcon.Warning)

    def quit(self) -> None:
        """Hide the icon and exit the tray process."""
        self.stop()
        app = QApplication.instance()
        if app is not None:
            app.quit()


def _wait_for_tray(app: QApplication, tray: Tray, *,
                   interval_ms: int = TRAY_WAIT_INTERVAL_MS,
                   timeout_s: float = TRAY_WAIT_TIMEOUT_S) -> QTimer | None:
    """Start `tray` once a system tray host is available, or quit after `timeout_s`."""
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray.start()
        return None

    deadline = time.monotonic() + timeout_s
    wait_timer = QTimer(tray)
    wait_timer.setInterval(interval_ms)

    def check() -> None:
        if QSystemTrayIcon.isSystemTrayAvailable():
            wait_timer.stop()
            tray.start()
            return
        if time.monotonic() >= deadline:
            wait_timer.stop()
            log.warning(
                "System tray is not available after %d seconds; exiting. "
                "On GNOME, install and enable the AppIndicator and KStatusNotifierItem Support extension.",
                int(timeout_s),
            )
            app.quit()

    wait_timer.timeout.connect(check)
    wait_timer.start()
    return wait_timer


def run(config_path: Path | None = None) -> int:
    """Run the system tray process until Quit or no tray host appears."""
    app = make_app()
    app.setQuitOnLastWindowClosed(False)
    client = DaemonClient()

    from .single import claim

    try:
        owned = client.submit(claim(lambda _token: None, bus_name=TRAY_BUS_NAME)).result(timeout=5)
    except Exception:
        owned = None
    if owned is False:
        client.close()
        return 0

    tray = Tray(client, config_path)
    _wait_timer = _wait_for_tray(app, tray)
    try:
        return app.exec()
    finally:
        if _wait_timer is not None:
            _wait_timer.stop()
        tray.stop()
        if owned is not None and hasattr(owned, "disconnect"):
            try:
                owned.disconnect()
            except Exception:
                pass
        client.close()
