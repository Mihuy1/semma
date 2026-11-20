"""Tests for the system tray icon (`semma tray`) and its pure view logic."""

from __future__ import annotations

import os
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from semma.cli import build_parser, cmd_tray  # noqa: E402
from semma.daemon import DaemonNotRunning  # noqa: E402
from semma.gui.app import make_app  # noqa: E402
from semma.gui.tray import (  # noqa: E402
    BUSY_POLL_MS,
    IDLE_POLL_MS,
    Tray,
    icon_path,
    load_tray_icon,
    tray_view,
)


def _done(value: Any = None, exc: BaseException | None = None) -> Future[Any]:
    f: Future[Any] = Future()
    if exc is not None:
        f.set_exception(exc)
    else:
        f.set_result(value)
    return f


class FakeTrayClient:
    """Minimal stand-in for `DaemonClient` implementing `.state()` and `.call()`."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state_value = state
        self.state_error: BaseException | None = None
        self.call_error: BaseException | None = None
        self.pending_state_future: Future[Any] | None = None
        self.state_calls: int = 0
        self.calls: list[str] = []

    def state(self) -> Future[Any]:
        self.state_calls += 1
        if self.pending_state_future is not None:
            return self.pending_state_future
        if self.state_error is not None:
            return _done(exc=self.state_error)
        return _done(self.state_value)

    def call(self, method: str) -> Future[Any]:
        self.calls.append(method)
        if self.call_error is not None:
            return _done(exc=self.call_error)
        return _done(None)


@pytest.fixture(scope="module")
def app() -> QtWidgets.QApplication:
    return make_app([])


@pytest.fixture
def make_tray(app: QtWidgets.QApplication):
    created: list[Tray] = []

    def _build(client: FakeTrayClient, config_path: Path | None = None) -> Tray:
        try:
            tray = Tray(client, config_path)
        except Exception as exc:
            pytest.skip(f"QSystemTrayIcon cannot be constructed offscreen: {exc}")
        created.append(tray)
        return tray

    yield _build
    for tray in created:
        tray.stop()


def test_tray_view_offline() -> None:
    v = tray_view(None)
    assert v.kind == "offline"
    assert v.status_line == "Semma service is not running"
    assert "Semma service is not running" in v.tooltip
    assert not v.can_pause and not v.can_resume


def test_tray_view_setup_pending() -> None:
    v = tray_view({"state": "idle", "setup_pending": True, "paused": False})
    assert v.kind == "setup"
    assert v.status_line == "Setup not finished"
    assert "Setup not finished" in v.tooltip
    assert not v.can_pause and not v.can_resume


def test_tray_view_idle_and_download_error() -> None:
    v = tray_view({"state": "idle", "paused": False, "setup_pending": False})
    assert v.kind == "idle"
    assert v.status_line == "Idle"
    assert v.can_pause and not v.can_resume

    err = tray_view({"state": "idle", "paused": False, "download_error": "connection timed out"})
    assert err.kind == "idle"
    assert "connection timed out" in err.status_line
    assert "connection timed out" in err.tooltip
    assert err.can_pause and not err.can_resume


def test_tray_view_busy_indexing() -> None:
    v = tray_view({
        "state": "indexing",
        "paused": False,
        "run_done": 120,
        "run_total": 4000,
        "run_failed": 0,
        "run_rate": 12.5,
        "current_file": "/home/alice/Documents/quarterly-report.pdf",
    })
    assert v.kind == "busy"
    assert v.status_line == "Indexing 120 / 4,000 files"
    assert "Indexing 120 / 4,000 files" in v.tooltip
    assert "quarterly-report.pdf" in v.tooltip
    assert "/home/alice/Documents" not in v.tooltip
    assert v.can_pause and not v.can_resume


def test_tray_view_busy_other_states() -> None:
    assert tray_view({"state": "indexing", "run_done": 0, "run_total": 0}).kind == "busy"
    assert tray_view({"state": "scanning"}).kind == "busy"
    assert tray_view({"state": "loading"}).kind == "busy"
    dl = tray_view({"state": "downloading", "download_file": "model.onnx"})
    assert dl.kind == "busy"
    assert "model.onnx" in dl.tooltip
    assert dl.can_pause and not dl.can_resume


def test_tray_view_paused() -> None:
    v = tray_view({"state": "paused", "paused": True, "pause_reason": "on battery"})
    assert v.kind == "paused"
    assert "on battery" in v.status_line
    assert "on battery" in v.tooltip
    assert not v.can_pause and v.can_resume

    plain = tray_view({"state": "paused", "paused": True, "pause_reason": ""})
    assert plain.kind == "paused"
    assert plain.status_line == "Paused"
    assert not plain.can_pause and plain.can_resume


@pytest.mark.parametrize("kind", ["idle", "busy", "paused", "offline"])
def test_tray_svg_icons_exist_and_load(app: QtWidgets.QApplication, kind: str) -> None:
    path = icon_path(kind)
    assert path.is_file()
    assert "currentColor" in path.read_text(encoding="utf-8")
    icon = load_tray_icon(kind)
    assert not icon.isNull()


def test_tray_updates_menu_icon_and_poll_interval(app: QtWidgets.QApplication, make_tray) -> None:
    client = FakeTrayClient(None)
    tray = make_tray(client)

    actions = tray.menu.actions()
    assert actions[0] is tray.status_action and not tray.status_action.isEnabled()
    assert actions[1].isSeparator()
    assert actions[2] is tray.open_action and tray.open_action.text() == "Open Semma"
    assert actions[3] is tray.pause_action
    assert actions[4] is tray.rescan_action and tray.rescan_action.text() == "Rescan now"
    assert actions[5].isSeparator()
    assert actions[6] is tray.quit_action and tray.quit_action.text() == "Quit"

    assert tray.icon_kind == "offline"
    assert not tray.pause_action.isEnabled()
    assert not tray.rescan_action.isEnabled()
    assert tray.timer.interval() == IDLE_POLL_MS

    tray.set_state({"state": "idle", "setup_pending": True})
    assert tray.view.kind == "setup"
    assert tray.icon_kind == "offline"
    assert tray.status_action.text() == "Setup not finished"
    assert not tray.pause_action.isEnabled()
    assert not tray.rescan_action.isEnabled()

    tray.set_state({
        "state": "indexing",
        "run_done": 120,
        "run_total": 4000,
        "current_file": "/tmp/notes.md",
    })
    assert tray.icon_kind == "busy"
    assert tray.status_action.text() == "Indexing 120 / 4,000 files"
    assert tray.pause_action.text() == "Pause indexing"
    assert tray.pause_action.isEnabled()
    assert tray.rescan_action.isEnabled()
    assert tray.timer.interval() == BUSY_POLL_MS

    tray.set_state({"state": "paused", "paused": True, "pause_reason": "paused by request"})
    assert tray.icon_kind == "paused"
    assert tray.pause_action.text() == "Resume indexing"
    assert tray.pause_action.isEnabled()
    assert tray.rescan_action.isEnabled()
    assert tray.timer.interval() == IDLE_POLL_MS

    tray.set_state({"state": "idle", "paused": False})
    assert tray.icon_kind == "idle"
    assert tray.pause_action.text() == "Pause indexing"
    assert tray.pause_action.isEnabled()
    assert tray.timer.interval() == IDLE_POLL_MS


def test_tray_actions_call_daemon_and_repoll(app: QtWidgets.QApplication, make_tray) -> None:
    client = FakeTrayClient({"state": "idle", "paused": False})
    tray = make_tray(client)
    tray.poll()
    app.processEvents()
    assert client.state_calls == 1

    tray.toggle_pause()
    app.processEvents()
    assert client.calls[-1] == "Pause"
    assert client.state_calls == 2

    tray.set_state({"state": "paused", "paused": True, "pause_reason": "paused by request"})
    tray.toggle_pause()
    app.processEvents()
    assert client.calls[-1] == "Resume"
    assert client.state_calls == 3

    tray.rescan()
    app.processEvents()
    assert client.calls[-1] == "Rescan"
    assert client.state_calls == 4


def test_tray_call_error_is_reported(app: QtWidgets.QApplication, make_tray, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeTrayClient({"state": "idle", "paused": False})
    tray = make_tray(client)
    tray.set_state({"state": "idle", "paused": False})

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(tray.tray, "showMessage", lambda title, msg, *_a: shown.append((title, msg)))

    client.call_error = DaemonNotRunning("offline")
    tray.rescan()
    app.processEvents()
    assert "not running" in tray.last_error.lower()
    assert tray.status_action.text() == tray.last_error
    assert shown and shown[0][1] == tray.last_error


def test_tray_does_not_stack_polls(app: QtWidgets.QApplication, make_tray) -> None:
    pending: Future[Any] = Future()
    client = FakeTrayClient()
    client.pending_state_future = pending
    tray = make_tray(client)

    tray.poll()
    tray.poll()
    tray.poll()
    assert client.state_calls == 1

    client.pending_state_future = None
    client.state_value = {"state": "idle", "paused": False}
    pending.set_result({"state": "scanning"})
    app.processEvents()
    assert tray.view.kind == "busy"

    tray.poll()
    app.processEvents()
    assert client.state_calls == 2
    assert tray.view.kind == "idle"


def test_tray_open_semma_and_trigger(app: QtWidgets.QApplication, make_tray, monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[tuple[str, list[str]]] = []

    def fake_start_detached(program: str, args: list[str]) -> tuple[bool, int]:
        launched.append((program, list(args)))
        return True, 12345

    monkeypatch.setattr("semma.gui.tray.QProcess.startDetached", staticmethod(fake_start_detached))

    cfg = Path("/tmp/custom-semma.toml")
    tray = make_tray(FakeTrayClient(), config_path=cfg)
    assert tray.open_semma() is True
    assert launched[-1] == (sys.executable, ["-m", "semma", "--config", str(cfg), "gui"])

    tray_default = make_tray(FakeTrayClient())
    tray_default._on_activated(QtWidgets.QSystemTrayIcon.ActivationReason.Trigger)
    assert launched[-1] == (sys.executable, ["-m", "semma", "gui"])


def test_cli_parses_tray_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["tray"])
    assert args.func is cmd_tray
    assert args.config is None

    args_cfg = parser.parse_args(["--config", "/tmp/semma.toml", "tray"])
    assert args_cfg.func is cmd_tray
    assert args_cfg.config == Path("/tmp/semma.toml")
