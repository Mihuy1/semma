"""The window's widgets, offscreen, with a fake daemon client. Skipped without PySide6."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from gui_fakes import FakeClient, status  # noqa: E402

from semma.config_write import read_raw  # noqa: E402
from semma.gui.app import MainWindow, make_app  # noqa: E402
from semma.gui.settings.page import RESTART_NOTICE, SettingsPage  # noqa: E402
from semma.gui.status_page import StatusPage  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QtWidgets.QApplication:
    return make_app([])


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    for d in ("Documents", "Pictures", ".config/semma"):
        (h / d).mkdir(parents=True)
    return h


def config_path(home: Path) -> Path:
    return home / ".config" / "semma" / "config.toml"


def test_window_builds_and_shows_status(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status(state="indexing", indexed=10, pending=30, run_rate=2.0,
                               current_file=str(home / "Documents" / "a.md"),
                               indexed_by_kind={"text": 10}, pending_by_kind={"pdf": 30}))
    w = MainWindow(client, config_path(home), home)
    w.show()
    app.processEvents()
    page = w.status_page
    assert "Status" in client.calls  # polled when shown
    assert page.running is True
    assert page.headline.text() == "Indexing: 10 of 40 files"
    assert page.detail.fullText() == "~/Documents/a.md"
    assert page.progress.maximum() == 40 and page.progress.value() == 10
    assert page.pause_button.text() == "Pause"
    assert [r[0].text() for r in page._kind_rows] == ["Text", "PDF", "Total"]
    assert [r[1].text() for r in page._kind_rows] == ["10", "0", "10"]
    assert w.tabs.count() == 2
    w.close()


def test_status_updates_paused_and_failed(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status())
    page = StatusPage(client, home)
    page.set_status(status(state="paused", paused=True, pause_reason="paused by request", indexed=5, pending=5,
                           failed=2, failed_by_kind={"pdf": 2}, last_scan_at=1000.0), now=1000.0 + 120)
    assert page.headline.text() == "Paused by you"
    assert page.pause_button.text() == "Resume"
    assert page.failed_label.text() == "2 files could not be indexed."
    assert not page.show_failed_button.isHidden() and not page.retry_button.isHidden()
    assert page.last_scan.text() == "Last scan: 2 minutes ago"
    page.toggle_pause()
    assert client.calls[-1] == "Resume"
    page.set_status(status(indexed=10))
    assert page.headline.text() == "Up to date: 10 files indexed"
    assert page.pause_button.text() == "Pause" and page.progress.isHidden()
    assert page.show_failed_button.isHidden()


def test_status_of_an_older_daemon(app: QtWidgets.QApplication, home: Path) -> None:
    page = StatusPage(FakeClient(), home)
    page.set_status({"state": "idle", "paused": False, "indexed": 3, "pending": 0, "failed": 0})
    assert page.headline.text() == "Up to date: 3 files indexed"
    assert [r[0].text() for r in page._kind_rows] == ["All files"] and page.no_files.isHidden()


def test_failed_dialog_lists_and_retries(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status(failed=1))
    client.failed_rows = [(str(home / "Documents" / "bad.pdf"), "corrupt file\nmore detail")]
    page = StatusPage(client, home)
    dialog = page.show_failed()
    app.processEvents()
    assert dialog.tree.topLevelItemCount() == 1
    item = dialog.tree.topLevelItem(0)
    assert item.text(0) == "~/Documents/bad.pdf" and item.text(1) == "corrupt file"
    dialog.retry()
    assert "RetryFailed" in client.calls
    assert "1 file queued" in dialog.message.text()
    dialog.close()
    app.processEvents()


def test_daemon_not_running_and_start_service(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(None)
    page = StatusPage(client, home)
    page.poll = lambda: None  # type: ignore[method-assign]  # not visible: driven by hand
    page.set_status(None)
    assert page.running is False and page.pages.currentIndex() == 2
    client.systemctl_result = (1, "Unit semma.service not found.")
    page.start_service()
    assert client.calls[-1] == "systemctl start"
    assert "Unit semma.service not found." in page.start_message.text()
    assert page.start_button.isEnabled()
    # the daemon appears: the page recovers on the next answer
    page.set_status(status(indexed=1))
    assert page.running is True and page.pages.currentIndex() == 1


def test_status_errors_are_shown_inline(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status())
    page = StatusPage(client, home)
    page._should_poll = lambda: True  # type: ignore[method-assign]
    client.status_error = TimeoutError()
    page.poll()
    assert "did not answer in time" in page.error.text()
    client.status_error = None
    page.poll()
    assert page.error.text() == ""


def make_settings(home: Path, client: FakeClient, text: str | None = None) -> SettingsPage:
    if text is not None:
        config_path(home).write_text(text)
    return SettingsPage(client, config_path(home), home)


def test_settings_apply_writes_config_and_reloads(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status())
    page = make_settings(home, client, '# keep me\n[scan]\nroots = ["~/Documents"]\n')
    assert [page.nav.item(i).text() for i in range(page.nav.count())] == [
        "Folders", "File types", "Priorities", "Media", "Performance"]
    folders = page.select("folders")
    assert not page.apply_button.isEnabled() and not page.revert_button.isEnabled()
    folders.roots.add(str(home / "Pictures"))
    folders.roots.add(str(home / "Pictures"))  # twice: refused
    assert "already" in folders.message.text()
    folders.skip_hidden.setChecked(False)
    assert page.apply_button.isEnabled() and page.revert_button.isEnabled()
    page.apply()
    assert read_raw(config_path(home)) == {"scan": {"roots": ["~/Documents", "~/Pictures"], "skip_hidden": False}}
    assert config_path(home).read_text().startswith("# keep me")
    assert client.calls[-1] == "ReloadConfig"
    assert page.notice.kind == "info" and not page.apply_button.isEnabled()


def test_settings_apply_with_an_older_daemon_asks_for_a_restart(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status())
    client.reload_result = None
    page = make_settings(home, client)
    page.select("folders").include_mounts.setChecked(False)
    page.apply()
    assert read_raw(config_path(home)) == {"scan": {"include_mounts": False}}
    assert page.notice.text() == RESTART_NOTICE
    assert page.notice.action_button().text() == "Restart service"
    page.notice.action_button().click()
    assert client.calls[-1] == "systemctl restart"
    assert "restarted" in page.notice.text()


def test_settings_apply_restart_names(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(status())
    client.reload_result = ["scan.roots"]
    page = make_settings(home, client)
    page.select("folders").skip_hidden.setChecked(False)
    page.apply()
    assert page.notice.text().startswith(RESTART_NOTICE) and "scan.roots" in page.notice.text()


def test_settings_apply_without_daemon(app: QtWidgets.QApplication, home: Path) -> None:
    client = FakeClient(None)
    page = make_settings(home, client)
    page.select("folders").skip_hidden.setChecked(False)
    page.apply()
    assert read_raw(config_path(home)) == {"scan": {"skip_hidden": False}}
    assert page.notice.kind == "info" and page.notice.action_button().isHidden()


def test_settings_revert(app: QtWidgets.QApplication, home: Path) -> None:
    page = make_settings(home, FakeClient(status()))
    folders = page.select("folders")
    folders.excludes.add(str(home / "Documents"))
    assert page.apply_button.isEnabled()
    page.revert()
    assert folders.excludes.folders() == [] and not page.apply_button.isEnabled()
    assert not config_path(home).exists()


def test_settings_invalid_config_is_shown_and_nothing_written(app: QtWidgets.QApplication, home: Path) -> None:
    bad = "[scan\nroots = 1\n"
    page = make_settings(home, FakeClient(status()), bad)
    assert page.ctx is None and page.error.kind == "error" and "cannot be read" in page.error.text()
    assert not page.apply_button.isEnabled()
    page.apply()
    assert config_path(home).read_text() == bad


def test_settings_write_error_is_shown(app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page = make_settings(home, FakeClient(status()), "[scan]\nskip_hidden = true\n")
    page.select("folders").skip_hidden.setChecked(False)
    # the file is broken by hand after the page loaded it
    config_path(home).write_text("[scan\n")
    page.apply()
    assert "were not saved" in page.error.text()
    assert config_path(home).read_text() == "[scan\n"
    assert "ReloadConfig" not in page.client.calls


def test_single_instance_activates_the_first_window(app: QtWidgets.QApplication) -> None:
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        pytest.skip("no session bus")
    from semma.gui.single import claim

    name = f"io.github.Semma.WindowTest{os.getpid()}"
    tokens: list[str] = []

    async def main() -> None:
        first = await claim(tokens.append, bus_name=name)
        assert first not in (None, False)
        try:
            second = await claim(lambda t: None, bus_name=name)
            assert second is False
        finally:
            first.disconnect()

    asyncio.run(main())
    assert tokens == [os.environ.get("XDG_ACTIVATION_TOKEN", "")]


def test_close_prompts_on_unapplied_changes_and_handles_cancel_discard_apply(
    app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import Future

    QMessageBox = QtWidgets.QMessageBox
    prompts: list[tuple[str, str, QMessageBox.StandardButton]] = []

    def fake_exec(reply: QMessageBox.StandardButton):
        def run(self: QtWidgets.QMessageBox) -> int:
            prompts.append((self.text(), self.informativeText(), self.standardButtons()))
            return int(reply)
        return run

    client = FakeClient(status())
    w = MainWindow(client, config_path(home), home)
    w.show()
    app.processEvents()

    # No changes: closes without prompting.
    monkeypatch.setattr(QMessageBox, "exec", fake_exec(QMessageBox.StandardButton.Cancel))
    assert w.close() is True
    assert prompts == []

    # Reopen, edit Media while currently viewing Status, then try to close -> Cancel.
    w.show()
    media = w.settings_page.select("media")
    media.max_video_gb.setValue(10.0)
    w.settings_page.select("folders")  # view a different section
    w.tabs.setCurrentWidget(w.status_page)  # and switch back to Status

    monkeypatch.setattr(QMessageBox, "exec", fake_exec(QMessageBox.StandardButton.Cancel))
    assert w.close() is False
    assert len(prompts) == 1
    assert "unapplied" in prompts[-1][0].lower()
    assert "Media" in prompts[-1][1]
    assert prompts[-1][2] == (
        QMessageBox.StandardButton.Apply
        | QMessageBox.StandardButton.Discard
        | QMessageBox.StandardButton.Cancel
    )
    assert w.isVisible()
    assert w.tabs.currentWidget() is w.settings_page
    assert w.settings_page.stack.currentIndex() == w.settings_page.sections.index(media)
    assert not config_path(home).exists()

    # Discard: closes without saving or reloading.
    monkeypatch.setattr(QMessageBox, "exec", fake_exec(QMessageBox.StandardButton.Discard))
    assert w.close() is True
    assert not w.isVisible()
    assert not config_path(home).exists()
    assert "ReloadConfig" not in client.calls

    # Reopen (reloads clean defaults), edit Media, Apply -> saves, reloads asynchronously, then closes.
    w.settings_page.reload()
    w.show()
    w.settings_page.select("media").max_video_gb.setValue(12.0)
    pending_reload: Future[list[str] | None] = Future()
    monkeypatch.setattr(client, "reload_config", lambda: pending_reload)
    monkeypatch.setattr(QMessageBox, "exec", fake_exec(QMessageBox.StandardButton.Apply))
    assert w.close() is False  # waiting for ReloadConfig
    assert w.isVisible()
    assert read_raw(config_path(home)) == {"scan": {"max_file_mb": {"video": 12288}}}

    pending_reload.set_result([])
    app.processEvents()
    assert not w.isVisible()


def test_close_apply_keeps_window_open_when_restart_needed(
    app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    QMessageBox = QtWidgets.QMessageBox
    client = FakeClient(status())
    client.reload_result = ["scan.roots"]
    w = MainWindow(client, config_path(home), home)
    w.show()
    w.settings_page.select("folders").skip_hidden.setChecked(False)
    w.tabs.setCurrentWidget(w.status_page)

    monkeypatch.setattr(QMessageBox, "exec", lambda self: int(QMessageBox.StandardButton.Apply))
    assert w.close() is False
    assert w.isVisible()
    assert w.tabs.currentWidget() is w.settings_page
    assert RESTART_NOTICE in w.settings_page.notice.text()
    assert read_raw(config_path(home)) == {"scan": {"skip_hidden": False}}

    # No unapplied changes left -> closing again succeeds immediately.
    assert w.close() is True


def test_close_with_blocking_problems_hides_apply_button(
    app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    QMessageBox = QtWidgets.QMessageBox
    seen_buttons: list[QMessageBox.StandardButton] = []

    def fake_exec(self: QtWidgets.QMessageBox) -> int:
        seen_buttons.append(self.standardButtons())
        return int(QMessageBox.StandardButton.Discard)

    w = MainWindow(FakeClient(status()), config_path(home), home)
    w.show()
    ft_sec = w.settings_page.select("filetypes")
    for cb in ft_sec.kind_checks.values():
        cb.setChecked(False)
    assert w.settings_page.is_modified() and not w.settings_page.can_apply()

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    assert w.close() is True
    assert seen_buttons == [QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel]


def test_close_apply_write_error_keeps_window_open(
    app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    QMessageBox = QtWidgets.QMessageBox
    config_path(home).write_text("[scan]\nskip_hidden = true\n")
    w = MainWindow(FakeClient(status()), config_path(home), home)
    w.show()
    w.settings_page.select("folders").skip_hidden.setChecked(False)
    w.tabs.setCurrentWidget(w.status_page)
    config_path(home).write_text("[scan\n")  # broken on disk before Apply

    monkeypatch.setattr(QMessageBox, "exec", lambda self: int(QMessageBox.StandardButton.Apply))
    assert w.close() is False
    assert w.isVisible()
    assert w.tabs.currentWidget() is w.settings_page
    assert "were not saved" in w.settings_page.error.text()
    assert w.settings_page.is_modified()
    monkeypatch.setattr(QMessageBox, "exec", lambda self: int(QMessageBox.StandardButton.Discard))
    assert w.close() is True
