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
        "Folders", "File types", "Priorities", "Media", "Screenshots", "Performance"]
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


def _icon_avg_channel(icon: object, mode: object) -> tuple[float, float]:
    """Return (weighted average luminance, average alpha) over visible pixels of a 22x22 icon."""
    from PySide6.QtGui import QIcon, QImage

    assert isinstance(icon, QIcon)
    pm = icon.pixmap(22, 22, mode)  # type: ignore[arg-type]
    assert not pm.isNull()
    img = pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    lum_sum = 0.0
    alpha_sum = 0
    count = 0
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.alpha() > 16:
                lum_sum += (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) * (c.alpha() / 255.0)
                alpha_sum += c.alpha()
                count += 1
    assert count > 0
    return lum_sum / count, alpha_sum / count


def test_disabled_button_icons_are_muted_not_bright_white(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    from PySide6.QtGui import QIcon

    from semma.gui import theme
    from semma.gui.onboarding import OnboardingWindow

    theme.ensure_icon_theme_paths()
    old_pal = app.palette()
    old_theme = QIcon.themeName()
    try:
        app.setPalette(theme.build_gnome_palette(dark=True))
        if Path("/usr/share/icons/breeze-dark/index.theme").is_file():
            QIcon.setThemeName("breeze-dark")
        page = make_settings(home, FakeClient(status()))
        folders = page.select("folders")
        priorities = page.select("priorities")
        wiz = OnboardingWindow(FakeClient(status()), config_path(home), home)

        buttons = [
            page.revert_button,
            page.apply_button,
            folders.roots.remove_button,
            priorities.remove_button,
            priorities.reset_button,
            wiz.back_button,
        ]
        for btn in buttons:
            assert not btn.isEnabled()
            icon = btn.icon()
            assert not icon.isNull()
            normal_lum, normal_alpha = _icon_avg_channel(icon, QIcon.Mode.Normal)
            disabled_lum, disabled_alpha = _icon_avg_channel(icon, QIcon.Mode.Disabled)
            # Either luminance is noticeably darker on dark mode or alpha is significantly reduced.
            assert disabled_lum < normal_lum - 30 or disabled_alpha < normal_alpha * 0.65, (
                f"{btn.text()} disabled icon not muted: normal=({normal_lum:.1f}, {normal_alpha:.1f}) "
                f"disabled=({disabled_lum:.1f}, {disabled_alpha:.1f})"
            )
    finally:
        app.setPalette(old_pal)
        QIcon.setThemeName(old_theme)


def test_gnome_palette_has_distinct_surfaces_and_disabled_colors(app: QtWidgets.QApplication) -> None:
    from PySide6.QtGui import QPalette

    from semma.gui import theme

    for dark in (True, False):
        pal = theme.build_gnome_palette(dark=dark)
        win = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Window)
        base = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Base)
        btn = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Button)
        assert win != base and win != btn and base != btn

        btn_text = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText)
        dis_text = pal.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
        assert abs(btn_text.lightness() - dis_text.lightness()) >= 80


def test_all_settings_sections_resolve_icons_in_adwaita(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    from PySide6.QtGui import QIcon

    from semma.gui import theme

    theme.ensure_icon_theme_paths()
    old_theme = QIcon.themeName()
    try:
        if Path("/usr/share/icons/Adwaita/index.theme").is_file():
            QIcon.setThemeName("Adwaita")
        page = make_settings(home, FakeClient(status()))
        assert page.nav.count() == 6
        for i in range(page.nav.count()):
            icon = page.nav.item(i).icon()
            assert not icon.isNull(), f"missing icon for section {page.nav.item(i).text()}"
            assert not icon.pixmap(22, 22).isNull()
    finally:
        QIcon.setThemeName(old_theme)


def test_theme_detection_and_platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from semma.gui import theme

    assert theme.is_gnome_desktop({"XDG_CURRENT_DESKTOP": "ubuntu:GNOME"})
    assert not theme.is_gnome_desktop({"XDG_CURRENT_DESKTOP": "KDE"})
    assert theme.parse_gtk_font("Inter Variable  10") == ("Inter Variable", 10.0)
    assert theme.parse_gtk_font("Adwaita Sans 11") == ("Adwaita Sans", 11.0)
    assert theme.parse_gtk_font("") is None

    monkeypatch.delenv("QT_QPA_PLATFORMTHEME", raising=False)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    monkeypatch.setattr(theme, "has_gtk3_platform_plugin", lambda: False)
    theme.prepare_platform_env()
    assert os.environ["QT_QPA_PLATFORMTHEME"] == "xdgdesktopportal"

    detected_dark = theme.detect_desktop_theme(
        portal={
            "org.freedesktop.appearance": {"color-scheme": 1, "accent-color": (0.2, 0.5, 0.9)},
            "org.gnome.desktop.interface": {"font-name": "Adwaita Sans 11"},
        },
        gsettings={},
        ini={},
        env={"XDG_CURRENT_DESKTOP": "GNOME"},
    )
    assert detected_dark.is_gnome is True
    assert detected_dark.dark is True
    assert detected_dark.accent == (51, 128, 230)
    assert detected_dark.font_name == "Adwaita Sans 11"

    detected_light = theme.detect_desktop_theme(
        portal={},
        gsettings={"color-scheme": "prefer-light", "accent-color": "teal"},
        ini={},
        env={"XDG_CURRENT_DESKTOP": "GNOME", "GTK_THEME": "Adwaita"},
    )
    assert detected_light.dark is False
    assert detected_light.accent == (33, 144, 164)


def test_live_theme_switch_refreshes_icons_and_stylesheet(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    from PySide6.QtGui import QIcon

    from semma.gui import theme
    from semma.gui.onboarding import OnboardingWindow
    from semma.gui.status_page import FailedDialog

    theme.ensure_icon_theme_paths()
    old_pal = app.palette()
    old_css = app.styleSheet()
    old_theme = QIcon.themeName()
    try:
        if Path("/usr/share/icons/Adwaita/index.theme").is_file():
            QIcon.setThemeName("Adwaita")
        app.setPalette(theme.build_gnome_palette(dark=False))
        app.processEvents()

        client = FakeClient(status(state="indexing", indexed=5, pending=5, failed=1,
                                   indexed_by_kind={"text": 5}, failed_by_kind={"pdf": 1}))
        w = MainWindow(client, config_path(home), home)
        w.show()
        w.status_page.error.show_message("warning", "Transient warning")
        dialog = FailedDialog(client, home, w)
        dialog.show()
        wiz = OnboardingWindow(client, config_path(home), home)
        wiz.show()
        app.processEvents()

        light_lums = {
            "tab0": _icon_avg_channel(w.tabs.tabIcon(0), QIcon.Mode.Normal)[0],
            "pause": _icon_avg_channel(w.status_page.pause_button.icon(), QIcon.Mode.Normal)[0],
            "rescan": _icon_avg_channel(w.status_page.rescan_button.icon(), QIcon.Mode.Normal)[0],
            "retry": _icon_avg_channel(w.status_page.retry_button.icon(), QIcon.Mode.Normal)[0],
            "start": _icon_avg_channel(w.status_page.start_button.icon(), QIcon.Mode.Normal)[0],
            "failed_retry": _icon_avg_channel(dialog.retry_button.icon(), QIcon.Mode.Normal)[0],
            "msg_close": _icon_avg_channel(w.status_page.error._close.icon(), QIcon.Mode.Normal)[0],
            "settings_apply": _icon_avg_channel(w.settings_page.apply_button.icon(), QIcon.Mode.Normal)[0],
            "wiz_next": _icon_avg_channel(wiz.next_button.icon(), QIcon.Mode.Normal)[0],
        }
        for name, lum in light_lums.items():
            assert lum < 100, f"{name} light icon should be dark, got lum={lum:.1f}"

        # Live switch to dark palette (which triggers ApplicationPaletteChange -> stylesheet rebuild + PaletteChange)
        app.setPalette(theme.build_gnome_palette(dark=True))
        app.processEvents()

        assert "#242424" in app.styleSheet() or "#2d2d2d" in app.styleSheet()
        dark_lums = {
            "tab0": _icon_avg_channel(w.tabs.tabIcon(0), QIcon.Mode.Normal)[0],
            "pause": _icon_avg_channel(w.status_page.pause_button.icon(), QIcon.Mode.Normal)[0],
            "rescan": _icon_avg_channel(w.status_page.rescan_button.icon(), QIcon.Mode.Normal)[0],
            "retry": _icon_avg_channel(w.status_page.retry_button.icon(), QIcon.Mode.Normal)[0],
            "start": _icon_avg_channel(w.status_page.start_button.icon(), QIcon.Mode.Normal)[0],
            "failed_retry": _icon_avg_channel(dialog.retry_button.icon(), QIcon.Mode.Normal)[0],
            "msg_close": _icon_avg_channel(w.status_page.error._close.icon(), QIcon.Mode.Normal)[0],
            "settings_apply": _icon_avg_channel(w.settings_page.apply_button.icon(), QIcon.Mode.Normal)[0],
            "wiz_next": _icon_avg_channel(wiz.next_button.icon(), QIcon.Mode.Normal)[0],
        }
        for name, lum in dark_lums.items():
            assert lum > 180, f"{name} dark icon did not refresh to bright, got lum={lum:.1f}"

        wiz._completed = True
        wiz.close()
        dialog.close()
        w.close()
    finally:
        app.setPalette(old_pal)
        app.setStyleSheet(old_css)
        QIcon.setThemeName(old_theme)


def test_priorities_spinbox_cells_and_messagebar_styling(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    from PySide6.QtGui import QPalette

    from semma.gui import theme

    old_pal = app.palette()
    old_css = app.styleSheet()
    try:
        for dark in (False, True):
            app.setPalette(theme.build_gnome_palette(dark=dark))
            app.processEvents()
            page = make_settings(home, FakeClient(status()))
            priorities = page.select("priorities")
            page.resize(800, 600)
            page.show()
            app.processEvents()

            spin = next(iter(priorities.score_spins.values()))
            # Child QSpinBox must keep its palette Base background, not transparent from parent stylesheet.
            expected_base = app.palette().color(QPalette.ColorRole.Base)
            assert spin.palette().color(QPalette.ColorRole.Base) == expected_base
            assert priorities.table.rowHeight(0) >= spin.sizeHint().height() + 4
            assert priorities.table.columnWidth(1) >= spin.sizeHint().width() + 12
            page.close()
    finally:
        app.setPalette(old_pal)
        app.setStyleSheet(old_css)


def test_status_not_running_labels_are_not_clipped(app: QtWidgets.QApplication, home: Path) -> None:
    w = MainWindow(FakeClient(None), config_path(home), home)
    w.show()
    w.status_page.set_status(None)
    w.status_page.start_message.setText("The service could not be started: Unit semma.service not found.")
    app.processEvents()
    not_running = w.status_page.pages.widget(2)
    for lbl in not_running.findChildren(QtWidgets.QLabel):
        if lbl.wordWrap() and lbl.text():
            assert lbl.height() >= lbl.heightForWidth(lbl.width())
    w.close()


def test_onboarding_keeps_load_error_visible(app: QtWidgets.QApplication, home: Path) -> None:
    from semma.gui.onboarding import OnboardingWindow

    cfg = config_path(home)
    cfg.write_text("[scan\nbroken = 1\n")
    wiz = OnboardingWindow(FakeClient(status()), cfg, home)
    assert "cannot be read" in wiz.problems.text()
    assert not wiz.problems.isHidden()
    assert not wiz.next_button.isEnabled()
    wiz._completed = True
    wiz.close()


def test_main_and_onboarding_share_exact_window_background_and_transparent_welcome_icon(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QIcon, QPalette

    from semma.gui import theme
    from semma.gui.onboarding import OnboardingWindow

    theme.ensure_icon_theme_paths()
    old_pal = app.palette()
    old_css = app.styleSheet()
    old_theme = QIcon.themeName()
    try:
        for dark, icon_theme in ((True, "breeze-dark"), (False, "breeze"), (True, "Adwaita")):
            if Path(f"/usr/share/icons/{icon_theme}/index.theme").is_file():
                QIcon.setThemeName(icon_theme)
            pal = theme.build_gnome_palette(dark=dark)
            app.setPalette(pal)
            app.setStyleSheet(theme.build_app_stylesheet(pal))
            app.processEvents()

            expected_bg = pal.color(QPalette.ColorRole.Window)
            client = FakeClient(status(state="indexing"))
            w = MainWindow(client, config_path(home), home)
            w.show()
            app.processEvents()
            img_status = w.grab().toImage()
            assert img_status.pixelColor(10, 80) == expected_bg

            w.tabs.setCurrentWidget(w.settings_page)
            app.processEvents()
            img_settings = w.grab().toImage()
            assert img_settings.pixelColor(10, 200) == expected_bg
            assert img_settings.pixelColor(240, 40) == expected_bg

            wiz = OnboardingWindow(client, config_path(home), home)
            wiz.show()
            app.processEvents()
            for page_idx in range(len(wiz._forms)):
                wiz._show_page(page_idx)
                app.processEvents()
                img_wiz = wiz.grab().toImage()
                assert img_wiz.pixelColor(10, 10) == expected_bg
                assert img_wiz.pixelColor(10, img_wiz.height() - 10) == expected_bg
                scroll = wiz.stack.currentWidget()
                assert scroll is not None
                vp_pt = scroll.mapTo(wiz, QPoint(scroll.width() - 30, 6))
                assert img_wiz.pixelColor(vp_pt) == expected_bg

            # Check _WelcomePage icon label corners have exact Window background (no dark square).
            wiz._show_page(0)
            app.processEvents()
            img_welcome = wiz.grab().toImage()
            icon_lbl = wiz.welcome._icon_label
            for corner in (
                QPoint(0, 0),
                QPoint(icon_lbl.width() - 1, 0),
                QPoint(0, icon_lbl.height() - 1),
                QPoint(icon_lbl.width() - 1, icon_lbl.height() - 1),
            ):
                pt = icon_lbl.mapTo(wiz, corner)
                assert img_welcome.pixelColor(pt) == expected_bg

            wiz._completed = True
            wiz.close()
            w.close()

        # Live switch on breeze-dark must re-tint monochrome welcome icon for light vs dark palette.
        if Path("/usr/share/icons/breeze-dark/index.theme").is_file():
            QIcon.setThemeName("breeze-dark")
            app.setPalette(theme.build_gnome_palette(dark=False))
            app.processEvents()
            wiz = OnboardingWindow(FakeClient(status()), config_path(home), home)
            wiz.show()
            app.processEvents()
            pm_light = wiz.welcome._icon_label.pixmap()
            assert not pm_light.isNull()
            img_light = pm_light.toImage()
            light_lums = [
                img_light.pixelColor(x, y).lightness()
                for y in range(img_light.height())
                for x in range(img_light.width())
                if img_light.pixelColor(x, y).alpha() > 128
            ]
            assert light_lums and (sum(light_lums) / len(light_lums)) < 100

            app.setPalette(theme.build_gnome_palette(dark=True))
            app.processEvents()
            pm_dark = wiz.welcome._icon_label.pixmap()
            img_dark = pm_dark.toImage()
            dark_lums = [
                img_dark.pixelColor(x, y).lightness()
                for y in range(img_dark.height())
                for x in range(img_dark.width())
                if img_dark.pixelColor(x, y).alpha() > 128
            ]
            assert dark_lums and (sum(dark_lums) / len(dark_lums)) > 180
            wiz._completed = True
            wiz.close()
    finally:
        app.setPalette(old_pal)
        app.setStyleSheet(old_css)
        QIcon.setThemeName(old_theme)


def test_tinting_keeps_full_color_sizes_of_mixed_icons(app: QtWidgets.QApplication) -> None:
    """Breeze's `folder` is monochrome at 16 px but blue at 32 px+; only the monochrome sizes get tinted."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon

    from semma.gui import theme, widgets

    if not Path("/usr/share/icons/breeze/places/48/folder.svg").is_file():
        pytest.skip("Breeze icon theme not installed")
    theme.ensure_icon_theme_paths()
    old_theme = QIcon.themeName()
    try:
        QIcon.setThemeName("breeze")
        icon = widgets.theme_icon("folder")
        assert widgets._is_monochrome_pixmap(icon.pixmap(QSize(16, 16)))
        assert not widgets._is_monochrome_pixmap(icon.pixmap(QSize(48, 48)))
    finally:
        QIcon.setThemeName(old_theme)
