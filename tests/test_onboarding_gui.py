"""The setup wizard and Run setup again, offscreen, with a fake daemon client."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from gui_fakes import FakeClient, status  # noqa: E402

from semma.config_write import read_raw  # noqa: E402
from semma.gui.app import MainWindow, make_app  # noqa: E402
from semma.gui.onboarding import OnboardingWindow  # noqa: E402


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


def test_wizard_blocks_empty_scope_discards_on_close_and_writes_on_finish(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    finished: list[bool] = []
    client = FakeClient(status(download_file="a.gguf", download_all_done=100, download_all_total=400,
                               model_files_ready=False))
    wizard = OnboardingWindow(client, config_path(home), home)
    wizard.finished_ok.connect(lambda: finished.append(True))
    wizard.show()
    app.processEvents()
    assert wizard.footer_label.text().startswith("Downloading the model")
    assert wizard.footer_bar.isVisible()
    assert wizard.next_button.text() == "Next" and not wizard.back_button.isEnabled()

    wizard.next_button.click()  # folders
    assert wizard.folders.isVisible() or wizard.stack.currentIndex() == 1
    wizard.folders.roots.set_folders([])
    wizard.folders.include_mounts.setChecked(False)
    assert not wizard.next_button.isEnabled()
    assert "Nothing is indexed" in wizard.problems.text()
    wizard.folders.include_mounts.setChecked(True)
    assert wizard.next_button.isEnabled()

    wizard.next_button.click()  # file types
    for box in wizard.filetypes.kind_checks.values():
        box.setChecked(False)
    assert not wizard.next_button.isEnabled()
    next(iter(wizard.filetypes.kind_checks.values())).setChecked(True)
    assert wizard.next_button.isEnabled()

    wizard.folders.skip_hidden.setChecked(False)
    wizard.confirm_close = lambda: "stay"  # closing asks first; "stay" keeps the wizard open
    wizard.close()
    app.processEvents()
    assert wizard.isVisible()
    wizard.confirm_close = lambda: "later"
    wizard.close()
    app.processEvents()
    assert not wizard.isVisible()
    assert finished == []
    assert not config_path(home).exists()

    client.calls.clear()
    wizard = OnboardingWindow(client, config_path(home), home)
    wizard.finished_ok.connect(lambda: finished.append(True))
    wizard.show()
    for _ in range(4):
        wizard.next_button.click()
    assert wizard.next_button.text() == "Start indexing"
    combo = wizard.device.device_combo
    for i in range(combo.count()):
        if combo.itemData(i) == "cpu":
            combo.setCurrentIndex(i)
    wizard.next_button.click()
    app.processEvents()
    assert finished == [True]
    assert "ReloadConfig" in client.calls
    raw = read_raw(config_path(home))
    assert raw["onboarding_complete"] is True
    assert raw["model"] == {"device": "cpu"}
    assert "scan" not in raw


def test_wizard_footer_when_the_service_is_down(app: QtWidgets.QApplication, home: Path) -> None:
    wizard = OnboardingWindow(FakeClient(None), config_path(home), home)
    wizard.show()
    app.processEvents()
    assert wizard.footer_label.text() == "The Semma service is not running."
    assert wizard.start_button.isVisible()


def test_run_setup_again_pauses_and_opens_the_wizard(app: QtWidgets.QApplication, home: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    config_path(home).write_text('onboarding_complete = true\n[scan]\nroots = ["~/Documents"]\n', encoding="utf-8")
    client = FakeClient(status())
    window = MainWindow(client, config_path(home), home)
    window.show()
    app.processEvents()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: QtWidgets.QMessageBox.StandardButton.Cancel)
    window.settings_page.setup_button.click()
    assert read_raw(config_path(home))["onboarding_complete"] is True
    assert window._wizard is None

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: QtWidgets.QMessageBox.StandardButton.Yes)
    window.settings_page.setup_button.click()
    app.processEvents()
    raw = read_raw(config_path(home))
    assert raw["onboarding_complete"] is False
    assert raw["scan"]["roots"] == ["~/Documents"]
    assert "ReloadConfig" in client.calls
    wizard = window._wizard
    assert wizard is not None and wizard.isVisible()
    assert not window.isVisible()
    # Prefilled with the saved folders, and closing does not turn indexing back on.
    assert wizard.folders.roots.folders() == ["~/Documents"]
    assert wizard.rerun
    wizard.confirm_close = lambda: "later"
    wizard.close()
    app.processEvents()
    assert window.isVisible()
    assert window.tabs.currentWidget() is window.status_page
    assert read_raw(config_path(home))["onboarding_complete"] is False

    # Run it again and close with "Keep current settings": indexing resumes, page edits are dropped.
    client.calls.clear()
    window.settings_page.setup_button.click()
    app.processEvents()
    wizard = window._wizard
    assert wizard is not None and wizard.isVisible()
    wizard.folders.skip_hidden.setChecked(not wizard.folders.skip_hidden.isChecked())
    wizard.confirm_close = lambda: "skip"
    wizard.close()
    app.processEvents()
    assert not wizard.isVisible() and window.isVisible()
    raw = read_raw(config_path(home))
    assert raw["onboarding_complete"] is True
    assert raw["scan"] == {"roots": ["~/Documents"]}
    assert client.calls.count("ReloadConfig") == 2  # paused, then resumed


def test_closing_the_first_run_wizard_with_defaults_starts_indexing(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    finished: list[bool] = []
    closed: list[bool] = []
    client = FakeClient(status())
    wizard = OnboardingWindow(client, config_path(home), home)
    wizard.finished_ok.connect(lambda: finished.append(True))
    wizard.closed_early.connect(lambda: closed.append(True))
    wizard.show()
    wizard.next_button.click()
    wizard.folders.skip_hidden.setChecked(not wizard.folders.skip_hidden.isChecked())
    wizard.confirm_close = lambda: "skip"
    wizard.close()
    app.processEvents()
    assert finished == [True] and closed == []
    assert "ReloadConfig" in client.calls
    assert read_raw(config_path(home)) == {"onboarding_complete": True}


def test_closing_after_a_save_that_needs_a_restart_counts_as_finished(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    finished: list[bool] = []
    closed: list[bool] = []
    client = FakeClient(status())
    client.reload_result = ["model.device"]
    wizard = OnboardingWindow(client, config_path(home), home)
    wizard.finished_ok.connect(lambda: finished.append(True))
    wizard.closed_early.connect(lambda: closed.append(True))
    wizard.confirm_close = lambda: pytest.fail("a saved setup must not ask")
    wizard.show()
    for _ in range(5):
        wizard.next_button.click()
    app.processEvents()
    assert wizard.next_button.text() == "Close" and finished == []
    wizard.close()
    app.processEvents()
    assert finished == [True] and closed == []
