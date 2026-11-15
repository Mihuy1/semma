"""Tests for the File types, Priorities, Media and Performance settings sections and models."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from semma import filetypes as ft
from semma.config_write import read_raw, update_config
from semma.gui.settings import filetypes_model as ftm
from semma.gui.settings import priorities_model as pm
from semma.gui.settings.context import SettingsContext


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    for d in ("Documents", "Pictures", "Downloads", "Work", ".config/semma"):
        (h / d).mkdir(parents=True)
    return h


def config_path(home: Path) -> Path:
    return home / ".config" / "semma" / "config.toml"


def ctx_for(home: Path, text: str | None = None) -> SettingsContext:
    path = config_path(home)
    if text is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return SettingsContext.load(path, home)


# -- File types model ----------------------------------------------------------------------

def test_filetypes_load_defaults(home: Path) -> None:
    ctx = ctx_for(home)
    v = ftm.load(ctx)
    assert v == ftm.FiletypesValues(list(ft.KINDS), [])


def test_filetypes_load_from_config(home: Path) -> None:
    ctx = ctx_for(home, '[scan]\nkinds = ["pdf", "text"]\nexclude_extensions = [".MD", "tif", "md"]\n')
    v = ftm.load(ctx)
    assert v.kinds == ["text", "pdf"]
    assert v.exclude_extensions == ["md", "tif"]


def test_filetypes_parse_extensions() -> None:
    assert ftm.parse_extensions(" .MD, txt  .TXT , Rst, , . ") == ["md", "rst", "txt"]
    assert ftm.parse_extensions([".JPG", "png, gif", "jpg"]) == ["gif", "jpg", "png"]
    assert ftm.parse_extensions("") == []


def test_filetypes_changes_and_back_to_default(home: Path) -> None:
    saved = ftm.load(ctx_for(home))
    assert ftm.changes(saved, saved.copy()) == {}

    edited = saved.copy()
    edited.kinds = ["text", "pdf"]
    edited.exclude_extensions = ftm.parse_extensions(".MD, tif, md")
    assert ftm.changes(saved, edited) == {
        "scan": {"kinds": ["text", "pdf"], "exclude_extensions": ["md", "tif"]}
    }

    ctx = ctx_for(home, '[scan]\nkinds = ["text"]\nexclude_extensions = ["md"]\n')
    saved_custom = ftm.load(ctx)
    back = ftm.FiletypesValues(list(ft.KINDS), [])
    assert ftm.changes(saved_custom, back) == {
        "scan": {"kinds": None, "exclude_extensions": None}
    }
    update_config(ctx.config_path, ftm.changes(saved_custom, back))
    assert read_raw(ctx.config_path) == {}


def test_filetypes_problems(home: Path) -> None:
    ctx = ctx_for(home)
    assert ftm.problems(ftm.FiletypesValues(list(ft.KINDS), ["md", "jpg"]), ctx) == []

    probs = ftm.problems(ftm.FiletypesValues([], []), ctx)
    assert len(probs) == 1 and probs[0].blocking and "Nothing would be indexed" in probs[0].text

    hints = ftm.problems(ftm.FiletypesValues(["text"], [".bak", "md", "exe"]), ctx)
    assert len(hints) == 2
    assert all(not p.blocking for p in hints)
    assert any("bak" in p.text for p in hints) and any("exe" in p.text for p in hints)


def test_filetypes_models_do_not_import_pyside6() -> None:
    code = (
        "import sys, semma.gui.settings.filetypes_model, semma.gui.settings.priorities_model; "
        "bad = [m for m in sys.modules if m.startswith(('PySide6', 'shiboken6'))]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# -- Priorities model ----------------------------------------------------------------------

def test_priorities_load_defaults(home: Path) -> None:
    ctx = ctx_for(home)
    v = pm.load(ctx)
    assert v.locations == pm.DEFAULT_LOCATIONS
    assert v.other_location == 10
    assert pm.changes(v, v.copy()) == {}


def test_priorities_load_and_changes_only_diff_from_defaults(home: Path) -> None:
    ctx = ctx_for(home)
    saved = pm.load(ctx)
    edited = saved.copy()
    edited.locations["~/Documents"] = 60
    edited.locations, msg = pm.add_location(edited.locations, str(home / "Work"), ctx, score=55)
    assert msg is None
    edited.other_location = 5

    ch = pm.changes(saved, edited)
    assert ch == {
        "priority": {
            "locations": {"~/Documents": 60, "~/Work": 55},
            "other_location": 5,
        }
    }
    update_config(ctx.config_path, ch)
    assert read_raw(ctx.config_path) == {
        "priority": {
            "locations": {"~/Documents": 60, "~/Work": 55},
            "other_location": 5,
        }
    }

    # Reload and remove the user-added folder: config.toml replaces the old dict so ~/Work is gone
    reloaded = pm.load(ctx_for(home))
    assert reloaded.locations["~/Documents"] == 60
    assert reloaded.locations["~/Work"] == 55
    after_remove = reloaded.copy()
    after_remove.locations = pm.remove_location(after_remove.locations, "~/Work", after_remove.defaults)
    # Default folders cannot be removed
    after_remove.locations = pm.remove_location(after_remove.locations, "~/Documents", after_remove.defaults)
    assert "~/Documents" in after_remove.locations

    ch2 = pm.changes(reloaded, after_remove)
    assert ch2 == {"priority": {"locations": {"~/Documents": 60}}}
    update_config(ctx.config_path, ch2)
    assert read_raw(ctx.config_path) == {
        "priority": {"locations": {"~/Documents": 60}, "other_location": 5}
    }
    reloaded2 = pm.load(ctx_for(home))
    assert "~/Work" not in reloaded2.locations
    assert reloaded2.locations["~/Documents"] == 60

    # Reset to defaults -> None removes the keys from config.toml
    reset = pm.reset_locations(reloaded2)
    reset.other_location = pm.DEFAULT_OTHER_LOCATION
    ch3 = pm.changes(reloaded2, reset)
    assert ch3 == {"priority": {"locations": None, "other_location": None}}
    update_config(ctx.config_path, ch3)
    assert read_raw(ctx.config_path) == {}


def test_priorities_add_duplicate_and_problems(home: Path) -> None:
    ctx = ctx_for(home)
    v = pm.load(ctx)
    same, msg = pm.add_location(v.locations, str(home / "Documents"), ctx)
    assert same == v.locations and msg == "~/Documents is in the list already."

    assert pm.problems(v, ctx) == []
    bad = v.copy()
    bad.other_location = 120
    probs = pm.problems(bad, ctx)
    assert len(probs) == 1 and probs[0].blocking


# -- Widgets -------------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from gui_fakes import FakeClient, status  # noqa: E402

from semma.gui.app import make_app  # noqa: E402
from semma.gui.settings.page import SettingsPage  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QtWidgets.QApplication:
    return make_app([])


def test_filetypes_section_widget(app: QtWidgets.QApplication, home: Path) -> None:
    page = SettingsPage(FakeClient(status()), config_path(home), home)
    sec = page.select("filetypes")
    assert not sec.is_modified()
    assert all(cb.isChecked() for cb in sec.kind_checks.values())

    sec.kind_checks["video"].setChecked(False)
    sec.exclude_extensions.setText(" .TIF, md  tif ")
    assert sec.is_modified()
    assert sec.changes() == {
        "scan": {
            "kinds": ["text", "pdf", "office", "image", "audio"],
            "exclude_extensions": ["md", "tif"],
        }
    }
    assert page.apply_button.isEnabled()

    for cb in sec.kind_checks.values():
        cb.setChecked(False)
    assert not page.apply_button.isEnabled()
    assert "Nothing would be indexed" in page.problems_label.text()

    sec.kind_checks["text"].setChecked(True)
    assert page.apply_button.isEnabled()
    page.apply()
    assert read_raw(config_path(home)) == {
        "scan": {"kinds": ["text"], "exclude_extensions": ["md", "tif"]}
    }


def test_priorities_section_widget(app: QtWidgets.QApplication, home: Path) -> None:
    page = SettingsPage(FakeClient(status()), config_path(home), home)
    sec = page.select("priorities")
    assert not sec.is_modified()
    assert not sec.reset_button.isEnabled()

    # Selecting a default row keeps Remove disabled
    sec.table.selectRow(0)
    assert not sec.remove_button.isEnabled()

    # Adding a custom folder enables Remove and Reset to defaults
    sec.choose_folder = lambda _start: str(home / "Work")
    sec.add_button.click()
    assert sec.is_modified()
    assert sec.remove_button.isEnabled()
    assert sec.reset_button.isEnabled()

    # Duplicate folder shows message
    sec.add_button.click()
    assert "already" in sec.message.text()

    sec.score_spins["~/Documents"].setValue(65)
    sec.score_spins["~/Work"].setValue(55)
    sec.other_location.setValue(8)
    assert sec.changes() == {
        "priority": {
            "locations": {"~/Documents": 65, "~/Work": 55},
            "other_location": 8,
        }
    }
    page.apply()
    assert read_raw(config_path(home)) == {
        "priority": {
            "locations": {"~/Documents": 65, "~/Work": 55},
            "other_location": 8,
        }
    }

    # Remove the custom folder and reset defaults
    sec.table.selectRow(sec.table.rowCount() - 1)
    assert sec.remove_button.isEnabled()
    sec.remove_button.click()
    assert "~/Work" not in sec.locations()
    sec.reset_button.click()
    assert sec.locations() == pm.DEFAULT_LOCATIONS
    page.apply()
    assert read_raw(config_path(home)) == {"priority": {"other_location": 8}}

