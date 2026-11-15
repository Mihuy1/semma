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
        "import sys, semma.gui.settings.filetypes_model; "
        "bad = [m for m in sys.modules if m.startswith(('PySide6', 'shiboken6'))]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
