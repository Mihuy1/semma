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
from semma.gui.settings import media_model as mm
from semma.gui.settings import performance_model as pfm
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
        "import sys, semma.gui.settings.filetypes_model, semma.gui.settings.priorities_model, "
        "semma.gui.settings.media_model, semma.gui.settings.performance_model; "
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


# -- Media model ---------------------------------------------------------------------------

def test_media_load_changes_and_problems(home: Path) -> None:
    ctx = ctx_for(home)
    saved = mm.load(ctx)
    assert saved == mm.MediaValues(60, 4.0, 5.0)
    assert mm.changes(saved, saved.copy()) == {}

    edited = mm.MediaValues(45, 1.0, 20.0)
    assert mm.changes(saved, edited) == {
        "media": {"max_minutes": 45},
        "scan": {"max_file_mb": {"audio": 1024, "video": 20480}},
    }
    update_config(ctx.config_path, mm.changes(saved, edited))
    assert read_raw(ctx.config_path) == {
        "media": {"max_minutes": 45},
        "scan": {"max_file_mb": {"audio": 1024, "video": 20480}},
    }

    reloaded_ctx = ctx_for(home)
    assert reloaded_ctx.config.max_media_seconds == 45 * 60
    assert reloaded_ctx.config.max_bytes["audio"] == 1024 * 2**20
    assert reloaded_ctx.config.max_bytes["video"] == 20480 * 2**20
    reloaded = mm.load(reloaded_ctx)
    assert reloaded == mm.MediaValues(45, 1.0, 20.0)

    back = mm.MediaValues(60, 4.0, 5.0)
    assert mm.changes(reloaded, back) == {
        "media": {"max_minutes": None},
        "scan": {"max_file_mb": None},
    }
    update_config(ctx.config_path, mm.changes(reloaded, back))
    assert read_raw(ctx.config_path) == {}

    assert mm.problems(mm.MediaValues(60, 4.0, 5.0), ctx) == []
    probs = mm.problems(mm.MediaValues(0, 0.0, 2000.0), ctx)
    assert len(probs) == 3 and all(p.blocking for p in probs)


def test_media_keeps_other_kind_overrides_in_max_file_mb(home: Path) -> None:
    ctx = ctx_for(home, "[scan]\nmax_file_mb = { pdf = 1000, audio = 2048 }\n")
    saved = mm.load(ctx)
    assert saved == mm.MediaValues(60, 2.0, 5.0)
    assert saved.other_max_file_mb == {"pdf": 1000}

    # Changing only max_minutes leaves scan.max_file_mb untouched in changes()
    only_min = saved.copy()
    only_min.max_minutes = 20
    assert mm.changes(saved, only_min) == {"media": {"max_minutes": 20}}

    # Changing video limit keeps pdf override and existing audio override
    with_video = saved.copy()
    with_video.max_video_gb = 10.0
    assert mm.changes(saved, with_video) == {
        "scan": {"max_file_mb": {"pdf": 1000, "audio": 2048, "video": 10240}},
    }

    # Returning audio and video to their defaults keeps pdf override (not None)
    back_to_defaults = saved.copy()
    back_to_defaults.max_audio_gb = 4.0
    ch = mm.changes(saved, back_to_defaults)
    assert ch == {"scan": {"max_file_mb": {"pdf": 1000}}}
    update_config(ctx.config_path, ch)
    assert read_raw(ctx.config_path) == {"scan": {"max_file_mb": {"pdf": 1000}}}


# -- Performance model ---------------------------------------------------------------------

def test_performance_load_defaults_and_custom_device(home: Path) -> None:
    ctx = ctx_for(home)
    saved = pfm.load(ctx)
    assert saved == pfm.PerformanceValues(30, None, 24, 10, True)
    assert pfm.device_choices(None) == [
        ("Automatic", None),
        ("Graphics card", "gpu"),
        ("Processor", "cpu"),
    ]

    ctx_custom = ctx_for(
        home,
        '[model]\ndevice = "vulkan:1"\n[daemon]\npause_when_gpu_busy = false\n'
        "gpu_busy_threshold = 45\nrescan_hours = 12\nkrunner_max_results = 15\n",
    )
    loaded = pfm.load(ctx_custom)
    assert loaded == pfm.PerformanceValues(45, "vulkan:1", 12, 15, False)
    assert ("vulkan:1", "vulkan:1") in pfm.device_choices(loaded.device)
    assert pfm.changes(loaded, loaded.copy()) == {}


def test_performance_changes_and_back_to_default(home: Path) -> None:
    ctx = ctx_for(home)
    saved = pfm.load(ctx)

    # Turning pause_when_gpu_busy off without touching threshold writes only pause_when_gpu_busy.
    pause_off = saved.copy()
    pause_off.pause_when_gpu_busy = False
    assert pfm.changes(saved, pause_off) == {"daemon": {"pause_when_gpu_busy": False}}

    edited = pfm.PerformanceValues(
        gpu_busy_threshold=50,
        device="cpu",
        rescan_hours=6,
        krunner_max_results=20,
        pause_when_gpu_busy=False,
    )
    ch = pfm.changes(saved, edited)
    assert ch == {
        "model": {"device": "cpu"},
        "daemon": {
            "pause_when_gpu_busy": False,
            "gpu_busy_threshold": 50,
            "rescan_hours": 6,
            "krunner_max_results": 20,
        },
    }
    update_config(ctx.config_path, ch)
    assert read_raw(ctx.config_path) == ch

    reloaded = pfm.load(ctx_for(home))
    assert reloaded == edited

    back = pfm.PerformanceValues()
    ch_back = pfm.changes(reloaded, back)
    assert ch_back == {
        "model": {"device": None},
        "daemon": {
            "pause_when_gpu_busy": None,
            "gpu_busy_threshold": None,
            "rescan_hours": None,
            "krunner_max_results": None,
        },
    }
    update_config(ctx.config_path, ch_back)
    assert read_raw(ctx.config_path) == {}


def test_performance_problems(home: Path) -> None:
    ctx = ctx_for(home)
    assert pfm.problems(pfm.PerformanceValues(), ctx) == []
    assert pfm.problems(pfm.PerformanceValues(device="vulkan:1"), ctx) == []
    bad = pfm.PerformanceValues(gpu_busy_threshold=0, device="tpu", rescan_hours=0, krunner_max_results=0)
    probs = pfm.problems(bad, ctx)
    assert len(probs) == 4 and all(p.blocking for p in probs)


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


def test_media_section_widget(app: QtWidgets.QApplication, home: Path) -> None:
    config_path(home).write_text("[scan]\nmax_file_mb = { pdf = 800 }\n")
    page = SettingsPage(FakeClient(status()), config_path(home), home)
    sec = page.select("media")
    assert not sec.is_modified()
    assert sec.max_minutes.value() == 60
    assert sec.max_audio_gb.value() == pytest.approx(4.0)
    assert sec.max_video_gb.value() == pytest.approx(5.0)

    sec.max_minutes.setValue(90)
    sec.max_audio_gb.setValue(1.5)
    sec.max_video_gb.setValue(20.0)
    assert sec.is_modified()
    assert sec.changes() == {
        "media": {"max_minutes": 90},
        "scan": {"max_file_mb": {"pdf": 800, "audio": 1536, "video": 20480}},
    }
    page.apply()
    assert read_raw(config_path(home)) == {
        "media": {"max_minutes": 90},
        "scan": {"max_file_mb": {"pdf": 800, "audio": 1536, "video": 20480}},
    }

    sec.max_minutes.setValue(60)
    sec.max_audio_gb.setValue(4.0)
    sec.max_video_gb.setValue(5.0)
    assert sec.changes() == {
        "media": {"max_minutes": None},
        "scan": {"max_file_mb": {"pdf": 800}},
    }
    page.apply()
    assert read_raw(config_path(home)) == {"scan": {"max_file_mb": {"pdf": 800}}}


def test_performance_section_widget(app: QtWidgets.QApplication, home: Path) -> None:
    config_path(home).write_text('[model]\ndevice = "vulkan:1"\n')
    page = SettingsPage(FakeClient(status()), config_path(home), home)
    sec = page.select("performance")
    assert not sec.is_modified()
    assert sec.selected_device() == "vulkan:1"
    assert sec.device_combo.count() == 4
    assert sec.pause_when_gpu_busy.isChecked()
    assert sec.gpu_busy_threshold.isEnabled()

    # Unchecking pause_when_gpu_busy disables the spin box and writes pause_when_gpu_busy = false
    sec.pause_when_gpu_busy.setChecked(False)
    assert not sec.gpu_busy_threshold.isEnabled()
    assert sec.changes() == {"daemon": {"pause_when_gpu_busy": False}}
    page.apply()
    assert read_raw(config_path(home)) == {
        "model": {"device": "vulkan:1"},
        "daemon": {"pause_when_gpu_busy": False},
    }
    assert not sec.gpu_busy_threshold.isEnabled()

    # Re-checking enables the spin box and removes pause_when_gpu_busy; keep custom device while editing daemon fields
    sec.pause_when_gpu_busy.setChecked(True)
    assert sec.gpu_busy_threshold.isEnabled()
    sec.gpu_busy_threshold.setValue(40)
    sec.rescan_hours.setValue(12)
    sec.krunner_max_results.setValue(15)
    assert sec.is_modified()
    assert sec.changes() == {
        "daemon": {
            "pause_when_gpu_busy": None,
            "gpu_busy_threshold": 40,
            "rescan_hours": 12,
            "krunner_max_results": 15,
        }
    }
    page.apply()
    assert read_raw(config_path(home)) == {
        "model": {"device": "vulkan:1"},
        "daemon": {
            "gpu_busy_threshold": 40,
            "rescan_hours": 12,
            "krunner_max_results": 15,
        },
    }

    # Change device to Automatic -> removes model.device
    sec.set_device(None)
    assert sec.changes() == {"model": {"device": None}}
    page.apply()
    assert read_raw(config_path(home)) == {
        "daemon": {
            "gpu_busy_threshold": 40,
            "rescan_hours": 12,
            "krunner_max_results": 15,
        }
    }



