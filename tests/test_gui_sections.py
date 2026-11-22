"""Tests for the File types, Priorities, Media and Performance settings sections and models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from semma import filetypes as ft
from semma.config_write import read_raw, update_config
from semma.gui.settings import filetypes_model as ftm
from semma.gui.settings import media_model as mm
from semma.gui.settings import performance_model as pfm
from semma.gui.settings import priorities_model as pm
from semma.gui.settings import screenshots_model as sm
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

    reloaded = pm.load(ctx_for(home))
    assert reloaded.locations["~/Documents"] == 60
    assert reloaded.locations["~/Work"] == 55
    after_remove = reloaded.copy()
    after_remove.locations = pm.remove_location(after_remove.locations, "~/Work", after_remove.defaults)
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

    only_min = saved.copy()
    only_min.max_minutes = 20
    assert mm.changes(saved, only_min) == {"media": {"max_minutes": 20}}

    with_video = saved.copy()
    with_video.max_video_gb = 10.0
    assert mm.changes(saved, with_video) == {
        "scan": {"max_file_mb": {"pdf": 1000, "audio": 2048, "video": 10240}},
    }

    back_to_defaults = saved.copy()
    back_to_defaults.max_audio_gb = 4.0
    ch = mm.changes(saved, back_to_defaults)
    assert ch == {"scan": {"max_file_mb": {"pdf": 1000}}}
    update_config(ctx.config_path, ch)
    assert read_raw(ctx.config_path) == {"scan": {"max_file_mb": {"pdf": 1000}}}


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


def _fake_ocr_facts(
    *,
    library_available: bool = True,
    installed_languages: tuple[str, ...] = ("deu", "eng", "fin"),
    default_languages: tuple[str, ...] = ("fin", "eng"),
    hint_prefix: str = "sudo dnf install",
) -> sm.OcrFacts:
    def _hint(code: str | None = None) -> str:
        if not hint_prefix:
            return ""
        if code is None:
            return f"{hint_prefix} tesseract-libs tesseract-langpack-eng"
        return f"{hint_prefix} tesseract-langpack-{code}"

    return sm.OcrFacts(
        library_available=library_available,
        installed_languages=installed_languages,
        default_languages=default_languages,
        install_hint=_hint,
    )


def test_screenshots_load_defaults_and_configured(home: Path) -> None:
    facts = _fake_ocr_facts()
    ctx = ctx_for(home)
    saved = sm.load(ctx, facts)
    assert saved == sm.ScreenshotsValues(enabled=False, languages=["fin", "eng"])
    assert sm.changes(saved, saved.copy(), facts) == {}

    ctx_cfg = ctx_for(home, '[ocr]\nenabled = true\nlanguages = ["deu", "eng"]\n')
    loaded = sm.load(ctx_cfg, facts)
    assert loaded == sm.ScreenshotsValues(enabled=True, languages=["deu", "eng"])
    assert sm.changes(loaded, loaded.copy(), facts) == {}


def test_screenshots_changes_automatic_and_back_to_default(home: Path) -> None:
    facts = _fake_ocr_facts()
    ctx = ctx_for(home)
    saved = sm.load(ctx, facts)

    turn_on = saved.copy()
    turn_on.enabled = True
    ch_on = sm.changes(saved, turn_on, facts)
    assert ch_on == {"ocr": {"enabled": True}}
    update_config(ctx.config_path, ch_on)
    assert read_raw(ctx.config_path) == {"ocr": {"enabled": True}}

    reloaded = sm.load(ctx_for(home), facts)
    edited = reloaded.copy()
    edited.languages = ["deu", "fin", "deu", "eng", "spa"]
    ch_langs = sm.changes(reloaded, edited, facts)
    assert ch_langs == {"ocr": {"languages": ["deu", "fin", "eng"]}}
    update_config(ctx.config_path, ch_langs)
    assert read_raw(ctx.config_path) == {"ocr": {"enabled": True, "languages": ["deu", "fin", "eng"]}}

    reloaded2 = sm.load(ctx_for(home), facts)
    back_auto = reloaded2.copy()
    back_auto.languages = ["fin", "eng"]
    ch_auto = sm.changes(reloaded2, back_auto, facts)
    assert ch_auto == {"ocr": {"languages": None}}
    update_config(ctx.config_path, ch_auto)
    assert read_raw(ctx.config_path) == {"ocr": {"enabled": True}}

    reloaded3 = sm.load(ctx_for(home), facts)
    turn_off = reloaded3.copy()
    turn_off.enabled = False
    ch_off = sm.changes(reloaded3, turn_off, facts)
    assert ch_off == {"ocr": {"enabled": None}}
    update_config(ctx.config_path, ch_off)
    assert read_raw(ctx.config_path) == {}


def test_screenshots_problems_and_language_choices(home: Path) -> None:
    facts = _fake_ocr_facts(installed_languages=("eng", "fin"), default_languages=("eng",))
    ctx = ctx_for(home, '[ocr]\nenabled = true\nlanguages = ["swe", "eng"]\n')
    saved = sm.load(ctx, facts)
    assert saved.languages == ["swe", "eng"]

    choices = sm.language_choices(saved.languages, facts)
    assert choices[0] == ("None", None)
    assert ("English", "eng") in choices
    assert ("Finnish", "fin") in choices
    assert ("Swedish (not installed)", "swe") in choices

    probs = sm.problems(saved, ctx, facts)
    assert len(probs) == 1
    assert not probs[0].blocking
    assert "Swedish" in probs[0].text
    assert "tesseract-langpack-swe" in probs[0].text

    none_selected = sm.ScreenshotsValues(enabled=True, languages=[])
    p_empty = sm.problems(none_selected, ctx, facts)
    assert len(p_empty) == 1 and not p_empty[0].blocking
    assert "no text will be read" in p_empty[0].text

    only_missing = sm.ScreenshotsValues(enabled=True, languages=["swe"])
    p_missing = sm.problems(only_missing, ctx, facts)
    assert len(p_missing) == 2 and all(not p.blocking for p in p_missing)
    assert any("Swedish" in p.text and "tesseract-langpack-swe" in p.text for p in p_missing)
    assert any("no text will be read" in p.text for p in p_missing)

    no_hint_facts = _fake_ocr_facts(installed_languages=("eng",), default_languages=("eng",), hint_prefix="")
    p_no_hint = sm.problems(only_missing, ctx, no_hint_facts)
    assert p_no_hint[0].text == "The Swedish language pack is not installed."


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

    sec.table.selectRow(0)
    assert not sec.remove_button.isEnabled()

    sec.choose_folder = lambda _start: str(home / "Work")
    sec.add_button.click()
    assert sec.is_modified()
    assert sec.remove_button.isEnabled()
    assert sec.reset_button.isEnabled()

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

    sec.pause_when_gpu_busy.setChecked(False)
    assert not sec.gpu_busy_threshold.isEnabled()
    assert sec.changes() == {"daemon": {"pause_when_gpu_busy": False}}
    page.apply()
    assert read_raw(config_path(home)) == {
        "model": {"device": "vulkan:1"},
        "daemon": {"pause_when_gpu_busy": False},
    }
    assert not sec.gpu_busy_threshold.isEnabled()

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


def test_performance_device_help_popover_and_no_restart_note(
    app: QtWidgets.QApplication, home: Path,
) -> None:
    page = SettingsPage(FakeClient(status()), config_path(home), home)
    page.show()
    sec = page.select("performance")
    app.processEvents()

    label_texts = [lbl.text().lower() for lbl in sec.hw_box.findChildren(QtWidgets.QLabel)]
    assert not any("restart" in text for text in label_texts)

    btn = sec.device_help_button
    tooltip = btn.toolTip()
    assert tooltip
    assert "Automatic" in tooltip
    assert "Vulkan" in tooltip
    assert "CPU" in tooltip

    popover = sec.device_help_popover
    assert not popover.isVisible()
    btn.click()
    app.processEvents()
    assert popover.isVisible()
    popover_text = popover.text()
    assert "Automatic" in popover_text
    assert "Graphics card" in popover_text
    assert "Processor" in popover_text
    assert "Vulkan" in popover_text
    assert "CPU" in popover_text

    popover.hide()
    app.processEvents()
    assert not popover.isVisible()
    page.close()


def _select_combo_data(combo: QtWidgets.QComboBox, value: object) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"combo value {value!r} not found")


def test_screenshots_section_widget(
    app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtCore import Qt

    facts = _fake_ocr_facts(installed_languages=("deu", "eng", "fin"), default_languages=("fin", "eng"))
    monkeypatch.setattr(sm.OcrFacts, "detect", lambda: facts)

    page = SettingsPage(FakeClient(status()), config_path(home), home)
    sec = page.select("screenshots")
    assert not sec.is_modified()
    assert sec.enabled_check.isEnabled()
    assert not sec.enabled_check.isChecked()
    assert all(not combo.isEnabled() for combo in sec.language_combos)
    assert all(not lbl.isEnabled() for lbl in sec.language_labels)
    assert sec.missing_note.isHidden()
    assert "for example: " in sec.more_languages_note.text()
    assert "tesseract-langpack-deu" in sec.more_languages_note.text()
    assert sec.more_languages_note.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse

    assert [c.currentData() for c in sec.language_combos] == ["fin", "eng", None]

    sec.enabled_check.setChecked(True)
    assert all(combo.isEnabled() for combo in sec.language_combos)
    assert all(lbl.isEnabled() for lbl in sec.language_labels)
    assert sec.changes() == {"ocr": {"enabled": True}}

    _select_combo_data(sec.language_combos[0], None)
    _select_combo_data(sec.language_combos[1], "deu")
    _select_combo_data(sec.language_combos[2], "deu")
    assert sec.values().languages == ["deu"]
    assert sec.changes() == {"ocr": {"enabled": True, "languages": ["deu"]}}

    page.apply()
    assert read_raw(config_path(home)) == {"ocr": {"enabled": True, "languages": ["deu"]}}

    _select_combo_data(sec.language_combos[0], "fin")
    _select_combo_data(sec.language_combos[1], "eng")
    _select_combo_data(sec.language_combos[2], None)
    assert sec.changes() == {"ocr": {"languages": None}}
    sec.enabled_check.setChecked(False)
    assert sec.changes() == {"ocr": {"enabled": None, "languages": None}}
    page.apply()
    assert read_raw(config_path(home)) == {}


def test_screenshots_section_missing_library_and_uninstalled_language(
    app: QtWidgets.QApplication, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtCore import Qt

    missing_lib = _fake_ocr_facts(library_available=False, installed_languages=(), default_languages=())
    monkeypatch.setattr(sm.OcrFacts, "detect", lambda: missing_lib)
    page = SettingsPage(FakeClient(status()), config_path(home), home)
    sec = page.select("screenshots")
    assert not sec.enabled_check.isChecked()
    assert not sec.enabled_check.isEnabled()
    assert not sec.missing_note.isHidden()
    assert "Tesseract" in sec.missing_note.text()
    assert "tesseract-libs" in sec.missing_command.text()
    assert sec.missing_command.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse

    config_path(home).write_text('[ocr]\nenabled = true\nlanguages = ["swe"]\n', encoding="utf-8")
    page.reload()
    assert sec.enabled_check.isChecked()
    assert sec.enabled_check.isEnabled()
    assert sec.language_combos[0].currentText() == "Swedish (not installed)"
    assert sec.language_combos[0].currentData() == "swe"
    assert "Swedish" in page.problems_label.text()
    assert "no text will be read" in page.problems_label.text()
    assert page.apply_button.isEnabled() is False

    sec.enabled_check.setChecked(False)
    assert page.apply_button.isEnabled()
    page.apply()
    assert read_raw(config_path(home)) == {"ocr": {"languages": ["swe"]}}
    assert not sec.enabled_check.isChecked()
    assert not sec.enabled_check.isEnabled()

    no_langs = _fake_ocr_facts(
        library_available=True, installed_languages=(), default_languages=(), hint_prefix="",
    )
    monkeypatch.setattr(sm.OcrFacts, "detect", lambda: no_langs)
    config_path(home).write_text("", encoding="utf-8")
    page.reload()
    assert not sec.enabled_check.isChecked()
    assert not sec.enabled_check.isEnabled()
    assert not sec.missing_note.isHidden()
    assert sec.missing_command.isHidden()
    assert sec.more_languages_note.text() == (
        "More languages can be added by installing their Tesseract language pack."
    )


