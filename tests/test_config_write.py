from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from semma import config
from semma.config_write import ConfigWriteError, collapse_home, read_raw, update_config


def test_creates_missing_file_and_parents(tmp_path: Path) -> None:
    cfg_path = tmp_path / "nested" / "semma" / "config.toml"
    assert not cfg_path.exists()

    update_config(
        cfg_path,
        {
            "": {"db_path": "~/.local/share/semma/custom.db"},
            "scan": {"roots": ["~/Documents"], "skip_hidden": False},
            "daemon": {"rescan_hours": 12},
        },
    )

    assert cfg_path.is_file()
    raw = read_raw(cfg_path)
    assert raw == {
        "db_path": "~/.local/share/semma/custom.db",
        "scan": {"roots": ["~/Documents"], "skip_hidden": False},
        "daemon": {"rescan_hours": 12},
    }


def test_preserves_comments_blank_lines_key_order_and_unknown_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    original = (
        "# Global configuration\n"
        'db_path = "~/old.db" # inline db comment\n'
        "\n"
        "[scan]\n"
        "# Which directories to index\n"
        'roots = ["~/Documents"]\n'
        "future_flag = true\n"
        "skip_hidden = true # keep comment\n"
        "\n"
        "[custom_unknown]\n"
        'note = "leave me alone"\n'
    )
    cfg_path.write_text(original, encoding="utf-8")

    update_config(
        cfg_path,
        {
            "scan": {"skip_hidden": False, "include_mounts": False},
            "daemon": {"rescan_hours": 6},
        },
    )

    updated = cfg_path.read_text(encoding="utf-8")
    assert "# Global configuration" in updated
    assert "# inline db comment" in updated
    assert "# Which directories to index" in updated
    assert "future_flag = true" in updated
    assert "# keep comment" in updated
    assert "[custom_unknown]\n" in updated
    assert 'note = "leave me alone"' in updated

    assert updated.index("roots =") < updated.index("future_flag =") < updated.index("skip_hidden =")
    assert updated.index("[custom_unknown]") < updated.index("[daemon]")

    raw = read_raw(cfg_path)
    assert raw["scan"]["skip_hidden"] is False
    assert raw["scan"]["include_mounts"] is False
    assert raw["scan"]["future_flag"] is True
    assert raw["custom_unknown"] == {"note": "leave me alone"}
    assert raw["daemon"] == {"rescan_hours": 6}


def test_none_removes_keys_and_emptied_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        'db_path = "~/custom.db"\n'
        "\n"
        "[scan]\n"
        "skip_hidden = false\n"
        "\n"
        "[daemon]\n"
        "rescan_hours = 12\n"
        "watch_debounce_seconds = 3\n",
        encoding="utf-8",
    )

    update_config(
        cfg_path,
        {
            "": {"db_path": None},
            "scan": {"skip_hidden": None},
            "daemon": {"rescan_hours": None},
            "model": {"device": None},
        },
    )

    text = cfg_path.read_text(encoding="utf-8")
    assert "[scan]" not in text
    assert "[model]" not in text
    assert "db_path" not in text
    assert read_raw(cfg_path) == {"daemon": {"watch_debounce_seconds": 3}}

    update_config(cfg_path, {"daemon": {"watch_debounce_seconds": None}})
    assert read_raw(cfg_path) == {}
    assert "[daemon]" not in cfg_path.read_text(encoding="utf-8")


def test_none_keeps_section_when_unknown_key_remains(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[scan]\n"
        "skip_hidden = false\n"
        "future_setting = 123\n",
        encoding="utf-8",
    )

    update_config(cfg_path, {"scan": {"skip_hidden": None}})
    assert read_raw(cfg_path) == {"scan": {"future_setting": 123}}


def test_dict_values_inline_table_and_subtable_keep_form_and_replace_keys(tmp_path: Path) -> None:
    inline_path = tmp_path / "inline.toml"
    inline_path.write_text(
        "[priority]\n"
        'locations = {"~/Documents" = 40, "~/Old" = 5}\n',
        encoding="utf-8",
    )
    update_config(
        inline_path,
        {"priority": {"locations": {"~/Documents": 55, "~/Music": 35}}},
    )
    inline_text = inline_path.read_text(encoding="utf-8")
    assert "[priority.locations]" not in inline_text
    assert "locations = {" in inline_text
    assert read_raw(inline_path) == {"priority": {"locations": {"~/Documents": 55, "~/Music": 35}}}

    update_config(inline_path, {"priority": {"locations": {}}})
    empty_inline_text = inline_path.read_text(encoding="utf-8")
    assert "[priority.locations]" not in empty_inline_text
    assert "locations = {" in empty_inline_text
    assert read_raw(inline_path) == {"priority": {"locations": {}}}

    subtable_path = tmp_path / "subtable.toml"
    subtable_path.write_text(
        "[priority]\n"
        "other_location = 10\n"
        "\n"
        "[priority.locations]\n"
        "# location weights\n"
        '"~/Documents" = 40\n'
        '"~/Old" = 5\n',
        encoding="utf-8",
    )
    update_config(
        subtable_path,
        {"priority": {"locations": {"~/Documents": 60, "~/Pictures": 30}}},
    )
    subtable_text = subtable_path.read_text(encoding="utf-8")
    assert "[priority.locations]" in subtable_text
    assert "# location weights" in subtable_text
    assert "locations = {" not in subtable_text
    assert read_raw(subtable_path) == {
        "priority": {
            "other_location": 10,
            "locations": {"~/Documents": 60, "~/Pictures": 30},
        }
    }

    dotted_path = tmp_path / "dotted.toml"
    dotted_path.write_text(
        "[index]\n"
        "prefetch_chunks = 8\n"
        "batch_sizes.text = 8\n"
        "batch_sizes.image = 8\n",
        encoding="utf-8",
    )
    update_config(dotted_path, {"index": {"batch_sizes": {"text": 16, "video": 2}}})
    assert read_raw(dotted_path) == {
        "index": {"prefetch_chunks": 8, "batch_sizes": {"text": 16, "video": 2}}
    }
    update_config(dotted_path, {"index": {"batch_sizes": {}}})
    assert read_raw(dotted_path) == {"index": {"prefetch_chunks": 8, "batch_sizes": {}}}


def test_invalid_toml_raises_and_leaves_file_untouched(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    broken = b"[scan\nroots = [1, 2\n"
    cfg_path.write_bytes(broken)

    with pytest.raises(ConfigWriteError):
        update_config(cfg_path, {"scan": {"skip_hidden": False}})

    assert cfg_path.read_bytes() == broken
    with pytest.raises(ConfigWriteError):
        read_raw(cfg_path)


def test_unknown_section_or_key_raises_without_touching_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    original = b"[scan]\nskip_hidden = true\n"
    cfg_path.write_bytes(original)

    for bad_changes in (
        {"bogus_section": {"foo": 1}},
        {"scan": {"bogus_key": 1}},
        {"": {"bogus_top": 1}},
        {"": {"scan": {"skip_hidden": False}}},
    ):
        with pytest.raises(ConfigWriteError):
            update_config(cfg_path, bad_changes)
        assert cfg_path.read_bytes() == original

    missing_path = tmp_path / "missing_dir" / "config.toml"
    with pytest.raises(ConfigWriteError):
        update_config(missing_path, {"scan": {"not_a_key": 1}})
    assert not missing_path.parent.exists()


def test_preserves_file_mode_and_cleans_temp_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[scan]\nskip_hidden = true\n", encoding="utf-8")
    os.chmod(cfg_path, 0o640)

    update_config(cfg_path, {"scan": {"skip_hidden": False}})

    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_read_raw_missing_and_plain_types(tmp_path: Path) -> None:
    cfg_path = tmp_path / "missing.toml"
    assert read_raw(cfg_path) == {}

    cfg_path.write_text(
        "[scan]\n"
        'roots = ["~/Documents", "~/Notes"]\n'
        "[priority.locations]\n"
        '"~/Documents" = 40\n',
        encoding="utf-8",
    )
    raw = read_raw(cfg_path)
    assert type(raw) is dict
    assert type(raw["scan"]) is dict
    assert type(raw["scan"]["roots"]) is list
    assert type(raw["priority"]["locations"]) is dict


def test_round_trip_with_load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_home = tmp_path / "home"
    tmp_home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_home / ".local" / "share"))

    cfg_path = tmp_home / ".config" / "semma" / "config.toml"
    docs = tmp_home / "Documents"
    excluded = docs / "Private"
    extra = Path("/mnt/data/archive")

    update_config(
        cfg_path,
        {
            "scan": {
                "roots": [collapse_home(docs, tmp_home), collapse_home(extra, tmp_home)],
                "exclude_paths": [collapse_home(excluded, tmp_home)],
                "skip_hidden": False,
            },
            "priority": {
                "locations": {collapse_home(docs, tmp_home): 85},
            },
            "daemon": {
                "rescan_hours": 12.5,
            },
        },
    )

    loaded = config.load_config(cfg_path, home=tmp_home)
    assert loaded.roots == [config.canonical(docs), config.canonical(extra)]
    assert loaded.exclude_paths == [config.canonical(excluded)]
    assert loaded.skip_hidden is False
    assert loaded.locations[config.canonical(docs)] == 85.0
    assert loaded.rescan_hours == 12.5

    update_config(cfg_path, {"daemon": {"rescan_hours": None}})
    reloaded = config.load_config(cfg_path, home=tmp_home)
    assert reloaded.rescan_hours == config.Config(home=tmp_home, db_path=loaded.db_path, roots=[]).rescan_hours


def test_collapse_home_and_expand_are_inverses(tmp_path: Path) -> None:
    home = tmp_path / "alice"
    cases = [
        home,
        home / "Documents",
        home / "Documents" / "sub" / "notes.md",
        Path("/mnt/drive/files"),
        tmp_path / "alice2" / "Documents",
    ]

    for original in cases:
        collapsed = collapse_home(original, home)
        assert config._expand(collapsed, home) == original

    assert collapse_home(home, home) == "~"
    assert collapse_home(home / "Documents" / "sub", home) == "~/Documents/sub"
    assert collapse_home(Path("/mnt/drive"), home) == "/mnt/drive"
    assert collapse_home(tmp_path / "alice2", home) == str(tmp_path / "alice2")


def test_top_level_key_added_before_existing_sections_and_scalar_updates(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[scan]\n"
        'roots = ["~/Documents"]\n'
        "\n"
        "[search]\n"
        "knn_k = 1_000\n"
        "z_gate = 2.0\n",
        encoding="utf-8",
    )

    update_config(
        cfg_path,
        {
            "": {"db_path": "~/.local/share/semma/index.db"},
            "scan": {"roots": ["~/Documents", "~/Notes"]},
            "search": {"knn_k": 1000, "z_gate": 2.5},
        },
    )

    text = cfg_path.read_text(encoding="utf-8")
    assert "knn_k = 1_000" in text
    assert text.index("db_path =") < text.index("[scan]")
    assert read_raw(cfg_path) == {
        "db_path": "~/.local/share/semma/index.db",
        "scan": {"roots": ["~/Documents", "~/Notes"]},
        "search": {"knn_k": 1000, "z_gate": 2.5},
    }


def test_onboarding_complete_round_trips(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[scan]\nroots = [\"~/Documents\"]\n", encoding="utf-8")
    update_config(cfg_path, {"": {"onboarding_complete": False}})
    assert read_raw(cfg_path)["onboarding_complete"] is False
    assert "roots" in read_raw(cfg_path)["scan"]
    loaded = config.load_config(cfg_path, home=tmp_path)
    assert loaded.onboarding_complete is False
    update_config(cfg_path, {"": {"onboarding_complete": True}})
    assert config.load_config(cfg_path, home=tmp_path).onboarding_complete is True


def test_unserializable_value_raises_without_modifying_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    original = b"[scan]\nskip_hidden = true\n"
    cfg_path.write_bytes(original)

    with pytest.raises(ConfigWriteError):
        update_config(cfg_path, {"scan": {"roots": [None]}})

    assert cfg_path.read_bytes() == original


def test_symlink_to_existing_file_keeps_symlink_comments_and_mode(tmp_path: Path) -> None:
    target = tmp_path / "dotfiles" / "semma.toml"
    target.parent.mkdir()
    target.write_text(
        "# Dotfiles managed config\n"
        "[scan]\n"
        "skip_hidden = true # keep this comment\n",
        encoding="utf-8",
    )
    os.chmod(target, 0o640)

    cfg_dir = tmp_path / "config" / "semma"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "config.toml"
    cfg_path.symlink_to(target)

    update_config(cfg_path, {"scan": {"skip_hidden": False}, "daemon": {"rescan_hours": 8}})

    assert cfg_path.is_symlink()
    assert cfg_path.readlink() == target
    updated = target.read_text(encoding="utf-8")
    assert "# Dotfiles managed config" in updated
    assert "# keep this comment" in updated
    assert read_raw(target) == {"scan": {"skip_hidden": False}, "daemon": {"rescan_hours": 8}}
    assert read_raw(cfg_path) == {"scan": {"skip_hidden": False}, "daemon": {"rescan_hours": 8}}
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert [p.name for p in cfg_dir.iterdir()] == ["config.toml"]
    assert [p.name for p in target.parent.iterdir()] == ["semma.toml"]


def test_dangling_symlink_creates_target_and_keeps_symlink(tmp_path: Path) -> None:
    target = tmp_path / "dotfiles" / "nested" / "semma.toml"
    assert not target.parent.exists()

    cfg_dir = tmp_path / "config" / "semma"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "config.toml"
    cfg_path.symlink_to(target)
    assert cfg_path.is_symlink()
    assert not cfg_path.exists()

    update_config(cfg_path, {"scan": {"roots": ["~/Documents"], "skip_hidden": False}})

    assert cfg_path.is_symlink()
    assert cfg_path.readlink() == target
    assert target.is_file()
    assert cfg_path.exists()
    assert read_raw(target) == {"scan": {"roots": ["~/Documents"], "skip_hidden": False}}
    assert read_raw(cfg_path) == {"scan": {"roots": ["~/Documents"], "skip_hidden": False}}


def test_relative_symlink_updates_target_and_keeps_relative_link(tmp_path: Path) -> None:
    dotfiles_dir = tmp_path / "dotfiles"
    dotfiles_dir.mkdir()
    target = dotfiles_dir / "semma.toml"
    target.write_text("# relative target\n[scan]\nskip_hidden = true\n", encoding="utf-8")

    cfg_dir = tmp_path / "config" / "semma"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "config.toml"
    rel_target = Path("..") / ".." / "dotfiles" / "semma.toml"
    cfg_path.symlink_to(rel_target)

    update_config(cfg_path, {"scan": {"skip_hidden": False}})

    assert cfg_path.is_symlink()
    assert cfg_path.readlink() == rel_target
    assert not cfg_path.readlink().is_absolute()
    assert "# relative target" in target.read_text(encoding="utf-8")
    assert read_raw(target) == {"scan": {"skip_hidden": False}}
