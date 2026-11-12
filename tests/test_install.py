from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

import pytest

from gemmasearch import dbus_names
from gemmasearch.install import install_kde, status_kde, uninstall_kde


@pytest.fixture()
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    home = tmp_path / "home"
    data_home = home / ".local" / "share"
    config_home = home / ".config"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    return data_home, config_home


def test_install_kde_writes_files_and_runs_commands_in_order(
    isolated_xdg: tuple[Path, Path],
) -> None:
    data_home, config_home = isolated_xdg
    commands: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    fake_py = "/opt/gemma/venv/bin/python"
    written = install_kde(python=fake_py, runner=fake_runner)

    expected_desktop = data_home / "krunner" / "dbusplugins" / "gemmasearch.desktop"
    expected_dbus = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
    expected_unit = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT

    assert written == [expected_desktop, expected_dbus, expected_unit]
    for p in written:
        assert p.is_file()

    # Parse .desktop with configparser (case-sensitive keys) and verify every required key
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str  # type: ignore[assignment]
    cp.read(expected_desktop, encoding="utf-8")
    assert "Desktop Entry" in cp
    entry = cp["Desktop Entry"]
    assert entry["Name"] == "GemmaSearch"
    assert entry["Comment"] == "Semantic search of your files with EmbeddingGemma"
    assert entry["Icon"] == "system-search"
    assert entry["Type"] == "Service"
    assert entry["X-KDE-ServiceTypes"] == "Plasma/Runner"
    assert entry["X-Plasma-API"] == "DBus"
    assert entry["X-Plasma-DBusRunner-Service"] == dbus_names.BUS_NAME
    assert entry["X-Plasma-DBusRunner-Path"] == dbus_names.RUNNER_PATH
    assert entry["X-KDE-PluginInfo-Name"] == dbus_names.RUNNER_ID
    assert entry["X-KDE-PluginInfo-EnabledByDefault"] == "true"
    assert entry["X-Plasma-Runner-Min-Letter-Count"] == "3"
    assert entry["X-Plasma-Request-Actions-Once"] == "true"

    # Check D-Bus activation file
    dbus_cp = configparser.ConfigParser(interpolation=None)
    dbus_cp.optionxform = str  # type: ignore[assignment]
    dbus_cp.read(expected_dbus, encoding="utf-8")
    dbus_sec = dbus_cp["D-BUS Service"]
    assert dbus_sec["Name"] == dbus_names.BUS_NAME
    assert dbus_sec["Exec"] == f"{fake_py} -m gemmasearch daemon"
    assert dbus_sec["SystemdService"] == dbus_names.SYSTEMD_UNIT

    # Check systemd user unit
    unit_cp = configparser.ConfigParser(interpolation=None, strict=False)
    unit_cp.optionxform = str  # type: ignore[assignment]
    unit_cp.read(expected_unit, encoding="utf-8")
    assert unit_cp["Unit"]["Description"] == "GemmaSearch semantic file search daemon"
    assert unit_cp["Unit"]["After"] == "graphical-session.target"
    assert unit_cp["Unit"]["PartOf"] == "graphical-session.target"
    assert unit_cp["Service"]["Type"] == "dbus"
    assert unit_cp["Service"]["BusName"] == dbus_names.BUS_NAME
    assert unit_cp["Service"]["ExecStart"] == f"{fake_py} -m gemmasearch daemon"
    assert unit_cp["Service"]["Restart"] == "on-failure"
    assert unit_cp["Service"]["RestartSec"] == "5"
    assert unit_cp["Service"]["TimeoutStopSec"] == "30"
    assert unit_cp["Install"]["WantedBy"] == "graphical-session.target"
    unit_text = expected_unit.read_text(encoding="utf-8")
    assert "Environment=PYTHONUNBUFFERED=1\n" in unit_text
    assert "Environment=HF_HUB_OFFLINE=1\n" in unit_text

    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "gemmasearch.service"],
        ["kquitapp6", "krunner"],
    ]


def test_install_idempotency_and_uninstall_removes_everything(
    isolated_xdg: tuple[Path, Path],
) -> None:
    commands: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    first = install_kde(python="/usr/bin/python3", runner=fake_runner)
    second = install_kde(python="/usr/bin/python3", runner=fake_runner)
    assert first == second
    for p in first:
        assert p.is_file()
        # Ensure no leftover temp files in parent directory
        assert list(p.parent.glob(".*.tmp")) == []

    commands.clear()
    removed = uninstall_kde(runner=fake_runner)
    assert removed == first
    for p in first:
        assert not p.exists()

    assert commands == [
        ["systemctl", "--user", "disable", "--now", "gemmasearch.service"],
        ["systemctl", "--user", "daemon-reload"],
        ["kquitapp6", "krunner"],
    ]

    # Second uninstall is a safe no-op for missing files
    assert uninstall_kde(runner=fake_runner) == []


def test_paths_with_spaces_are_quoted_in_unit_and_dbus_service(
    isolated_xdg: tuple[Path, Path],
) -> None:
    _, config_home = isolated_xdg
    spaced_py = "/home/user/My Projects/Gemma Search/.venv/bin/python"
    written = install_kde(
        python=spaced_py,
        enable=False,
        restart_krunner=False,
        runner=lambda cmd: 0,
    )
    _, dbus_path, unit_path = written
    assert unit_path == config_home / "systemd" / "user" / "gemmasearch.service"

    unit_text = unit_path.read_text(encoding="utf-8")
    dbus_text = dbus_path.read_text(encoding="utf-8")
    expected_exec = f'"{spaced_py}" -m gemmasearch daemon'
    assert f"ExecStart={expected_exec}\n" in unit_text
    assert f"Exec={expected_exec}\n" in dbus_text


def test_status_kde_reports_files_and_service_state(
    isolated_xdg: tuple[Path, Path],
) -> None:
    st_before = status_kde(runner=lambda cmd: 1)
    assert st_before["installed"] is False
    assert st_before["enabled"] is False
    assert st_before["active"] is False
    assert st_before["files"] == {
        "desktop": False,
        "dbus_service": False,
        "systemd_unit": False,
    }

    install_kde(runner=lambda cmd: 0)
    st_after = status_kde(runner=lambda cmd: 0)
    assert st_after["installed"] is True
    assert st_after["enabled"] is True
    assert st_after["active"] is True
    assert st_after["files"] == {
        "desktop": True,
        "dbus_service": True,
        "systemd_unit": True,
    }


def test_python_m_gemmasearch_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "gemmasearch", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "gemmasearch" in proc.stdout
