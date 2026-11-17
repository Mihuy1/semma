from __future__ import annotations

import configparser
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from semma import dbus_names
from semma.cli import main as cli_main
from semma.install import (
    _atomic_write,
    detect_desktop,
    install_gnome,
    install_kde,
    render_gnome_search_provider_ini,
    shared_service_kept_for_gnome,
    shared_service_kept_for_kde,
    status_gnome,
    status_kde,
    uninstall_gnome,
    uninstall_kde,
)


@pytest.fixture()
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    from semma import install as install_mod

    home = tmp_path / "home"
    data_home = home / ".local" / "share"
    config_home = home / ".config"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setattr(install_mod, "DEFAULT_SYSTEM_DATA_DIR", tmp_path / "default_sys_share")
    return data_home, config_home


def test_install_kde_writes_files_and_runs_commands_in_order(
    isolated_xdg: tuple[Path, Path],
) -> None:
    data_home, config_home = isolated_xdg
    commands: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    fake_py = "/opt/semma/venv/bin/python"
    written = install_kde(python=fake_py, runner=fake_runner)

    expected_desktop = data_home / "krunner" / "dbusplugins" / "semma.desktop"
    expected_app_desktop = data_home / "applications" / dbus_names.GNOME_DESKTOP_ID
    expected_dbus = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
    expected_unit = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT

    assert written == [expected_desktop, expected_app_desktop, expected_dbus, expected_unit]
    for p in written:
        assert p.is_file()

    # Parse .desktop with configparser (case-sensitive keys) and verify every required key
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str  # type: ignore[assignment]
    cp.read(expected_desktop, encoding="utf-8")
    assert "Desktop Entry" in cp
    entry = cp["Desktop Entry"]
    assert entry["Name"] == "Semma"
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

    # Check shared application menu entry
    app_cp = configparser.ConfigParser(interpolation=None)
    app_cp.optionxform = str  # type: ignore[assignment]
    app_cp.read(expected_app_desktop, encoding="utf-8")
    assert "Desktop Entry" in app_cp
    app_entry = app_cp["Desktop Entry"]
    assert app_entry["Type"] == "Application"
    assert app_entry["Name"] == "Semma"
    assert app_entry["Comment"] == "Semantic file search: status and settings"
    assert app_entry["Icon"] == "system-search"
    assert app_entry["Exec"] == f"{fake_py} -m semma gui"
    assert app_entry["Terminal"] == "false"
    assert app_entry["Categories"] == "Utility;"
    assert app_entry["StartupNotify"] == "true"
    assert "OnlyShowIn" not in app_entry
    assert "NoDisplay" not in app_entry

    # Check D-Bus activation file
    dbus_cp = configparser.ConfigParser(interpolation=None)
    dbus_cp.optionxform = str  # type: ignore[assignment]
    dbus_cp.read(expected_dbus, encoding="utf-8")
    dbus_sec = dbus_cp["D-BUS Service"]
    assert dbus_sec["Name"] == dbus_names.BUS_NAME
    assert dbus_sec["Exec"] == f"{fake_py} -m semma daemon"
    assert dbus_sec["SystemdService"] == dbus_names.SYSTEMD_UNIT

    # Check systemd user unit
    unit_cp = configparser.ConfigParser(interpolation=None, strict=False)
    unit_cp.optionxform = str  # type: ignore[assignment]
    unit_cp.read(expected_unit, encoding="utf-8")
    assert unit_cp["Unit"]["Description"] == "Semma semantic file search daemon"
    assert unit_cp["Unit"]["After"] == "graphical-session.target"
    assert unit_cp["Unit"]["PartOf"] == "graphical-session.target"
    assert unit_cp["Service"]["Type"] == "dbus"
    assert unit_cp["Service"]["BusName"] == dbus_names.BUS_NAME
    assert unit_cp["Service"]["ExecStart"] == f"{fake_py} -m semma daemon"
    assert unit_cp["Service"]["Restart"] == "on-failure"
    assert unit_cp["Service"]["RestartSec"] == "5"
    assert unit_cp["Service"]["TimeoutStopSec"] == "30"
    assert unit_cp["Install"]["WantedBy"] == "graphical-session.target"
    unit_text = expected_unit.read_text(encoding="utf-8")
    assert "Environment=PYTHONUNBUFFERED=1\n" in unit_text
    assert "Environment=HF_HUB_OFFLINE=1\n" in unit_text

    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "semma.service"],
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
        ["systemctl", "--user", "disable", "--now", "semma.service"],
        ["systemctl", "--user", "daemon-reload"],
        ["kquitapp6", "krunner"],
    ]

    # Second uninstall is a safe no-op for missing files
    assert uninstall_kde(runner=fake_runner) == []


def test_paths_with_spaces_are_quoted_in_unit_and_dbus_service(
    isolated_xdg: tuple[Path, Path],
) -> None:
    _, config_home = isolated_xdg
    spaced_py = "/home/user/My Projects/Semma Search/.venv/bin/python"
    written = install_kde(
        python=spaced_py,
        enable=False,
        restart_krunner=False,
        runner=lambda cmd: 0,
    )
    _, app_desktop_path, dbus_path, unit_path = written
    assert unit_path == config_home / "systemd" / "user" / "semma.service"

    unit_text = unit_path.read_text(encoding="utf-8")
    dbus_text = dbus_path.read_text(encoding="utf-8")
    app_desktop_text = app_desktop_path.read_text(encoding="utf-8")
    expected_exec = f'"{spaced_py}" -m semma daemon'
    assert f"ExecStart={expected_exec}\n" in unit_text
    assert f"Exec={expected_exec}\n" in dbus_text
    assert f'Exec="{spaced_py}" -m semma gui\n' in app_desktop_text


def test_status_kde_reports_files_and_service_state(
    isolated_xdg: tuple[Path, Path],
) -> None:
    st_before = status_kde(runner=lambda cmd: 1)
    assert st_before["installed"] is False
    assert st_before["enabled"] is False
    assert st_before["active"] is False
    assert st_before["files"] == {
        "desktop": False,
        "app_desktop": False,
        "dbus_service": False,
        "systemd_unit": False,
    }

    written = install_kde(runner=lambda cmd: 0)
    st_after = status_kde(runner=lambda cmd: 0)
    assert st_after["installed"] is True
    assert st_after["enabled"] is True
    assert st_after["active"] is True
    assert st_after["files"] == {
        "desktop": True,
        "app_desktop": True,
        "dbus_service": True,
        "systemd_unit": True,
    }

    # Missing application entry marks KDE integration incomplete.
    app_desktop = written[1]
    app_desktop.unlink()
    st_missing_app = status_kde(runner=lambda cmd: 0)
    assert st_missing_app["installed"] is False
    assert st_missing_app["files"]["app_desktop"] is False


def test_python_m_semma_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "semma", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "semma" in proc.stdout


def test_gnome_renderers_contents() -> None:
    ini_text = render_gnome_search_provider_ini()
    ini_cp = configparser.ConfigParser(interpolation=None)
    ini_cp.optionxform = str  # type: ignore[assignment]
    ini_cp.read_string(ini_text)
    assert "Shell Search Provider" in ini_cp
    sec = ini_cp["Shell Search Provider"]
    assert sec["DesktopId"] == "io.github.Semma.desktop"
    assert sec["BusName"] == "io.github.Semma"
    assert dbus_names.GNOME_PATH == "/io/github/Semma/SearchProvider"
    assert sec["ObjectPath"] == "/io/github/Semma/SearchProvider"
    assert sec["Version"] == "2"


def test_install_gnome_writable_system_dir(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "sys_share"
    commands: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> int:
        commands.append(cmd)
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return 3
        return 0

    res = install_gnome(
        python="/opt/semma/venv/bin/python",
        system_data_dir=sys_data,
        runner=fake_runner,
    )

    expected_desktop = data_home / "applications" / dbus_names.GNOME_DESKTOP_ID
    expected_dbus = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
    expected_unit = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT
    expected_ini = sys_data / "gnome-shell" / "search-providers" / "io.github.Semma.search-provider.ini"

    assert res.written == [expected_desktop, expected_dbus, expected_unit, expected_ini]
    assert res.pending_ini is None
    assert res.sudo_command is None
    assert res.service_files_kept is False
    assert res.kept_service_python is None
    for p in res.written:
        assert p.is_file()
    assert not (config_home / "environment.d").exists()

    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "is-active", dbus_names.SYSTEMD_UNIT],
        ["systemctl", "--user", "enable", "--now", dbus_names.SYSTEMD_UNIT],
    ]

    st = status_gnome(system_data_dir=sys_data, runner=lambda cmd: 0)
    assert st["installed"] is True
    assert st["enabled"] is True
    assert st["active"] is True
    assert st["files"] == {
        "desktop": True,
        "ini": True,
        "dbus_service": True,
        "systemd_unit": True,
    }


def test_install_gnome_readonly_system_dir_stages_ini_and_returns_sudo(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, _ = isolated_xdg
    sys_data = tmp_path / "ro_sys_share"
    sys_data.mkdir()
    sys_data.chmod(0o555)
    try:
        commands: list[list[str]] = []
        res = install_gnome(
            python="/usr/bin/python3",
            enable=False,
            system_data_dir=sys_data,
            runner=lambda cmd: commands.append(cmd) or 0,
        )
        expected_ini = sys_data / "gnome-shell" / "search-providers" / "io.github.Semma.search-provider.ini"
        expected_staged = data_home / "semma" / "io.github.Semma.search-provider.ini"

        assert res.pending_ini == expected_staged
        assert expected_staged.is_file()
        assert not expected_ini.exists()
        assert res.sudo_command is not None
        assert res.sudo_command.startswith("sudo install -Dm644 ")
        assert str(expected_staged) in res.sudo_command
        assert str(expected_ini) in res.sudo_command
        assert commands == [["systemctl", "--user", "daemon-reload"]]
    finally:
        sys_data.chmod(0o755)


def test_uninstall_gnome_keeps_shared_service_when_kde_present(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "sys_share"
    kde_commands: list[list[str]] = []
    install_kde(python="/usr/bin/python3", runner=lambda cmd: kde_commands.append(cmd) or 0)
    install_gnome(python="/usr/bin/python3", system_data_dir=sys_data, runner=lambda cmd: 0)

    commands: list[list[str]] = []
    res = uninstall_gnome(system_data_dir=sys_data, runner=lambda cmd: commands.append(cmd) or 0)

    gnome_desktop = data_home / "applications" / dbus_names.GNOME_DESKTOP_ID
    gnome_ini = sys_data / "gnome-shell" / "search-providers" / "io.github.Semma.search-provider.ini"
    kde_desktop = data_home / "krunner" / "dbusplugins" / f"{dbus_names.RUNNER_ID}.desktop"
    dbus_path = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
    unit_path = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT

    assert res.sudo_command is None
    assert res.service_kept is True
    assert res.removed == [gnome_ini]
    assert gnome_desktop.is_file()
    assert not gnome_ini.exists()
    assert kde_desktop.is_file()
    assert dbus_path.is_file()
    assert unit_path.is_file()
    assert commands == []

    # Uninstalling KDE afterwards removes the shared application entry and service files too.
    kde_removed = uninstall_kde(system_data_dir=sys_data, runner=lambda cmd: commands.append(cmd) or 0)
    assert kde_removed == [kde_desktop, gnome_desktop, dbus_path, unit_path]
    assert not gnome_desktop.exists()
    assert not dbus_path.exists()
    assert not unit_path.exists()


def test_uninstall_gnome_removes_shared_service_when_kde_absent_and_handles_ro_ini(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "sys_share"
    providers_dir = sys_data / "gnome-shell" / "search-providers"
    providers_dir.mkdir(parents=True)

    install_gnome(python="/usr/bin/python3", system_data_dir=sys_data, runner=lambda cmd: 0)
    ini_path = providers_dir / "io.github.Semma.search-provider.ini"
    assert ini_path.is_file()

    providers_dir.chmod(0o555)
    try:
        commands: list[list[str]] = []
        res = uninstall_gnome(system_data_dir=sys_data, runner=lambda cmd: commands.append(cmd) or 0)

        dbus_path = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
        unit_path = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT
        gnome_desktop = data_home / "applications" / dbus_names.GNOME_DESKTOP_ID

        assert not gnome_desktop.exists()
        assert not dbus_path.exists()
        assert not unit_path.exists()
        assert gnome_desktop in res.removed
        assert dbus_path in res.removed
        assert unit_path in res.removed
        assert res.sudo_command == f"sudo rm -f {ini_path}"
        assert res.service_kept is False
        assert commands == [
            ["systemctl", "--user", "disable", "--now", dbus_names.SYSTEMD_UNIT],
            ["systemctl", "--user", "daemon-reload"],
        ]
    finally:
        providers_dir.chmod(0o755)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"XDG_CURRENT_DESKTOP": "KDE"}, "kde"),
        ({"XDG_CURRENT_DESKTOP": "plasma"}, "kde"),
        ({"XDG_CURRENT_DESKTOP": "GNOME"}, "gnome"),
        ({"XDG_CURRENT_DESKTOP": "ubuntu:GNOME"}, "gnome"),
        ({"XDG_CURRENT_DESKTOP": "GNOME-Classic"}, "gnome"),
        ({"XDG_CURRENT_DESKTOP": "GNOME-Flashback:GNOME"}, None),
        ({"XDG_CURRENT_DESKTOP": "GNOME-Flashback:GNOME", "XDG_SESSION_DESKTOP": "gnome"}, None),
        ({"DESKTOP_SESSION": "/usr/share/xsessions/plasma"}, "kde"),
        ({"DESKTOP_SESSION": "/usr/share/xsessions/gnome-flashback"}, None),
        ({"XDG_CURRENT_DESKTOP": "XFCE"}, None),
        ({"XDG_CURRENT_DESKTOP": "", "XDG_SESSION_DESKTOP": "plasma"}, "kde"),
        ({"DESKTOP_SESSION": "plasmawayland"}, "kde"),
        ({}, None),
    ],
)
def test_detect_desktop(env: dict[str, str], expected: str | None) -> None:
    assert detect_desktop(env) == expected


def test_cli_install_unknown_desktop_returns_2_and_writes_nothing(
    isolated_xdg: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_home, config_home = isolated_xdg
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "XFCE")
    monkeypatch.delenv("XDG_SESSION_DESKTOP", raising=False)
    monkeypatch.delenv("DESKTOP_SESSION", raising=False)

    rc = cli_main(["install"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "Could not detect a supported desktop (KDE Plasma or GNOME). Use --desktop kde|gnome." in captured.err
    assert not data_home.exists()
    assert not config_home.exists()

    rc_un = cli_main(["uninstall"])
    assert rc_un == 2


def test_cli_install_gnome_prints_sudo_and_restart_note(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from semma import install as install_mod

    sys_data = tmp_path / "ro_sys"
    sys_data.mkdir()
    sys_data.chmod(0o555)
    try:
        orig_install_gnome = install_mod.install_gnome

        def patched_install_gnome(*, python=None, enable=True, system_data_dir=sys_data, runner=None):
            return orig_install_gnome(
                python=python,
                enable=enable,
                system_data_dir=sys_data,
                runner=lambda cmd: 0,
            )

        monkeypatch.setattr(install_mod, "install_gnome", patched_install_gnome)
        rc = cli_main(["install", "--desktop", "gnome", "--no-enable"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Run this once to let GNOME Shell find the provider:" in out
        assert "sudo install -Dm644" in out
        assert "GNOME Shell" in out
        assert "started" not in out.lower()
    finally:
        sys_data.chmod(0o755)


def test_uninstall_kde_keeps_shared_service_when_gnome_present(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "sys_share"
    install_kde(python="/usr/bin/python3", runner=lambda cmd: 0)
    install_gnome(python="/usr/bin/python3", system_data_dir=sys_data, runner=lambda cmd: 0)

    assert shared_service_kept_for_kde(system_data_dir=sys_data) is True
    assert shared_service_kept_for_gnome() is True

    commands: list[list[str]] = []
    removed = uninstall_kde(system_data_dir=sys_data, runner=lambda cmd: commands.append(cmd) or 0)

    kde_desktop = data_home / "krunner" / "dbusplugins" / f"{dbus_names.RUNNER_ID}.desktop"
    assert removed == [kde_desktop]
    assert not kde_desktop.exists()
    assert (data_home / "applications" / dbus_names.GNOME_DESKTOP_ID).is_file()
    assert (data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service").is_file()
    assert (config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT).is_file()
    assert commands == [["kquitapp6", "krunner"]]
    assert shared_service_kept_for_gnome() is False


def test_uninstall_kde_keeps_shared_files_when_only_staged_gnome_ini_present(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "ro_sys_share"
    sys_data.mkdir()
    sys_data.chmod(0o555)
    try:
        install_kde(python="/usr/bin/python3", runner=lambda cmd: 0)
        res = install_gnome(python="/usr/bin/python3", system_data_dir=sys_data, runner=lambda cmd: 0)
        assert res.pending_ini is not None
        assert shared_service_kept_for_kde(system_data_dir=sys_data) is True

        removed = uninstall_kde(system_data_dir=sys_data, runner=lambda cmd: 0)
        kde_desktop = data_home / "krunner" / "dbusplugins" / f"{dbus_names.RUNNER_ID}.desktop"
        assert removed == [kde_desktop]
        assert (data_home / "applications" / dbus_names.GNOME_DESKTOP_ID).is_file()
        assert (data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service").is_file()
        assert (config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT).is_file()
    finally:
        sys_data.chmod(0o755)


def test_atomic_write_sets_mode_0644(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "test.ini"
    _atomic_write(target, "[Test]\nKey=1\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_install_gnome_try_restarts_when_already_active(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    sys_data = tmp_path / "sys_share"
    commands: list[list[str]] = []

    res = install_gnome(
        python="/opt/semma/venv/bin/python",
        system_data_dir=sys_data,
        runner=lambda cmd: commands.append(cmd) or 0,
    )
    assert res.service_files_kept is False
    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "is-active", dbus_names.SYSTEMD_UNIT],
        ["systemctl", "--user", "enable", "--now", dbus_names.SYSTEMD_UNIT],
        ["systemctl", "--user", "try-restart", dbus_names.SYSTEMD_UNIT],
    ]


def test_install_gnome_keeps_different_service_files_unless_python_passed(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "sys_share"
    old_py = "/opt/old/venv/bin/python"
    new_py = "/opt/new/venv/bin/python"

    install_kde(python=old_py, runner=lambda cmd: 0)
    dbus_path = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
    unit_path = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT
    old_dbus_text = dbus_path.read_text(encoding="utf-8")
    old_unit_text = unit_path.read_text(encoding="utf-8")

    commands: list[list[str]] = []
    res_kept = install_gnome(
        python=None,
        system_data_dir=sys_data,
        runner=lambda cmd: commands.append(cmd) or 0,
    )
    assert res_kept.service_files_kept is True
    assert res_kept.kept_service_python == old_py
    assert dbus_path not in res_kept.written
    assert unit_path not in res_kept.written
    assert dbus_path.read_text(encoding="utf-8") == old_dbus_text
    assert unit_path.read_text(encoding="utf-8") == old_unit_text
    # Service files were not rewritten, so try-restart must not run.
    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", dbus_names.SYSTEMD_UNIT],
    ]

    commands.clear()
    res_replaced = install_gnome(
        python=new_py,
        system_data_dir=sys_data,
        runner=lambda cmd: commands.append(cmd) or 0,
    )
    assert res_replaced.service_files_kept is False
    assert res_replaced.kept_service_python is None
    assert dbus_path in res_replaced.written
    assert unit_path in res_replaced.written
    assert new_py in dbus_path.read_text(encoding="utf-8")
    assert new_py in unit_path.read_text(encoding="utf-8")
    assert commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "is-active", dbus_names.SYSTEMD_UNIT],
        ["systemctl", "--user", "enable", "--now", dbus_names.SYSTEMD_UNIT],
        ["systemctl", "--user", "try-restart", dbus_names.SYSTEMD_UNIT],
    ]


def test_install_gnome_rerun_skips_staging_when_system_ini_identical(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    data_home, _ = isolated_xdg
    sys_data = tmp_path / "ro_sys_share"
    providers_dir = sys_data / "gnome-shell" / "search-providers"
    providers_dir.mkdir(parents=True)
    ini_path = providers_dir / f"{dbus_names.BUS_NAME}.search-provider.ini"
    ini_path.write_text(render_gnome_search_provider_ini(), encoding="utf-8")

    staged_path = data_home / "semma" / f"{dbus_names.BUS_NAME}.search-provider.ini"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_text("stale staged content", encoding="utf-8")

    providers_dir.chmod(0o555)
    try:
        res = install_gnome(
            python="/usr/bin/python3",
            enable=False,
            system_data_dir=sys_data,
            runner=lambda cmd: 0,
        )
        assert res.pending_ini is None
        assert res.sudo_command is None
        assert ini_path not in res.written
        assert not staged_path.exists()
        st = status_gnome(system_data_dir=sys_data, runner=lambda cmd: 0)
        assert st["installed"] is True
    finally:
        providers_dir.chmod(0o755)


def test_cli_uninstall_reports_kept_shared_service(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from semma import install as install_mod

    data_home, _ = isolated_xdg
    sys_data = tmp_path / "sys_share"
    monkeypatch.setattr(install_mod, "_default_runner", lambda cmd: 0)
    monkeypatch.setattr(install_mod, "DEFAULT_SYSTEM_DATA_DIR", sys_data)

    install_kde(python="/usr/bin/python3", runner=lambda cmd: 0)
    install_gnome(python="/usr/bin/python3", system_data_dir=sys_data, runner=lambda cmd: 0)

    rc_kde = cli_main(["uninstall-kde"])
    out_kde = capsys.readouterr().out
    assert rc_kde == 0
    assert (
        "The semma service was kept because the GNOME search provider is still installed; "
        "run `semma uninstall-gnome` to remove it."
    ) in out_kde
    assert (data_home / "applications" / dbus_names.GNOME_DESKTOP_ID).is_file()

    install_kde(python="/usr/bin/python3", runner=lambda cmd: 0)
    rc_gnome = cli_main(["uninstall-gnome"])
    out_gnome = capsys.readouterr().out
    assert rc_gnome == 0
    assert (
        "The semma service was kept because the KDE KRunner integration is still installed; "
        "run `semma uninstall-kde` to remove it."
    ) in out_gnome
    assert (data_home / "applications" / dbus_names.GNOME_DESKTOP_ID).is_file()


def test_cli_kde_and_gnome_status_report_app_desktop(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from semma import install as install_mod

    sys_data = tmp_path / "sys_share"
    monkeypatch.setattr(install_mod, "DEFAULT_SYSTEM_DATA_DIR", sys_data)
    monkeypatch.setattr("semma.cli._quiet_runner", lambda cmd: 0)

    install_kde(python="/usr/bin/python3", runner=lambda cmd: 0)
    assert cli_main(["kde-status"]) == 0
    out_kde = capsys.readouterr().out
    assert "app_desktop   present  ~/.local/share/applications/io.github.Semma.desktop" in out_kde

    install_gnome(python="/usr/bin/python3", system_data_dir=sys_data, runner=lambda cmd: 0)
    assert cli_main(["gnome-status"]) == 0
    out_gnome = capsys.readouterr().out
    assert "desktop       present  ~/.local/share/applications/io.github.Semma.desktop" in out_gnome


def test_cli_install_gnome_kept_service_notice_and_python_flag(
    isolated_xdg: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from semma import install as install_mod

    data_home, config_home = isolated_xdg
    sys_data = tmp_path / "sys_share"
    old_py = "/opt/old/venv/bin/python"
    new_py = "/opt/new/venv/bin/python"
    install_kde(python=old_py, runner=lambda cmd: 0)

    orig_install_gnome = install_mod.install_gnome
    monkeypatch.setattr(
        install_mod,
        "install_gnome",
        lambda *, python=None, enable=True, system_data_dir=sys_data, runner=None: orig_install_gnome(
            python=python,
            enable=enable,
            system_data_dir=sys_data,
            runner=lambda cmd: 0,
        ),
    )

    rc1 = cli_main(["install-gnome", "--no-enable"])
    out1 = capsys.readouterr().out
    assert rc1 == 0
    assert f"kept existing service files that point to {old_py}; pass --python to replace them" in out1
    assert "started" not in out1.lower()

    rc2 = cli_main(["install", "--desktop", "gnome", "--python", new_py])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert "kept existing service files" not in out2
    assert "Service enabled and started." in out2
    assert new_py in (data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service").read_text(
        encoding="utf-8"
    )
    assert new_py in (config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT).read_text(
        encoding="utf-8"
    )
