"""Install, uninstall, and inspect KDE Plasma KRunner D-Bus plugin and systemd user service."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import dbus_names
from .config import xdg_config_home, xdg_data_home


def _default_runner(cmd: list[str]) -> int:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode
    except OSError:
        return 1


def _resolve_python(python: str | None) -> str:
    raw = python or sys.executable
    # Keep symlink name (e.g. .venv/bin/python) intact so virtualenv activation works,
    # while resolving relative paths and ~ to an absolute path.
    return os.path.abspath(os.path.expanduser(raw))


def _quote_exec_path(path: str) -> str:
    """Quote an executable path for systemd ExecStart= and D-Bus Exec= if it contains spaces/quotes."""
    if any(ch.isspace() or ch in ('"', "\\") for ch in path):
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return path


def _target_paths(home: Path | None = None) -> tuple[Path, Path, Path]:
    base_home = home or Path.home()
    data_home = xdg_data_home(base_home)
    config_home = xdg_config_home(base_home)
    desktop_path = data_home / "krunner" / "dbusplugins" / f"{dbus_names.RUNNER_ID}.desktop"
    dbus_service_path = data_home / "dbus-1" / "services" / f"{dbus_names.BUS_NAME}.service"
    systemd_unit_path = config_home / "systemd" / "user" / dbus_names.SYSTEMD_UNIT
    return desktop_path, dbus_service_path, systemd_unit_path


def render_desktop_file() -> str:
    return (
        "[Desktop Entry]\n"
        "Name=GemmaSearch\n"
        "Comment=Semantic search of your files with EmbeddingGemma\n"
        "Icon=system-search\n"
        "Type=Service\n"
        "X-KDE-ServiceTypes=Plasma/Runner\n"
        "X-Plasma-API=DBus\n"
        f"X-Plasma-DBusRunner-Service={dbus_names.BUS_NAME}\n"
        f"X-Plasma-DBusRunner-Path={dbus_names.RUNNER_PATH}\n"
        f"X-KDE-PluginInfo-Name={dbus_names.RUNNER_ID}\n"
        "X-KDE-PluginInfo-EnabledByDefault=true\n"
        "X-Plasma-Runner-Min-Letter-Count=3\n"
        "X-Plasma-Request-Actions-Once=true\n"
    )


def render_dbus_service_file(python_path: str) -> str:
    quoted_py = _quote_exec_path(python_path)
    return (
        "[D-BUS Service]\n"
        f"Name={dbus_names.BUS_NAME}\n"
        f"Exec={quoted_py} -m gemmasearch daemon\n"
        f"SystemdService={dbus_names.SYSTEMD_UNIT}\n"
    )


def render_systemd_unit_file(python_path: str) -> str:
    quoted_py = _quote_exec_path(python_path)
    return (
        "[Unit]\n"
        "Description=GemmaSearch semantic file search daemon\n"
        "After=graphical-session.target\n"
        "PartOf=graphical-session.target\n"
        "\n"
        "[Service]\n"
        "Type=dbus\n"
        f"BusName={dbus_names.BUS_NAME}\n"
        f"ExecStart={quoted_py} -m gemmasearch daemon\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        "Environment=HF_HUB_OFFLINE=1\n"
        "TimeoutStopSec=30\n"
        "\n"
        "[Install]\n"
        "WantedBy=graphical-session.target\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def install_kde(
    *,
    python: str | None = None,
    enable: bool = True,
    restart_krunner: bool = True,
    runner: Callable[[list[str]], int] | None = None,
) -> list[Path]:
    """Write KRunner desktop file, D-Bus activation file, and systemd user unit, then reload."""
    run = runner or _default_runner
    py_path = _resolve_python(python)
    desktop_path, dbus_service_path, systemd_unit_path = _target_paths()

    _atomic_write(desktop_path, render_desktop_file())
    _atomic_write(dbus_service_path, render_dbus_service_file(py_path))
    _atomic_write(systemd_unit_path, render_systemd_unit_file(py_path))

    written = [desktop_path, dbus_service_path, systemd_unit_path]

    run(["systemctl", "--user", "daemon-reload"])
    if enable:
        run(["systemctl", "--user", "enable", "--now", dbus_names.SYSTEMD_UNIT])
    if restart_krunner:
        run(["kquitapp6", "krunner"])

    return written


def uninstall_kde(
    *,
    restart_krunner: bool = True,
    runner: Callable[[list[str]], int] | None = None,
) -> list[Path]:
    """Disable the systemd user unit, remove the installed KDE/D-Bus/systemd files, and reload."""
    run = runner or _default_runner
    desktop_path, dbus_service_path, systemd_unit_path = _target_paths()

    run(["systemctl", "--user", "disable", "--now", dbus_names.SYSTEMD_UNIT])

    removed: list[Path] = []
    for path in (desktop_path, dbus_service_path, systemd_unit_path):
        if path.exists() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed.append(path)

    run(["systemctl", "--user", "daemon-reload"])
    if restart_krunner:
        run(["kquitapp6", "krunner"])

    return removed


def status_kde(
    *,
    runner: Callable[[list[str]], int] | None = None,
) -> dict[str, Any]:
    """Report which KDE/systemd integration files exist and whether the user unit is enabled/active."""
    run = runner or _default_runner
    desktop_path, dbus_service_path, systemd_unit_path = _target_paths()

    files = {
        "desktop": desktop_path.is_file(),
        "dbus_service": dbus_service_path.is_file(),
        "systemd_unit": systemd_unit_path.is_file(),
    }
    paths = {
        "desktop": desktop_path,
        "dbus_service": dbus_service_path,
        "systemd_unit": systemd_unit_path,
    }
    enabled = run(["systemctl", "--user", "is-enabled", dbus_names.SYSTEMD_UNIT]) == 0
    active = run(["systemctl", "--user", "is-active", dbus_names.SYSTEMD_UNIT]) == 0

    return {
        "files": files,
        "paths": paths,
        "desktop": files["desktop"],
        "dbus_service": files["dbus_service"],
        "systemd_unit": files["systemd_unit"],
        "installed": all(files.values()),
        "enabled": enabled,
        "active": active,
    }
