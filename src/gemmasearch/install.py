"""Install, uninstall, and inspect KDE Plasma KRunner / GNOME Shell D-Bus integration and systemd user service."""

from __future__ import annotations

import errno
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import dbus_names
from .config import xdg_config_home, xdg_data_home

DEFAULT_SYSTEM_DATA_DIR = Path("/usr/local/share")


@dataclass
class GnomeInstallResult:
    """Result of installing the GNOME Shell search provider integration."""

    written: list[Path]
    pending_ini: Path | None
    sudo_command: str | None
    service_files_kept: bool = False
    kept_service_python: str | None = None


@dataclass
class GnomeUninstallResult:
    """Result of uninstalling the GNOME Shell search provider integration."""

    removed: list[Path]
    sudo_command: str | None
    service_kept: bool = False


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


def _gnome_target_paths(
    home: Path | None = None,
    system_data_dir: Path = DEFAULT_SYSTEM_DATA_DIR,
) -> tuple[Path, Path, Path]:
    base_home = home or Path.home()
    data_home = xdg_data_home(base_home)
    ini_name = f"{dbus_names.BUS_NAME}.search-provider.ini"
    desktop_path = data_home / "applications" / dbus_names.GNOME_DESKTOP_ID
    ini_path = Path(system_data_dir) / "gnome-shell" / "search-providers" / ini_name
    staged_ini_path = data_home / "gemmasearch" / ini_name
    return desktop_path, ini_path, staged_ini_path


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


def render_gnome_desktop_file(python_path: str | None = None) -> str:
    """Render the GNOME .desktop entry required by GNOME Shell's SearchProvider2 DesktopId."""
    quoted_py = _quote_exec_path(_resolve_python(python_path))
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=GemmaSearch\n"
        "Comment=Semantic search of your files with EmbeddingGemma\n"
        "Icon=system-search\n"
        f"Exec={quoted_py} -m gemmasearch ctl status\n"
        "Terminal=false\n"
        "OnlyShowIn=GNOME;\n"
        "Categories=Utility;\n"
    )


def render_gnome_search_provider_ini() -> str:
    """Render the GNOME Shell search-provider .ini descriptor."""
    return (
        "[Shell Search Provider]\n"
        f"DesktopId={dbus_names.GNOME_DESKTOP_ID}\n"
        f"BusName={dbus_names.BUS_NAME}\n"
        f"ObjectPath={dbus_names.GNOME_PATH}\n"
        "Version=2\n"
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
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _is_permission_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or exc.errno == errno.EROFS


def _read_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _extract_exec_python(unit_text: str | None, dbus_text: str | None) -> str | None:
    """Extract the Python executable path from an existing systemd unit or D-Bus service file."""
    for text, prefix in ((unit_text, "ExecStart="), (dbus_text, "Exec=")):
        if not text:
            continue
        for line in text.splitlines():
            if line.startswith(prefix):
                raw = line[len(prefix):].strip()
                if not raw:
                    continue
                try:
                    parts = shlex.split(raw)
                except ValueError:
                    parts = raw.split()
                if parts:
                    return parts[0]
    return None


def _write_shared_service_files(
    py_path: str,
    dbus_service_path: Path,
    systemd_unit_path: Path,
) -> None:
    _atomic_write(dbus_service_path, render_dbus_service_file(py_path))
    _atomic_write(systemd_unit_path, render_systemd_unit_file(py_path))


def _reload_and_enable(
    run: Callable[[list[str]], int],
    *,
    enable: bool,
    try_restart_if_active: bool = False,
) -> None:
    run(["systemctl", "--user", "daemon-reload"])
    if enable:
        was_active = (
            try_restart_if_active
            and run(["systemctl", "--user", "is-active", dbus_names.SYSTEMD_UNIT]) == 0
        )
        run(["systemctl", "--user", "enable", "--now", dbus_names.SYSTEMD_UNIT])
        if was_active:
            run(["systemctl", "--user", "try-restart", dbus_names.SYSTEMD_UNIT])


def _remove_existing(paths: Iterable[Path]) -> list[Path]:
    removed: list[Path] = []
    for path in paths:
        if path.exists() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def _build_status(
    path_map: dict[str, Path],
    run: Callable[[list[str]], int],
) -> dict[str, Any]:
    files = {name: path.is_file() for name, path in path_map.items()}
    enabled = run(["systemctl", "--user", "is-enabled", dbus_names.SYSTEMD_UNIT]) == 0
    active = run(["systemctl", "--user", "is-active", dbus_names.SYSTEMD_UNIT]) == 0
    return {
        "files": files,
        "paths": path_map,
        **files,
        "installed": all(files.values()),
        "enabled": enabled,
        "active": active,
    }


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
    _write_shared_service_files(py_path, dbus_service_path, systemd_unit_path)

    written = [desktop_path, dbus_service_path, systemd_unit_path]

    _reload_and_enable(run, enable=enable)
    if restart_krunner:
        run(["kquitapp6", "krunner"])

    return written


def shared_service_kept_for_kde() -> bool:
    """Return True if the shared service is kept because the GNOME integration is installed."""
    gnome_desktop_path, _, _ = _gnome_target_paths()
    return gnome_desktop_path.exists() or gnome_desktop_path.is_symlink()


def shared_service_kept_for_gnome() -> bool:
    """Return True if the shared service is kept because the KDE integration is installed."""
    kde_desktop_path, _, _ = _target_paths()
    return kde_desktop_path.exists() or kde_desktop_path.is_symlink()


def uninstall_kde(
    *,
    restart_krunner: bool = True,
    runner: Callable[[list[str]], int] | None = None,
) -> list[Path]:
    """Disable the systemd user unit, remove the installed KDE/D-Bus/systemd files, and reload."""
    run = runner or _default_runner
    desktop_path, dbus_service_path, systemd_unit_path = _target_paths()
    # The service and its D-Bus/systemd files are shared with the GNOME provider: keep them
    # while that is still installed.
    gnome_installed = shared_service_kept_for_kde()

    if gnome_installed:
        removed = _remove_existing((desktop_path,))
    else:
        run(["systemctl", "--user", "disable", "--now", dbus_names.SYSTEMD_UNIT])
        removed = _remove_existing((desktop_path, dbus_service_path, systemd_unit_path))
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
    return _build_status(
        {
            "desktop": desktop_path,
            "dbus_service": dbus_service_path,
            "systemd_unit": systemd_unit_path,
        },
        run,
    )


def install_gnome(
    *,
    python: str | None = None,
    enable: bool = True,
    system_data_dir: Path = DEFAULT_SYSTEM_DATA_DIR,
    runner: Callable[[list[str]], int] | None = None,
) -> GnomeInstallResult:
    """Write GNOME desktop/search-provider files, D-Bus service, and systemd unit, then reload."""
    run = runner or _default_runner
    py_path = _resolve_python(python)
    _, dbus_service_path, systemd_unit_path = _target_paths()
    gnome_desktop_path, ini_path, staged_ini_path = _gnome_target_paths(system_data_dir=system_data_dir)

    _atomic_write(gnome_desktop_path, render_gnome_desktop_file(py_path))
    written = [gnome_desktop_path]

    desired_dbus = render_dbus_service_file(py_path)
    desired_unit = render_systemd_unit_file(py_path)
    existing_dbus = _read_if_file(dbus_service_path)
    existing_unit = _read_if_file(systemd_unit_path)
    dbus_differs = existing_dbus is not None and existing_dbus != desired_dbus
    unit_differs = existing_unit is not None and existing_unit != desired_unit

    service_files_kept = False
    kept_service_python: str | None = None
    service_files_written = False
    if python is None and (dbus_differs or unit_differs):
        service_files_kept = True
        kept_service_python = _extract_exec_python(existing_unit, existing_dbus)
    else:
        _write_shared_service_files(py_path, dbus_service_path, systemd_unit_path)
        written.extend((dbus_service_path, systemd_unit_path))
        service_files_written = True

    ini_content = render_gnome_search_provider_ini()
    pending_ini: Path | None = None
    sudo_command: str | None = None
    if _read_if_file(ini_path) == ini_content:
        staged_ini_path.unlink(missing_ok=True)
    else:
        try:
            _atomic_write(ini_path, ini_content)
            written.append(ini_path)
            staged_ini_path.unlink(missing_ok=True)
        except OSError as exc:
            if not _is_permission_error(exc):
                raise
            _atomic_write(staged_ini_path, ini_content)
            written.append(staged_ini_path)
            pending_ini = staged_ini_path
            sudo_command = f"sudo install -Dm644 {shlex.quote(str(staged_ini_path))} {shlex.quote(str(ini_path))}"

    _reload_and_enable(run, enable=enable, try_restart_if_active=service_files_written)
    return GnomeInstallResult(
        written=written,
        pending_ini=pending_ini,
        sudo_command=sudo_command,
        service_files_kept=service_files_kept,
        kept_service_python=kept_service_python,
    )


def uninstall_gnome(
    *,
    system_data_dir: Path = DEFAULT_SYSTEM_DATA_DIR,
    runner: Callable[[list[str]], int] | None = None,
) -> GnomeUninstallResult:
    """Remove GNOME files and, if KDE is not installed, disable and remove shared service files."""
    run = runner or _default_runner
    _, dbus_service_path, systemd_unit_path = _target_paths()
    gnome_desktop_path, ini_path, staged_ini_path = _gnome_target_paths(system_data_dir=system_data_dir)
    kde_installed = shared_service_kept_for_gnome()

    if not kde_installed:
        run(["systemctl", "--user", "disable", "--now", dbus_names.SYSTEMD_UNIT])

    removed = _remove_existing((gnome_desktop_path, staged_ini_path))
    sudo_command: str | None = None
    try:
        removed.extend(_remove_existing((ini_path,)))
    except OSError as exc:
        if not _is_permission_error(exc):
            raise
        sudo_command = f"sudo rm -f {shlex.quote(str(ini_path))}"

    if not kde_installed:
        removed.extend(_remove_existing((dbus_service_path, systemd_unit_path)))
        run(["systemctl", "--user", "daemon-reload"])

    return GnomeUninstallResult(
        removed=removed,
        sudo_command=sudo_command,
        service_kept=kde_installed,
    )


def status_gnome(
    *,
    system_data_dir: Path = DEFAULT_SYSTEM_DATA_DIR,
    runner: Callable[[list[str]], int] | None = None,
) -> dict[str, Any]:
    """Report which GNOME/systemd integration files exist and whether the user unit is enabled/active."""
    run = runner or _default_runner
    _, dbus_service_path, systemd_unit_path = _target_paths()
    gnome_desktop_path, ini_path, _ = _gnome_target_paths(system_data_dir=system_data_dir)
    return _build_status(
        {
            "desktop": gnome_desktop_path,
            "ini": ini_path,
            "dbus_service": dbus_service_path,
            "systemd_unit": systemd_unit_path,
        },
        run,
    )


def _match_desktop_token(token: str) -> str | None:
    lowered = token.strip().lower()
    if not lowered:
        return None
    if lowered.startswith("gnome-flashback"):
        return None
    if lowered in ("kde", "plasma") or lowered.startswith(("kde-", "plasma")):
        return "kde"
    if lowered == "gnome" or lowered.startswith("gnome-"):
        return "gnome"
    return None


def detect_desktop(env: Mapping[str, str] | None = None) -> str | None:
    """Detect KDE Plasma or GNOME from standard XDG/desktop session environment variables."""
    environ = os.environ if env is None else env
    for var in ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION"):
        raw = environ.get(var, "")
        if var == "DESKTOP_SESSION" and "/" in raw:
            raw = os.path.basename(raw.rstrip("/"))
        tokens = [t.strip() for t in raw.split(":") if t.strip()]
        if not tokens:
            continue
        if any(t.lower().startswith("gnome-flashback") for t in tokens):
            return None
        for token in tokens:
            match = _match_desktop_token(token)
            if match is not None:
                return match
    return None
