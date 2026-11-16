"""Cross-desktop (KDE & GNOME/GTK) theme, palette, font, icon and stylesheet integration."""

from __future__ import annotations

import asyncio
import configparser
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QLibraryInfo, QObject, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import QApplication

log = logging.getLogger(__name__)

_GNOME_DESKTOPS = ("gnome", "ubuntu", "unity", "pantheon", "budgie", "cinnamon", "mate", "xfce")

DEFAULT_ACCENT: tuple[int, int, int] = (53, 132, 228)  # Adwaita blue #3584e4

_NAMED_ACCENTS: dict[str, tuple[int, int, int]] = {
    "blue": (53, 132, 228),
    "teal": (33, 144, 164),
    "green": (58, 148, 74),
    "yellow": (200, 136, 0),
    "orange": (237, 91, 0),
    "red": (230, 45, 66),
    "pink": (213, 97, 153),
    "purple": (145, 65, 172),
    "slate": (111, 131, 150),
}


@dataclass(frozen=True)
class DesktopTheme:
    is_gnome: bool
    dark: bool
    accent: tuple[int, int, int]
    font_name: str | None = None
    icon_theme: str | None = None


def is_gnome_desktop(env: Mapping[str, str] | None = None) -> bool:
    """True when the current desktop session is GNOME or another GTK-based desktop."""
    environ = env if env is not None else os.environ
    for var in ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION"):
        value = environ.get(var, "").lower()
        if any(name in value for name in _GNOME_DESKTOPS):
            return True
    return False


def has_gtk3_platform_plugin() -> bool:
    """True when Qt's GTK3 platformtheme plugin (`libqgtk3.so`) is installed."""
    plugins_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    return (Path(plugins_dir) / "platformthemes" / "libqgtk3.so").is_file()


def prepare_platform_env() -> None:
    """Set `QT_QPA_PLATFORMTHEME=xdgdesktopportal` on GNOME when `libqgtk3.so` is absent.

    Must be called before `QApplication` is constructed.
    """
    if os.environ.get("QT_QPA_PLATFORMTHEME"):
        return
    if not is_gnome_desktop():
        return
    if not has_gtk3_platform_plugin():
        os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"


def _unwrap_variant(value: Any) -> Any:
    while hasattr(value, "value") and hasattr(value, "signature"):
        value = value.value
    if isinstance(value, dict):
        return {k: _unwrap_variant(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap_variant(v) for v in value)
    return value


async def _fetch_portal_settings(timeout: float) -> dict[str, dict[str, Any]]:
    from dbus_fast import Message, MessageType
    from dbus_fast.aio import MessageBus

    bus = await asyncio.wait_for(MessageBus().connect(), timeout)
    try:
        msg = Message(
            destination="org.freedesktop.portal.Desktop",
            path="/org/freedesktop/portal/desktop",
            interface="org.freedesktop.portal.Settings",
            member="ReadAll",
            signature="as",
            body=[["org.freedesktop.appearance", "org.gnome.desktop.interface"]],
        )
        reply = await asyncio.wait_for(bus.call(msg), timeout)
        if reply is None or reply.message_type == MessageType.ERROR or not reply.body:
            return {}
        raw = _unwrap_variant(reply.body[0])
        return raw if isinstance(raw, dict) else {}
    finally:
        bus.disconnect()


def read_portal_settings(timeout: float = 0.3) -> dict[str, dict[str, Any]]:
    """Read appearance and GNOME interface settings from xdg-desktop-portal (bounded)."""
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return {}
    try:
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:
            # Not `with`: its exit waits for the worker, which would defeat the timeout below.
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                return pool.submit(asyncio.run, _fetch_portal_settings(timeout)).result(timeout=2 * timeout + 0.2)
            finally:
                pool.shutdown(wait=False)
        return asyncio.run(_fetch_portal_settings(timeout))
    except Exception as exc:
        log.debug("portal settings unavailable: %s", exc)
        return {}


def read_gsettings_interface(timeout: float = 0.3) -> dict[str, str]:
    """Fallback: read `org.gnome.desktop.interface` keys via `gsettings`."""
    if not shutil.which("gsettings"):
        return {}
    try:
        proc = subprocess.run(
            ["gsettings", "list-recursively", "org.gnome.desktop.interface"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return {}
    except Exception as exc:
        log.debug("gsettings failed: %s", exc)
        return {}
    out: dict[str, str] = {}
    prefix = "org.gnome.desktop.interface "
    for line in proc.stdout.splitlines():
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix):]
        parts = rest.split(" ", 1)
        if len(parts) != 2:
            continue
        key, val = parts[0].strip(), parts[1].strip()
        if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
            val = val[1:-1]
        out[key] = val
    return out


def read_gtk_settings_ini(paths: list[Path] | None = None) -> dict[str, str]:
    """Read GTK 3/4 `settings.ini` `[Settings]` keys."""
    if paths is None:
        cfg_home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        paths = [cfg_home / "gtk-3.0" / "settings.ini", cfg_home / "gtk-4.0" / "settings.ini"]
    merged: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        parser = configparser.ConfigParser()
        try:
            parser.read(path, encoding="utf-8")
        except Exception:
            continue
        if parser.has_section("Settings"):
            for k, v in parser.items("Settings"):
                merged[k.lower()] = v.strip()
    return merged


def _gtk_theme_dark_hint(theme_name: str) -> bool | None:
    name = theme_name.strip().lower()
    if not name:
        return None
    if name.endswith(":dark") or name.endswith("-dark"):
        return True
    if name.endswith(":light") or name.endswith("-light"):
        return False
    return None


def _parse_portal_accent(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    try:
        rgb = tuple(float(c) for c in value)
    except (TypeError, ValueError):
        return None
    if any(not (0.0 <= c <= 1.0) for c in rgb):
        return None
    return (round(rgb[0] * 255), round(rgb[1] * 255), round(rgb[2] * 255))


def parse_gtk_font(font_name: str) -> tuple[str, float] | None:
    """Parse a Pango/GTK font string like `'Adwaita Sans 11'` into `(family, point_size)`."""
    text = font_name.strip()
    if not text:
        return None
    parts = text.rsplit(None, 1)
    if len(parts) != 2:
        return None
    family, size_str = parts[0].strip(), parts[1].strip()
    if not family:
        return None
    try:
        size = float(size_str)
    except ValueError:
        return None
    if size <= 0:
        return None
    return family, size


def detect_desktop_theme(
    app: QApplication | None = None,
    *,
    portal: dict[str, dict[str, Any]] | None = None,
    gsettings: dict[str, str] | None = None,
    ini: dict[str, str] | None = None,
    env: Mapping[str, str] | None = None,
) -> DesktopTheme:
    """Detect desktop kind, dark/light preference, accent color, font and icon theme."""
    environ = env if env is not None else os.environ
    semma_theme = environ.get("SEMMA_THEME", "").strip().lower()
    gnome = is_gnome_desktop(environ) or semma_theme in ("gnome-dark", "gnome-light")

    portal_data = portal if portal is not None else (read_portal_settings() if gnome else {})
    appearance = portal_data.get("org.freedesktop.appearance", {})
    gnome_iface = portal_data.get("org.gnome.desktop.interface", {})

    if gsettings is None and gnome and not gnome_iface:
        gsettings = read_gsettings_interface()
    gsettings_data = gsettings or {}
    ini_data = ini if ini is not None else (read_gtk_settings_ini() if gnome else {})

    # 1. Dark / light mode
    dark: bool | None = None
    if semma_theme == "gnome-dark":
        dark = True
    elif semma_theme == "gnome-light":
        dark = False

    if dark is None:
        gtk_theme = environ.get("GTK_THEME", "")
        dark = _gtk_theme_dark_hint(gtk_theme)

    if dark is None:
        portal_cs = appearance.get("color-scheme")
        if portal_cs == 1:
            dark = True
        elif portal_cs == 2:
            dark = False

    if dark is None:
        iface_cs = str(gnome_iface.get("color-scheme") or gsettings_data.get("color-scheme") or "").lower()
        if iface_cs == "prefer-dark":
            dark = True
        elif iface_cs == "prefer-light":
            dark = False

    if dark is None:
        prefer_dark = ini_data.get("gtk-application-prefer-dark-theme", "").lower()
        if prefer_dark in ("1", "true", "yes"):
            dark = True
        elif prefer_dark in ("0", "false", "no"):
            dark = False
        if dark is None:
            dark = _gtk_theme_dark_hint(ini_data.get("gtk-theme-name", ""))

    if dark is None and app is not None:
        try:
            cs = app.styleHints().colorScheme()
            if cs == Qt.ColorScheme.Dark:
                dark = True
            elif cs == Qt.ColorScheme.Light:
                dark = False
        except Exception:
            pass

    if dark is None and app is not None:
        dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

    if dark is None:
        dark = False

    # 2. Accent color
    accent = _parse_portal_accent(appearance.get("accent-color"))
    if accent is None:
        named = str(gnome_iface.get("accent-color") or gsettings_data.get("accent-color") or "").strip().lower()
        accent = _NAMED_ACCENTS.get(named, DEFAULT_ACCENT)

    # 3. Font & icon theme
    font_raw = gnome_iface.get("font-name") or gsettings_data.get("font-name") or ini_data.get("gtk-font-name")
    font_name = str(font_raw).strip() if font_raw else None

    icon_raw = gnome_iface.get("icon-theme") or gsettings_data.get("icon-theme") or ini_data.get("gtk-icon-theme-name")
    icon_theme = str(icon_raw).strip() if icon_raw else None

    return DesktopTheme(is_gnome=gnome, dark=dark, accent=accent, font_name=font_name, icon_theme=icon_theme)


def build_gnome_palette(dark: bool, accent: tuple[int, int, int] | None = None) -> QPalette:
    """Build a cohesive Libadwaita / GTK4-matched QPalette for dark or light mode."""
    accent_color = QColor(*accent) if accent is not None else QColor(*DEFAULT_ACCENT)
    pal = QPalette()

    if dark:
        active_colors: dict[QPalette.ColorRole, QColor] = {
            QPalette.ColorRole.Window: QColor("#242424"),
            QPalette.ColorRole.WindowText: QColor("#ffffff"),
            QPalette.ColorRole.Base: QColor("#1e1e1e"),
            QPalette.ColorRole.AlternateBase: QColor("#282828"),
            QPalette.ColorRole.Text: QColor("#ffffff"),
            QPalette.ColorRole.Button: QColor("#303030"),
            QPalette.ColorRole.ButtonText: QColor("#ffffff"),
            QPalette.ColorRole.BrightText: QColor("#ffffff"),
            QPalette.ColorRole.Light: QColor("#454545"),
            QPalette.ColorRole.Midlight: QColor("#3a3a3a"),
            QPalette.ColorRole.Mid: QColor("#2c2c2c"),
            QPalette.ColorRole.Dark: QColor("#181818"),
            QPalette.ColorRole.Shadow: QColor("#0d0d0d"),
            QPalette.ColorRole.Highlight: accent_color,
            QPalette.ColorRole.HighlightedText: QColor("#ffffff"),
            QPalette.ColorRole.PlaceholderText: QColor("#9a9996"),
            QPalette.ColorRole.ToolTipBase: QColor("#303030"),
            QPalette.ColorRole.ToolTipText: QColor("#ffffff"),
            QPalette.ColorRole.Link: QColor("#78aeed"),
            QPalette.ColorRole.LinkVisited: QColor("#b583e8"),
        }
        disabled_colors: dict[QPalette.ColorRole, QColor] = {
            QPalette.ColorRole.WindowText: QColor("#77767b"),
            QPalette.ColorRole.Text: QColor("#77767b"),
            QPalette.ColorRole.ButtonText: QColor("#77767b"),
            QPalette.ColorRole.Button: QColor("#2a2a2a"),
            QPalette.ColorRole.Base: QColor("#202020"),
            QPalette.ColorRole.Window: QColor("#242424"),
            QPalette.ColorRole.Highlight: QColor("#383838"),
            QPalette.ColorRole.HighlightedText: QColor("#9a9996"),
            QPalette.ColorRole.PlaceholderText: QColor("#5e5c64"),
        }
    else:
        active_colors = {
            QPalette.ColorRole.Window: QColor("#fafafa"),
            QPalette.ColorRole.WindowText: QColor("#1e1e1e"),
            QPalette.ColorRole.Base: QColor("#ffffff"),
            QPalette.ColorRole.AlternateBase: QColor("#f3f3f5"),
            QPalette.ColorRole.Text: QColor("#1e1e1e"),
            QPalette.ColorRole.Button: QColor("#ebebed"),
            QPalette.ColorRole.ButtonText: QColor("#1e1e1e"),
            QPalette.ColorRole.BrightText: QColor("#ffffff"),
            QPalette.ColorRole.Light: QColor("#ffffff"),
            QPalette.ColorRole.Midlight: QColor("#f4f4f6"),
            QPalette.ColorRole.Mid: QColor("#d8d8dc"),
            QPalette.ColorRole.Dark: QColor("#b8b8be"),
            QPalette.ColorRole.Shadow: QColor("#77767b"),
            QPalette.ColorRole.Highlight: accent_color,
            QPalette.ColorRole.HighlightedText: QColor("#ffffff"),
            QPalette.ColorRole.PlaceholderText: QColor("#77767b"),
            QPalette.ColorRole.ToolTipBase: QColor("#242424"),
            QPalette.ColorRole.ToolTipText: QColor("#ffffff"),
            QPalette.ColorRole.Link: QColor("#1b6acb"),
            QPalette.ColorRole.LinkVisited: QColor("#7b42b8"),
        }
        disabled_colors = {
            QPalette.ColorRole.WindowText: QColor("#9a9996"),
            QPalette.ColorRole.Text: QColor("#9a9996"),
            QPalette.ColorRole.ButtonText: QColor("#9a9996"),
            QPalette.ColorRole.Button: QColor("#f0f0f2"),
            QPalette.ColorRole.Base: QColor("#f6f6f8"),
            QPalette.ColorRole.Window: QColor("#fafafa"),
            QPalette.ColorRole.Highlight: QColor("#e0e0e4"),
            QPalette.ColorRole.HighlightedText: QColor("#77767b"),
            QPalette.ColorRole.PlaceholderText: QColor("#b0afb4"),
        }

    if hasattr(QPalette.ColorRole, "Accent"):
        active_colors[QPalette.ColorRole.Accent] = accent_color

    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        for role, color in active_colors.items():
            pal.setColor(group, role, color)
    for role, color in disabled_colors.items():
        pal.setColor(QPalette.ColorGroup.Disabled, role, color)
    return pal


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        round(a.red() * (1 - t) + b.red() * t),
        round(a.green() * (1 - t) + b.green() * t),
        round(a.blue() * (1 - t) + b.blue() * t),
        round(a.alpha() * (1 - t) + b.alpha() * t),
    )


def _rgba(color: QColor, alpha: float | None = None) -> str:
    a = color.alpha() if alpha is None else max(0, min(255, round(alpha * 255)))
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {a})"


def build_app_stylesheet(palette: QPalette) -> str:
    """Build a palette-aware stylesheet that polishes cards, nav sidebar, tabs and progress bars."""
    win = palette.color(QPalette.ColorRole.Window)
    base = palette.color(QPalette.ColorRole.Base)
    alt_base = palette.color(QPalette.ColorRole.AlternateBase)
    text = palette.color(QPalette.ColorRole.WindowText)
    highlight = palette.color(QPalette.ColorRole.Highlight)
    highlighted_text = palette.color(QPalette.ColorRole.HighlightedText)
    dark = win.lightness() < 128

    if dark:
        card_bg = _blend(win, QColor("#ffffff"), 0.04)
        border = _rgba(QColor("#ffffff"), 0.11)
        subtle_border = _rgba(QColor("#ffffff"), 0.08)
        hover_bg = _rgba(QColor("#ffffff"), 0.07)
        track_bg = _rgba(QColor("#ffffff"), 0.12)
        header_bg = _blend(base, QColor("#ffffff"), 0.05)
    else:
        card_bg = _blend(win, QColor("#ffffff"), 0.75)
        border = _rgba(QColor("#000000"), 0.12)
        subtle_border = _rgba(QColor("#000000"), 0.09)
        hover_bg = _rgba(QColor("#000000"), 0.05)
        track_bg = _rgba(QColor("#000000"), 0.11)
        header_bg = _blend(base, QColor("#000000"), 0.03)

    return f"""
    QMainWindow {{
        background-color: {win.name()};
    }}
    QScrollArea {{
        background-color: transparent;
        border: none;
    }}
    QScrollArea > QWidget#qt_scrollarea_viewport {{
        background-color: transparent;
    }}
    QScrollArea > QWidget#qt_scrollarea_viewport > QWidget {{
        background-color: transparent;
    }}
    QGroupBox {{
        background-color: {card_bg.name()};
        border: 1px solid {border};
        border-radius: 8px;
        margin-top: 19px;
        padding: 10px 14px 10px 14px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 2px;
        top: 1px;
        padding: 0 2px;
        font-weight: 600;
        color: {text.name()};
    }}
    QListWidget#settingsNav {{
        background: transparent;
        border: none;
        outline: none;
        padding: 4px 0px;
    }}
    QListWidget#settingsNav::item {{
        border-radius: 6px;
        padding: 7px 10px;
        margin: 2px 4px;
    }}
    QListWidget#settingsNav::item:hover:!selected {{
        background-color: {hover_bg};
    }}
    QListWidget#settingsNav::item:selected {{
        background-color: {highlight.name()};
        color: {highlighted_text.name()};
    }}
    QListWidget, QTreeWidget, QTableWidget {{
        border: 1px solid {subtle_border};
        border-radius: 6px;
    }}
    QListWidget::item {{
        padding: 3px 6px;
    }}
    QHeaderView::section {{
        background-color: {header_bg.name()};
        padding: 5px 8px;
        border: none;
        border-bottom: 1px solid {subtle_border};
        font-weight: 600;
    }}
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background-color: {track_bg};
        min-height: 8px;
        max-height: 8px;
    }}
    QProgressBar::chunk {{
        border-radius: 4px;
        background-color: {highlight.name()};
    }}
    QTabBar::tab {{
        padding: 7px 18px;
    }}
    QPushButton {{
        min-height: 24px;
        padding: 4px 12px;
    }}
    QFrame#statusSummaryCard {{
        background-color: {card_bg.name()};
        border: 1px solid {border};
        border-radius: 8px;
    }}
    QFrame#messageBar {{
        background-color: {alt_base.name()};
        border: 1px solid {border};
        border-radius: 6px;
    }}
    """


def ensure_icon_theme_paths() -> None:
    """Ensure standard XDG icon theme directories are present in `QIcon.themeSearchPaths()`."""
    candidates: list[str] = []
    xdg_data_home = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    candidates.append(str(xdg_data_home / "icons"))
    candidates.append(str(Path.home() / ".icons"))
    xdg_data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    for entry in xdg_data_dirs.split(":"):
        if entry:
            candidates.append(str(Path(entry) / "icons"))
    candidates.append("/usr/share/icons")

    current = QIcon.themeSearchPaths()
    added = False
    for path in candidates:
        if path not in current and Path(path).is_dir():
            current.append(path)
            added = True
    if added:
        QIcon.setThemeSearchPaths(current)


def _theme_exists(theme_name: str) -> bool:
    for base in QIcon.themeSearchPaths():
        if base.startswith(":"):
            continue
        if (Path(base) / theme_name / "index.theme").is_file():
            return True
    return False


def _apply_gnome_theme(app: QApplication, info: DesktopTheme) -> None:
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    palette = build_gnome_palette(dark=info.dark, accent=info.accent)
    app.setPalette(palette)

    current_icon_theme = QIcon.themeName()
    semma_theme = os.environ.get("SEMMA_THEME", "").strip().lower()
    if semma_theme in ("gnome-dark", "gnome-light") and _theme_exists("Adwaita"):
        QIcon.setThemeName("Adwaita")
    elif not current_icon_theme or current_icon_theme.lower() == "hicolor":
        chosen = info.icon_theme if info.icon_theme and _theme_exists(info.icon_theme) else "Adwaita"
        if _theme_exists(chosen):
            QIcon.setThemeName(chosen)
    elif not info.dark and current_icon_theme.lower().endswith("-dark"):
        light_variant = current_icon_theme[:-5]
        if _theme_exists(light_variant):
            QIcon.setThemeName(light_variant)
        elif _theme_exists("Adwaita"):
            QIcon.setThemeName("Adwaita")
    elif info.dark and current_icon_theme.lower() == "breeze" and _theme_exists("breeze-dark"):
        QIcon.setThemeName("breeze-dark")
    if _theme_exists("Adwaita"):
        QIcon.setFallbackThemeName("Adwaita")

    use_manual_font = (
        not has_gtk3_platform_plugin()
        or os.environ.get("QT_QPA_PLATFORMTHEME", "").lower() == "xdgdesktopportal"
    )
    if use_manual_font and info.font_name:
        parsed = parse_gtk_font(info.font_name)
        if parsed is not None:
            family, size = parsed
            font = QFont(family)
            font.setPointSizeF(size)
            app.setFont(font)


def _refresh_stylesheet(app: QApplication) -> None:
    css = build_app_stylesheet(app.palette())
    if css != app.styleSheet():
        app.setStyleSheet(css)


class _ThemeEventFilter(QObject):
    """Fallback for Qt builds without `QGuiApplication.paletteChanged`.

    A filter on the application object sees every event of every object, so it is only installed
    when the signal is missing.
    """

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app
        self._updating = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._app and event.type() == QEvent.Type.ApplicationPaletteChange and not self._updating:
            self._updating = True
            try:
                _refresh_stylesheet(self._app)
            finally:
                self._updating = False
        return False


def apply_desktop_theme(app: QApplication) -> None:
    """Apply cross-desktop palette, font, icon fallbacks and cohesive stylesheet."""
    ensure_icon_theme_paths()
    if _theme_exists("Adwaita") and not QIcon.fallbackThemeName():
        QIcon.setFallbackThemeName("Adwaita")

    info = detect_desktop_theme(app)
    if info.is_gnome:
        _apply_gnome_theme(app, info)
    app.setStyleSheet(build_app_stylesheet(app.palette()))

    if not app.property("_semma_theme_connected"):
        app.setProperty("_semma_theme_connected", True)
        # Rebuild the palette-aware stylesheet when the palette changes (e.g. a KDE color scheme).
        palette_changed = getattr(app, "paletteChanged", None)
        if palette_changed is not None:
            palette_changed.connect(lambda _palette: _refresh_stylesheet(app))
        else:
            app.installEventFilter(_ThemeEventFilter(app))
        try:
            hints = app.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                def _on_scheme_changed(_scheme: Qt.ColorScheme) -> None:
                    updated = detect_desktop_theme(app)
                    if updated.is_gnome:
                        _apply_gnome_theme(app, updated)
                    _refresh_stylesheet(app)

                hints.colorSchemeChanged.connect(_on_scheme_changed)
        except Exception:
            log.debug("color scheme change notifications unavailable", exc_info=True)
