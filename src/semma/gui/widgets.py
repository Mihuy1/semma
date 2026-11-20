"""Small widgets shared by the pages."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPalette, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QWidget,
)

_STANDARD_ICONS = {
    "dialog-information": QStyle.StandardPixmap.SP_MessageBoxInformation,
    "dialog-warning": QStyle.StandardPixmap.SP_MessageBoxWarning,
    "dialog-error": QStyle.StandardPixmap.SP_MessageBoxCritical,
    "window-close": QStyle.StandardPixmap.SP_DialogCloseButton,
    "folder": QStyle.StandardPixmap.SP_DirIcon,
    "view-refresh": QStyle.StandardPixmap.SP_BrowserReload,
    "media-playback-pause": QStyle.StandardPixmap.SP_MediaPause,
    "media-playback-start": QStyle.StandardPixmap.SP_MediaPlay,
    "list-add": QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "list-remove": QStyle.StandardPixmap.SP_TrashIcon,
    "document-save": QStyle.StandardPixmap.SP_DialogApplyButton,
    "edit-undo": QStyle.StandardPixmap.SP_DialogResetButton,
    "go-previous": QStyle.StandardPixmap.SP_ArrowBack,
    "go-next": QStyle.StandardPixmap.SP_ArrowForward,
}

_COLORFUL_ICONS = frozenset({
    "io.github.Semma",
    "system-search",
    "dialog-information",
    "dialog-warning",
    "dialog-error",
})

_ICON_SIZES = (16, 22, 24, 32, 48, 64)


def _is_monochrome_pixmap(pm: QPixmap) -> bool:
    """True when every visible pixel in `pm` is near-achromatic (chroma <= 25).

    Large pixmaps are scaled down first so the per-pixel Python loop stays cheap; chroma survives
    smooth scaling well enough to tell a symbolic glyph from a full-color icon.
    """
    if pm.isNull():
        return False
    img = pm.toImage()
    if img.width() > 24 or img.height() > 24:
        img = img.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    raw = bytes(img.constBits())
    stride = img.bytesPerLine()
    row_bytes = img.width() * 4
    has_opaque = False
    for y in range(img.height()):
        start = y * stride
        for i in range(start, start + row_bytes, 4):
            if raw[i + 3] <= 16:
                continue
            has_opaque = True
            r, g, b = raw[i], raw[i + 1], raw[i + 2]
            if max(r, g, b) - min(r, g, b) > 25:
                return False
    return has_opaque


def _tint_pixmap(pm: QPixmap, color: QColor) -> QPixmap:
    out = QPixmap(pm.size())
    out.setDevicePixelRatio(pm.devicePixelRatio())
    out.fill(Qt.GlobalColor.transparent)
    logical = QRect(QPoint(0, 0), pm.deviceIndependentSize().toSize())
    painter = QPainter(out)
    painter.drawPixmap(0, 0, pm)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(logical, color)
    painter.end()
    return out


def _dim_pixmap(pm: QPixmap, opacity: float = 0.4) -> QPixmap:
    out = QPixmap(pm.size())
    out.setDevicePixelRatio(pm.devicePixelRatio())
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    painter.setOpacity(opacity)
    painter.drawPixmap(0, 0, pm)
    painter.end()
    return out


def _effective_palette(widget: QWidget | None = None) -> QPalette:
    if widget is not None:
        return widget.palette()
    app = QApplication.instance()
    return app.palette() if isinstance(app, QApplication) else QPalette()


def tinted_icon(
    icon: QIcon,
    palette: QPalette | None = None,
    *,
    monochrome: bool | None = None,
    prefer_color: bool = False,
    selected: bool = False,
) -> QIcon:
    """Return a copy of `icon` with proper Normal/Active/Disabled/Selected pixmaps.

    Monochrome/symbolic icons are tinted to ButtonText (Normal/Active), Disabled ButtonText
    (Disabled), and HighlightedText (Selected). When `selected` is True, Normal/Active/Disabled
    use HighlightedText so icons inside widgets that paint with Normal mode (e.g. selected
    QTabBar tabs) stay legible on Highlight backgrounds. Colorful icons keep their Normal
    colors and get a translucent Disabled pixmap.

    When `prefer_color` is True and `monochrome` is None, if the icon provides any full-color
    size (for example Breeze's `folder`, which is monochrome at 16-24 px and colored at 32 px+),
    smaller monochrome sizes are rendered by downscaling a full-color size instead of tinting the
    monochrome glyph.
    """
    if icon.isNull():
        return icon
    pal = palette if palette is not None else _effective_palette()

    normal_role = QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.ButtonText
    normal_color = pal.color(QPalette.ColorGroup.Active, normal_role)
    disabled_color = QColor(pal.color(QPalette.ColorGroup.Disabled, normal_role))
    if disabled_color.alpha() == 255 and abs(normal_color.lightness() - disabled_color.lightness()) < 40:
        disabled_color.setAlphaF(0.45)
    selected_color = pal.color(QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText)

    entries: list[tuple[int, float, QPixmap, bool]] = []
    for sz in _ICON_SIZES:
        for dpr in (1.0, 2.0):
            pm = icon.pixmap(QSize(sz, sz), dpr, QIcon.Mode.Normal, QIcon.State.Off)
            if pm.isNull():
                continue
            is_mono = monochrome if monochrome is not None else _is_monochrome_pixmap(pm)
            entries.append((sz, dpr, pm, is_mono))

    colored_by_px = sorted(
        (pm for _, _, pm, is_mono in entries if not is_mono),
        key=lambda p: (p.width(), p.height()),
    )
    use_color_upscale = prefer_color and monochrome is None and bool(colored_by_px)

    out = QIcon()
    for sz, dpr, pm, is_mono in entries:
        if is_mono and use_color_upscale:
            target_px = round(sz * dpr)
            src = next(
                (c for c in colored_by_px if c.width() >= target_px and c.height() >= target_px),
                colored_by_px[-1],
            )
            pm = src.scaled(
                target_px,
                target_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            pm.setDevicePixelRatio(dpr)
            is_mono = False
        if is_mono:
            normal_pm = _tint_pixmap(pm, normal_color)
            disabled_pm = _tint_pixmap(pm, disabled_color)
            selected_pm = _tint_pixmap(pm, selected_color)
            out.addPixmap(normal_pm, QIcon.Mode.Normal)
            out.addPixmap(normal_pm, QIcon.Mode.Active)
            out.addPixmap(disabled_pm, QIcon.Mode.Disabled)
            out.addPixmap(selected_pm, QIcon.Mode.Selected)
        else:
            out.addPixmap(pm, QIcon.Mode.Normal)
            out.addPixmap(pm, QIcon.Mode.Active)
            out.addPixmap(_dim_pixmap(pm), QIcon.Mode.Disabled)
            out.addPixmap(pm, QIcon.Mode.Selected)
    return out if not out.isNull() else icon


def _resolve_theme_icon(name: str, *fallbacks: str, widget: QWidget | None = None) -> tuple[QIcon, str | None]:
    names = (name, *fallbacks)
    theme_lower = QIcon.themeName().lower()
    prefer_symbolic = theme_lower.startswith("adwaita")

    if prefer_symbolic:
        for n in names:
            if n in _COLORFUL_ICONS:
                if QIcon.hasThemeIcon(n):
                    return QIcon.fromTheme(n), n
                continue
            sym = n if n.endswith("-symbolic") else f"{n}-symbolic"
            if QIcon.hasThemeIcon(sym):
                return QIcon.fromTheme(sym), sym

    for n in names:
        if QIcon.hasThemeIcon(n):
            return QIcon.fromTheme(n), n

    for n in names:
        if not n.endswith("-symbolic"):
            sym = f"{n}-symbolic"
            if QIcon.hasThemeIcon(sym):
                return QIcon.fromTheme(sym), sym

    for n in names:
        if n in _STANDARD_ICONS:
            style = widget.style() if widget is not None else QApplication.style()
            if style is not None:
                return style.standardIcon(_STANDARD_ICONS[n]), n

    return QIcon(), None


def theme_icon(
    name: str,
    *fallbacks: str,
    widget: QWidget | None = None,
    monochrome: bool | None = None,
    prefer_color: bool = False,
    selected: bool = False,
) -> QIcon:
    """The desktop theme's icon `name` (or the first fallback the theme has), tinted for the active palette."""
    icon, resolved = _resolve_theme_icon(name, *fallbacks, widget=widget)
    if icon.isNull():
        return icon
    if monochrome is None and resolved is not None and resolved.endswith("-symbolic"):
        monochrome = True
    return tinted_icon(
        icon, _effective_palette(widget), monochrome=monochrome, prefer_color=prefer_color, selected=selected,
    )


def scaled_font(widget: QWidget, factor: float, bold: bool = False) -> QFont:
    font = QFont(widget.font())
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * factor)
    else:
        font.setPixelSize(round(font.pixelSize() * factor))
    font.setBold(bold)
    return font


class ElidedLabel(QLabel):
    """A one-line label that shortens long text with "…" (in the middle, for paths) instead of
    widening the window."""

    def __init__(self, text: str = "", parent: QWidget | None = None,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle) -> None:
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full = text
        self.setToolTip(text if text else "")
        self._elide()

    def fullText(self) -> str:  # noqa: N802
        return self._full

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = self.contentsRect().width()
        shown = QFontMetrics(self.font()).elidedText(self._full, self._mode, width) if width > 0 else self._full
        super().setText(shown)


class MessageBar(QFrame):
    """An inline message (information, warning or error) with an optional action button and a
    close button, in place of a modal box. Hidden while empty."""

    ICONS = {"info": "dialog-information", "warning": "dialog-warning", "error": "dialog-error"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("messageBar")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.AlternateBase)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 8, 8)
        lay.setSpacing(8)
        self._icon = QLabel(self)
        self._text = QLabel(self)
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._action = QPushButton(self)
        self._close = QToolButton(self)
        self._close.setAutoRaise(True)
        self._close.setToolTip("Close")
        self._close.clicked.connect(self.clear)
        lay.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self._text, 1)
        lay.addWidget(self._action, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)
        self._callback: Callable[[], None] | None = None
        self._action.clicked.connect(self._on_action)
        self.kind = ""
        self._apply_icons()
        self.clear()

    def _apply_icons(self) -> None:
        if not hasattr(self, "_close"):
            return
        self._close.setIcon(theme_icon("window-close", widget=self))
        if self.kind:
            size = self.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
            self._icon.setPixmap(
                theme_icon(self.ICONS.get(self.kind, "dialog-information"), widget=self).pixmap(size, size)
            )

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def show_message(self, kind: str, text: str, action: str = "",
                     callback: Callable[[], None] | None = None) -> None:
        self.kind = kind
        self._apply_icons()
        self._text.setText(text)
        self._action.setText(action)
        self._action.setVisible(bool(action))
        self._callback = callback
        self.show()

    def text(self) -> str:
        return "" if self.isHidden() else self._text.text()

    def action_button(self) -> QPushButton:
        return self._action

    def clear(self) -> None:
        self.kind = ""
        self._text.setText("")
        self._callback = None
        self.hide()

    def _on_action(self) -> None:
        if self._callback is not None:
            self._callback()
