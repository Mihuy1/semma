"""Small widgets shared by the pages."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QPalette, QResizeEvent
from PySide6.QtWidgets import (
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
}


def theme_icon(name: str, *fallbacks: str, widget: QWidget | None = None) -> QIcon:
    """The desktop theme's icon `name` (or the first fallback the theme has), else Qt's own."""
    for n in (name, *fallbacks):
        if QIcon.hasThemeIcon(n):
            return QIcon.fromTheme(n)
    for n in (name, *fallbacks):
        if n in _STANDARD_ICONS:
            style = widget.style() if widget is not None else None
            if style is None:
                from PySide6.QtWidgets import QApplication
                style = QApplication.style()
            return style.standardIcon(_STANDARD_ICONS[n])
    return QIcon()


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
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.AlternateBase)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 6, 6)
        self._icon = QLabel(self)
        self._text = QLabel(self)
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._action = QPushButton(self)
        self._close = QToolButton(self)
        self._close.setAutoRaise(True)
        self._close.setIcon(theme_icon("window-close", widget=self))
        self._close.setToolTip("Close")
        self._close.clicked.connect(self.clear)
        lay.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self._text, 1)
        lay.addWidget(self._action, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)
        self._callback: Callable[[], None] | None = None
        self._action.clicked.connect(self._on_action)
        self.kind = ""
        self.clear()

    def show_message(self, kind: str, text: str, action: str = "",
                     callback: Callable[[], None] | None = None) -> None:
        self.kind = kind
        size = self.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
        self._icon.setPixmap(theme_icon(self.ICONS.get(kind, "dialog-information"), widget=self).pixmap(size, size))
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
