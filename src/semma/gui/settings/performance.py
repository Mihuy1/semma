"""Settings › Performance: GPU pausing, model device, periodic rescan and desktop search results."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import theme_icon
from . import performance_model as pfm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext

_DEVICE_HELP_TOOLTIP = (
    "Automatic: graphics card (Vulkan) when available, otherwise processor (CPU).\n"
    "Graphics card: always Vulkan.\n"
    "Processor: always CPU.\n"
    "Click for details."
)

_DEVICE_HELP_HTML = (
    "<p style='margin: 0 0 6px 0;'><b>Automatic</b>: Uses your graphics card (Vulkan) when "
    "available, and falls back to your processor (CPU) if not.</p>"
    "<p style='margin: 0 0 6px 0;'><b>Graphics card</b>: Always runs on your graphics card "
    "via Vulkan (fails if no compatible GPU or driver is available).</p>"
    "<p style='margin: 0;'><b>Processor</b>: Always runs on your processor (CPU).</p>"
)


class _HelpPopover(QFrame):
    """Lightweight popup frame displaying wrapped help text anchored near a button."""

    def __init__(self, html: str, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.ColorRole.Window)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        self._label = QLabel(html, self)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(360)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._label)

    def text(self) -> str:
        return self._label.text()

    def show_near(self, anchor: QWidget) -> None:
        self.adjustSize()
        pos = anchor.mapToGlobal(QPoint(0, anchor.height() + 4))
        screen = anchor.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = min(max(pos.x(), avail.left()), max(avail.left(), avail.right() - self.width() + 1))
            y = pos.y()
            if y + self.height() > avail.bottom() + 1:
                y = max(avail.top(), anchor.mapToGlobal(QPoint(0, 0)).y() - self.height() - 4)
            pos = QPoint(x, y)
        self.move(pos)
        self.show()


class GraphicsCardGroup(QGroupBox):
    """Model device selector plus the GPU-busy pause switch and threshold."""

    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Graphics card and processor", parent)
        hl = QVBoxLayout(self)
        hl.setSpacing(8)

        dev_row = QHBoxLayout()
        dev_label = QLabel("Run the model on:", self)
        self.device_combo = QComboBox(self)
        self.populate_devices(None)
        self.device_combo.currentIndexChanged.connect(self.edited)
        dev_label.setBuddy(self.device_combo)

        self.device_help_button = QToolButton(self)
        self.device_help_button.setAutoRaise(True)
        self.device_help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.device_help_button.setAccessibleName("About model device options")
        self.device_help_button.setToolTip(_DEVICE_HELP_TOOLTIP)
        self.device_help_button.clicked.connect(self._toggle_device_help)
        self.device_help_popover = _HelpPopover(_DEVICE_HELP_HTML, self)
        self._apply_icons()

        dev_row.addWidget(dev_label)
        dev_row.addWidget(self.device_combo)
        dev_row.addWidget(self.device_help_button)
        dev_row.addStretch(1)
        hl.addLayout(dev_row)

        hl.addSpacing(4)

        self.pause_when_gpu_busy = QCheckBox(
            "Pause indexing while other programs use the graphics card", self
        )
        self.pause_when_gpu_busy.setChecked(pfm.DEFAULT_PAUSE_WHEN_GPU_BUSY)
        self.pause_when_gpu_busy.toggled.connect(self._on_pause_toggled)
        hl.addWidget(self.pause_when_gpu_busy)

        gpu_row = QHBoxLayout()
        gpu_row.setContentsMargins(24, 0, 0, 0)
        self.gpu_busy_label = QLabel("when they use more than", self)
        self.gpu_busy_threshold = QSpinBox(self)
        self.gpu_busy_threshold.setRange(pfm.MIN_GPU_BUSY_THRESHOLD, pfm.MAX_GPU_BUSY_THRESHOLD)
        self.gpu_busy_threshold.setValue(pfm.DEFAULT_GPU_BUSY_THRESHOLD)
        self.gpu_busy_threshold.setSuffix("%")
        self.gpu_busy_threshold.valueChanged.connect(self.edited)
        self.gpu_busy_label.setBuddy(self.gpu_busy_threshold)
        gpu_row.addWidget(self.gpu_busy_label)
        gpu_row.addWidget(self.gpu_busy_threshold)
        gpu_row.addStretch(1)
        hl.addLayout(gpu_row)

    def populate_devices(self, current_device: str | None) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        choices = pfm.device_choices(current_device)
        norm = pfm.normalise_device(current_device)
        selected_idx = 0
        for idx, (label, val) in enumerate(choices):
            self.device_combo.addItem(label, val)
            if val == norm:
                selected_idx = idx
        self.device_combo.setCurrentIndex(selected_idx)
        self.device_combo.blockSignals(False)

    def load(self, values: pfm.PerformanceValues) -> None:
        self.populate_devices(values.device)
        self.pause_when_gpu_busy.blockSignals(True)
        self.pause_when_gpu_busy.setChecked(values.pause_when_gpu_busy)
        self.pause_when_gpu_busy.blockSignals(False)
        self._set_gpu_threshold_enabled(values.pause_when_gpu_busy)
        self.gpu_busy_threshold.blockSignals(True)
        self.gpu_busy_threshold.setValue(values.gpu_busy_threshold)
        self.gpu_busy_threshold.blockSignals(False)

    def selected_device(self) -> str | None:
        return pfm.normalise_device(self.device_combo.currentData())

    def set_device(self, device: str | None) -> None:
        norm = pfm.normalise_device(device)
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == norm:
                self.device_combo.setCurrentIndex(i)
                return
        self.populate_devices(norm)
        self.edited.emit()

    def apply_to(self, values: pfm.PerformanceValues) -> pfm.PerformanceValues:
        out = values.copy()
        out.device = self.selected_device()
        out.pause_when_gpu_busy = self.pause_when_gpu_busy.isChecked()
        out.gpu_busy_threshold = self.gpu_busy_threshold.value()
        return out

    def _set_gpu_threshold_enabled(self, enabled: bool) -> None:
        self.gpu_busy_label.setEnabled(enabled)
        self.gpu_busy_threshold.setEnabled(enabled)

    def _on_pause_toggled(self, checked: bool) -> None:
        self._set_gpu_threshold_enabled(checked)
        self.edited.emit()

    def _apply_icons(self) -> None:
        if not hasattr(self, "device_help_button"):
            return
        icon = theme_icon("help-contextual", "dialog-question", "help-about", widget=self)
        if not icon.isNull():
            self.device_help_button.setIcon(icon)
            self.device_help_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        else:
            self.device_help_button.setText("?")

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_icons()

    def _toggle_device_help(self) -> None:
        if self.device_help_popover.isVisible():
            self.device_help_popover.hide()
        else:
            self.device_help_popover.show_near(self.device_help_button)


class PerformanceSection(SettingsSection):
    key = "performance"
    title = "Performance"
    icon = "preferences-system-performance"
    icon_fallbacks = (
        "preferences-devices-cpu",
        "utilities-system-monitor",
        "speedometer",
        "preferences-system-symbolic",
        "applications-system-symbolic",
        "computer-symbolic",
        "preferences-system",
    )
    description = "How Semma uses the graphics card, processor and background scans."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = pfm.PerformanceValues()

        self.hw_box = GraphicsCardGroup(self)
        self.device_combo = self.hw_box.device_combo
        self.device = self.device_combo
        self.device_help_button = self.hw_box.device_help_button
        self.device_help_popover = self.hw_box.device_help_popover
        self.pause_when_gpu_busy = self.hw_box.pause_when_gpu_busy
        self.gpu_busy_label = self.hw_box.gpu_busy_label
        self.gpu_busy_threshold = self.hw_box.gpu_busy_threshold
        self.hw_box.edited.connect(self._edited)

        scan_box = QGroupBox("Scanning and desktop search", self)
        sl = QVBoxLayout(scan_box)
        sl.setSpacing(8)

        rescan_row = QHBoxLayout()
        rescan_prefix = QLabel("Check all folders again every", scan_box)
        self.rescan_hours = QSpinBox(scan_box)
        self.rescan_hours.setRange(pfm.MIN_RESCAN_HOURS, pfm.MAX_RESCAN_HOURS)
        self.rescan_hours.setSuffix(" hours")
        self.rescan_hours.valueChanged.connect(self._edited)
        rescan_prefix.setBuddy(self.rescan_hours)
        rescan_row.addWidget(rescan_prefix)
        rescan_row.addWidget(self.rescan_hours)
        rescan_row.addStretch(1)
        sl.addLayout(rescan_row)

        results_row = QHBoxLayout()
        results_label = QLabel("Results shown in KRunner and GNOME Shell:", scan_box)
        self.krunner_max_results = QSpinBox(scan_box)
        self.krunner_max_results.setRange(pfm.MIN_KRUNNER_MAX_RESULTS, pfm.MAX_KRUNNER_MAX_RESULTS)
        self.krunner_max_results.valueChanged.connect(self._edited)
        results_label.setBuddy(self.krunner_max_results)
        results_row.addWidget(results_label)
        results_row.addWidget(self.krunner_max_results)
        results_row.addStretch(1)
        sl.addLayout(results_row)

        self.body.addWidget(self.hw_box)
        self.body.addWidget(scan_box)

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = pfm.load(ctx)
        self.hw_box.load(self._saved)
        for spin, val in (
            (self.rescan_hours, self._saved.rescan_hours),
            (self.krunner_max_results, self._saved.krunner_max_results),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def selected_device(self) -> str | None:
        return self.hw_box.selected_device()

    def set_device(self, device: str | None) -> None:
        self.hw_box.set_device(device)

    def values(self) -> pfm.PerformanceValues:
        return self.hw_box.apply_to(
            pfm.PerformanceValues(
                rescan_hours=self.rescan_hours.value(),
                krunner_max_results=self.krunner_max_results.value(),
            )
        )

    def changes(self) -> Changes:
        return pfm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return pfm.problems(self.values(), self.ctx) if self.ctx is not None else []

    def _edited(self, *_args: object) -> None:
        self.edited.emit()
