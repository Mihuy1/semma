"""Settings › Performance: GPU pausing, model device, periodic rescan and desktop search results."""

from __future__ import annotations

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import performance_model as pfm
from .base import SettingsSection
from .context import Changes, Problem, SettingsContext


class PerformanceSection(SettingsSection):
    key = "performance"
    title = "Performance"
    icon = "speedometer"
    description = "How Semma uses the graphics card, processor and background scans."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._saved = pfm.PerformanceValues()

        # Model & GPU group
        hw_box = QGroupBox("Graphics card and processor", self)
        hl = QVBoxLayout(hw_box)

        dev_row = QHBoxLayout()
        dev_label = QLabel("Run the model on:", hw_box)
        self.device_combo = QComboBox(hw_box)
        self.device = self.device_combo
        self._populate_devices(None)
        self.device_combo.currentIndexChanged.connect(self._edited)
        dev_label.setBuddy(self.device_combo)
        dev_row.addWidget(dev_label)
        dev_row.addWidget(self.device_combo)
        dev_row.addStretch(1)
        hl.addLayout(dev_row)

        dev_note = QLabel("Changing this setting requires a service restart.", hw_box)
        dev_note.setWordWrap(True)
        dev_note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        hl.addWidget(dev_note)

        hl.addSpacing(4)

        self.pause_when_gpu_busy = QCheckBox(
            "Pause indexing while other programs use the graphics card", hw_box
        )
        self.pause_when_gpu_busy.setChecked(pfm.DEFAULT_PAUSE_WHEN_GPU_BUSY)
        self.pause_when_gpu_busy.toggled.connect(self._on_pause_toggled)
        hl.addWidget(self.pause_when_gpu_busy)

        gpu_row = QHBoxLayout()
        gpu_row.setContentsMargins(24, 0, 0, 0)
        self.gpu_busy_label = QLabel("when they use more than", hw_box)
        self.gpu_busy_threshold = QSpinBox(hw_box)
        self.gpu_busy_threshold.setRange(pfm.MIN_GPU_BUSY_THRESHOLD, pfm.MAX_GPU_BUSY_THRESHOLD)
        self.gpu_busy_threshold.setValue(pfm.DEFAULT_GPU_BUSY_THRESHOLD)
        self.gpu_busy_threshold.setSuffix("%")
        self.gpu_busy_threshold.valueChanged.connect(self._edited)
        self.gpu_busy_label.setBuddy(self.gpu_busy_threshold)
        gpu_row.addWidget(self.gpu_busy_label)
        gpu_row.addWidget(self.gpu_busy_threshold)
        gpu_row.addStretch(1)
        hl.addLayout(gpu_row)

        # Scanning & search group
        scan_box = QGroupBox("Scanning and desktop search", self)
        sl = QVBoxLayout(scan_box)

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

        self.body.addWidget(hw_box)
        self.body.addWidget(scan_box)

    def _populate_devices(self, current_device: str | None) -> None:
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

    def load(self, ctx: SettingsContext) -> None:
        super().load(ctx)
        self._saved = pfm.load(ctx)
        self._populate_devices(self._saved.device)
        self.pause_when_gpu_busy.blockSignals(True)
        self.pause_when_gpu_busy.setChecked(self._saved.pause_when_gpu_busy)
        self.pause_when_gpu_busy.blockSignals(False)
        self._set_gpu_threshold_enabled(self._saved.pause_when_gpu_busy)
        for spin, val in (
            (self.gpu_busy_threshold, self._saved.gpu_busy_threshold),
            (self.rescan_hours, self._saved.rescan_hours),
            (self.krunner_max_results, self._saved.krunner_max_results),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def selected_device(self) -> str | None:
        return pfm.normalise_device(self.device_combo.currentData())

    def set_device(self, device: str | None) -> None:
        norm = pfm.normalise_device(device)
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == norm:
                self.device_combo.setCurrentIndex(i)
                return
        self._populate_devices(norm)
        self.edited.emit()

    def values(self) -> pfm.PerformanceValues:
        return pfm.PerformanceValues(
            gpu_busy_threshold=self.gpu_busy_threshold.value(),
            device=self.selected_device(),
            rescan_hours=self.rescan_hours.value(),
            krunner_max_results=self.krunner_max_results.value(),
            pause_when_gpu_busy=self.pause_when_gpu_busy.isChecked(),
        )

    def changes(self) -> Changes:
        return pfm.changes(self._saved, self.values())

    def problems(self) -> list[Problem]:
        return pfm.problems(self.values(), self.ctx) if self.ctx is not None else []

    def _set_gpu_threshold_enabled(self, enabled: bool) -> None:
        self.gpu_busy_label.setEnabled(enabled)
        self.gpu_busy_threshold.setEnabled(enabled)

    def _on_pause_toggled(self, checked: bool) -> None:
        self._set_gpu_threshold_enabled(checked)
        self._edited()

    def _edited(self, *_args: object) -> None:
        self.edited.emit()
