"""The Performance section's settings (daemon.pause_when_gpu_busy,
daemon.gpu_busy_threshold, model.device, daemon.rescan_hours,
daemon.krunner_max_results) as plain values. No Qt here."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...config import _is_valid_device_spec
from .context import Changes, Problem, SettingsContext

DAEMON_SECTION = "daemon"
MODEL_SECTION = "model"

DEFAULT_PAUSE_WHEN_GPU_BUSY = True
DEFAULT_GPU_BUSY_THRESHOLD = 30
MIN_GPU_BUSY_THRESHOLD = 1
MAX_GPU_BUSY_THRESHOLD = 100

DEFAULT_DEVICE: str | None = None
STANDARD_DEVICES: tuple[tuple[str, str | None], ...] = (
    ("Automatic", None),
    ("Graphics card", "gpu"),
    ("Processor", "cpu"),
)
STANDARD_DEVICE_VALUES: frozenset[str | None] = frozenset(v for _, v in STANDARD_DEVICES)

DEFAULT_RESCAN_HOURS = 24
MIN_RESCAN_HOURS = 1
MAX_RESCAN_HOURS = 720

DEFAULT_KRUNNER_MAX_RESULTS = 10
MIN_KRUNNER_MAX_RESULTS = 1
MAX_KRUNNER_MAX_RESULTS = 100


def normalise_device(device: str | None) -> str | None:
    if device is None:
        return None
    val = device.strip().lower()
    if val in ("", "auto"):
        return None
    return val


def device_choices(current_device: str | None = None) -> list[tuple[str, str | None]]:
    """Combo-box (label, value) pairs, adding `current_device` when it is a custom spec like
    'vulkan:1' or 'cuda'."""
    choices = list(STANDARD_DEVICES)
    norm = normalise_device(current_device)
    if norm not in STANDARD_DEVICE_VALUES:
        choices.append((norm or "", norm))
    return choices


@dataclass
class PerformanceValues:
    gpu_busy_threshold: int = DEFAULT_GPU_BUSY_THRESHOLD
    device: str | None = DEFAULT_DEVICE
    rescan_hours: int = DEFAULT_RESCAN_HOURS
    krunner_max_results: int = DEFAULT_KRUNNER_MAX_RESULTS
    pause_when_gpu_busy: bool = DEFAULT_PAUSE_WHEN_GPU_BUSY

    def copy(self) -> PerformanceValues:
        return replace(self)


DEFAULTS = PerformanceValues()


def load(ctx: SettingsContext) -> PerformanceValues:
    cfg = ctx.config
    threshold = max(MIN_GPU_BUSY_THRESHOLD, min(MAX_GPU_BUSY_THRESHOLD, round(cfg.gpu_busy_threshold)))
    rescan = max(MIN_RESCAN_HOURS, round(cfg.rescan_hours))
    max_results = max(MIN_KRUNNER_MAX_RESULTS, int(cfg.krunner_max_results))
    return PerformanceValues(
        gpu_busy_threshold=threshold,
        device=normalise_device(cfg.device),
        rescan_hours=rescan,
        krunner_max_results=max_results,
        pause_when_gpu_busy=bool(cfg.pause_when_gpu_busy),
    )


def changes(saved: PerformanceValues, edited: PerformanceValues) -> Changes:
    """Only keys the user changed; returning a key to its default yields None."""
    daemon_out: dict[str, object] = {}
    model_out: dict[str, object] = {}

    old_pause = bool(saved.pause_when_gpu_busy)
    new_pause = bool(edited.pause_when_gpu_busy)
    if new_pause != old_pause:
        daemon_out["pause_when_gpu_busy"] = (
            None if new_pause is DEFAULT_PAUSE_WHEN_GPU_BUSY else new_pause
        )

    old_thresh = round(float(saved.gpu_busy_threshold))
    new_thresh = round(float(edited.gpu_busy_threshold))
    if new_thresh != old_thresh:
        daemon_out["gpu_busy_threshold"] = (
            None if new_thresh == DEFAULT_GPU_BUSY_THRESHOLD else new_thresh
        )

    old_rescan = round(float(saved.rescan_hours))
    new_rescan = round(float(edited.rescan_hours))
    if new_rescan != old_rescan:
        daemon_out["rescan_hours"] = None if new_rescan == DEFAULT_RESCAN_HOURS else new_rescan

    old_results = int(saved.krunner_max_results)
    new_results = int(edited.krunner_max_results)
    if new_results != old_results:
        daemon_out["krunner_max_results"] = (
            None if new_results == DEFAULT_KRUNNER_MAX_RESULTS else new_results
        )

    old_dev = normalise_device(saved.device)
    new_dev = normalise_device(edited.device)
    if new_dev != old_dev:
        model_out["device"] = None if new_dev == DEFAULT_DEVICE else new_dev

    out: Changes = {}
    if model_out:
        out[MODEL_SECTION] = model_out
    if daemon_out:
        out[DAEMON_SECTION] = daemon_out
    return out


def problems(values: PerformanceValues, ctx: SettingsContext | None = None) -> list[Problem]:
    out: list[Problem] = []
    if not MIN_GPU_BUSY_THRESHOLD <= values.gpu_busy_threshold <= MAX_GPU_BUSY_THRESHOLD:
        out.append(
            Problem(
                f"The GPU threshold must be between {MIN_GPU_BUSY_THRESHOLD}% and {MAX_GPU_BUSY_THRESHOLD}%.",
                blocking=True,
            )
        )
    dev = normalise_device(values.device)
    if dev is not None and not _is_valid_device_spec(dev):
        out.append(Problem(f"Unknown model device {values.device!r}.", blocking=True))
    if values.rescan_hours < MIN_RESCAN_HOURS:
        out.append(Problem(f"The rescan interval must be at least {MIN_RESCAN_HOURS} hour.", blocking=True))
    if values.krunner_max_results < MIN_KRUNNER_MAX_RESULTS:
        out.append(Problem("At least 1 search result must be shown.", blocking=True))
    return out
