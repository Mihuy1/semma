"""GPU activity monitor using nvidia-smi to pause indexing while foreign GPU workloads run."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterable

DEFAULT_IGNORED_COMMANDS: frozenset[str] = frozenset({
    "kwin_wayland",
    "Xwayland",
    "plasmashell",
})

DEFAULT_IGNORED_MAX_SM: float = 50.0
UNAVAILABLE_TTL_SECONDS: float = 600.0
COMMAND_TIMEOUT_SECONDS: float = 5.0


def _default_own_pids() -> set[int]:
    return {os.getpid()}


def parse_pmon_output(
    output: str,
    *,
    own_pids: set[int],
    ignored_commands: frozenset[str] = DEFAULT_IGNORED_COMMANDS,
    ignored_max_sm: float = DEFAULT_IGNORED_MAX_SM,
    gpu_index: int = 0,
) -> float:
    """Parse `nvidia-smi pmon -c 1 -s u` output and sum foreign SM % on *gpu_index*."""
    total_sm = 0.0
    cmd_col = 9

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line.lstrip("#").split()
            if "gpu" in header and "pid" in header and "sm" in header and "command" in header:
                cmd_col = header.index("command")
            continue

        parts = line.split(None, cmd_col)
        if len(parts) < 4:
            continue

        try:
            gpu = int(parts[0])
        except ValueError:
            continue
        if gpu != gpu_index:
            continue

        pid_str = parts[1]
        if pid_str == "-":
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue

        if pid in own_pids:
            continue

        sm_str = parts[3]
        if sm_str == "-":
            sm = 0.0
        else:
            try:
                sm = float(sm_str)
            except ValueError:
                continue

        cmd_raw = (parts[cmd_col] if len(parts) > cmd_col else parts[-1]).strip()
        cmd_token = cmd_raw.split()[0] if cmd_raw else ""
        cmd_base = os.path.basename(cmd_token)

        if (cmd_raw in ignored_commands or cmd_base in ignored_commands) and sm <= ignored_max_sm:
            continue

        total_sm += sm

    return total_sm


class GpuMonitor:
    """Monitor NVIDIA GPU SM utilization of non-Semma processes."""

    def __init__(
        self,
        own_pids: Callable[[], set[int]] | None = None,
        threshold: float = 30.0,
        *,
        gpu_index: int = 0,
        ignored_commands: Iterable[str] = DEFAULT_IGNORED_COMMANDS,
        ignored_max_sm: float = DEFAULT_IGNORED_MAX_SM,
        unavailable_ttl: float = UNAVAILABLE_TTL_SECONDS,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self.own_pids: Callable[[], set[int]] = own_pids or _default_own_pids
        self.threshold = float(threshold)
        self.gpu_index = int(gpu_index)
        self.ignored_commands: frozenset[str] = frozenset(ignored_commands)
        self.ignored_max_sm = float(ignored_max_sm)
        self.unavailable_ttl = float(unavailable_ttl)
        self.timeout = float(timeout)
        self._unavailable_until: float | None = None

    def _is_unavailable(self) -> bool:
        if self._unavailable_until is None:
            return False
        if time.monotonic() < self._unavailable_until:
            return True
        self._unavailable_until = None
        return False

    def _mark_unavailable(self) -> None:
        self._unavailable_until = time.monotonic() + self.unavailable_ttl

    def _run_nvidia_smi(self, args: list[str]) -> str | None:
        if self._is_unavailable():
            return None
        try:
            proc = subprocess.run(
                ["nvidia-smi", *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._mark_unavailable()
            return None

        if proc.returncode != 0:
            self._mark_unavailable()
            return None
        return proc.stdout

    def other_load(self) -> float | None:
        """Return sum of SM % across non-own processes on the monitored GPU, or None if unavailable."""
        stdout = self._run_nvidia_smi(["pmon", "-c", "1", "-s", "u"])
        if stdout is None:
            return None
        try:
            own = set(self.own_pids())
        except Exception:
            own = {os.getpid()}
        return parse_pmon_output(
            stdout,
            own_pids=own,
            ignored_commands=self.ignored_commands,
            ignored_max_sm=self.ignored_max_sm,
            gpu_index=self.gpu_index,
        )

    def busy(self, threshold: float | None = None) -> bool:
        """Return True if foreign GPU SM load meets or exceeds *threshold* (False if unavailable)."""
        limit = self.threshold if threshold is None else float(threshold)
        load = self.other_load()
        if load is None:
            return False
        return load >= limit

    def gpu_utilization(self) -> float | None:
        """Query overall GPU utilization % via `nvidia-smi --query-gpu=utilization.gpu`."""
        stdout = self._run_nvidia_smi(
            ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        )
        if stdout is None:
            return None
        valid_lines: list[float] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                valid_lines.append(float(line.split(",")[0].strip()))
            except ValueError:
                continue
        if self.gpu_index < 0 or self.gpu_index >= len(valid_lines):
            return None
        return valid_lines[self.gpu_index]
