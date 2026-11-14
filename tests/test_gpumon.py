from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from semma.gpumon import GpuMonitor, parse_pmon_output

SAMPLE_PMON = """\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0      15152   C+G     12      5      -      -      -      -    kwin_wayland
    0      15179     G      4      2      -      -      -      -    plasmashell
    0      15200     G      8      1      -      -      -      -    /usr/bin/Xwayland
    0      22000     C     45     30      -      -      -      -    python3
    0      33000   C+G     38     20      -      -      -      -    steam_game
    1      44000   C+G     95     80      -      -      -      -    other_gpu_job
"""


def test_parse_pmon_excludes_own_pids_compositors_and_other_gpus() -> None:
    # pid 22000 is our own pid; kwin_wayland (12%), plasmashell (4%), Xwayland (8%) are <= 50%;
    # GPU 1 (95%) is ignored on GPU 0; only steam_game (38%) counts.
    load = parse_pmon_output(SAMPLE_PMON, own_pids={22000})
    assert load == pytest.approx(38.0)

    # If pid 22000 is NOT in own_pids, python3 (45%) + steam_game (38%) = 83%
    load_all = parse_pmon_output(SAMPLE_PMON, own_pids=set())
    assert load_all == pytest.approx(83.0)

    # Querying GPU 1 returns 95%
    load_gpu1 = parse_pmon_output(SAMPLE_PMON, own_pids={22000}, gpu_index=1)
    assert load_gpu1 == pytest.approx(95.0)


def test_compositor_above_max_sm_is_counted_and_ignore_list_configurable() -> None:
    pmon_high_kwin = """\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0      15152   C+G     65     10      -      -      -      -    kwin_wayland
    0      15179     G      -      -      -      -      -      -    plasmashell
"""
    assert parse_pmon_output(pmon_high_kwin, own_pids=set()) == pytest.approx(65.0)

    # Custom ignore list
    custom = """\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
    0       1000     G     25      5      -      -      -      -    custom_shell
"""
    assert parse_pmon_output(
        custom, own_pids=set(), ignored_commands=frozenset({"custom_shell"})
    ) == pytest.approx(0.0)


def test_malformed_and_idle_lines_skipped() -> None:
    messy = """\
# comment line
    0          -     -      -      -      -      -      -      -    -
malformed line
    0       notapid  G     40      -      -      -      -      -    badpid
    0       5000     G   badsm     -      -      -      -      -    badsm
    0       6000     C     15      -      -      -      -      -    worker
"""
    assert parse_pmon_output(messy, own_pids=set()) == pytest.approx(15.0)


def test_gpu_monitor_busy_and_own_pids(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):
        assert cmd == ["nvidia-smi", "pmon", "-c", "1", "-s", "u"]
        return SimpleNamespace(returncode=0, stdout=SAMPLE_PMON, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    mon = GpuMonitor(own_pids=lambda: {22000}, threshold=30.0)
    assert mon.other_load() == pytest.approx(38.0)
    assert mon.busy() is True
    assert mon.busy(threshold=50.0) is False

    mon_both_own = GpuMonitor(own_pids=lambda: {22000, 33000}, threshold=30.0)
    assert mon_both_own.other_load() == pytest.approx(0.0)
    assert mon_both_own.busy() is False


def test_unavailable_caching_on_missing_or_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def failing_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", failing_run)

    mon = GpuMonitor(unavailable_ttl=600.0)
    assert mon.other_load() is None
    assert mon.busy() is False
    assert mon.other_load() is None
    assert mon.gpu_utilization() is None
    # Should have only spawned once because unavailability is cached for 10 min
    assert calls == 1


def test_timeout_marks_unavailable_until_ttl_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    should_timeout = True

    def fake_run(cmd, **kwargs):
        nonlocal calls
        calls += 1
        if should_timeout:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)
        return SimpleNamespace(returncode=0, stdout="42\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    now = [1000.0]
    monkeypatch.setattr("semma.gpumon.time.monotonic", lambda: now[0])

    mon = GpuMonitor(unavailable_ttl=600.0)
    assert mon.other_load() is None
    assert calls == 1

    # 5 minutes later: still cached as unavailable
    now[0] += 300.0
    assert mon.gpu_utilization() is None
    assert calls == 1

    # 10 minutes + 1s later: cache expires and nvidia-smi is tried again
    should_timeout = False
    now[0] += 301.0
    assert mon.gpu_utilization() == pytest.approx(42.0)
    assert calls == 2


def test_gpu_utilization_multi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):
        assert cmd == [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        return SimpleNamespace(returncode=0, stdout="17\n88\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert GpuMonitor(gpu_index=0).gpu_utilization() == pytest.approx(17.0)
    assert GpuMonitor(gpu_index=1).gpu_utilization() == pytest.approx(88.0)
    assert GpuMonitor(gpu_index=2).gpu_utilization() is None
