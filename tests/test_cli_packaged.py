"""Desktop-integration sub-commands exist in a source checkout but not in a package install."""

from __future__ import annotations

import pytest

from semma import cli

INTEGRATION = [
    "install",
    "uninstall",
    "install-kde",
    "uninstall-kde",
    "kde-status",
    "install-gnome",
    "uninstall-gnome",
    "gnome-status",
]
EVERYWHERE = [["search", "x"], ["status"], ["ctl", "status"]]


def test_packaged_rejects_integration_commands(monkeypatch, capsys):
    monkeypatch.setattr("sys.base_prefix", "/usr/lib/semma/python")
    assert cli.is_packaged()
    parser = cli.build_parser()
    for name in INTEGRATION:
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([name])
        assert exc.value.code == 2, name
    capsys.readouterr()
    for argv in EVERYWHERE:
        assert callable(parser.parse_args(argv).func)


def test_other_prefix_accepts_integration_commands(monkeypatch):
    monkeypatch.setattr("sys.base_prefix", "/usr")
    assert not cli.is_packaged()
    parser = cli.build_parser()
    for name in INTEGRATION:
        assert callable(parser.parse_args([name]).func), name
    for argv in EVERYWHERE:
        assert callable(parser.parse_args(argv).func)
