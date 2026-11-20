from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import lint_comments


def test_allows_directives_docstrings_and_hash_in_strings(tmp_path: pathlib.Path) -> None:
    sample = tmp_path / "clean.py"
    sample.write_text(
        '#!/usr/bin/env python3\n'
        '"""Module docstring with # inside."""\n'
        'x = "not # a comment"\n'
        'y = 1  # noqa: E501\n'
        'z: int = 2  # type: ignore[assignment]\n',
        encoding="utf-8",
    )
    baseline = tmp_path / ".comment-baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")

    errors, counts = lint_comments.check_ratchet(tmp_path, baseline)
    assert errors == []
    assert counts == {}


def test_rejects_new_prose_comment(tmp_path: pathlib.Path) -> None:
    sample = tmp_path / "bad.py"
    sample.write_text("x = 1  # explains nothing\n", encoding="utf-8")
    baseline = tmp_path / ".comment-baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")

    errors, counts = lint_comments.check_ratchet(tmp_path, baseline)
    assert counts == {"bad.py": 1}
    assert any("bad.py: found 1 prose comment(s), allowed 0" in e for e in errors)


def test_ratchet_allows_baselined_and_rejects_increase_or_stale(tmp_path: pathlib.Path) -> None:
    sample = tmp_path / "legacy.py"
    sample.write_text("# first\n# second\nx = 1\n", encoding="utf-8")
    baseline = tmp_path / ".comment-baseline.json"

    assert lint_comments.main(["--root", str(tmp_path), "--update-baseline"]) == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {"legacy.py": 2}
    assert lint_comments.main(["--root", str(tmp_path)]) == 0

    sample.write_text("# first\n# second\n# third\nx = 1\n", encoding="utf-8")
    assert lint_comments.main(["--root", str(tmp_path)]) == 1

    sample.write_text("# first\nx = 1\n", encoding="utf-8")
    assert lint_comments.main(["--root", str(tmp_path)]) == 1

    assert lint_comments.main(["--root", str(tmp_path), "--update-baseline"]) == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {"legacy.py": 1}
    assert lint_comments.main(["--root", str(tmp_path)]) == 0


def test_shell_shebang_and_heredoc_preserved(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# shellcheck disable=SC2034\n"
        "cat <<'EOF'\n"
        "# inside heredoc\n"
        "EOF\n",
        encoding="utf-8",
    )
    baseline = tmp_path / ".comment-baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")

    errors, counts = lint_comments.check_ratchet(tmp_path, baseline)
    assert errors == []
    assert counts == {}
