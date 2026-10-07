import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    (tmp_path / "test_other.py").write_text("from calc import add\n\n\ndef test_zero():\n    assert add(0, 0) == 0\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git = ["git", "-c", "user.email=a@b.c", "-c", "user.name=n"]
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run([*git, *cmd], cwd=tmp_path, check=True)
    return tmp_path
