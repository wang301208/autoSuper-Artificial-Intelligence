import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from pathlib import Path
import subprocess

import logging
import pytest

from capability.skill_library import SkillLibrary
from execution import Executor
from execution.executor import SkillExecutionError


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)


def test_executor_flow(tmp_path: Path) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo)

    lib.add_skill("hello", "def hello():\n    return 'hi'\n", {"lang": "python"})
    lib.add_skill("goodbye", "def goodbye():\n    return 'bye'\n", {"lang": "python"})

    executor = Executor(lib)
    results = executor.execute("hello then goodbye")

    assert list(results.keys()) == ["hello", "goodbye"]
    assert results["hello"] == "hi"
    assert results["goodbye"] == "bye"


def test_call_skill_logs_exception(tmp_path: Path, caplog) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo)
    lib.add_skill(
        "fail",
        "def fail():\n    raise RuntimeError('boom')\n",
        {"lang": "python"},
    )

    executor = Executor(lib)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SkillExecutionError) as exc_info:
            executor._call_skill("local", "fail")

    assert "fail" in str(exc_info.value)
    assert "boom" in str(exc_info.value)
    assert any(
        "fail" in record.message and "boom" in record.message
        for record in caplog.records
    )
