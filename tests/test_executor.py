import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from pathlib import Path
import subprocess

from capability.skill_library import SkillLibrary
from execution import Executor


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
