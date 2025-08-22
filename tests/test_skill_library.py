import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import json
import subprocess
from pathlib import Path

from capability.skill_library import SkillLibrary


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)


def test_add_and_get_skill(tmp_path: Path) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo)
    code = "def hello():\n    return 'hi'\n"
    metadata = {"lang": "python"}

    lib.add_skill("hello", code, metadata)
    retrieved_code, retrieved_meta = lib.get_skill("hello")

    assert "return 'hi'" in retrieved_code
    assert retrieved_meta == metadata

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert "Add skill hello" in log.stdout
