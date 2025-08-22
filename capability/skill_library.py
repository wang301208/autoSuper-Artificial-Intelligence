from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List


class SkillLibrary:
    """Store and retrieve skill source code and metadata in a Git repository."""

    def __init__(self, repo_path: str | Path, storage_dir: str = "skills") -> None:
        self.repo_path = Path(repo_path)
        self.storage_dir = self.repo_path / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def add_skill(self, name: str, code: str, metadata: Dict) -> None:
        """Add a skill to the library and commit the change to Git."""
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        skill_file.write_text(code, encoding="utf-8")
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(["git", "add", str(skill_file), str(meta_file)], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", f"Add skill {name}"], cwd=self.repo_path, check=True)

    def get_skill(self, name: str) -> Tuple[str, Dict]:
        """Retrieve a skill's source code and metadata."""
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        code = skill_file.read_text(encoding="utf-8")
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        return code, metadata

    def list_skills(self) -> List[str]:
        """List all available skills."""
        return [p.stem for p in self.storage_dir.glob("*.py")]

    def history(self, name: str) -> str:
        """Return the Git commit history for a skill file."""
        skill_file = self.storage_dir / f"{name}.py"
        result = subprocess.run(
            ["git", "log", "--", str(skill_file)],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
