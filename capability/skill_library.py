from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, List


class SkillLibrary:
    """Store and retrieve skill source code and metadata in a Git repository."""

    def __init__(self, repo_path: str | Path, storage_dir: str = "skills", buffer_size: int = 65536) -> None:
        self.repo_path = Path(repo_path)
        self.storage_dir = self.repo_path / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # Preallocate a reusable buffer to avoid repeated temporary allocations when
        # reading files from disk.
        self._read_buffer = bytearray(buffer_size)

    def add_skill(self, name: str, code: str, metadata: Dict) -> None:
        """Add a skill to the library and commit the change to Git."""
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        skill_file.write_text(code, encoding="utf-8")
        if name.startswith("MetaSkill_") and "active" not in metadata:
            metadata["active"] = False
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(["git", "add", str(skill_file), str(meta_file)], cwd=self.repo_path, check=True)
        subprocess.run(["git", "commit", "-m", f"Add skill {name}"], cwd=self.repo_path, check=True)
        # Clear cached entries so subsequent lookups see the new skill immediately.
        self._load_skill.cache_clear()

    def _read_file(self, path: Path) -> str:
        """Read text from ``path`` using a preallocated buffer."""
        size = path.stat().st_size
        if size > len(self._read_buffer):
            # Grow the buffer if needed for larger files.
            self._read_buffer = bytearray(size)
        with open(path, "rb") as f:
            n = f.readinto(self._read_buffer)
        return self._read_buffer[:n].decode("utf-8")

    @lru_cache(maxsize=128)
    def _load_skill(self, name: str) -> Tuple[str, Dict]:
        """Load a skill's source and metadata from disk with caching."""
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        code = self._read_file(skill_file)
        metadata = json.loads(self._read_file(meta_file))
        if name.startswith("MetaSkill_") and not metadata.get("active"):
            raise PermissionError("Meta-skill version not activated by System Architect")
        return code, metadata

    def get_skill(self, name: str) -> Tuple[str, Dict]:
        """Retrieve a skill's source code and metadata using an in-memory cache."""
        return self._load_skill(name)

    def activate_meta_skill(self, name: str) -> None:
        """Mark a meta-skill as active and commit the change to Git."""
        meta_file = self.storage_dir / f"{name}.json"
        metadata = json.loads(self._read_file(meta_file))
        metadata["active"] = True
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(["git", "add", str(meta_file)], cwd=self.repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Activate meta-skill {name}"],
            cwd=self.repo_path,
            check=True,
        )
        # Ensure cache is invalidated so future reads get the updated metadata.
        self._load_skill.cache_clear()

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
