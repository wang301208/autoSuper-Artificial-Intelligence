from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple, List

import aiofiles


class SkillLibrary:
    """Store and retrieve skill source code and metadata in a Git repository."""

    def __init__(
        self,
        repo_path: str | Path,
        storage_dir: str = "skills",
        cache_size: int = 128,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.storage_dir = self.repo_path / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache_size = cache_size
        self._cache: "OrderedDict[str, Tuple[str, Dict]]" = OrderedDict()

    def add_skill(self, name: str, code: str, metadata: Dict) -> None:
        """Add a skill to the library and commit the change to Git."""
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        skill_file.write_text(code, encoding="utf-8")
        if name.startswith("MetaSkill_") and "active" not in metadata:
            metadata["active"] = False
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(
            ["git", "add", str(skill_file), str(meta_file)],
            cwd=self.repo_path,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add skill {name}"],
            cwd=self.repo_path,
            check=True,
        )
        # Remove any stale cached entry for this skill.
        self._cache.pop(name, None)

    async def _read_file(self, path: Path) -> str:
        """Read text from ``path`` asynchronously."""
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _load_skill(self, name: str) -> Tuple[str, Dict]:
        """Load a skill's source and metadata from disk with caching."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        code = await self._read_file(skill_file)
        metadata = json.loads(await self._read_file(meta_file))
        if name.startswith("MetaSkill_") and not metadata.get("active"):
            raise PermissionError(
                "Meta-skill version not activated by System Architect"
            )
        self._cache[name] = (code, metadata)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return code, metadata

    async def get_skill(self, name: str) -> Tuple[str, Dict]:
        """Retrieve a skill's source code and metadata using an in-memory cache."""
        return await self._load_skill(name)

    async def activate_meta_skill(self, name: str) -> None:
        """Mark a meta-skill as active and commit the change to Git."""
        meta_file = self.storage_dir / f"{name}.json"
        metadata = json.loads(await self._read_file(meta_file))
        metadata["active"] = True
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(["git", "add", str(meta_file)], cwd=self.repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Activate meta-skill {name}"],
            cwd=self.repo_path,
            check=True,
        )
        # Ensure cache is invalidated so future reads get the updated metadata.
        self._cache.pop(name, None)

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
