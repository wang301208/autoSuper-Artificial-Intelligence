from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple, List

import aiofiles
from cachetools import LRUCache, TTLCache


logger = logging.getLogger(__name__)


class SkillLibrary:
    """Store and retrieve skill source code and metadata in a Git repository."""

    def __init__(
        self,
        repo_path: str | Path,
        storage_dir: str = "skills",
        cache_size: int = 128,
        cache_ttl: int | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.storage_dir = self.repo_path / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._cache_ttl = cache_ttl
        if cache_ttl is not None:
            self._cache: "TTLCache[str, Tuple[str, Dict]]" = TTLCache(
                maxsize=cache_size, ttl=cache_ttl
            )
        else:
            self._cache: "LRUCache[str, Tuple[str, Dict]]" = LRUCache(maxsize=cache_size)

        self.hits = 0
        self.misses = 0

        self.persist_path = Path(persist_path or (self.storage_dir / "cache.sqlite"))
        self._init_persist()
        self._warm_cache()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _init_persist(self) -> None:
        self._db = sqlite3.connect(self.persist_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (name TEXT PRIMARY KEY, code TEXT, metadata TEXT, timestamp REAL)"
        )
        self._db.commit()

    def _save_to_persist(self, name: str, code: str, metadata: Dict) -> None:
        ts = time.time()
        self._db.execute(
            "INSERT OR REPLACE INTO cache (name, code, metadata, timestamp) VALUES (?, ?, ?, ?)",
            (name, code, json.dumps(metadata), ts),
        )
        self._db.commit()

    def _load_from_persist(self, name: str) -> Tuple[str, Dict] | None:
        cur = self._db.execute(
            "SELECT code, metadata, timestamp FROM cache WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if not row:
            return None
        code, metadata_json, ts = row
        if self._cache_ttl is not None and time.time() - ts > self._cache_ttl:
            self._db.execute("DELETE FROM cache WHERE name = ?", (name,))
            self._db.commit()
            return None
        return code, json.loads(metadata_json)

    def _warm_cache(self) -> None:
        cur = self._db.execute(
            "SELECT name, code, metadata, timestamp FROM cache ORDER BY timestamp"
        )
        rows = cur.fetchall()
        now = time.time()
        for name, code, metadata_json, ts in rows:
            if self._cache_ttl is not None and now - ts > self._cache_ttl:
                self._db.execute("DELETE FROM cache WHERE name = ?", (name,))
                continue
            self._cache[name] = (code, json.loads(metadata_json))
        self._db.commit()

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()

    # ------------------------------------------------------------------
    # Public API
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
        self.invalidate(name)

    async def _read_file(self, path: Path) -> str:
        """Read text from ``path`` asynchronously."""
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _load_skill(self, name: str) -> Tuple[str, Dict]:
        """Load a skill's source and metadata from disk with caching."""
        if name in self._cache:
            self.hits += 1
            return self._cache[name]

        self.misses += 1
        persisted = self._load_from_persist(name)
        if persisted:
            self._cache[name] = persisted
            return persisted

        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        code = await self._read_file(skill_file)
        metadata = json.loads(await self._read_file(meta_file))
        if name.startswith("MetaSkill_") and not metadata.get("active"):
            logger.warning(
                "Meta-skill %s requested while inactive; activating automatically.",
                name,
            )
            try:
                await self.activate_meta_skill(name)
            except Exception as err:  # pragma: no cover - best effort logging
                logger.error(
                    "Failed to auto-activate meta-skill %s: %s", name, err
                )
                raise PermissionError(
                    "Meta-skill version could not be activated"
                ) from err
            metadata["active"] = True
        self._cache[name] = (code, metadata)
        self._save_to_persist(name, code, metadata)
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
        self.invalidate(name)

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

    # ------------------------------------------------------------------
    # Cache utilities
    def cache_stats(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "currsize": self._cache.currsize,
            "maxsize": self._cache.maxsize,
        }

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._cache.clear()
            self._db.execute("DELETE FROM cache")
        else:
            self._cache.pop(name, None)
            self._db.execute("DELETE FROM cache WHERE name = ?", (name,))
        self._db.commit()

