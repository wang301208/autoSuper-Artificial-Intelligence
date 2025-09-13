import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import json
import subprocess
from pathlib import Path
import asyncio
import time

import pytest

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
    retrieved_code, retrieved_meta = asyncio.run(lib.get_skill("hello"))

    assert "return 'hi'" in retrieved_code
    assert retrieved_meta == metadata

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert "Add skill hello" in log.stdout
    lib.close()


def test_meta_skill_auto_activation_and_caching(tmp_path: Path) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo)
    code = "def foo():\n    return 1\n"
    metadata = {
        "name": "MetaSkill_Test",
        "version": "1.0",
        "description": "test",
        "protected": False,
    }
    lib.add_skill("MetaSkill_Test", code, metadata)

    # Requesting the inactive meta-skill should transparently activate it.
    _, meta = asyncio.run(lib.get_skill("MetaSkill_Test"))
    assert meta["active"] is True
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert "Activate meta-skill MetaSkill_Test" in log.stdout

    # Repeated calls should use the cached, already activated version.
    _, meta2 = asyncio.run(lib.get_skill("MetaSkill_Test"))
    assert meta2["active"] is True
    stats = lib.cache_stats()
    assert stats["hits"] == 1
    log2 = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert log.stdout == log2.stdout
    lib.close()


def test_cache_eviction_and_reload(tmp_path: Path) -> None:
    repo = tmp_path
    init_repo(repo)
    persist = repo / "cache.sqlite"
    lib = SkillLibrary(repo, cache_size=2, persist_path=persist)
    for i in range(3):
        code = f"def s{i}():\n    return {i}\n"
        lib.add_skill(f"skill{i}", code, {"i": i})
        asyncio.run(lib.get_skill(f"skill{i}"))
    stats = lib.cache_stats()
    assert stats["misses"] == 3
    lib.close()

    lib2 = SkillLibrary(repo, cache_size=2, persist_path=persist)
    asyncio.run(lib2.get_skill("skill1"))
    asyncio.run(lib2.get_skill("skill2"))
    stats = lib2.cache_stats()
    assert stats["hits"] == 2
    asyncio.run(lib2.get_skill("skill0"))
    stats = lib2.cache_stats()
    assert stats["misses"] == 1
    lib2.close()


def test_cache_ttl_expiration(tmp_path: Path) -> None:
    repo = tmp_path
    init_repo(repo)
    lib = SkillLibrary(repo, cache_ttl=1)
    code = "def hello():\n    return 'hi'\n"
    lib.add_skill("hello", code, {})
    asyncio.run(lib.get_skill("hello"))
    time.sleep(1.1)
    asyncio.run(lib.get_skill("hello"))
    stats = lib.cache_stats()
    assert stats["misses"] == 2
    lib.close()
