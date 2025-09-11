import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repository root on path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[3]))
from backend.memory.long_term import LongTermMemory


def test_tag_filtering(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        mem.add("news", "item1", tags=["urgent", "important"])
        mem.add("news", "item2", tags=["routine"])
        assert list(mem.get(tags=["urgent"])) == ["item1"]
    finally:
        mem.close()


def test_time_range_query(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    now = datetime.utcnow().timestamp()
    try:
        mem.add("news", "old", timestamp=now - 10)
        mem.add("news", "current", timestamp=now)
        mem.add("news", "new", timestamp=now + 10)
        results = list(mem.get(start_ts=now - 1, end_ts=now + 1))
        assert results == ["current"]
    finally:
        mem.close()


def test_large_dataset_performance(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        for i in range(1000):
            tag = "target" if i == 999 else "other"
            mem.add("cat", f"content {i}", tags=[tag])
        start = time.time()
        result = list(mem.get(tags=["target"]))
        duration = time.time() - start
        assert result == ["content 999"]
        assert duration < 1.0
    finally:
        mem.close()


def test_embedding_storage(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        mem.add_embedding("key", [0.1, 0.2, 0.3], {"tag": "test"})
        stored = mem.get_embedding("key")
        assert stored is not None
        vector, meta = stored
        assert vector == [0.1, 0.2, 0.3]
        assert meta == {"tag": "test"}
    finally:
        mem.close()
