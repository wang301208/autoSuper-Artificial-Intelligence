import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reasoning.multi_hop import MultiHopAssociator
from backend.knowledge import UnifiedKnowledgeBase
from backend.knowledge.unified import KnowledgeSource
from backend.memory import LongTermMemory
from backend.reflection import ReflectionModule


def test_unified_knowledge_base():
    kb = UnifiedKnowledgeBase()
    kb.add_source(KnowledgeSource(name="science", data={"atom": "basic unit"}))
    kb.add_source(KnowledgeSource(name="art", data={"atom": "indivisible style"}))
    result = kb.query("atom")
    assert result == {"science": "basic unit", "art": "indivisible style"}


def test_long_term_memory(tmp_path):
    db_path = tmp_path / "memory.db"
    memory = LongTermMemory(db_path)
    memory.add("dialogue", "hello")
    memory.add("task", "write code")
    assert list(memory.get("dialogue")) == ["hello"]
    assert sorted(memory.get()) == ["hello", "write code"]
    memory.close()


def test_multi_hop_associator():
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
    }
    assoc = MultiHopAssociator(graph)
    assert assoc.find_path("A", "C") == ["A", "B", "C"]


def test_reflection_module():
    module = ReflectionModule()
    evaluation, revised = module.reflect("test response")
    assert "response_length" in evaluation
    assert revised.endswith("[revised]")
