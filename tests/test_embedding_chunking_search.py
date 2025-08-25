
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'autogpts' / 'autogpt'))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'autogpts' / 'autogpt' / 'memory' / 'vector'))

import asyncio
import importlib.util, pathlib as _p
spec = importlib.util.spec_from_file_location("memory_item", (_p.Path(__file__).resolve().parents[1] / "autogpts" / "autogpt" / "autogpt" / "memory" / "vector" / "memory_item.py"))
memory_item = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_item)
MemoryItem = memory_item.MemoryItem
MemoryItemRelevance = memory_item.MemoryItemRelevance
from autogpt.memory.vector.utils import get_embedding
from autogpt.processing.text import chunk_code_by_structure


class SimpleEmbeddingProvider:
    class _Tokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()
        def decode(self, tokens: list[str]) -> str:
            return " ".join(tokens)
    def get_tokenizer(self, model_name: str):
        return self._Tokenizer()
    def count_tokens(self, text: str, model_name: str) -> int:
        return len(self.get_tokenizer(model_name).encode(text))
    def get_token_limit(self, model_name: str) -> int:
        return 8192
    async def create_embedding(self, text: str, model_name: str, embedding_parser, **kwargs):
        import numpy as np
        tokens = self.get_tokenizer(model_name).encode(text.lower())
        dims = 64
        vec = np.zeros(dims, dtype=np.float32)
        for tok in tokens:
            vec[hash(tok) % dims] += 1.0
        class R: pass
        r = R(); r.embedding = embedding_parser(vec.tolist()); r.prompt_tokens_used = len(tokens); r.completion_tokens_used = 0; r.model_info=None
        return r

class DummyConfig:
    embedding_model = "test"
    plugins = []


def test_local_embedding_provider_generates_embeddings():
    provider = SimpleEmbeddingProvider()
    cfg = DummyConfig()
    emb = asyncio.run(get_embedding("hello world", cfg, provider))
    assert isinstance(emb, list)
    assert len(emb) == 64


def test_chunk_code_by_structure_splits_functions():
    provider = SimpleEmbeddingProvider()
    tokenizer = provider.get_tokenizer("test")
    code = """def foo():
    pass

class Bar:
    def baz(self):
        pass
"""
    chunks = list(chunk_code_by_structure(code, 100, tokenizer, with_overlap=False))
    assert len(chunks) == 2
    assert chunks[0][0].startswith("def foo")
    assert chunks[1][0].startswith("class Bar")


def test_search_retrieval_prefers_relevant_item():
    provider = SimpleEmbeddingProvider()
    cfg = DummyConfig()
    text1 = "The sky is blue and bright."
    text2 = "Computers execute algorithms quickly."
    e1 = asyncio.run(get_embedding(text1, cfg, provider))
    e2 = asyncio.run(get_embedding(text2, cfg, provider))
    item1 = MemoryItem(
        raw_content=text1,
        summary=text1,
        chunks=[text1],
        chunk_summaries=[text1],
        e_summary=e1,
        e_chunks=[e1],
        metadata={},
    )
    item2 = MemoryItem(
        raw_content=text2,
        summary=text2,
        chunks=[text2],
        chunk_summaries=[text2],
        e_summary=e2,
        e_chunks=[e2],
        metadata={},
    )
    query = "What color is the sky?"
    e_query = asyncio.run(get_embedding(query, cfg, provider))
    rel1 = MemoryItemRelevance.of(item1, query, e_query)
    rel2 = MemoryItemRelevance.of(item2, query, e_query)
    assert rel1.score > rel2.score
