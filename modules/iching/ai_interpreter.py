"""AI-enhanced I Ching interpreter."""
from typing import Callable, Dict


class AIEnhancedInterpreter:
    """Interpret hexagrams using a simple knowledge base and optional LLM.

    The interpreter expands on traditional hexagram texts using either an
    injected large language model (LLM) callable or a built-in heuristic
    knowledge base. The goal is to provide context-aware interpretations for
    modern usage.
    """

    def __init__(self, knowledge_base: Dict[str, str] | None = None, llm: Callable[[str, str], str] | None = None):
        self.knowledge_base = knowledge_base or {
            "乾": "乾为天，元亨利贞，象征创造与领导。",
            "坤": "坤为地，厚德载物，象征包容与顺承。",
        }
        self.llm = llm

    def interpret(self, hexagram: str, context: str | None = None) -> str:
        """Return an interpretation of *hexagram* given *context*.

        Parameters
        ----------
        hexagram:
            Name of the hexagram to interpret.
        context:
            Optional context such as "career" or "relationships" to tailor the
            reading. When an LLM callable is provided, it will be used to
            generate the context specific advice. Otherwise a simple heuristic
            expansion from the knowledge base is used.
        """
        base = self.knowledge_base.get(hexagram, "未知卦象")
        ctx = (context or "").lower()

        if self.llm:
            enhanced = self.llm(hexagram, ctx)
            return f"{base}\n{enhanced}".strip()

        if "career" in ctx:
            addition = "在事业方面，要积极进取，勇于承担责任。"
        elif "relationship" in ctx or "love" in ctx or "感情" in ctx:
            addition = "在人际关系中，应当真诚相待，维护和谐。"
        else:
            addition = "保持正直和谦逊，将迎来吉祥。"

        return f"{base}（{addition}）"
