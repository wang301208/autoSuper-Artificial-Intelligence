"""Test configuration and shared fixtures."""

import sys
import types

# Some modules imported by the codebase are optional during tests.
# Provide lightweight stubs so imports succeed without installing heavy deps.
sys.modules.setdefault(
    "auto_gpt_plugin_template", types.ModuleType("auto_gpt_plugin_template")
)
sys.modules["auto_gpt_plugin_template"].AutoGPTPluginTemplate = type(
    "AutoGPTPluginTemplate", (), {}
)

openai_mod = types.ModuleType("openai")
exceptions_mod = types.ModuleType("openai._exceptions")
class APIStatusError(Exception):
    pass
class RateLimitError(Exception):
    pass
exceptions_mod.APIStatusError = APIStatusError
exceptions_mod.RateLimitError = RateLimitError
openai_mod._exceptions = exceptions_mod
sys.modules.setdefault("openai", openai_mod)
sys.modules.setdefault("openai._exceptions", exceptions_mod)
types_mod = types.ModuleType("openai.types")
openai_mod.types = types_mod
sys.modules.setdefault("openai.types", types_mod)
types_mod.CreateEmbeddingResponse = type("CreateEmbeddingResponse", (), {})
chat_mod = types.ModuleType("openai.types.chat")
chat_mod.__getattr__ = lambda name: type(name, (), {})
types_mod.chat = chat_mod
sys.modules.setdefault("openai.types.chat", chat_mod)

