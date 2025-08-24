import sys, os, types
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))
_openapi_client = types.ModuleType("openapi_python_client")
_openapi_config = types.ModuleType("config")
setattr(_openapi_config, "Config", type("Config", (), {}))
_openapi_client.config = _openapi_config
sys.modules["openapi_python_client"] = _openapi_client
sys.modules["openapi_python_client.config"] = _openapi_config
sys.modules.setdefault("docx", types.ModuleType("docx"))
sys.modules.setdefault("pypdf", types.ModuleType("pypdf"))
_pylatexenc = types.ModuleType("pylatexenc")
_latex2text = types.ModuleType("latex2text")
setattr(_latex2text, "LatexNodes2Text", type("LatexNodes2Text", (), {}))
_pylatexenc.latex2text = _latex2text
sys.modules["pylatexenc"] = _pylatexenc
sys.modules["pylatexenc.latex2text"] = _latex2text

from autogpt.core.configuration.learning import LearningConfiguration
from autogpt.core.learning import ExperienceLearner
from autogpt.models.action_history import (
    EpisodicActionHistory,
    Action,
    ActionSuccessResult,
)
from autogpt.models.command import Command
from autogpt.models.command_registry import CommandRegistry


def test_learning_updates_command_priority():
    history = EpisodicActionHistory()
    history.register_action(Action(name="b", args={}, reasoning=""))
    history.register_result(ActionSuccessResult(outputs="ok"))

    learner = ExperienceLearner(
        memory=history, config=LearningConfiguration(enabled=True)
    )
    weights = learner.learn_from_experience()
    assert weights["b"] == 1.0

    registry = CommandRegistry()

    def cmd_a(*, agent=None):
        return "A"

    def cmd_b(*, agent=None):
        return "B"

    registry.register(Command("a", "cmd a", cmd_a, []))
    registry.register(Command("b", "cmd b", cmd_b, []))

    before = [c.name for c in registry.list_available_commands(agent=None)]
    assert before == ["a", "b"]

    for name, weight in weights.items():
        registry.get_command(name).priority = weight

    after = [c.name for c in registry.list_available_commands(agent=None)]
    assert after[0] == "b"
