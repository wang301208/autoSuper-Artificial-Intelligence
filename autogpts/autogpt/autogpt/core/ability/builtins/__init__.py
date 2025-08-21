from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility
from autogpt.core.ability.builtins.file_operations import ReadFile, WriteFile
from autogpt.core.ability.builtins.generate_tests import GenerateTests
from autogpt.core.ability.builtins.query_language_model import QueryLanguageModel
from autogpt.core.ability.builtins.run_tests import RunTests

BUILTIN_ABILITIES = {
    QueryLanguageModel.name(): QueryLanguageModel,
    CreateNewAbility.name(): CreateNewAbility,
    ReadFile.name(): ReadFile,
    WriteFile.name(): WriteFile,
    RunTests.name(): RunTests,
    GenerateTests.name(): GenerateTests,
}

__all__ = [
    "BUILTIN_ABILITIES",
    "CreateNewAbility",
    "QueryLanguageModel",
    "ReadFile",
    "WriteFile",
    "RunTests",
    "GenerateTests",
]
