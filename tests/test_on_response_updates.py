from types import SimpleNamespace
import sys
from pathlib import Path
import types
from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend/autogpt"))

# Stub autogpt.config to avoid heavy configuration imports
config_stub = types.ModuleType("autogpt.config")
from pydantic import BaseModel, Field
class AIProfile(BaseModel):
    ai_name: str = "GPT"
class AIDirectives(BaseModel):
    @classmethod
    def from_file(cls, *args, **kwargs):
        return cls()
class Config(BaseModel):
    pass
class ConfigBuilder:
    default_settings = SimpleNamespace(prompt_settings_file="")
config_stub.AIProfile = AIProfile
config_stub.AIDirectives = AIDirectives
config_stub.Config = Config
config_stub.ConfigBuilder = ConfigBuilder
sys.modules["autogpt.config"] = config_stub

ai_directives_module = types.ModuleType("autogpt.config.ai_directives")
ai_directives_module.AIDirectives = AIDirectives
sys.modules["autogpt.config.ai_directives"] = ai_directives_module

ai_profile_module = types.ModuleType("autogpt.config.ai_profile")
ai_profile_module.AIProfile = AIProfile
sys.modules["autogpt.config.ai_profile"] = ai_profile_module

action_history_module = types.ModuleType("autogpt.models.action_history")
class Action(BaseModel):
    name: str
    args: dict
    reasoning: str
class ActionResult(BaseModel):
    pass
class Episode(BaseModel):
    action: Action
    result: ActionResult | None = None
class EpisodicActionHistory(BaseModel):
    episodes: list[Episode] = Field(default_factory=list)
    def __len__(self):
        return len(self.episodes)
    def register_action(self, action):
        self.episodes.append(Episode(action=action))
    @property
    def current_episode(self):
        return self.episodes[-1] if self.episodes else None
action_history_module.Action = Action
action_history_module.ActionResult = ActionResult
action_history_module.EpisodicActionHistory = EpisodicActionHistory
sys.modules["autogpt.models.action_history"] = action_history_module

context_item_module = types.ModuleType("autogpt.models.context_item")
class StaticContextItem:
    def __init__(self, description: str, source, content: str):
        self.item_description = description
        self.item_source = source
        self.item_content = content
    @property
    def description(self):
        return self.item_description
    @property
    def source(self):
        return self.item_source
    @property
    def content(self):
        return self.item_content
    def fmt(self):
        return self.item_content
context_item_module.StaticContextItem = StaticContextItem
sys.modules["autogpt.models.context_item"] = context_item_module

class AgentContext:
    def __init__(self):
        self.items = []
    def add(self, item):
        self.items.append(item)
    def __bool__(self):
        return len(self.items) > 0

# Stub brain modules to avoid torch dependency
brain_module = types.ModuleType("autogpt.core.brain")
brain_config_module = types.ModuleType("autogpt.core.brain.config")
class TransformerBrainConfig:
    dim = 0
brain_transformer_module = types.ModuleType("autogpt.core.brain.transformer_brain")
class TransformerBrain:
    def __init__(self, *args, **kwargs):
        pass
brain_transformer_module.TransformerBrain = TransformerBrain
brain_config_module.TransformerBrainConfig = TransformerBrainConfig
brain_module.config = brain_config_module
brain_module.transformer_brain = brain_transformer_module
sys.modules["autogpt.core.brain"] = brain_module
sys.modules["autogpt.core.brain.config"] = brain_config_module
sys.modules["autogpt.core.brain.transformer_brain"] = brain_transformer_module

# Stub external dependencies required for importing BaseAgent
events_module = types.ModuleType("events")
class EventBus: ...
def create_event_bus(*args, **kwargs):
    return EventBus()
events_module.EventBus = EventBus
events_module.create_event_bus = create_event_bus
sys.modules["events"] = events_module

events_client_module = types.ModuleType("events.client")
class EventClient:
    def __init__(self, *args, **kwargs):
        pass
    def publish(self, *args, **kwargs):
        pass
events_client_module.EventClient = EventClient
sys.modules["events.client"] = events_client_module

events_coord_module = types.ModuleType("events.coordination")
class TaskStatus: ...
class TaskStatusEvent:
    def __init__(self, *args, **kwargs):
        pass
    def to_dict(self):
        return {}
events_coord_module.TaskStatus = TaskStatus
events_coord_module.TaskStatusEvent = TaskStatusEvent
sys.modules["events.coordination"] = events_coord_module

forge_module = types.ModuleType("forge")
forge_sdk_module = types.ModuleType("forge.sdk")
forge_sdk_model_module = types.ModuleType("forge.sdk.model")
class Task(BaseModel):
    input: str = ""
forge_sdk_model_module.Task = Task
forge_sdk_module.model = forge_sdk_model_module
forge_module.sdk = forge_sdk_module
sys.modules["forge"] = forge_module
sys.modules["forge.sdk"] = forge_sdk_module
sys.modules["forge.sdk.model"] = forge_sdk_model_module

sentry_sdk = types.ModuleType("sentry_sdk")
def capture_exception(*args, **kwargs):
    pass
sentry_sdk.capture_exception = capture_exception
sys.modules["sentry_sdk"] = sentry_sdk

# Stub resource provider package to avoid executing heavy __init__
schema_path = ROOT / "backend/autogpt/autogpt/core/resource/model_providers/schema.py"
schema_module = SourceFileLoader(
    "autogpt.core.resource.model_providers.schema", str(schema_path)
).load_module()
resource_pkg = types.ModuleType("autogpt.core.resource.model_providers")
resource_pkg.schema = schema_module
resource_pkg.openai = None  # placeholder
sys.modules["autogpt.core.resource.model_providers"] = resource_pkg
sys.modules["autogpt.core.resource.model_providers.schema"] = schema_module

# Stub OpenAI provider modules to avoid heavy dependencies
resource_openai_module = types.ModuleType(
    "autogpt.core.resource.model_providers.openai"
)
class OpenAIModelName(str):
    GPT3_16k = "gpt3"
    GPT4 = "gpt4"
resource_openai_module.OpenAIModelName = OpenAIModelName
resource_openai_module.OPEN_AI_CHAT_MODELS = {
    OpenAIModelName.GPT3_16k: SimpleNamespace(name="gpt3"),
    OpenAIModelName.GPT4: SimpleNamespace(name="gpt4"),
}
sys.modules[
    "autogpt.core.resource.model_providers.openai"
] = resource_openai_module
resource_pkg.openai = resource_openai_module

# Stub Google Cloud logging dependencies
google_module = types.ModuleType("google")
cloud_module = types.ModuleType("google.cloud")
logging_v2_module = types.ModuleType("google.cloud.logging_v2")
handlers_module = types.ModuleType("google.cloud.logging_v2.handlers")
class CloudLoggingFilter: ...
class StructuredLogHandler: ...
handlers_module.CloudLoggingFilter = CloudLoggingFilter
handlers_module.StructuredLogHandler = StructuredLogHandler
logging_v2_module.handlers = handlers_module
cloud_module.logging_v2 = logging_v2_module
google_module.cloud = cloud_module
sys.modules["google"] = google_module
sys.modules["google.cloud"] = cloud_module
sys.modules["google.cloud.logging_v2"] = logging_v2_module
sys.modules["google.cloud.logging_v2.handlers"] = handlers_module

# Stub autogpt.logs to avoid speech and other heavy deps
logs_module = types.ModuleType("autogpt.logs")
logs_config_module = types.ModuleType("autogpt.logs.config")
def configure_chat_plugins(*args, **kwargs):
    pass
def configure_logging(*args, **kwargs):
    pass
logs_config_module.configure_chat_plugins = configure_chat_plugins
logs_config_module.configure_logging = configure_logging
logs_helpers_module = types.ModuleType("autogpt.logs.helpers")
def request_user_double_check(*args, **kwargs):
    return ""
logs_helpers_module.request_user_double_check = request_user_double_check
logs_module.config = logs_config_module
logs_module.helpers = logs_helpers_module
sys.modules["autogpt.logs"] = logs_module
sys.modules["autogpt.logs.config"] = logs_config_module
sys.modules["autogpt.logs.helpers"] = logs_helpers_module

dotenv_module = types.ModuleType("dotenv")
def load_dotenv(*args, **kwargs):
    pass
dotenv_module.load_dotenv = load_dotenv
sys.modules["dotenv"] = dotenv_module
llm_openai_module = types.ModuleType("autogpt.llm.providers.openai")
def get_openai_command_specs(commands):
    return []
llm_openai_module.get_openai_command_specs = get_openai_command_specs
sys.modules["autogpt.llm.providers.openai"] = llm_openai_module

from autogpt.agents.base import BaseAgent
from autogpt.agents.utils.prompt_scratchpad import PromptScratchpad
from autogpt.core.prompting.schema import ChatPrompt
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatModelResponse,
    ChatModelInfo,
)
from autogpt.models.action_history import EpisodicActionHistory


def test_on_response_updates_memory_and_context():
    agent = SimpleNamespace()
    agent.event_history = EpisodicActionHistory()
    agent.context = AgentContext()
    agent.config = SimpleNamespace(plugins=[])

    parsed = (
        "test_cmd",
        {"arg": "value"},
        {"thoughts": {"reasoning": "because"}},
    )
    message = AssistantChatMessage(content="some content")
    model_info = ChatModelInfo(
        name="gpt",
        provider_name="openai",
        max_tokens=1000,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
    )
    response = ChatModelResponse(
        response=message,
        parsed_result=parsed,
        prompt_tokens_used=0,
        completion_tokens_used=0,
        model_info=model_info,
    )

    prompt = ChatPrompt(messages=[])
    scratchpad = PromptScratchpad()

    BaseAgent.on_response(agent, response, prompt, scratchpad)

    assert len(agent.event_history) == 1
    assert agent.event_history.episodes[0].action.name == "test_cmd"
    assert len(agent.context.items) == 1
    assert agent.context.items[0].content == "some content"
