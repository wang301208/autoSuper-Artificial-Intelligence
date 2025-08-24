import logging
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from autogpt.core.ability import (
    AbilityRegistrySettings,
    AbilityResult,
    SimpleAbilityRegistry,
)
from autogpt.core.agent.layered import LayeredAgent
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.planning import PlannerSettings, SimplePlanner, Task, TaskStatus
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    OpenAIProvider,
    OpenAISettings,
    AssistantChatMessage,
)
from autogpt.core.resource.model_providers.schema import ChatModelResponse
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings
from autogpt.config import Config


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    systems: AgentSystems
    self_assess_frequency: int = 5


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        self.agent.configuration.name = agent_goals["agent_name"]
        self.agent.configuration.role = agent_goals["agent_role"]
        self.agent.configuration.goals = agent_goals["agent_goals"]


class PerformanceEvaluator:
    """Score ability results based on success, cost, and duration."""

    def __init__(
        self,
        success_weight: float = 1.0,
        cost_weight: float = 0.1,
        duration_weight: float = 0.1,
    ) -> None:
        self._success_weight = success_weight
        self._cost_weight = cost_weight
        self._duration_weight = duration_weight

    def score(self, result: AbilityResult, cost: float, duration: float) -> float:
        success_score = 1.0 if result.success else 0.0
        return (
            self._success_weight * success_score
            - self._cost_weight * cost
            - self._duration_weight * duration
        )


class SimpleAgent(LayeredAgent, Configurable):
    default_settings = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            name="Entrepreneur-GPT",
            role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            cycle_count=0,
            max_task_cycle_count=3,
            creation_time="",
            systems=AgentSystems(
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.ability.SimpleAbilityRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route=(
                        "autogpt.core.resource.model_providers.OpenAIProvider"
                    ),
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
            self_assess_frequency=5,
        ),
    )

    def __init__(
        self,
        settings: AgentSystemSettings,
        logger: logging.Logger,
        ability_registry: SimpleAbilityRegistry,
        memory: SimpleMemory,
        openai_provider: OpenAIProvider,
        planning: SimplePlanner,
        workspace: SimpleWorkspace,
        next_layer: Optional[LayeredAgent] = None,
        optimize_abilities: bool = False,
    ):
        super().__init__(next_layer=next_layer)
        self._configuration = settings.configuration
        self._logger = logger
        self._ability_registry = ability_registry
        self._memory = memory
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._planning = planning
        self._workspace = workspace
        self._task_queue = []
        self._completed_tasks = []
        self._current_task = None
        self._next_ability = None
        self._performance_evaluator = PerformanceEvaluator()
        self._optimize_abilities = optimize_abilities
        self._ability_metrics: dict[str, list[float]] = {}

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
        optimize_abilities: bool = False,
    ) -> "SimpleAgent":
        agent_settings = SimpleWorkspace.load_agent_settings(workspace_path)
        agent_args = {}

        agent_args["settings"] = agent_settings.agent
        agent_args["logger"] = logger
        agent_args["workspace"] = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger,
        )
        agent_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger,
        )
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
        )
        agent_args["memory"] = cls._get_system_instance(
            "memory",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
        )

        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["openai_provider"]},
        )

        return cls(**agent_args, optimize_abilities=optimize_abilities)

    async def build_initial_plan(self) -> dict:
        plan = await self._planning.make_initial_plan(
            agent_name=self._configuration.name,
            agent_role=self._configuration.role,
            agent_goals=self._configuration.goals,
            abilities=self._ability_registry.list_abilities(),
        )
        tasks = [Task.parse_obj(task) for task in plan.parsed_result["task_list"]]

        # TODO: Should probably do a step to evaluate the quality of the generated tasks
        #  and ensure that they have actionable ready and acceptance criteria

        self._task_queue.extend(tasks)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        self._task_queue[-1].context.status = TaskStatus.READY
        return plan.parsed_result

    def route_task(self, task: Task, *args, **kwargs):
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        self._task_queue[-1].context.status = TaskStatus.READY
        return task

    async def determine_next_ability(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._configuration.cycle_count += 1
        task = self._task_queue.pop()
        self._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        next_ability = await self._choose_next_ability(
            task,
            self._ability_registry.dump_abilities(),
        )
        self._current_task = task
        self._next_ability = next_ability.parsed_result
        return self._current_task, self._next_ability

    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability_name = self._next_ability["next_ability"]
            ability_args = self._next_ability["ability_arguments"]
            ability = self._ability_registry.get_ability(ability_name)

            filename = (
                ability_args.get("filename") if ability_name == "write_file" else None
            )
            start_time = time.perf_counter()
            ability_response = await ability(**ability_args)
            duration = time.perf_counter() - start_time
            cost = float(ability_response.ability_args.get("cost", 0))
            self._ability_metrics.setdefault(ability_name, []).append(duration)
            hint = getattr(getattr(ability, "_configuration", None), "performance_hint", None)
            if (
                self._optimize_abilities
                and hint is not None
                and duration > hint
            ):
                self._ability_registry.optimize_ability(
                    ability_name, {"duration": duration}
                )

            if ability_name == "write_file" and ability_response.success:
                lint_ability = self._ability_registry.get_ability("lint_code")
                lint_result = await lint_ability(file_path=filename)
                self._logger.info(
                    f"Static analysis for {filename}: {lint_result.message}"
                )
                self._memory.add(
                    f"Static analysis for {filename}: {lint_result.message}"
                )
                ability_response.message += f" Lint: {lint_result.message}"
                if not lint_result.success:
                    subprocess.run(["git", "checkout", "--", filename], check=False)
                    if hasattr(self._workspace, "refresh"):
                        self._workspace.refresh()
                    ability_response.message += (
                        " Static analysis failed. Changes reverted."
                    )
                else:
                    generate_tests = self._ability_registry.get_ability(
                        "generate_tests"
                    )
                    tests_code = await generate_tests(file_path=filename)
                    test_filename = None
                    if tests_code.success and tests_code.message:
                        test_filename = str(
                            Path("tests") / f"test_{Path(filename).stem}.py"
                        )
                        await ability(
                            filename=test_filename, contents=tests_code.message
                        )
                        ability_response.message += (
                            f" Test file generated at {test_filename}."
                        )
                    run_tests = self._ability_registry.get_ability("run_tests")
                    tests_result = await run_tests()
                    critique = await self._ability_registry.perform(
                        "query_language_model",
                        query=(
                            "Given these test results, provide feedback on "
                            "the change:\n"
                            + tests_result.message
                        ),
                    )
                    test_status = "passed" if tests_result.success else "failed"
                    if not tests_result.success:
                        self._memory.add(
                            f"Test failed for {filename}:\n{tests_result.message}\n"
                            f"Critique: {critique.message}"
                        )
                        subprocess.run(["git", "checkout", "--", filename], check=False)
                        if test_filename:
                            subprocess.run(
                                ["git", "checkout", "--", test_filename], check=False
                            )
                            if subprocess.run(
                                ["git", "ls-files", test_filename, "--error-unmatch"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            ).returncode:
                                Path(test_filename).unlink(missing_ok=True)
                        if hasattr(self._workspace, "refresh"):
                            self._workspace.refresh()
                        ability_response.message += " Tests failed. Changes reverted."
                    else:
                        ability_response.message += " Tests passed."
                        evaluate_metrics = self._ability_registry.get_ability(
                            "evaluate_metrics"
                        )
                        metrics_result = await evaluate_metrics(file_path=filename)
                        self._memory.add(
                            f"Metrics for {filename}: {metrics_result.message}"
                        )
                        ability_response.message += (
                            f" Metrics: {metrics_result.message}"
                        )
                        subprocess.run(["git", "add", filename], check=False)
                        if test_filename:
                            subprocess.run(["git", "add", test_filename], check=False)
                        diff = subprocess.check_output(
                            ["git", "diff", "--cached"], text=True
                        )
                        summary = await self._ability_registry.perform(
                            "query_language_model",
                            query=(
                                "Summarize the following changes for ",
                                "a commit message:\n"
                                + diff
                            ),
                        )
                        commit_message = (
                            f"Auto-update: {summary.message}"
                            if summary.success and summary.message
                            else f"Auto-update: modify {filename}"
                        )
                        subprocess.run(
                            ["git", "commit", "-m", commit_message], check=False
                        )
                        commit_hash = subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], text=True
                        ).strip()
                        self._memory.add(
                            f"Commit {commit_hash} - {commit_message} - ",
                            f"Test {test_status} for {filename}:\n",
                            f"{tests_result.message}\nCritique: {critique.message}",
                        )

            await self._update_tasks_and_memory(ability_response)
            task_desc = getattr(self._current_task, "description", self._current_task.objective)
            task_id = str(getattr(self._current_task, "id", id(self._current_task)))
            score = self._performance_evaluator.score(
                ability_response, cost=cost, duration=duration
            )
            self._memory.log_score(
                task_id=task_id,
                task_description=task_desc,
                ability=ability_name,
                score=score,
            )
            if (
                self._configuration.cycle_count % self._configuration.self_assess_frequency
                == 0
            ):
                assessment = await self._ability_registry.perform(
                    "self_assess", limit=5
                )
                self._memory.add(f"Self-assessment: {assessment.message}")
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                self._task_queue.append(self._current_task)
            self._current_task = None
            self._next_ability = None

            return ability_response.dict()
        else:
            raise NotImplementedError

    async def _evaluate_task_and_add_context(self, task: Task) -> Task:
        """Evaluate the task and add context to it."""
        if task.context.status == TaskStatus.IN_PROGRESS:
            # Nothing to do here
            return task
        else:
            self._logger.debug(f"Evaluating task {task} and adding relevant context.")
            query = getattr(task, "description", task.objective)
            k = 5
            try:
                config = Config()
                relevant_memories = self._memory.get_relevant(query, k, config)

                class _ContextMemory:
                    def __init__(self, item):
                        self._item = item

                    def summary(self) -> str:
                        return self._item.summary

                for memory in relevant_memories:
                    task.context.memories.append(_ContextMemory(memory.memory_item))
            except Exception as e:
                self._logger.debug(f"Failed to get relevant memories: {e}")

            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    async def _choose_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
    ):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")

        degraded_modules = self._get_degraded_modules()
        if degraded_modules:
            module_path = degraded_modules[0]
            return ChatModelResponse(
                response=AssistantChatMessage(
                    content="evaluate_metrics due to performance regression"
                ),
                parsed_result={
                    "next_ability": "evaluate_metrics",
                    "ability_arguments": {"file_path": module_path},
                },
            )

        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task"
            #  with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task"
            #  with an appropriate reason
            raise NotImplementedError
        else:
            task_desc = getattr(task, "description", task.objective)
            ability_specs.sort(
                key=lambda spec: self._average_score(task_desc, spec.name),
                reverse=True,
            )
            next_ability = await self._planning.determine_next_ability(
                task, ability_specs
            )
            return next_ability

    def _average_score(self, task_desc: str, ability_name: str) -> float:
        scores = self._memory.get_scores_for_task(task_desc, ability_name)
        return sum(scores) / len(scores) if scores else 0.0

    def _get_degraded_modules(self) -> list[str]:
        """Parse memory for performance metrics and return modules with regressions."""
        records: dict[str, list[tuple[float, float]]] = {}
        for msg in self._memory.get():
            match = re.match(
                r"Metrics for (.*): complexity=([0-9.]+), runtime=([0-9.]+)",
                msg,
            )
            if match:
                file = match.group(1)
                complexity = float(match.group(2))
                runtime = float(match.group(3))
                records.setdefault(file, []).append((complexity, runtime))

        degraded = []
        for file, values in records.items():
            if len(values) >= 2:
                prev_c, prev_r = values[-2]
                curr_c, curr_r = values[-1]
                if curr_c > prev_c or curr_r > prev_r:
                    degraded.append(file)
        return degraded

    async def _update_tasks_and_memory(self, ability_result: AbilityResult):
        self._current_task.context.cycle_count += 1
        self._current_task.context.prior_actions.append(ability_result)
        if ability_result.ability_name == "evaluate_metrics":
            file_path = ability_result.ability_args.get("file_path", "")
            self._memory.add(f"Metrics for {file_path}: {ability_result.message}")
        # TODO: Summarize new knowledge
        # TODO: store knowledge and summaries in memory and in relevant tasks
        # TODO: evaluate whether the task is complete

    def __repr__(self):
        return "SimpleAgent()"

    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        configuration_dict = {
            "agent": cls.get_user_config(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.build_agent_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.build_agent_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger=logger,
        )
        logger.debug("Loading agent planner.")
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        logger.debug("determining agent name and goals.")
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response.parsed_result

    @classmethod
    def provision_agent(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ):
        agent_settings.agent.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: SimpleWorkspace = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger=logger,
        )
        return workspace.setup_workspace(agent_settings, logger)

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        system_locations = agent_settings.agent.configuration.systems.dict()

        system_settings = getattr(agent_settings, system_name)
        system_class = SimplePluginService.get_plugin(system_locations[system_name])
        system_instance = system_class(
            system_settings,
            *args,
            logger=logger.getChild(system_name),
            **kwargs,
        )
        return system_instance


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty
    dictionaries at the leaves.

    Args:
        d: The dictionary to prune.

    Returns:
        The pruned dictionary.
    """
    pruned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            pruned_value = _prune_empty_dicts(value)
            if (
                pruned_value
            ):  # if the pruned dictionary is not empty, add it to the result
                pruned[key] = pruned_value
        else:
            pruned[key] = value
    return pruned
