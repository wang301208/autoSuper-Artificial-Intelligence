"""Cognition adapters bridging :class:`SimpleAgent` and neuromorphic backends."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Tuple

from pydantic import Field

from autogpt.core.brain.config import WholeBrainConfig
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.resource.model_providers.schema import CompletionModelFunction
from modules.brain.state import (
    BrainCycleResult,
    CognitiveIntent,
    EmotionSnapshot,
    FeelingSnapshot,
    PersonalityProfile,
)
from modules.brain.whole_brain import WholeBrainSimulation


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


class BrainAdapterConfiguration(SystemConfiguration):
    """Configuration wrapper for the simple cognition adapter."""

    whole_brain: WholeBrainConfig = Field(default_factory=WholeBrainConfig)


class SimpleBrainAdapterSettings(SystemSettings):
    configuration: BrainAdapterConfiguration


class SimpleBrainAdapter(Configurable):
    """Drive ``SimpleAgent`` cognition via :class:`WholeBrainSimulation`."""

    default_settings = SimpleBrainAdapterSettings(
        name="simple_brain_adapter",
        description=(
            "Routes planning and action selection through the WholeBrain "
            "neuromorphic simulation."
        ),
        configuration=BrainAdapterConfiguration(),
    )

    #: Mapping from cognitive intention to preferred ability names.
    _INTENTION_ABILITY_PREFERENCES: Mapping[str, tuple[str, ...]] = {
        "observe": ("self_assess", "lint_code", "run_tests"),
        "explore": ("run_tests", "self_assess", "create_new_ability"),
        "approach": ("run_tests", "write_file", "generate_tests"),
        "withdraw": ("self_assess", "evaluate_metrics", "lint_code"),
    }

    def __init__(
        self,
        settings: SimpleBrainAdapterSettings,
        logger: logging.Logger,
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        brain_kwargs = self._configuration.whole_brain.to_simulation_kwargs()
        self._brain = WholeBrainSimulation(**brain_kwargs)
        self._last_cycle: BrainCycleResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def build_initial_plan(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        ability_specs: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a bootstrap plan from the current brain state."""

        input_payload = self._compose_cycle_input(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            abilities=ability_specs,
        )
        brain_result = self._brain.process_cycle(input_payload)
        self._last_cycle = brain_result
        metadata = self._summarise_cycle(brain_result)

        plan_steps = brain_result.intent.plan or ["clarify_objective"]
        plan_dict = {
            "task_list": self._plan_steps_to_tasks(plan_steps, agent_goals),
            "backend": "whole_brain",
            "intention": brain_result.intent.intention,
            "confidence": float(brain_result.intent.confidence),
            "thoughts": metadata,
        }
        return plan_dict, metadata

    async def determine_next_ability(
        self,
        *,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        task: Any | None,
        ability_specs: list[CompletionModelFunction],
        cycle_index: int,
        backlog_size: int,
        completed: int,
        state_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select the next ability using the neuromorphic backend."""

        input_payload = self._compose_cycle_input(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            task=task,
            abilities=[spec.name for spec in ability_specs],
            cycle_index=cycle_index,
            backlog_size=backlog_size,
            completed=completed,
            state_context=state_context,
        )
        brain_result = self._brain.process_cycle(input_payload)
        self._last_cycle = brain_result
        metadata = self._summarise_cycle(brain_result)

        ability_name, ability_args = self._select_ability(
            brain_result.intent,
            ability_specs,
        )
        payload = {
            "next_ability": ability_name,
            "ability_arguments": ability_args,
            "backend": "whole_brain",
            "confidence": float(brain_result.intent.confidence),
            "plan": list(brain_result.intent.plan),
            "reasoning": metadata.get("analysis"),
        }
        return payload, metadata

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compose_cycle_input(
        self,
        *,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: Iterable[str],
        task: Any | None = None,
        cycle_index: int = 0,
        backlog_size: int = 0,
        completed: int = 0,
        state_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare structured inputs for :meth:`WholeBrainSimulation.process_cycle`."""

        state_context = state_context or {}
        ability_list = list(abilities)

        text_fragments: list[str] = [f"Role: {agent_role}"]
        if agent_goals:
            text_fragments.append("Goals: " + "; ".join(agent_goals[:3]))
        if ability_list:
            text_fragments.append("Abilities: " + ", ".join(ability_list))

        context = {
            "cycle_count": float(max(0, cycle_index)),
            "backlog": float(max(0, backlog_size)),
            "completed": float(max(0, completed)),
            "ability_count": float(len(ability_list)),
        }
        context.update(
            {
                f"state_{key}": _safe_float(value)
                for key, value in state_context.items()
                if isinstance(value, (int, float))
            }
        )

        vision = [
            min(1.0, context["cycle_count"] / 10.0),
            min(1.0, context["backlog"] / 10.0),
            min(1.0, context["completed"] / (backlog_size + completed + 1 or 1)),
        ]
        auditory = [
            min(1.0, len(agent_goals) / 5.0),
            min(1.0, len(ability_list) / 5.0),
            min(1.0, backlog_size / 5.0),
        ]
        somatosensory = [
            min(1.0, cycle_index / 8.0),
            min(1.0, backlog_size / 8.0),
            1.0 if state_context.get("enough_info") else 0.0,
        ]

        if task is not None:
            description = getattr(task, "objective", str(task))
            text_fragments.append(f"Task: {description}")
            context["task_priority"] = _safe_float(getattr(task, "priority", 0))
            context["task_cycles"] = _safe_float(
                getattr(getattr(task, "context", None), "cycle_count", 0)
            )
            ready = getattr(task, "ready_criteria", []) or []
            acceptance = getattr(task, "acceptance_criteria", []) or []
            if ready:
                text_fragments.append("Ready: " + "; ".join(map(str, ready)))
            if acceptance:
                text_fragments.append("Done when: " + "; ".join(map(str, acceptance)))

        return {
            "agent_id": agent_name,
            "text": "\n".join(text_fragments),
            "context": context,
            "vision": vision,
            "auditory": auditory,
            "somatosensory": somatosensory,
            "is_salient": bool(backlog_size > 0 and cycle_index > 0),
        }

    def _plan_steps_to_tasks(
        self, steps: Iterable[str], agent_goals: list[str]
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for index, raw_step in enumerate(steps, start=1):
            step = str(raw_step)
            normalized = step.replace("_", " ").replace("-", " ").strip()
            objective = normalized.capitalize() or "Clarify objective"
            task_type = self._infer_task_type(step)
            ready_hint = f"Outline how to {normalized.lower()}"
            acceptance_hint = f"Summarise the outcome of {normalized.lower()}"
            tasks.append(
                {
                    "objective": objective,
                    "type": task_type,
                    "priority": index,
                    "ready_criteria": [ready_hint],
                    "acceptance_criteria": [acceptance_hint],
                }
            )
        if not tasks:  # pragma: no cover - defensive
            tasks.append(
                {
                    "objective": "Review goals",
                    "type": "plan",
                    "priority": 1,
                    "ready_criteria": ["List current goals"],
                    "acceptance_criteria": [
                        "Document a concrete action supporting the primary goal"
                    ],
                }
            )
        if agent_goals:
            tasks[0]["ready_criteria"].append(
                f"Ensure alignment with goal: {agent_goals[0]}"
            )
        return tasks

    def _infer_task_type(self, step: str) -> str:
        lowered = step.lower()
        if any(keyword in lowered for keyword in ("write", "engage", "create")):
            return "write"
        if any(keyword in lowered for keyword in ("test", "verify", "run")):
            return "test"
        if any(keyword in lowered for keyword in ("code", "implement", "fix")):
            return "code"
        if any(keyword in lowered for keyword in ("scan", "assess", "observe", "review")):
            return "research"
        return "plan"

    def _select_ability(
        self,
        intent: CognitiveIntent,
        ability_specs: list[CompletionModelFunction],
    ) -> Tuple[str, dict[str, Any]]:
        ability_by_name = {spec.name: spec for spec in ability_specs}
        preference = self._INTENTION_ABILITY_PREFERENCES.get(
            intent.intention, ("self_assess",)
        )

        for name in preference:
            spec = ability_by_name.get(name)
            if spec and self._callable_without_required_args(spec):
                return name, {}

        for spec in ability_specs:
            if self._callable_without_required_args(spec):
                return spec.name, {}

        if ability_specs:
            self._logger.debug(
                "No ability without required arguments available; returning '%s'",
                ability_specs[0].name,
            )
            return ability_specs[0].name, {}

        return "self_assess", {}

    def _callable_without_required_args(self, spec: CompletionModelFunction) -> bool:
        if not spec.parameters:
            return True
        return not any(param.required for param in spec.parameters.values())

    def _summarise_cycle(self, result: BrainCycleResult) -> dict[str, Any]:
        def _emotion_payload(emotion: EmotionSnapshot) -> dict[str, float | str]:
            return {
                "primary": getattr(emotion.primary, "value", str(emotion.primary)),
                "intensity": float(emotion.intensity),
                "mood": float(emotion.mood),
                "dimensions": {k: float(v) for k, v in emotion.dimensions.items()},
                "context": {k: float(v) for k, v in emotion.context.items()},
                "decay": float(emotion.decay),
            }

        payload = {
            "backend": "whole_brain",
            "intention": result.intent.intention,
            "plan": list(result.intent.plan),
            "confidence": float(result.intent.confidence),
            "weights": {k: float(v) for k, v in result.intent.weights.items()},
            "tags": list(result.intent.tags),
            "analysis": "; ".join(result.intent.plan)
            if result.intent.plan
            else result.intent.intention,
            "emotion": _emotion_payload(result.emotion),
            "curiosity": asdict(result.curiosity),
            "personality": asdict(result.personality),
            "metrics": {k: float(v) for k, v in (result.metrics or {}).items()},
            "metadata": {k: v for k, v in (result.metadata or {}).items() if v is not None},
        }
        if result.thoughts:
            payload["thoughts"] = {
                "focus": result.thoughts.focus,
                "summary": result.thoughts.summary,
                "plan": list(result.thoughts.plan),
                "tags": list(result.thoughts.tags),
            }
        if result.feeling:
            payload["feeling"] = {
                "descriptor": result.feeling.descriptor,
                "valence": float(result.feeling.valence),
                "arousal": float(result.feeling.arousal),
                "mood": float(result.feeling.mood),
                "confidence": float(result.feeling.confidence),
                "context_tags": list(result.feeling.context_tags),
            }
        return payload


__all__ = [
    "SimpleBrainAdapter",
    "SimpleBrainAdapterSettings",
    "BrainAdapterConfiguration",
]
