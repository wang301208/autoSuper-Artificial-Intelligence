from __future__ import annotations

from typing import Any, Dict, List, Tuple

import json

from backend.reflection.reflection import ReflectionModule, ReflectionResult
from backend.memory.long_term import LongTermMemory


class SelfModel:
    """Estimate the agent's own state using environment predictions."""

    def __init__(self, memory: LongTermMemory | None = None) -> None:
        self._reflection = ReflectionModule(max_passes=2, quality_threshold=2.0)
        self._history: List[str] = []
        self._memory = memory
        self._last_summary: str | None = None
        self._self_state: Dict[str, Any] = {
            "goals": [],  # list of {"goal": str, "subgoals": List[str]}
            "capabilities": {},  # name -> confidence
            "mood": "neutral",
        }

    def estimate(self, data: Dict[str, float], env_pred: Dict[str, float]) -> Dict[str, float]:
        """Return corrected CPU and memory predictions.

        The correction subtracts 10% of the average environment load from the
        agent's current usage as a simplistic feedback mechanism.
        """

        adjustment_cpu = env_pred.get("avg_cpu", 0.0) * 0.1
        adjustment_mem = env_pred.get("avg_memory", 0.0) * 0.1
        return {
            "cpu": max(data.get("cpu", 0.0) - adjustment_cpu, 0.0),
            "memory": max(data.get("memory", 0.0) - adjustment_mem, 0.0),
        }

    # ------------------------------------------------------------------
    def add_goal(self, goal: str, subgoals: List[str] | None = None) -> None:
        self._self_state["goals"].append({"goal": goal, "subgoals": subgoals or []})

    def set_capability(self, name: str, confidence: float) -> None:
        self._self_state["capabilities"][name] = confidence

    def update_state(self, events: List[str]) -> None:
        """Update the internal ``self_state`` based on ``events``."""

        for event in events:
            e = event.lower()
            if e.startswith("goal:"):
                self.add_goal(event.split(":", 1)[1].strip())
            elif e.startswith("subgoal:") and self._self_state["goals"]:
                self._self_state["goals"][-1]["subgoals"].append(
                    event.split(":", 1)[1].strip()
                )
            elif e.startswith("capability:"):
                spec = event.split(":", 1)[1].strip()
                if "(" in spec and ")" in spec:
                    name, conf = spec.split("(", 1)
                    try:
                        confidence = float(conf.strip(" )"))
                    except ValueError:
                        confidence = 1.0
                    self.set_capability(name.strip(), confidence)
                else:
                    self.set_capability(spec, 1.0)
            if any(word in e for word in ("error", "fail")):
                self._self_state["mood"] = "frustrated"
            elif any(word in e for word in ("success", "completed", "done")):
                self._self_state["mood"] = "satisfied"
            if self._memory and ("decision" in e or "outcome" in e):
                self._memory.add("decision_outcome", event)

    def assess_state(
        self,
        data: Dict[str, float],
        env_pred: Dict[str, float],
        last_action: str,
    ) -> Tuple[Dict[str, float], str]:
        """Return corrected metrics and an introspective summary.

        The summary is generated using :class:`ReflectionModule` and recent
        results are stored for future reference.
        """

        past_context = ""
        if self._memory:
            past = list(self._memory.get("decision_outcome"))
            if past:
                past_context = "; ".join(past[-3:])

        self.update_state([last_action])
        metrics = self.estimate(data, env_pred)
        base = (
            f"cpu={metrics['cpu']:.2f}, memory={metrics['memory']:.2f}; "
            f"last_action={last_action}"
        )
        if past_context:
            base += f"; past={past_context}"
        evaluation, revised = self._reflection.reflect(base)
        if last_action in self._self_state["capabilities"] and evaluation.sentiment == "negative":
            self._self_state["capabilities"][last_action] *= evaluation.confidence
        summary = (
            f"conf={evaluation.confidence:.2f},sent={evaluation.sentiment} | {revised}"
        )

        goals_str = ", ".join(g["goal"] for g in self._self_state["goals"]) or "none"
        caps_str = ", ".join(
            f"{n}:{c:.2f}" for n, c in self._self_state["capabilities"].items()
        ) or "none"
        narrative_base = (
            f"mood={self._self_state['mood']}; goals={goals_str}; capabilities={caps_str}"
        )
        if past_context:
            narrative_base += f"; past={past_context}"
        n_eval, n_revised = self._reflection.reflect(narrative_base)
        narrative = (
            f"conf={n_eval.confidence:.2f},sent={n_eval.sentiment} | {n_revised}"
        )
        summary = f"{summary} | {narrative}"

        self._history.append(summary)
        self._history = self._history[-5:]
        self._last_summary = summary
        if self._memory:
            self._memory.add("self_awareness", summary)
            self._memory.add("self_narrative", narrative)
            self._memory.add(
                "decision_outcome",
                f"{last_action}:{metrics['cpu']:.2f}/{metrics['memory']:.2f}",
            )
            self._memory.add("reflection_scores", json.dumps(evaluation.__dict__))
            self._memory.add("reflection_scores", json.dumps(n_eval.__dict__))
        return metrics, summary

    # ------------------------------------------------------------------
    def introspect(
        self, data: Dict[str, float], env_pred: Dict[str, float], last_action: str
    ) -> Dict[str, float | str]:
        """Return CPU/memory estimates together with a reflection summary."""

        metrics, summary = self.assess_state(data, env_pred, last_action)
        return {"cpu": metrics["cpu"], "memory": metrics["memory"], "summary": summary}

    @property
    def history(self) -> List[str]:
        return list(self._history)

    @property
    def last_summary(self) -> str | None:
        """Return the most recent introspection summary."""

        return self._last_summary


class SelfPlanner:
    """Simple planning interface leveraging :class:`SelfModel`."""

    def __init__(self, model: SelfModel) -> None:
        self._model = model

    def set_goal(self, goal: str, subgoals: List[str] | None = None) -> None:
        self._model.add_goal(goal, subgoals)

    def revise_goals(self) -> List[str]:
        goals_desc = "; ".join(
            f"{g['goal']}:{','.join(g['subgoals'])}" for g in self._model._self_state["goals"]
        ) or "none"
        caps_desc = ", ".join(
            f"{n}:{c:.2f}" for n, c in self._model._self_state["capabilities"].items()
        ) or "none"
        prompt = f"goals={goals_desc}; capabilities={caps_desc}"
        _eval, revised = self._model._reflection.reflect(prompt)
        new_goals = [g.strip() for g in revised.replace(" [revised]", "").split(",") if g.strip()]
        self._model._self_state["goals"] = [
            {"goal": g, "subgoals": []} for g in new_goals
        ]
        if self._model._memory:
            self._model._memory.add("planning", revised)
        return new_goals
