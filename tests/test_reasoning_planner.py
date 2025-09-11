import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reasoning.planner import ReasoningPlanner
from backend.reasoning.solvers import RuleProbabilisticSolver


class DummySource:
    def __init__(self):
        self.calls = 0

    def query(self, statement: str):
        self.calls += 1
        return [f"{statement}_evidence"]


def test_planner_caching_and_explanation():
    source = DummySource()
    solver = RuleProbabilisticSolver({"A": [("B", 0.7)]})
    planner = ReasoningPlanner([source], solver)

    first = planner.infer("A")
    second = planner.infer("A")

    assert first == ("B", 0.7)
    assert second == ("B", 0.7)
    assert source.calls == 1
    explanation = planner.explain()
    assert explanation[0]["statement"] == "A"
    assert explanation[0]["conclusion"] == "B"
    assert explanation[0]["probability"] == 0.7


def test_planner_chain():
    source = DummySource()
    solver = RuleProbabilisticSolver({"A": [("B", 0.7)], "B": [("C", 0.5)]})
    planner = ReasoningPlanner([source], solver)

    results = planner.chain(["A", "B"])
    assert results == [("B", 0.7), ("C", 0.5)]
    assert source.calls == 2
