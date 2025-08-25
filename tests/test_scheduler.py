import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from execution import Scheduler  # noqa: E402
from execution.task_graph import TaskGraph  # noqa: E402


def test_per_agent_task_counts() -> None:
    scheduler = Scheduler()
    scheduler.add_agent("a1")
    scheduler.add_agent("a2")

    graph = TaskGraph()
    graph.add_task("t1", "task 1", skill="s")
    graph.add_task("t2", "task 2", skill="s")

    def worker(agent: str, skill: str):
        return agent

    scheduler.submit(graph, worker)

    counts = scheduler.task_counts
    assert counts["a1"] + counts["a2"] == 2
