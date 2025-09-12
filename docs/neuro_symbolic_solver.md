# NeuroSymbolicSolver

`NeuroSymbolicSolver` combines predictions from a neural model with symbolic
rules. Neural outputs provide probabilistic suggestions, while the symbolic
knowledge base can introduce or override conclusions deterministically.

```python
from backend.reasoning.planner import ReasoningPlanner
from backend.reasoning.solvers import NeuroSymbolicSolver

def model(statement, evidence):
    return {"sunny": 0.8, "rain": 0.2}

solver = NeuroSymbolicSolver(model, {"take umbrella": "rain"})
planner = ReasoningPlanner(solver_config={"name": "neuro_symbolic", "params": {"neural_model": model, "knowledge_base": {"take umbrella": "rain"}}})
conclusion, probability = planner.infer("take umbrella")
# conclusion -> "rain", probability -> 1.0
```

Use :func:`load_neural_model` and :func:`load_symbolic_kb` to load resources
from disk.

