import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.consciousness import ConsciousnessModel
from modules.brain.consciousness_advanced import ConsciousnessAdvanced, AdaptiveAttention


def _dataset():
    return [
        {"score": 0.55, "ground_truth": 0},
        {"score": 0.60, "ground_truth": 0},
        {"score": 0.70, "ground_truth": 1},
        {"score": 0.80, "ground_truth": 1},
        {"score": 0.58, "ground_truth": 0},
        {"score": 0.90, "ground_truth": 1},
    ]


def test_hierarchical_model_improves_task_performance():
    data = _dataset()

    simple = ConsciousnessModel()
    preds = []
    for item in data:
        info = {"is_salient": item["score"] > 0.5}
        preds.append(simple.conscious_access(info))
    simple_acc = sum(int(p == bool(d["ground_truth"])) for p, d in zip(preds, data)) / len(data)

    advanced = ConsciousnessAdvanced(attention=AdaptiveAttention())
    adv_acc = advanced.evaluate_dataset(data)

    assert adv_acc > simple_acc
    assert len(advanced.global_workspace()) >= sum(d["ground_truth"] for d in data)
    assert advanced.metacognitive_accuracy() == adv_acc


def test_visualizer_hook_tracks_attention_and_memory():
    model = ConsciousnessAdvanced()
    snapshots = []

    def hook(data):
        snapshots.append(data)

    model.add_visualizer(hook)
    model.conscious_access({"score": 0.9, "ground_truth": 1})

    assert snapshots
    snap = snapshots[-1]
    assert "attention_scores" in snap and "memory" in snap
