import os
import sys

import pytest

# Ensure the repository root is on the Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.reasoning import CommonSenseReasoner


class DummyReasoner:
    def infer(self, text):
        return []


def evaluate(reasoner):
    dataset = [
        ("What is a dog?", "mammal"),
        ("What is an apple?", "fruit"),
    ]
    correct = 0
    for question, expected in dataset:
        results = reasoner.infer(question)
        if any(expected in r["conclusion"].lower() for r in results):
            correct += 1
    return correct / len(dataset)


def test_commonsense_reasoner_improves_accuracy():
    baseline = evaluate(DummyReasoner())
    commonsense = evaluate(CommonSenseReasoner())
    assert commonsense > baseline
    assert commonsense >= 0.5
