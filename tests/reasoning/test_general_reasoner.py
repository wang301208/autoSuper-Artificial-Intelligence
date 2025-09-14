from modules.brain.reasoning import GeneralReasoner


def test_reason_about_unknown_produces_hypotheses():
    reasoner = GeneralReasoner()
    steps = reasoner.reason_about_unknown("Investigate flarblax anomalies")
    assert steps, "should return at least one step"
    for step in steps:
        assert step["hypothesis"]
        assert step["verification"]


def test_reason_about_unknown_uses_available_knowledge():
    reasoner = GeneralReasoner()
    reasoner.add_concept_relation("acid", "base")
    reasoner.analogical.add_knowledge(
        "default", {"subject": "acid", "object": "base"}, "acid neutralizes base"
    )
    reasoner.add_example("mixing acid with base", "produces salt")

    steps = reasoner.reason_about_unknown("acid reaction with unknown", max_steps=3)

    assert any("acid relates to base" in s["hypothesis"] for s in steps)
    assert any("analogy" in s["hypothesis"] for s in steps)
    assert any("produces salt" in s["verification"] for s in steps)
