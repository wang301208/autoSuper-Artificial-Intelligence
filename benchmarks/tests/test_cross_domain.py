"""Tests for the cross-domain benchmark suite."""

from benchmarks.run_cross_domain import CrossDomainBenchmark


def test_cross_domain_benchmark_adapts_strategy() -> None:
    benchmark = CrossDomainBenchmark()
    results = benchmark.run()

    assert len(results) == 2
    assert {r.domain for r in results} == {"logic", "knowledge"}
    # Ensure that the strategy changed after the first task
    assert results[0].strategy != results[1].strategy
