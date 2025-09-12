# Benchmarks

This directory contains simple benchmarking utilities for AutoGPT.

## Cross-domain reasoning suite

`run_cross_domain.py` executes a minimal benchmark across two reasoning
domains: a logic puzzle and a knowledge retrieval task. After each task the
benchmark reflects on its performance, adapts its strategy, and records the
results.

### Running the suite

```
python benchmarks/run_cross_domain.py
```

### Tests

The suite is exercised in continuous integration by
`benchmarks/tests/test_cross_domain.py`.

```
pytest benchmarks/tests/test_cross_domain.py
```

## PSO comparison

`pso_benchmark.py` contrasts the default static PSO parameters with an adaptive
schedule for the inertia, cognitive and social coefficients. It reports the
best objective value achieved by both configurations and logs the parameter
trajectory for the adaptive run.

### Running the benchmark

```
python benchmarks/pso_benchmark.py
```
