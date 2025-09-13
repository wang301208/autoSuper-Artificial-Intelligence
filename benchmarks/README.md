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

## A/B testing framework

The `ab_testing` module enables side-by-side evaluation of two algorithm
variants. `run_ab_test.py` executes both versions in parallel on a synthetic
classification dataset, collects accuracy and timing metrics, and performs
basic statistical analysis (confidence intervals and a paired t-test).

### Running the benchmark

```
python benchmarks/run_ab_test.py --algoA benchmarks.ab_testing.sample_algorithms:algo_random --algoB benchmarks.ab_testing.sample_algorithms:algo_knn
```

### CI integration

In continuous integration, run the script with your production algorithm as
`--algoA` and the candidate change as `--algoB`. Fail the pipeline if the
reported p-value indicates a significant drop in accuracy.
