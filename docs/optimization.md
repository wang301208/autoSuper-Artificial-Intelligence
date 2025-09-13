# Optimization Module

The optimization module records every algorithm run and recommends improved
parameters for future runs. It keeps a CSV file of historic parameter choices
and their resulting performance, enabling simple automatic tuning without manual
trial and error.

## Usage

```python
from modules.optimization import log_run, optimize_params

search_space = {"lr": [0.01, 0.1, 1.0], "batch": [16, 32, 64]}
params = optimize_params("my_algo", search_space)
# run your algorithm with ``params`` and compute a performance score
metrics = {"score": 0.8}
log_run("my_algo", params, metrics)
```

The first call samples random parameters from the provided search space. After
several runs the best performing configuration is returned, saving time otherwise
spent on manual tuning.
