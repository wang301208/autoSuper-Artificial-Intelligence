# Modules

This directory contains optional modules and utilities for AutoGPT.

## Installation

Install module-specific dependencies separately to avoid conflicts with the main
environment:

```bash
pip install -r requirements.txt
```

This installs only the packages required by the modules under this directory. The
project now includes a `ModernDependencyManager` which will attempt to resolve and
install missing packages automatically using `pip`/`importlib` when modules are
loaded.

## Optimization

The ``optimization`` module provides lightweight parameter tuning utilities. It
records the parameters and metrics of each algorithm run to a CSV file and can
recommend improved parameters for future runs.

See [docs/optimization.md](../docs/optimization.md) for usage details.
