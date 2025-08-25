# ResourceRL CPU/GPU Benchmark

This benchmark compares training and evaluation performance of the
``ResourceRL`` agent on CPU versus GPU.

## Hardware requirements

- **CPU:** Any modern processor.
- **Optional GPU:** NVIDIA GPU with CUDA or AMD GPU with ROCm. PyTorch must be
  installed with the appropriate backend and at least 1 GB of memory is
  recommended.

## Running the benchmark

```bash
python benchmark/resource_rl_benchmark.py
```

The script executes the workload on the CPU and, when available, on a
CUDA/ROCm-capable GPU.  It reports the runtime for both paths, allowing you to
compare the speedup achieved with hardware acceleration.

