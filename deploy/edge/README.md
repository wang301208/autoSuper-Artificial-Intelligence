# Edge Deployment

This directory contains examples and documentation for running AutoGPT on edge
devices. The utilities in `backend.edge` provide:

- **Model optimisation** using [ONNX Runtime](https://onnxruntime.ai/) or
  [TensorRT](https://developer.nvidia.com/tensorrt) to prune and quantise models
  into lightweight formats.
- **Edge I/O helpers** for simple sensor input and output handling.
- **Resource management** utilities to monitor CPU, memory and GPU availability.

## Prerequisites

Optional dependencies can be installed with:

```bash
pip install .[edge]
```

## Example

Run the sample script:

```bash
python example.py
```

The script demonstrates optimising a model and using the edge I/O and resource
management interfaces.
