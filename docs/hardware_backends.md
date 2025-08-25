# Hardware Backends

AutoGPT's machine learning utilities can operate on a variety of hardware
through a small pluggable backend system. Three backends are provided:

- **CPU** – uses [NumPy](https://numpy.org) and is always available.
- **GPU** – uses [CuPy](https://cupy.dev) for CUDA compatible devices.
- **TPU** – uses [JAX](https://github.com/google/jax) for TPU accelerators.

The backend is selected automatically by attempting GPU, then TPU, and finally
falling back to CPU. You can force a specific backend by setting the
`AUTOGPT_DEVICE` environment variable to `"gpu"`, `"tpu"` or `"cpu"`.

If the requested backend or its dependencies are not installed the system will
gracefully fall back to the CPU backend. This makes development on machines
without specialised hardware straightforward while still enabling acceleration
when available.

## Emulation and setup

For development without access to GPUs/TPUs you may install the relevant
libraries to emulate the device on the CPU. For example, installing `cupy` on a
CPU-only machine will allow the GPU backend to run using a slower CUDA emulator.

Refer to the respective library documentation for detailed installation
instructions and hardware driver requirements.
