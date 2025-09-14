"""Example script for running AutoGPT components on edge devices."""
from __future__ import annotations

from pathlib import Path

from backend.edge import EdgeIO, EdgeResourceManager, optimize_model


def main() -> None:
    io = EdgeIO("./data")
    resources = EdgeResourceManager()

    print("CPU usage:", resources.cpu_usage())
    print("Memory usage:", resources.memory_usage())
    print("GPU available:", resources.gpu_available())

    # Demonstrate sensor read/write
    io.write_output("status", "edge device running")
    print("Sensor value:", io.read_sensor("temperature"))

    # Demonstrate model optimisation
    model_path = Path("./model.onnx")
    if model_path.exists():
        try:
            out_path = optimize_model(model_path, backend="onnxruntime")
            print("Optimised model written to", out_path)
        except RuntimeError as exc:
            print("Optimisation skipped:", exc)
    else:
        print("Model file not found; skipping optimisation")


if __name__ == "__main__":
    main()
