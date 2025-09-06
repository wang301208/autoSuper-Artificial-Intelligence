from __future__ import annotations

import argparse
import yaml

from backend.ml.meta_learning.maml import MAML, load_task


def run_meta_learning(config_path: str) -> None:
    """Run meta-learning based on the given YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    tasks_cfg = cfg.get("tasks", [])
    if not tasks_cfg:
        raise ValueError("No tasks specified in configuration")

    tasks = [load_task(task["dataset"]) for task in tasks_cfg]
    input_dim = tasks[0].support_x.shape[1]

    maml = MAML(
        input_dim=input_dim,
        inner_lr=cfg.get("inner_lr", 0.01),
        meta_lr=cfg.get("meta_lr", 0.001),
        adapt_steps=cfg.get("adapt_steps", 1),
    )
    history = maml.meta_train(tasks, epochs=cfg.get("epochs", 1))
    for epoch, loss in enumerate(history, 1):
        print(f"Epoch {epoch}: loss={loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training utilities")
    parser.add_argument("--meta", action="store_true", help="Run meta-learning")
    parser.add_argument(
        "--config", type=str, default="config/meta_learning.yaml", help="Config path"
    )
    args = parser.parse_args()

    if args.meta:
        run_meta_learning(args.config)
    else:
        print("Standard training routine not implemented. Use --meta for meta-learning.")


if __name__ == "__main__":
    main()
