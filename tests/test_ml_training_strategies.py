import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.ml import TrainingConfig
from backend.ml.models import MLP
from backend.ml.continual_trainer import ContinualTrainer
from backend.ml.multitask_trainer import MultiTaskTrainer

try:  # pragma: no cover - optional dependency
    from lion_pytorch import Lion
except Exception:  # pragma: no cover
    Lion = None  # type: ignore


def _write_dataset(path: Path, rows: int = 5) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "target"])
        for i in range(rows):
            writer.writerow([f"sample {i}", i])


def test_continual_trainer_strategy_switch(tmp_path):
    log_file = tmp_path / "logs.csv"
    # initialise log with header
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        writer.writeheader()

    cfg_off = TrainingConfig(optimizer="adam", checkpoint_dir=tmp_path / "ckpt1")
    trainer = ContinualTrainer(cfg_off, log_file)
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        writer.writerow({"prompt": "q", "completion": "a"})
    trainer.train()
    assert trainer.optimizer == "adam"
    assert trainer.scheduler is None
    assert not trainer.adversarial_hook_called
    assert not trainer.curriculum_hook_called
    assert not trainer.early_stopped

    cfg_on = TrainingConfig(
        optimizer="adamw",
        lr_scheduler="linear",
        early_stopping_patience=1,
        use_adversarial=True,
        use_curriculum=True,
        use_ewc=True,
        use_orthogonal=True,
        checkpoint_dir=tmp_path / "ckpt2",
    )
    trainer_on = ContinualTrainer(cfg_on, log_file)
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        writer.writerow({"prompt": "q2", "completion": "a2"})
    trainer_on.train()
    assert trainer_on.optimizer == "adamw"
    assert trainer_on.scheduler == "linear"
    assert trainer_on.adversarial_hook_called
    assert trainer_on.curriculum_hook_called
    assert trainer_on.ewc_hook_called
    assert trainer_on.orthogonal_hook_called
    assert trainer_on.early_stopped


def test_multitask_trainer_strategy_switch(tmp_path):
    data = tmp_path / "task.csv"
    _write_dataset(data)

    cfg_off = TrainingConfig(optimizer="adam")
    trainer = MultiTaskTrainer({"t": str(data)}, cfg_off)
    trainer.train()
    assert trainer.optimizer == "adam"
    assert trainer.scheduler is None
    assert not trainer.adversarial_hook_called
    assert not trainer.curriculum_hook_called
    assert not trainer.early_stopped

    cfg_on = TrainingConfig(
        optimizer="lion",
        lr_scheduler="cosine",
        early_stopping_patience=1,
        use_adversarial=True,
        use_curriculum=True,
    )
    trainer_on = MultiTaskTrainer({"t": str(data)}, cfg_on)
    trainer_on.train()
    assert trainer_on.optimizer == "lion"
    assert trainer_on.scheduler == "cosine"
    assert trainer_on.adversarial_hook_called
    assert trainer_on.curriculum_hook_called
    assert trainer_on.early_stopped


def test_trainer_model_selection(tmp_path):
    data = tmp_path / "task.csv"
    _write_dataset(data)

    cfg = TrainingConfig(task_model_types={"t": "mlp"})
    trainer = MultiTaskTrainer({"t": str(data)}, cfg)
    results = trainer.train()
    model, _ = results["t"]
    assert isinstance(model, MLP)


@pytest.mark.parametrize(
    "opt_name,opt_cls",
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("lion", Lion),
    ],
)
def test_continual_trainer_optimizer_instances(tmp_path, opt_name, opt_cls):
    if opt_name == "lion" and Lion is None:
        pytest.skip("lion optimizer not available")
    log_file = tmp_path / f"logs_{opt_name}.csv"
    cfg = TrainingConfig(optimizer=opt_name, checkpoint_dir=tmp_path / f"ckpt_{opt_name}")
    trainer = ContinualTrainer(cfg, log_file)
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
        writer.writerow({"text": "sample", "reward": "1"})
    trainer.train()
    assert isinstance(trainer.torch_optimizer, opt_cls)


@pytest.mark.parametrize(
    "opt_name,opt_cls",
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("lion", Lion),
    ],
)
def test_multitask_trainer_optimizer_instances(tmp_path, opt_name, opt_cls):
    if opt_name == "lion" and Lion is None:
        pytest.skip("lion optimizer not available")
    data = tmp_path / f"task_{opt_name}.csv"
    with data.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "target"])
        for i in range(5):
            writer.writerow([f"sample {i}", i])
    cfg = TrainingConfig(optimizer=opt_name)
    trainer = MultiTaskTrainer({"t": str(data)}, cfg)
    trainer.train()
    assert isinstance(trainer.torch_optimizers["t"], opt_cls)


def test_unsupported_optimizer_raises(tmp_path):
    log_file = tmp_path / "logs.csv"
    cfg = TrainingConfig(optimizer="sgd", checkpoint_dir=tmp_path / "ckpt")
    trainer = ContinualTrainer(cfg, log_file)
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
        writer.writerow({"text": "sample", "reward": "1"})
    with pytest.raises(ValueError):
        trainer.train()
