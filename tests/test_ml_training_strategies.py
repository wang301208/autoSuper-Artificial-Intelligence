import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.ml import TrainingConfig
from backend.ml.models import MLP
from backend.ml.continual_trainer import ContinualTrainer
from backend.ml.multitask_trainer import MultiTaskTrainer


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
