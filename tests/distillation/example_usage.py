"""Example usage of TeacherStudentTrainer."""

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch missing
    torch = None

from backend.ml.distillation import DistillationConfig, TeacherStudentTrainer


if __name__ == "__main__" and torch is not None:
    # Create dummy dataset
    x = torch.randn(100, 10)
    y = (x.sum(dim=1) > 0).long()
    dl = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    # Teacher and student share architecture for simplicity
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    teacher, student = Net(), Net()

    # Pretend teacher is pre-trained
    cfg = DistillationConfig(feature_layers=["fc1"], temperature=2.0)
    trainer = TeacherStudentTrainer(teacher, student, cfg)
    trainer.train(dl, epochs=1)
    print("Distillation completed")
