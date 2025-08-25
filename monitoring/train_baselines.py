"""Train baseline models on log data.

This script expects a CSV file containing at least two columns:
``log`` with the raw log text and ``target`` with the numerical target value.
It splits the data into train/validation/test sets, trains a linear regression
and a random forest regressor, selects the best model on the validation set,
and serializes the model together with the feature extractor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .feature_extractor import FeatureExtractor


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path | str) -> Tuple[list[str], list[float]]:
    """Return log texts and targets from CSV *path*."""
    df = pd.read_csv(path)
    if "log" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must contain 'log' and 'target' columns")
    logs = df["log"].astype(str).tolist()
    targets = df["target"].tolist()
    return logs, targets


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_models(X, y) -> tuple:
    """Train baseline models and return the best-performing one."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores[name] = mean_squared_error(y_val, preds)

    best_name = min(scores, key=scores.get)
    best_model = models[best_name]
    test_score = mean_squared_error(y_test, best_model.predict(X_test))

    return best_name, best_model, scores[best_name], test_score


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on log data")
    parser.add_argument("csv", help="Path to CSV file containing log and target columns")
    parser.add_argument(
        "--output", default="monitoring/models", help="Directory to store serialized artifacts"
    )
    args = parser.parse_args()

    logs, targets = load_dataset(args.csv)

    extractor = FeatureExtractor()
    features = extractor.fit_transform(logs)

    model_name, model, val_mse, test_mse = train_models(features, targets)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    model_path = output_dir / f"{model_name}_{timestamp}.joblib"
    vec_path = output_dir / f"feature_extractor_{timestamp}.joblib"
    meta_path = output_dir / f"metadata_{timestamp}.json"

    joblib.dump(model, model_path)
    extractor.save(vec_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "model_path": model_path.name,
                "vectorizer_path": vec_path.name,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Validation MSE: {val_mse:.4f}; Test MSE: {test_mse:.4f}")


if __name__ == "__main__":
    main()
