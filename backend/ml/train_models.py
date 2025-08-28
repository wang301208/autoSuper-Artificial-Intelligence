"""Train baseline models on log data.

This script expects a CSV file with two columns:
``text`` containing the raw log message and ``target`` containing the numeric
value to predict.  It splits the dataset into train, validation and test sets,
trains a simple model and stores the model together with the fitted
``FeatureExtractor``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .feature_extractor import FeatureExtractor


def load_data(path: str):
    df = pd.read_csv(path)
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'target' columns")
    return df["text"].astype(str).tolist(), df["target"].values


def train_model(model_name: str, X_train, y_train):
    if model_name == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on log data")
    parser.add_argument("data", help="Path to CSV file with columns 'text' and 'target'")
    parser.add_argument("--model", choices=["linear", "random_forest"], default="linear")
    parser.add_argument("--version", default="v1", help="Version string for saved artifacts")
    args = parser.parse_args()

    texts, targets = load_data(args.data)

    extractor = FeatureExtractor()
    X = extractor.fit_transform(texts)

    X_train, X_temp, y_train, y_temp = train_test_split(X, targets, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = train_model(args.model, X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_mse = mean_squared_error(y_val, val_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    artifacts_dir = Path("artifacts") / args.version
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / f"{args.model}_model.joblib")
    extractor.save(str(artifacts_dir / "feature_extractor.joblib"))
    with open(artifacts_dir / "metrics.txt", "w") as f:
        f.write(f"Validation MSE: {val_mse}\nTest MSE: {test_mse}\n")


if __name__ == "__main__":
    main()
