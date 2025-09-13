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
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:  # Optional dependency for graph parsing
    import networkx as nx
except Exception:  # pragma: no cover - dependency may be missing at runtime
    nx = None  # type: ignore

from .feature_extractor import (
    FeatureExtractor,
    GraphFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


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


def _build_extractor(args) -> Any:
    if args.feature_type == "sentence":
        return FeatureExtractor(
            method="sentence",
            use_pca=args.use_pca,
            n_components=args.pca_components,
            use_feature_selection=args.use_feature_selection,
            var_threshold=args.var_threshold,
        )
    if args.feature_type == "time_series":
        return TimeSeriesFeatureExtractor(
            window_size=args.window_size, apply_fft=args.apply_fft
        )
    if args.feature_type == "graph":
        return GraphFeatureExtractor(dimensions=args.embedding_dim)
    return FeatureExtractor(
        method="tfidf",
        use_pca=args.use_pca,
        n_components=args.pca_components,
        use_feature_selection=args.use_feature_selection,
        var_threshold=args.var_threshold,
    )


def _prepare_inputs(texts, args, extractor) -> Any:
    if args.feature_type == "time_series":
        series_list = [list(map(float, t.split())) for t in texts]
        return extractor.fit_transform(series_list)
    if args.feature_type == "graph":
        if nx is None:  # type: ignore[name-defined]
            raise ImportError("networkx is required for graph features")
        graphs = []
        for t in texts:
            edges = [tuple(e.split("-")) for e in t.split()]
            g = nx.Graph()
            g.add_edges_from(edges)
            graphs.append(g)
        return extractor.fit_transform(graphs)
    return extractor.fit_transform(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on log data")
    parser.add_argument("data", help="Path to CSV file with columns 'text' and 'target'")
    parser.add_argument("--model", choices=["linear", "random_forest"], default="linear")
    parser.add_argument("--version", default="v1", help="Version string for saved artifacts")
    parser.add_argument(
        "--feature-type",
        choices=["tfidf", "sentence", "time_series", "graph"],
        default="tfidf",
    )
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--use-feature-selection", action="store_true")
    parser.add_argument("--var-threshold", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--apply-fft", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()

    texts, targets = load_data(args.data)

    extractor = _build_extractor(args)
    X = _prepare_inputs(texts, args, extractor)

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
