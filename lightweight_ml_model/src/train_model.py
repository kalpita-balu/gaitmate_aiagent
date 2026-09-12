"""Calibrate a transparent, dependency-free IMU threshold model.

This temporary Week 1 fallback replaces the blocked scikit-learn/SciPy stack.
It learns thresholds from labelled sessions but is not a clinical ML model.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_feature_table

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "synthetic_imu_streams.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "fall_threshold_model.json"

PRIMARY_FEATURES = ("acc_mag_max", "gyro_mag_max")


def separation_threshold(event_rows: pd.DataFrame, baseline_rows: pd.DataFrame, feature: str) -> float:
    """Place the cutoff midway between the event and baseline training ranges."""
    event_minimum = float(event_rows[feature].min())
    baseline_maximum = float(baseline_rows[feature].max())
    if event_minimum <= baseline_maximum:
        raise ValueError(f"Training ranges overlap for {feature}; a safe threshold cannot be inferred.")
    return (event_minimum + baseline_maximum) / 2


def predict_label(row: pd.Series, model: dict) -> str:
    fall = model["fall_thresholds"]
    instability = model["instability_thresholds"]
    if all(float(row[name]) >= fall[name] for name in PRIMARY_FEATURES):
        return "fall"
    if all(float(row[name]) >= instability[name] for name in instability):
        return "instability"
    return "normal"


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing {DATASET}. Run generate_synthetic_data.py first.")
    feature_table = build_feature_table(pd.read_csv(DATASET))
    # Hold out whole sessions. Every fourth synthetic session is evaluation data.
    session_number = feature_table["session_id"].str.extract(r"(\d+)$").astype(int)[0]
    train = feature_table[session_number % 4 != 0]
    test = feature_table[session_number % 4 == 0]
    model = {
        "model_type": "transparent_imu_threshold_baseline",
        "model_version": "week1-fallback-1",
        "primary_features": list(PRIMARY_FEATURES),
        "fall_thresholds": {
            feature: separation_threshold(train[train["label"] == "fall"], train[train["label"] != "fall"], feature)
            for feature in PRIMARY_FEATURES
        },
        "instability_thresholds": {
            "gyro_mag_max": separation_threshold(
                train[train["label"] == "instability"], train[train["label"] == "normal"], "gyro_mag_max"
            )
        },
        "notes": "Development-only baseline. Scores are threshold strengths, not probabilities.",
    }
    predictions = test.apply(lambda row: predict_label(row, model), axis=1)
    fall_rows = test["label"] == "fall"
    fall_recall = float((predictions[fall_rows] == "fall").mean())
    metrics = {
        "fall_recall": fall_recall,
        "test_sessions": int(len(test)),
        "fall_test_sessions": int(fall_rows.sum()),
        "fall_false_negatives": int((predictions[fall_rows] != "fall").sum()),
        "prediction_counts": {label: int((predictions == label).sum()) for label in ("normal", "instability", "fall")},
        "model_limitations": "Synthetic-data result only. Do not use for clinical or emergency decisions.",
    }
    ARTIFACTS.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Fall recall: {fall_recall:.3f}")
    print(f"Saved threshold model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
