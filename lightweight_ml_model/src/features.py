"""Feature extraction for fixed IMU windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "acc_mag_mean", "acc_mag_std", "acc_mag_max", "acc_mag_range",
    "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max", "gyro_mag_range",
    "acc_sma", "gyro_sma",
]


def extract_features(window: pd.DataFrame) -> dict[str, float]:
    """Return numeric motion features for one session-sized IMU window."""
    required = {"ax", "ay", "az", "gx", "gy", "gz"}
    missing = required.difference(window.columns)
    if missing:
        raise ValueError(f"IMU window is missing columns: {sorted(missing)}")

    acc = window[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyro = window[["gx", "gy", "gz"]].to_numpy(dtype=float)
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    return {
        "acc_mag_mean": float(acc_mag.mean()),
        "acc_mag_std": float(acc_mag.std()),
        "acc_mag_max": float(acc_mag.max()),
        "acc_mag_range": float(np.ptp(acc_mag)),
        "gyro_mag_mean": float(gyro_mag.mean()),
        "gyro_mag_std": float(gyro_mag.std()),
        "gyro_mag_max": float(gyro_mag.max()),
        "gyro_mag_range": float(np.ptp(gyro_mag)),
        "acc_sma": float(np.abs(acc).sum(axis=1).mean()),
        "gyro_sma": float(np.abs(gyro).sum(axis=1).mean()),
    }


def build_feature_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IMU samples into one labeled feature row per session."""
    rows = []
    for session_id, group in samples.groupby("session_id", sort=False):
        row = {"session_id": session_id, **extract_features(group)}
        row["label"] = group["label"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)
