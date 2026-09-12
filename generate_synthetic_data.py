"""Generate development-only synthetic IMU windows for GaitMate."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
SAMPLE_RATE_HZ = 50
WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthetic_imu_streams.csv"


def make_window(label: str, session_number: int) -> pd.DataFrame:
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE_HZ
    phase = RNG.uniform(0, 2 * np.pi)
    ax = 0.45 * np.sin(2 * np.pi * 1.4 * t + phase) + RNG.normal(0, 0.12, WINDOW_SAMPLES)
    ay = 0.30 * np.cos(2 * np.pi * 1.4 * t + phase) + RNG.normal(0, 0.10, WINDOW_SAMPLES)
    az = 9.81 + 0.70 * np.sin(2 * np.pi * 1.4 * t + phase) + RNG.normal(0, 0.15, WINDOW_SAMPLES)
    gx = 18 * np.sin(2 * np.pi * 1.4 * t + phase) + RNG.normal(0, 4, WINDOW_SAMPLES)
    gy = 12 * np.cos(2 * np.pi * 1.4 * t + phase) + RNG.normal(0, 3, WINDOW_SAMPLES)
    gz = 7 * np.sin(2 * np.pi * 0.7 * t + phase) + RNG.normal(0, 2, WINDOW_SAMPLES)

    if label == "instability":
        start = RNG.integers(50, 110)
        length = RNG.integers(35, 65)
        disturbance = np.hanning(length)
        ax[start:start + length] += 2.4 * disturbance
        ay[start:start + length] -= 1.8 * disturbance
        gx[start:start + length] += 110 * disturbance
        gy[start:start + length] -= 85 * disturbance
    elif label == "fall":
        start = RNG.integers(70, 120)
        length = 18
        impact = np.hanning(length)
        ax[start:start + length] += 9.5 * impact
        ay[start:start + length] -= 6.5 * impact
        az[start:start + length] += 14.0 * impact
        gx[start:start + length] += 420 * impact
        gy[start:start + length] -= 300 * impact
        gz[start:start + length] += 180 * impact

    return pd.DataFrame({
        "session_id": f"{label}_{session_number:03d}",
        "timestamp_ms": np.arange(WINDOW_SAMPLES) * (1000 // SAMPLE_RATE_HZ),
        "ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz,
        "label": label,
    })


def main() -> None:
    windows = [make_window(label, number) for label in ("normal", "instability", "fall") for number in range(1, 121)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(windows, ignore_index=True).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(windows)} labeled windows")


if __name__ == "__main__":
    main()
