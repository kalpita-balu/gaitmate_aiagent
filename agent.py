"""Week 2 handoff point for the GaitMate threshold baseline."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from lightweight_ml_model.src.features import extract_features


class GaitMateAgent:
    """Classify a completed IMU window without scikit-learn or SciPy.

    This temporary baseline uses trained thresholds. The returned score is a
    threshold-strength measure, not a calibrated probability or medical advice.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or Path(__file__).resolve().parent / "lightweight_ml_model" / "artifacts" / "fall_threshold_model.json"
        self.model = json.loads(path.read_text(encoding="utf-8"))
        self.primary_features = self.model["primary_features"]

    def _strength(self, features: dict[str, float], thresholds: dict[str, float]) -> float:
        return min(features[name] / thresholds[name] for name in thresholds)

    def assess_window(self, window: pd.DataFrame) -> dict[str, float | str]:
        features = extract_features(window)
        fall_strength = self._strength(features, self.model["fall_thresholds"])
        instability_strength = self._strength(features, self.model["instability_thresholds"])
        if fall_strength >= 1:
            label, severity, score = "fall", "CRITICAL", fall_strength
        elif instability_strength >= 1:
            label, severity, score = "instability", "WARNING", instability_strength
        else:
            label, severity, score = "normal", "NORMAL", instability_strength
        return {"label": label, "severity": severity, "decision_score": round(float(score), 3)}
