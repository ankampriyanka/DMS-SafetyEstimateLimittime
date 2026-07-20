"""
Alertness regression model.

Uses gradient-boosted quantile regression (LightGBM) to predict:
  - Median estimate of remaining safe driving time (q=0.50)
  - Lower bound (q=0.10) and upper bound (q=0.90) for uncertainty interval
  - Alertness score (0–100)
  - Prediction confidence derived from interval width

Saves trained models to models/ directory.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from feature_pipeline import FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class AlertnessModel:
    """
    Quantile regression ensemble predicting time-to-fatigue with uncertainty.
    Falls back to sklearn GradientBoostingRegressor if LightGBM unavailable.
    """

    QUANTILES = [0.10, 0.50, 0.90]

    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()
        self.trained = False

    def _make_lgb_model(self, alpha: float):
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            verbose=-1,
        )

    def _make_sklearn_model(self, alpha: float):
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
        )

    def fit(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].values
        y = df["time_to_fatigue_min"].values

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        for q in self.QUANTILES:
            if LGB_AVAILABLE:
                m = self._make_lgb_model(q)
            else:
                m = self._make_sklearn_model(q)
            m.fit(X_train, y_train)
            self.models[q] = m

        # Validation MAE on median
        val_pred = self.models[0.50].predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        print(f"Validation MAE (median): {mae:.2f} min")

        self.trained = True
        return self

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict alertness metrics for a single feature vector or batch.
        Returns dict with keys: estimate_min, lower_min, upper_min,
                                alertness_score, confidence_pct, break_window_min
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        X = self.scaler.transform(features)

        p10 = self.models[0.10].predict(X)
        p50 = self.models[0.50].predict(X)
        p90 = self.models[0.90].predict(X)

        # Clip to reasonable range
        p10 = np.clip(p10, 0, 120)
        p50 = np.clip(p50, 0, 120)
        p90 = np.clip(p90, 0, 120)

        interval_width = p90 - p10
        # Confidence: narrower interval → higher confidence (max 95%)
        confidence = np.clip(1.0 - (interval_width / 60.0), 0.50, 0.95) * 100

        # Alertness score derived from median estimate (capped at 100)
        alertness = np.clip((p50 / 60.0) * 100, 0, 100)

        # Recommended break window: slightly inside the lower bound
        break_window = np.clip(p10 * 0.85, 1, p50)

        if features.shape[0] == 1:
            return {
                "estimate_min": float(p50[0]),
                "lower_min": float(p10[0]),
                "upper_min": float(p90[0]),
                "alertness_score": float(alertness[0]),
                "confidence_pct": float(confidence[0]),
                "break_window_min": float(break_window[0]),
            }

        return {
            "estimate_min": p50,
            "lower_min": p10,
            "upper_min": p90,
            "alertness_score": alertness,
            "confidence_pct": confidence,
            "break_window_min": break_window,
        }

    def save(self, path: str = None):
        path = path or str(MODEL_DIR / "alertness_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "scaler": self.scaler}, f)
        print(f"Model saved to {path}")

    def load(self, path: str = None):
        path = path or str(MODEL_DIR / "alertness_model.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.models = obj["models"]
        self.scaler = obj["scaler"]
        self.trained = True
        return self


def train_and_save(label_dir: str):
    """End-to-end: build features from labels, train, save model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_pipeline import build_feature_matrix

    print("Building feature matrix...")
    df = build_feature_matrix(label_dir)
    print(f"  {len(df)} frames, columns: {list(df.columns)}")

    print("Training model...")
    model = AlertnessModel()
    model.fit(df)
    model.save()
    return model, df


if __name__ == "__main__":
    import sys
    label_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw"
    train_and_save(label_dir)
