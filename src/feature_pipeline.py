"""
Feature extraction pipeline.

Reads YOLO-format labels from data/raw, maps class IDs to driver state scores,
and builds rolling time-series windows used by the alertness model.

Class map (from State Farm / local labels):
  0 = neutral (alert)
  1 = microsleep
  2 = distraction
  3 = phone use
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Alertness penalty per class (0 = fully alert, 1 = fully impaired)
CLASS_IMPAIRMENT = {0: 0.0, 1: 0.95, 2: 0.55, 3: 0.70}
CLASS_NAMES = {0: "neutral", 1: "microsleep", 2: "distraction", 3: "phone_use"}

WINDOW_SIZES = [5, 10, 20]  # rolling window lengths in frames


def load_labels(label_dir: str) -> pd.DataFrame:
    """Parse all YOLO .txt label files into a flat DataFrame."""
    records = []
    files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    for i, fpath in enumerate(files):
        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            try:
                cls = int(parts[0])
            except ValueError:
                continue  # skip classes.txt or text-label rows
            cx, cy, w, h = map(float, parts[1:5])
            records.append({
                "frame_idx": i,
                "filename": Path(fpath).stem,
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, "unknown"),
                "bbox_cx": cx,
                "bbox_cy": cy,
                "bbox_w": w,
                "bbox_h": h,
                "impairment": CLASS_IMPAIRMENT.get(cls, 0.5),
            })
    return pd.DataFrame(records)


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per frame: dominant class + impairment score.
    When multiple detections exist for a frame, take the highest impairment.
    """
    ts = (
        df.groupby("frame_idx")
        .agg(
            filename=("filename", "first"),
            dominant_class=("class_id", lambda x: x.iloc[x.map(CLASS_IMPAIRMENT).argmax()]),
            impairment=("impairment", "max"),
        )
        .reset_index()
    )
    ts["alertness_raw"] = 1.0 - ts["impairment"]
    return ts


def add_rolling_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std of alertness and impairment event flags."""
    for w in WINDOW_SIZES:
        ts[f"alert_mean_{w}"] = ts["alertness_raw"].rolling(w, min_periods=1).mean()
        ts[f"alert_std_{w}"] = ts["alertness_raw"].rolling(w, min_periods=1).std().fillna(0)
        ts[f"impair_rate_{w}"] = (
            (ts["dominant_class"] != 0).astype(int).rolling(w, min_periods=1).mean()
        )

    # Microsleep flag rate
    for w in WINDOW_SIZES:
        ts[f"microsleep_rate_{w}"] = (
            (ts["dominant_class"] == 1).astype(int).rolling(w, min_periods=1).mean()
        )

    return ts


def estimate_time_to_fatigue(ts: pd.DataFrame, fps: float = 1.0) -> pd.DataFrame:
    """
    Heuristic ground-truth label for training:
    For each frame, look forward and find how many frames until cumulative
    impairment exceeds a fatigue threshold. Convert to minutes.
    """
    impair_vals = ts["impairment"].values
    n = len(impair_vals)
    fatigue_threshold = 0.6  # cumulative impairment trigger
    window_cap = 300         # max look-ahead frames (~5 min at 1 fps)

    ttf = []
    for i in range(n):
        cum = 0.0
        frames_ahead = 0
        for j in range(i, min(i + window_cap, n)):
            cum += impair_vals[j]
            frames_ahead = j - i + 1
            if cum >= fatigue_threshold:
                break
        else:
            frames_ahead = window_cap
        ttf.append(frames_ahead / fps / 60.0)  # convert to minutes

    ts["time_to_fatigue_min"] = ttf
    # Alertness score 0–100
    ts["alertness_score"] = (ts["alert_mean_10"] * 100).clip(0, 100).round(1)
    return ts


def build_feature_matrix(label_dir: str, fps: float = 1.0) -> pd.DataFrame:
    """Full pipeline: labels → feature matrix ready for modelling."""
    df = load_labels(label_dir)
    ts = build_time_series(df)
    ts = add_rolling_features(ts)
    ts = estimate_time_to_fatigue(ts, fps=fps)
    return ts


FEATURE_COLS = (
    [f"alert_mean_{w}" for w in WINDOW_SIZES]
    + [f"alert_std_{w}" for w in WINDOW_SIZES]
    + [f"impair_rate_{w}" for w in WINDOW_SIZES]
    + [f"microsleep_rate_{w}" for w in WINDOW_SIZES]
    + ["alertness_raw"]
)
