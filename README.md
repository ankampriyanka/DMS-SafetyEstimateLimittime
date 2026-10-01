> **Portfolio focus:** Automotive AI · Driver Monitoring Systems · Predictive Safety
>
> This repository contains the predictive driver-alertness / remaining-safe-time prototype, a distinct research stream from the AI Trust Score work.

---

# DMS SafetyEstimateLimit — Predictive Driver Alertness System

> A prototype that shifts Driver Monitoring Systems from **reactive fatigue detection** to **predictive alertness estimation** — quantifying how much safe driving time remains *before* fatigue onset, rather than alerting only after it begins.

---

## Table of Contents

1. [Business Problem](#1-business-problem)
2. [Solution Approach](#2-solution-approach)
3. [Architecture](#3-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Model](#5-model)
6. [Live Dashboard Output](#6-live-dashboard-output)
7. [Dataset](#7-dataset)
8. [Project Structure](#8-project-structure)
9. [Quick Start](#9-quick-start)
10. [Extending to Real-Time](#10-extending-to-real-time)
11. [Roadmap](#11-roadmap)
12. [Related Work](#12-related-work)

---

## 1. Business Problem

At SAE Level 2 (L2) ADAS, the human driver remains legally and physically responsible for vehicle control at all times. Driver Monitoring Systems are required to ensure the driver remains attentive — but **current DMS are almost entirely reactive**:

| Current state | Problem |
|---|---|
| Alert fires when eyes close | Fatigue onset has already happened — it is too late for a gradual warning |
| Binary awake / drowsy output | Gives no indication of *how long* a driver can continue safely |
| No temporal context | A single frame cannot distinguish a brief blink from a microsleep trend |
| No uncertainty estimate | OEM cannot communicate reliability of the alert to the driver or the ADAS stack |

The safety gap is significant: a driver who has been degrading for 15 minutes before the alert fires has far less time to safely pull over than a driver warned at the first sign of onset.

**The goal of this prototype is to predict the remaining safe alert driving time window** — a continuous, proactive metric that enables the vehicle to issue graduated warnings before the driver becomes impaired.

---

## 2. Solution Approach

The prototype transforms per-frame YOLO behaviour detections into a **time-series regression problem**:

1. **Extract temporal features** — rolling windows of impairment rates, alertness trends, and microsleep frequency over the last 5, 10, and 20 frames capture the trajectory of driver state, not just the instantaneous snapshot.

2. **Generate a time-to-fatigue label** — for each frame, a look-ahead algorithm computes how many frames until cumulative impairment crosses a fatigue threshold. This continuous regression target (in minutes) replaces the binary alert/no-alert label.

3. **Quantile regression** — three separate models predict the 10th, 50th, and 90th percentile of remaining safe time. The interval width becomes the confidence score — a narrow interval means the model is certain; a wide interval signals high variability.

4. **Cockpit dashboard** — a Streamlit app replays the dataset frame-by-frame through a car-style HUD with SVG gauges, warning lights, and a rolling trend chart — simulating what an in-vehicle display would show.

The framework is designed to be **model-agnostic**: the feature pipeline and time-to-fatigue label generation work with any regression backbone. LightGBM is used as the default; sklearn GradientBoostingRegressor is the fallback if LightGBM is unavailable.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                    │
│                                                                          │
│   data/raw/                                                              │
│   YOLO .txt label files  +  .jpg frames                                  │
│   Format: class_id  cx  cy  w  h  (one detection per line)              │
│   Classes: 0=neutral  1=microsleep  2=distraction  3=phone_use           │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     src/feature_pipeline.py                              │
│                                                                          │
│  load_labels()          Parse all .txt files → flat DataFrame            │
│        │                (frame_idx, class_id, impairment, bbox)          │
│        ▼                                                                 │
│  build_time_series()    One row per frame                                │
│        │                dominant_class = highest-impairment detection    │
│        │                alertness_raw  = 1.0 - impairment                │
│        ▼                                                                 │
│  add_rolling_features() Windows: 5 / 10 / 20 frames                     │
│        │                alert_mean_W   rolling mean alertness            │
│        │                alert_std_W    alertness variability             │
│        │                impair_rate_W  fraction of impaired frames       │
│        │                microsleep_rate_W  microsleep event rate         │
│        ▼                                                                 │
│  estimate_time_to_fatigue()                                              │
│                         Look-ahead: frames until cumulative impairment   │
│                         >= 0.60  (capped at 300 frames / 5 min)          │
│                         → time_to_fatigue_min  (regression target)       │
│                         → alertness_score 0–100  (display metric)        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        src/model.py                                      │
│                                                                          │
│  AlertnessModel                                                          │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │  q = 0.10    │   │  q = 0.50    │   │  q = 0.90    │                │
│  │  Lower bound │   │  Median est. │   │  Upper bound │                │
│  │  of safe time│   │  (displayed) │   │  of safe time│                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│          │                 │                  │                          │
│          └─────────────────┼──────────────────┘                         │
│                            │                                             │
│  Confidence = 1 - (interval_width / 60)   capped [50%, 95%]            │
│  Break window  = lower_bound × 0.85        (safety margin)              │
│                                                                          │
│  Backbone: LightGBM LGBMRegressor (objective="quantile")                │
│  Fallback: sklearn GradientBoostingRegressor (loss="quantile")          │
│  Scaler:   MinMaxScaler on all 13 feature columns                        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      dashboard/app.py                                    │
│                                                                          │
│  Streamlit cockpit — replays dataset frame-by-frame                      │
│                                                                          │
│  ┌──────────────┐   ┌────────────────────────────┐   ┌────────────────┐ │
│  │  ALERTNESS   │   │  Status banner             │   │  SAFE TIME     │ │
│  │  SVG gauge   │   │  DRIVER ALERT /            │   │  SVG gauge     │ │
│  │   0 – 100    │   │  FATIGUE DETECTED /         │   │   0 – 60 min  │ │
│  │              │   │  DROWSY PULL OVER           │   │               │ │
│  └──────────────┘   │  Warning lights (6 states)  │   └────────────────┘ │
│                     │  HUD cells: Confidence /    │                      │
│                     │  Break-in / Detected State  │                      │
│                     └────────────────────────────┘                      │
│                                                                          │
│  Rolling line chart: Alertness Score + Est. Safe Time (last 120 frames) │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Engineering

The feature matrix has **13 columns** derived from the raw impairment signal:

| Feature              | Window    | Description                                    |
| -------------------- | --------- | ---------------------------------------------- |
| `alert_mean_5`       | 5 frames  | Rolling mean alertness — short-term state      |
| `alert_mean_10`      | 10 frames | Rolling mean alertness — medium-term trend     |
| `alert_mean_20`      | 20 frames | Rolling mean alertness — session trend         |
| `alert_std_5`        | 5 frames  | Alertness variability — captures oscillation   |
| `alert_std_10`       | 10 frames | Medium-term variability                        |
| `alert_std_20`       | 20 frames | Long-term variability                          |
| `impair_rate_5`      | 5 frames  | Fraction of impaired frames in short window    |
| `impair_rate_10`     | 10 frames | Medium-term impairment rate                    |
| `impair_rate_20`     | 20 frames | Long-term impairment rate                      |
| `microsleep_rate_5`  | 5 frames  | Microsleep event rate — highest-weight signal  |
| `microsleep_rate_10` | 10 frames | Medium-term microsleep rate                    |
| `microsleep_rate_20` | 20 frames | Long-term microsleep rate                      |
| `alertness_raw`      | 1 frame   | Instantaneous alertness (1 - impairment)       |

### Class impairment weights

| Class       | ID | Impairment penalty |
| ----------- | -- | ------------------ |
| neutral     | 0  | 0.00               |
| microsleep  | 1  | 0.95               |
| distraction | 2  | 0.55               |
| phone_use   | 3  | 0.70               |

---

## 5. Model

**Quantile regression** is used because the target (remaining safe time) is inherently uncertain — the model outputs a calibrated interval rather than a single point estimate.

```
Input:  X ∈ R^13   (scaled feature vector)

Output: p10  →  lower bound  (pessimistic: 10% of outcomes are worse)
        p50  →  median estimate  (displayed as "Estimated Safe Time")
        p90  →  upper bound  (optimistic: 90% of outcomes are worse)

Confidence = clip(1 - (p90 - p10) / 60, 0.50, 0.95) × 100

Alertness Score = clip((p50 / 60) × 100, 0, 100)

Break Window = clip(p10 × 0.85, 1, p50)
```

### Training

```bash
cd src
python model.py ../data/raw
# Outputs: models/alertness_model.pkl
# Prints:  Validation MAE (median): X.XX min
```

LightGBM parameters: 300 estimators, learning rate 0.05, 31 leaves, min 10 child samples. All time estimates are clipped to [0, 120] minutes.

---

## 6. Live Dashboard Output

The Streamlit cockpit displays the following in real time:

| Widget | Description |
|---|---|
| **Alertness gauge** | 0–100 SVG arc gauge; green ≥70, yellow 45–69, red < 45 |
| **Safe Time gauge** | 0–60 min SVG arc gauge showing median estimate |
| **Status banner** | `DRIVER ALERT` / `FATIGUE DETECTED — PLAN BREAK` / `DROWSY — PULL OVER NOW` |
| **Warning lights** | 6 state indicators: ALERT / FATIGUE / DROWSY / MICROSLEEP / DISTRACTION / PHONE |
| **Confidence** | Prediction certainty % with colour-coded progress bar |
| **Break In** | Recommended break window (minutes) |
| **Detected State** | Current frame behaviour class |
| **Rolling chart** | Alertness score + estimated safe time over last 120 frames |

Example output for a healthy driver:

```
Current Alertness Score    :  82 / 100
Estimated Safe Driving Time:  24 ± 6 min
Prediction Confidence      :  91 %
Recommended Break Window   :  Within 20 min
```

---

## 7. Dataset

**State Farm Distracted Driver Detection** (Kaggle public competition dataset)

| Property | Value |
|---|---|
| Images | ~1,232 labeled frames (YOLO format) |
| Annotation format | YOLO `.txt` — `class_id cx cy w h` per detection |
| Classes | 4 behaviour classes (neutral, microsleep, distraction, phone_use) |
| Image format | `.jpg`, 1920×1080 px |
| Source | AMB82-Mini embedded board, co-pilot seat, facing driver |
| Conditions | Multiple vehicles, varied lighting |
| Privacy | Faces anonymised — no identifiable features |

> **Note:** The State Farm dataset is a single-frame collection with no temporal ordering between files. For time-series modelling the frames are treated as a sequence by sort order. The NTHU Driver Drowsiness Detection Dataset (36 drivers, 5 scene conditions, continuous video) is identified as the next integration target for genuine temporal regression.

### Roadmap datasets

| Dataset | Why |
|---|---|
| NTHU-DDD | 36 drivers, continuous video, binary alert/drowsy + eye/head/mouth annotations — enables genuine survival analysis on session time |
| DMD (Driver Monitoring Dataset) | Multi-task annotations including gaze, head pose, hands on wheel |
| AUC Distracted Driver | Larger demographic spread, more distraction classes |

---

## 8. Project Structure

```
DMS-SafetyEstimateLimittime/
│
├── src/
│   ├── feature_pipeline.py    Label ingestion, rolling feature windows,
│   │                          time-to-fatigue label generation
│   └── model.py               AlertnessModel — quantile regression (q10/q50/q90),
│                              train/predict/save/load
│
├── dashboard/
│   └── app.py                 Streamlit cockpit — SVG gauges, warning lights,
│                              HUD cells, rolling chart
│
├── data/
│   └── raw/                   YOLO .txt label files + .jpg images
│                              (not committed — add your dataset here)
│
├── models/
│   └── alertness_model.pkl    Saved model (generated on first run)
│
├── notebooks/                 Exploratory analysis (optional)
│
├── requirements.txt
└── README.md
```

---

## 9. Quick Start

### Prerequisites

```bash
python >= 3.10
pip install -r requirements.txt
```

Dependencies: `numpy` · `pandas` · `scikit-learn` · `lightgbm` · `streamlit` · `matplotlib` · `Pillow`

### 1. Add your dataset

Place YOLO `.txt` label files (and optionally `.jpg` images) in `data/raw/`. The label format is one detection per line:

```
class_id  cx  cy  w  h
0         0.32 0.27 0.19 0.36
```

### 2. Train the model

```bash
cd src
python model.py ../data/raw
# Saves model to models/alertness_model.pkl
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The model trains automatically on the first dashboard launch if `models/alertness_model.pkl` does not exist. Use the sidebar slider to control replay speed (1–10 frames/sec). Tick "Show raw feature table" to inspect the full feature matrix.

---

## 10. Extending to Real-Time

To replace the dataset replay loop with a live camera feed:

1. **Swap the data source** — replace `feature_pipeline.build_feature_matrix()` with a frame-by-frame YOLO inference call (e.g. `ultralytics` YOLOv8) on the webcam stream.

2. **Maintain a rolling buffer** — keep a `collections.deque(maxlen=20)` of the last 20 frame feature vectors. Pass the buffer to `add_rolling_features()` on each new frame.

3. **Call predict** — call `AlertnessModel.predict(feature_vector)` on each new frame. The model returns `estimate_min`, `lower_min`, `upper_min`, `alertness_score`, `confidence_pct`, `break_window_min`.

4. **Update the display** — pass the result dict directly to the dashboard HUD — the gauge and chart components are already parameterised by these keys.

```python
# Pseudocode
model = AlertnessModel().load()
cap   = cv2.VideoCapture(0)
buf   = deque(maxlen=20)

while True:
    frame = cap.read()
    dets  = yolo.predict(frame)          # → class_id, confidence, bbox
    feats = extract_features(dets, buf)  # → 13-dim vector
    buf.append(feats)
    preds = model.predict(np.array(feats))
    display(preds)
```

---

## 11. Roadmap

- [ ] Integrate NTHU-DDD dataset for continuous video sequences and genuine temporal labels
- [ ] Replace heuristic TTF label with survival analysis (Cox PH or Weibull AFT) on NTHU session data
- [ ] Add PERCLOS and blink-rate features via MediaPipe Face Mesh
- [ ] Live webcam mode — real-time inference in dashboard
- [ ] Per-driver personalisation baseline (driver ID → calibrated threshold)
- [ ] Export model to ONNX for embedded deployment (NVIDIA Jetson / AMB82-Mini)
- [ ] Connect to AI Trust Score framework for end-to-end trustworthiness evaluation

---

## 12. Related Work

| Repository | Description |
| ---------- | ----------- |
| [ai-trust-score-dms](https://github.com/ankampriyanka/ai-trust-score-dms) | Companion prototype — AHP-weighted Trust Scorecard evaluating this model on Safety, Robustness, Fairness, Explainability and Privacy KPIs |

---

*Part of the Walsh DBA — AI/ML Prototypes research series.*
