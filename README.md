# DMS · SafetyEstimate — Predictive Driver Alertness System

A prototype that shifts Driver Monitoring Systems (DMS) from **reactive fatigue detection** to **predictive alertness estimation** — estimating how much safe driving time remains before fatigue onset, rather than only alerting after it begins.

## Live Output

| Metric | Example |
|---|---|
| Current Alertness Score | 82 / 100 |
| Estimated Safe Driving Time | 24 ± 6 min |
| Prediction Confidence | 91% |
| Recommended Break Window | Within 20 min |

## Architecture

```
data/raw/          ← YOLO-format labeled images (State Farm dataset)
src/
  feature_pipeline.py   ← frame labels → rolling feature windows
  model.py              ← LightGBM quantile regression (q10/q50/q90)
dashboard/
  app.py                ← Streamlit real-time replay dashboard
models/
  alertness_model.pkl   ← saved trained model (generated on first run)
```

### Pipeline

1. **Feature extraction** — each YOLO label file is parsed into a driver state per frame (neutral / microsleep / distraction / phone_use). Rolling windows (5, 10, 20 frames) compute impairment rates and alertness trends.

2. **Time-to-fatigue label** — for each frame, a look-ahead heuristic computes how many frames until cumulative impairment crosses a fatigue threshold, converted to minutes. This becomes the regression target.

3. **Quantile regression** — LightGBM trains three models (q=0.10, 0.50, 0.90) to predict the lower bound, median estimate, and upper bound of remaining safe time. Interval width drives confidence score.

4. **Dashboard** — Streamlit replays the dataset frame-by-frame, showing live metric cards and a rolling alertness / time-estimate chart.

## Dataset

**State Farm Distracted Driver Detection** (Kaggle public competition)
- 1,232 labeled driver images
- Classes: `neutral (0)`, `microsleep (1)`, `distraction (2)`, `phone_use (3)`
- YOLO bounding-box format: `class cx cy w h`

## Setup

```bash
pip install -r requirements.txt
```

### Train the model

```bash
cd src
python model.py ../data/raw
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

The model trains automatically on first dashboard launch if no saved model exists.

## Extending to Real-Time

To replace the replay loop with a live camera feed:
1. Swap `feature_pipeline.build_feature_matrix()` with a frame-by-frame MediaPipe or YOLO inference call
2. Maintain a rolling deque of the last N frame feature vectors
3. Call `model.predict()` on each new frame

## Roadmap

- [ ] Integrate NTHU-DDD dataset for continuous video sequences and richer temporal labels
- [ ] Replace heuristic TTF label with survival analysis on NTHU session data
- [ ] Add PERCLOS and blink-rate extraction from raw images via MediaPipe Face Mesh
- [ ] Live webcam mode in dashboard
- [ ] Per-driver personalisation baseline
