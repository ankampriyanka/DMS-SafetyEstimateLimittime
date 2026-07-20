"""
DMS SafetyEstimate Dashboard — Streamlit app.

Simulates a real-time driver monitoring feed by replaying the dataset
frame-by-frame and showing live alertness predictions.

Run:  streamlit run dashboard/app.py
"""

import sys
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from feature_pipeline import build_feature_matrix, FEATURE_COLS, CLASS_NAMES
from model import AlertnessModel, MODEL_DIR, train_and_save

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = str(MODEL_DIR / "alertness_model.pkl")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DMS · SafetyEstimate",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
}
.metric-label { color: #a0a0b0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #ffffff; font-size: 38px; font-weight: 700; margin: 4px 0; }
.metric-sub   { color: #6e6e8e; font-size: 12px; }
.alert-green  { color: #4ade80; }
.alert-yellow { color: #facc15; }
.alert-red    { color: #f87171; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building feature matrix from dataset...")
def load_features(label_dir: str) -> pd.DataFrame:
    return build_feature_matrix(label_dir)


@st.cache_resource(show_spinner="Training alertness model...")
def get_model(label_dir: str) -> AlertnessModel:
    model = AlertnessModel()
    path = MODEL_PATH
    if Path(path).exists():
        model.load(path)
    else:
        model, _ = train_and_save(label_dir)
    return model


def score_color(score: float) -> str:
    if score >= 70:
        return "alert-green"
    if score >= 45:
        return "alert-yellow"
    return "alert-red"


def time_color(mins: float) -> str:
    if mins >= 30:
        return "alert-green"
    if mins >= 15:
        return "alert-yellow"
    return "alert-red"


def status_text(score: float, mins: float) -> tuple[str, str]:
    if score >= 70 and mins >= 30:
        return "ALERT", "#4ade80"
    if score >= 45 or mins >= 15:
        return "FATIGUED — PLAN A BREAK", "#facc15"
    return "DROWSY — STOP NOW", "#f87171"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("DMS · SafetyEstimate")
st.sidebar.caption("Predictive Driver Monitoring System")
st.sidebar.markdown("---")

replay_speed = st.sidebar.slider("Replay speed (frames/sec)", 1, 10, 3)
show_raw = st.sidebar.checkbox("Show raw feature table", False)
retrain = st.sidebar.button("Re-train model")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset:** State Farm Distracted Driver Detection
**Classes:** neutral · microsleep · distraction · phone_use
**Model:** LightGBM quantile regression (q10/q50/q90)
""")

# ── Load data & model ─────────────────────────────────────────────────────────
df = load_features(str(DATA_DIR))

if retrain and Path(MODEL_PATH).exists():
    Path(MODEL_PATH).unlink()
    st.cache_resource.clear()

model = get_model(str(DATA_DIR))

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Driver Monitoring System — Predictive Alertness Estimator")
st.caption("Real-time alertness scoring and safe driving time prediction")
st.markdown("---")

col_feed, col_metrics = st.columns([1.1, 1], gap="large")

# Left: simulated video feed placeholder + state timeline
with col_feed:
    st.subheader("Simulated Driver Feed")
    frame_placeholder = st.empty()
    timeline_placeholder = st.empty()
    class_placeholder = st.empty()

# Right: live metric cards
with col_metrics:
    st.subheader("Live Predictions")
    score_ph     = st.empty()
    estimate_ph  = st.empty()
    confidence_ph = st.empty()
    break_ph     = st.empty()
    status_ph    = st.empty()

st.markdown("---")
chart_ph = st.empty()

if show_raw:
    st.subheader("Feature Table (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)

# ── Replay loop ───────────────────────────────────────────────────────────────
history_score = []
history_ttf   = []
history_class = []

for idx, row in df.iterrows():
    features = row[FEATURE_COLS].values.astype(float)
    pred = model.predict(features)

    score    = pred["alertness_score"]
    est      = pred["estimate_min"]
    lower    = pred["lower_min"]
    upper    = pred["upper_min"]
    conf     = pred["confidence_pct"]
    brk      = pred["break_window_min"]
    cls_name = CLASS_NAMES.get(int(row["dominant_class"]), "unknown")

    history_score.append(score)
    history_ttf.append(est)
    history_class.append(cls_name)

    sc = score_color(score)
    tc = time_color(est)
    status_label, status_color = status_text(score, est)

    # ── Metric cards ──
    score_ph.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Current Alertness Score</div>
      <div class="metric-value {sc}">{score:.0f} / 100</div>
      <div class="metric-sub">Based on last 10 frames of driver state</div>
    </div>""", unsafe_allow_html=True)

    estimate_ph.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Estimated Safe Driving Time</div>
      <div class="metric-value {tc}">{est:.0f} <span style="font-size:20px">± {(upper-lower)/2:.0f} min</span></div>
      <div class="metric-sub">90% interval: {lower:.0f} – {upper:.0f} min</div>
    </div>""", unsafe_allow_html=True)

    confidence_ph.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Prediction Confidence</div>
      <div class="metric-value">{conf:.0f}%</div>
      <div class="metric-sub">Based on interval width relative to 60 min horizon</div>
    </div>""", unsafe_allow_html=True)

    break_ph.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Recommended Break Window</div>
      <div class="metric-value {tc}">Within {brk:.0f} min</div>
      <div class="metric-sub">Take a break before the lower bound estimate</div>
    </div>""", unsafe_allow_html=True)

    status_ph.markdown(f"""
    <div class="metric-card" style="border: 2px solid {status_color};">
      <div class="metric-label">Driver Status</div>
      <div class="metric-value" style="color:{status_color}; font-size:24px;">{status_label}</div>
    </div>""", unsafe_allow_html=True)

    # ── Frame info ──
    frame_placeholder.info(f"Frame {idx+1}/{len(df)}  |  Detected state: **{cls_name.upper()}**")
    class_placeholder.progress(
        min(int(score), 100),
        text=f"Alertness: {score:.0f}%"
    )

    # ── Rolling chart ──
    if len(history_score) > 1:
        chart_df = pd.DataFrame({
            "Alertness Score": history_score[-120:],
            "Est. Safe Time (min)": history_ttf[-120:],
        })
        chart_ph.line_chart(chart_df, use_container_width=True)

    time.sleep(1.0 / replay_speed)

st.success("Replay complete. Adjust speed or re-train model from the sidebar.")
