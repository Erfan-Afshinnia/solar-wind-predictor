# src/monitoring/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Solar Power — Model Monitor",
    page_icon="🌞",
    layout="wide"
)

_ROOT = Path(__file__).resolve().parents[2]
_REF  = _ROOT / "data" / "processed" / "reference_data.csv"
_CUR  = _ROOT / "data" / "processed" / "current_data.csv"
_RPT  = _ROOT / "data" / "processed" / "monitoring_report.html"


@st.cache_data
def load_data():
    ref = pd.read_csv(_REF)
    cur = pd.read_csv(_CUR)
    return ref, cur


def compute_metrics(df):
    mask = df["ACTUAL_AC_POWER"] > 0   # daytime only
    d    = df[mask]
    return {
        "MAE":  mean_absolute_error(d["ACTUAL_AC_POWER"],
                                    d["PREDICTED_AC_POWER"]),
        "R²":   r2_score(d["ACTUAL_AC_POWER"],
                         d["PREDICTED_AC_POWER"]),
        "Rows": len(df),
    }


# ── Load data ─────────────────────────────────────────────
ref, cur = load_data()
ref_m    = compute_metrics(ref)
cur_m    = compute_metrics(cur)

# ── Header ────────────────────────────────────────────────
st.title("🌞 Solar Power Predictor — Model Monitor")
st.caption("Comparing training (reference) vs test (current) data")

# ── Metrics row ───────────────────────────────────────────
st.subheader("📊 Performance Overview")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Reference MAE",
            f"{ref_m['MAE']:,.0f} kW")
col2.metric("Current MAE",
            f"{cur_m['MAE']:,.0f} kW",
            delta=f"{cur_m['MAE'] - ref_m['MAE']:+,.0f} kW",
            delta_color="inverse")
col3.metric("Reference R²",  f"{ref_m['R²']:.4f}")
col4.metric("Current R²",
            f"{cur_m['R²']:.4f}",
            delta=f"{cur_m['R²'] - ref_m['R²']:+.4f}",
            delta_color="normal")

st.divider()

# ── Actual vs Predicted ───────────────────────────────────
st.subheader("📈 Actual vs Predicted (Current Data)")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

daytime = cur[cur["ACTUAL_AC_POWER"] > 0]
axes[0].scatter(daytime["ACTUAL_AC_POWER"],
                daytime["PREDICTED_AC_POWER"],
                alpha=0.3, s=10, color="steelblue")
axes[0].plot([0, 30000], [0, 30000], "r--")
axes[0].set_xlabel("Actual AC Power (kW)")
axes[0].set_ylabel("Predicted AC Power (kW)")
axes[0].set_title("Actual vs Predicted")

residuals = daytime["ACTUAL_AC_POWER"] - daytime["PREDICTED_AC_POWER"]
axes[1].hist(residuals, bins=40, color="tomato", edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Residual (kW)")
axes[1].set_title("Residual Distribution")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── Feature drift ─────────────────────────────────────────
st.subheader("🔍 Feature Distribution: Reference vs Current")

feature = st.selectbox(
    "Select feature to inspect:",
    ["IRRADIATION", "MODULE_TEMPERATURE",
     "AMBIENT_TEMPERATURE", "HOUR"]
)

fig2, ax = plt.subplots(figsize=(9, 4))
ax.hist(ref[feature], bins=40, alpha=0.5,
        label="Reference (train)", color="steelblue")
ax.hist(cur[feature], bins=40, alpha=0.5,
        label="Current (test)",    color="orange")
ax.set_xlabel(feature)
ax.set_ylabel("Count")
ax.set_title(f"{feature} — Distribution Comparison")
ax.legend()
st.pyplot(fig2)

st.divider()

# ── Evidently HTML report ─────────────────────────────────
st.subheader("📋 Full Evidently Drift Report")
if _RPT.exists():
    with open(_RPT, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=600, scrolling=True)
else:
    st.warning("Report not generated yet. Run generate_report.py first.")
    if st.button("Generate Report Now"):
        from src.monitoring.generate_report import generate_drift_report
        with st.spinner("Generating..."):
            generate_drift_report()
        st.success("Report generated! Refresh the page.")