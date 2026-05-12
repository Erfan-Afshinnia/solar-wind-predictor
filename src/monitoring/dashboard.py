# src/monitoring/dashboard.py
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Solar Power — Model Monitor",
    page_icon="🌞",
    layout="wide"
)

_ROOT  = Path(__file__).resolve().parents[2]
_REF   = _ROOT / "data" / "processed" / "reference_data.csv"
_CUR   = _ROOT / "data" / "processed" / "current_data.csv"
_DRIFT = _ROOT / "data" / "processed" / "drift_report.json"


@st.cache_data
def load_data():
    return pd.read_csv(_REF), pd.read_csv(_CUR)


def compute_metrics(df):
    mask = df["ACTUAL_AC_POWER"] > 0
    d = df[mask]
    return {
        "MAE": mean_absolute_error(d["ACTUAL_AC_POWER"], d["PREDICTED_AC_POWER"]),
        "R²":  r2_score(d["ACTUAL_AC_POWER"], d["PREDICTED_AC_POWER"]),
    }


ref, cur = load_data()
ref_m    = compute_metrics(ref)
cur_m    = compute_metrics(cur)

# ── Header ────────────────────────────────────────────────
st.title("🌞 Solar Power Predictor — Model Monitor")
st.caption("Training (reference) vs Test (current) data comparison")

# ── Metrics ───────────────────────────────────────────────
st.subheader("📊 Performance Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reference MAE", f"{ref_m['MAE']:,.0f} kW")
c2.metric("Current MAE",   f"{cur_m['MAE']:,.0f} kW",
          delta=f"{cur_m['MAE'] - ref_m['MAE']:+,.0f} kW",
          delta_color="inverse")
c3.metric("Reference R²",  f"{ref_m['R²']:.4f}")
c4.metric("Current R²",    f"{cur_m['R²']:.4f}",
          delta=f"{cur_m['R²'] - ref_m['R²']:+.4f}")

st.divider()

# ── Actual vs Predicted ───────────────────────────────────
st.subheader("📈 Actual vs Predicted (Current Data)")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

daytime = cur[cur["ACTUAL_AC_POWER"] > 0]
axes[0].scatter(daytime["ACTUAL_AC_POWER"], daytime["PREDICTED_AC_POWER"],
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

# ── Feature distributions ─────────────────────────────────
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

# ── Drift Detection ───────────────────────────────────────
st.subheader("⚠️ Feature Drift Detection (KS Test)")
st.caption("p < 0.05 = drift detected")

if _DRIFT.exists():
    with open(_DRIFT) as f:
        drift = json.load(f)

    summary  = drift["summary"]
    dc1, dc2 = st.columns(2)
    dc1.metric("Features with Drift",
               f"{summary['drifted_features']} / {summary['total_features']}")
    dc2.metric("Drift %", f"{summary['drift_pct']}%")

    rows = []
    for feat, result in drift["features"].items():
        rows.append({
            "Feature":      feat,
            "KS Statistic": result["ks_statistic"],
            "P-Value":      result["p_value"],
            "Status":       "⚠️ DRIFT" if result["drift_detected"] else "✅ OK",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.warning("Drift report not found.")
    if st.button("Generate Drift Report"):
        from src.monitoring.generate_report import generate_drift_report
        with st.spinner("Running KS tests..."):
            generate_drift_report()
        st.success("Done! Refresh the page.")