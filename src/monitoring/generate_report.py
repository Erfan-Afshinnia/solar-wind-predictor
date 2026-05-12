# src/monitoring/generate_report.py
import pandas as pd
import json
from pathlib import Path
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
_REF  = _ROOT / "data" / "processed" / "reference_data.csv"
_CUR  = _ROOT / "data" / "processed" / "current_data.csv"
_OUT  = _ROOT / "data" / "processed" / "drift_report.json"

FEATURE_COLS = [
    "IRRADIATION", "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE", "HOUR", "MONTH",
    "DAY_OF_YEAR", "HOUR_SIN", "HOUR_COS",
]


def detect_drift(ref_series, cur_series, threshold=0.05):
    """KS test — p < threshold means drift detected."""
    stat, p_value = stats.ks_2samp(ref_series, cur_series)
    return {
        "ks_statistic":   round(float(stat), 4),
        "p_value":        round(float(p_value), 4),
        "drift_detected": bool(p_value < threshold),
    }


def generate_drift_report():
    """Run KS drift detection on all features and save JSON report."""
    reference = pd.read_csv(_REF)
    current   = pd.read_csv(_CUR)

    report  = {"features": {}, "summary": {}}
    drifted = 0

    print("Feature Drift Detection (KS Test)")
    print("─" * 50)

    for col in FEATURE_COLS:
        result = detect_drift(reference[col], current[col])
        report["features"][col] = result
        if result["drift_detected"]:
            drifted += 1
        status = "⚠️  DRIFT" if result["drift_detected"] else "✅  OK   "
        print(f"  {col:<25} {status}  p={result['p_value']:.4f}")

    report["summary"] = {
        "total_features":   len(FEATURE_COLS),
        "drifted_features": drifted,
        "drift_pct":        round(drifted / len(FEATURE_COLS) * 100, 1),
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("─" * 50)
    print(f"\n✅ Report saved → {_OUT}")
    print(f"   {drifted}/{len(FEATURE_COLS)} features show drift")
    return report


if __name__ == "__main__":
    generate_drift_report()