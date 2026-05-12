import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metrics_preset import (
    DataDriftPreset,
    RegressionPreset
)

_ROOT = Path(__file__).resolve().parents[2]
_REF  = _ROOT / "data" / "processed" / "reference_data.csv"
_CUR  = _ROOT / "data" / "processed" / "current_data.csv"
_OUT  = _ROOT / "data" / "processed" / "monitoring_report.html"

FEATURE_COLS = [
    "IRRADIATION", "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE", "HOUR", "MONTH",
    "DAY_OF_YEAR", "HOUR_SIN", "HOUR_COS"
]

def load_data():
    reference = pd.read_csv(_REF)
    current   = pd.read_csv(_CUR)
    return reference, current


def generate_drift_report():
    """Generate HTML drift report using Evidently."""
    reference, current = load_data()

    report = Report(metrics=[
        DataDriftPreset(),
        RegressionPreset(),
    ])

    # Evidently needs target column named "target" and prediction "prediction"
    ref = reference[FEATURE_COLS].copy()
    ref["target"]     = reference["ACTUAL_AC_POWER"]
    ref["prediction"] = reference["PREDICTED_AC_POWER"]

    cur = current[FEATURE_COLS].copy()
    cur["target"]     = current["ACTUAL_AC_POWER"]
    cur["prediction"] = current["PREDICTED_AC_POWER"]

    report.run(reference_data=ref, current_data=cur)
    report.save_html(str(_OUT))
    print(f"✅ Report saved to {_OUT}")
    return report
