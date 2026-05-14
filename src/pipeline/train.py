import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


_ROOT      = Path(__file__).resolve().parents[2]
_GEN       = _ROOT / "data" / "raw" / "Plant_1_Generation_Data.csv"
_WEATHER   = _ROOT / "data" / "raw" / "Plant_1_Weather_Sensor_Data.csv"
_CANDIDATE = _ROOT / "models" / "xgb_candidate.json"
_PARAMS    = _ROOT / "models" / "best_params.json"

FEATURES = [
    "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE",
    "HOUR", "MONTH", "DAY_OF_YEAR", "HOUR_SIN", "HOUR_COS"
]

from src.feature_store.retrieve import get_training_features

def train_candidate() -> dict:
    """
    Retrain XGBoost and save as candidate model.
    Returns evaluation metrics on test set.
    """
    print("── Loading features from feature store ─────────")
    X, y = get_training_features()
    print(f"   Train/test split from {len(X)} rows")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"     Train: {len(x_train)} rows | Test: {len(x_test)} rows")

    # Load best params from previous tuning
    if _PARAMS.exists():
        with open(_PARAMS) as f:
            params = json.load(f)
        print(f"    Using saved params: {params}")
    else:
        # Fallback to known best params from our tuning
        params = {
            "n_estimators":       300,
            "learning_rate":      0.1,
            "max_depth":          6,
            "subsample":          0.7,
            "colsample_bytree":   0.7,
        }
    print("\n── Training candidate model ────────────────────")
    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    # Evaluate on daytime only
    y_pred    = model.predict(x_test)
    mask      = y_test > 0
    mae       = mean_absolute_error(y_test[mask], y_pred[mask])
    r2        = r2_score(y_test[mask], y_pred[mask])

    print(f"   Candidate MAE: {mae:,.1f} kW")
    print(f"   Candidate R²:  {r2:.4f}")

    # ── Save candidate (not champion yet) ─────────────────
    model.save_model(str(_CANDIDATE))
    print(f"\n✅ Candidate saved → {_CANDIDATE}")

    return {"mae": round(mae, 2), "r2": round(r2, 4)}

if __name__ == "__main__":
    metrics = train_candidate()
    print(f"\nFinal metrics: {metrics}")

