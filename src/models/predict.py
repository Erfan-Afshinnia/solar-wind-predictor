# src/models/predict.py
import xgboost as xgb
import pandas as pd
from pathlib import Path

from src.features.build_features import build_features

_ROOT  = Path(__file__).resolve().parents[2]
_MODEL = _ROOT / "models" / "xgb_champion.json"


def load_model() -> xgb.Booster:
    """Load the trained XGBoost model from disk."""
    if not _MODEL.exists():
        raise FileNotFoundError(f"Model not found at {_MODEL}")
    booster = xgb.Booster()
    booster.load_model(str(_MODEL))
    return booster


def predict_single(
    irradiation: float,
    module_temperature: float,
    ambient_temperature: float,
    date_time: str,
) -> float:
    """
    Make a single prediction given sensor readings and timestamp.
    Returns predicted AC power output in kW.
    """
    model = load_model()

    df = pd.DataFrame([{
        "DATE_TIME":           pd.to_datetime(date_time),
        "IRRADIATION":         irradiation,
        "MODULE_TEMPERATURE":  module_temperature,
        "AMBIENT_TEMPERATURE": ambient_temperature,
    }])

    features = build_features(df)

    # Booster requires DMatrix input
    dmatrix = xgb.DMatrix(features)
    prediction = model.predict(dmatrix)[0]

    return float(max(0.0, prediction))

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for a batch of rows.
    Expects columns: DATA_TIME, IRRADIATION,
                    MODULE_TEMPERATURE, AMBIENT_TEMPERATURE
    Returns the input df with PREDICTED_AC_POWER_KW column added.
    """
    # Validate required columns
    required = {"DATE_TIME", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse datetime
    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    # Feature engineering
    features = build_features(df)

    # Predict
    model = load_model()
    dmatrix = xgb.DMatrix(features)
    preds = model.predict(dmatrix)

    # Clip negatives and attach to original df
    df["PREDICTED_AC_POWER_KW"] = [
        round(float(max(0.0, p)), 2) for p in preds
    ]
    return df