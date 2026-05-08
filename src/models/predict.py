import joblib
import pandas as pd
import xgboost as xgb
from pathlib import Path

from src.features.build_features import build_features, FEATURES

# Resolve paths relative to project root
_ROOT = Path(__file__).resolve().parents[2]
_MODEL = _ROOT / "models" / "xgb_champion.json"
_FEATS = _ROOT / "models" / "features_names.joblib"

def load_model():
    if not _MODEL.exists():
        raise FileNotFoundError(f"Model not found at {_MODEL}")
    model = xgb.XGBRegressor()
    model.load_model(str(_MODEL))
    return model

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
        "DATE_TIME": pd.to_datetime(date_time),
        "IRRADIATION": irradiation,
        "MODULE_TEMPERATURE": module_temperature,
        "AMBIENT_TEMPERATURE": ambient_temperature,
    }])

    features = build_features(df)
    prediction = model.predict(features)[0]

    return float(max(0.0, prediction))