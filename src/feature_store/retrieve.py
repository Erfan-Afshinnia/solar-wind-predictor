import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

_ROOT    = Path(__file__).resolve().parents[2]
_PARQUET = _ROOT / "feature_repo" / "data" / "features.parquet"

FEATURE_COLS = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "HOUR", "MONTH", "DAY_OF_YEAR", "HOUR_SIN", "HOUR_COS"]

def get_training_features() -> tuple:
    """
    Get features and lables for training.
    Single source of truth - same data always.
    """
    if not _PARQUET.exists():
        raise FileNotFoundError(
            "Feature store not materialised."
            "Run src/feature_store/materialize.py first"
        )
    df = pd.read_parquet(_PARQUET)

    x  = df[FEATURE_COLS]
    y  = df["AC_POWER"]

    print(f"✅ Retrieved {len(X)} training rows from feature store")
    return x, y


def get_inference_features(
        irradiation: float,
        module_temperature: float,
        ambient_temperature: float,
        date_time: str,
) -> pd.DataFrame:
    """
    Build features for a single prediction request.
    Uses identical logic to training features.
    """
    dt = pd.to_datetime(date_time)

    features = {
        "IRRADIATION":            irradiation,
        "MODULE_TEMPERATURE":     module_temperature,
        "AMBIENT_TEMPERATURE":    ambient_temperature,
        "HOUR":                   dt.hour,
        "MONTH":                  dt.month,
        "DAY_OF_YEAR":            dt.dayofyear,
        "HOUR_SIN":               float(np.sin(2 * np.pi * dt.hour / 24)),
        "HOUR_COS":               float(np.cos(2 * np.pi * dt.hour / 24)),
    }
    return pd.DataFrame([features])[FEATURE_COLS]