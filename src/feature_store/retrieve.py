import pandas as pd
from pathlib import Path

from src.features.build_features import build_features, FEATURES

_ROOT    = Path(__file__).resolve().parents[2]
_PARQUET = _ROOT / "feature_repo" / "data" / "features.parquet"

FEATURE_COLS = FEATURES

def get_training_features() -> tuple:
    """
    Get features and labels for training.
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

    print(f"✅ Retrieved {len(x)} training rows from feature store")
    return x, y


def get_inference_features(
    irradiation: float,
    module_temperature: float,
    ambient_temperature: float,
    date_time: str,
) -> pd.DataFrame:
    """
    Build features for a single prediction request
    using the same feature-engineering pipeline as training.
    """
    raw = pd.DataFrame([{
        "DATE_TIME": pd.to_datetime(date_time),
        "IRRADIATION": irradiation,
        "MODULE_TEMPERATURE": module_temperature,
        "AMBIENT_TEMPERATURE": ambient_temperature,
    }])

    return build_features(raw)[FEATURE_COLS]