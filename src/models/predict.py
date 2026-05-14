# src/models/predict.py
import xgboost as xgb
import pandas as pd
from pathlib import Path

from src.feature_store.retrieve import get_inference_features   # ← new import

_ROOT  = Path(__file__).resolve().parents[2]
_MODEL = _ROOT / "models" / "xgb_champion.json"


def load_model() -> xgb.Booster:
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
    model = load_model()

    # ── Features now come from feature store ──────────────
    # Guaranteed identical to training features
    features = get_inference_features(           # ← replaces build_features()
        irradiation=irradiation,
        module_temperature=module_temperature,
        ambient_temperature=ambient_temperature,
        date_time=date_time,
    )

    dmatrix    = xgb.DMatrix(features)
    prediction = model.predict(dmatrix)[0]
    return float(max(0.0, prediction))


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    required = {"DATE_TIME", "IRRADIATION",
                "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    results = []
    for _, row in df.iterrows():
        pred = predict_single(
            irradiation=row["IRRADIATION"],
            module_temperature=row["MODULE_TEMPERATURE"],
            ambient_temperature=row["AMBIENT_TEMPERATURE"],
            date_time=str(row["DATE_TIME"]),
        )
        results.append(round(pred, 2))

    df["PREDICTED_AC_POWER_KW"] = results
    return df