import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

_ROOT    = Path(__file__).resolve().parents[2]
_GEN     = _ROOT / "data" / "raw" / "Plant_1_Generation_Data.csv"
_WEATHER = _ROOT / "data" / "raw" / "Plant_1_Weather_Sensor_Data.csv"
_OUT_DIR = _ROOT / "feature_repo" / "data"          # ← changed
_PARQUET = _OUT_DIR / "features.parquet"


def build_feature_dataset() -> pd.DataFrame:
    """Build complete feature dataset and asve as parquet."""
    print("── Building feature dataset ─────────────────────")

    gen     = pd.read_csv(_GEN)
    weather = pd.read_csv(_WEATHER)

    gen["DATE_TIME"]     = pd.to_datetime(
        gen["DATE_TIME"], format="%d-%m-%Y %H:%M"
    )
    weather["DATE_TIME"]  = pd.to_datetime(weather["DATE_TIME"])

    gen_agg = (gen.groupby("DATE_TIME")
                  .agg(AC_POWER=("AC_POWER", "sum"))
                  .reset_index())
    
    df = pd.merge(
        gen_agg,
        weather[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]],
        on="DATE_TIME", how="inner"
    )

    # Time features
    df["HOUR"]          = df["DATE_TIME"].dt.hour
    df["MONTH"]         = df["DATE_TIME"].dt.month
    df["DAY_OF_YEAR"]   = df["DATE_TIME"].dt.dayofyear
    df["HOUR_SIN"]      = np.sin(2 * np.pi * df["HOUR"] / 24)
    df["HOUR_COS"]      = np.cos(2** np.pi * df["HOUR"] / 24)

    # Required Feast Columns
    # event_timestamp: when the feature was observed
    # plant_id: entity identifier
    df["event_timestamp"] = df["DATE_TIME"].dt.tz_localize("UTC")
    df["plant_id"]        = 1  # single plant

    # ── Save as parquet ───────────────────────────────────
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PARQUET, index=False)

    print(f"✅ Feature dataset: {len(df)} rows × {len(df.columns)} cols")
    print(f"✅ Saved → {_PARQUET}")
    print(f"\nColumns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    build_feature_dataset()