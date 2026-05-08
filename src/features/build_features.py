import numpy as np
import pandas as pd

FEATURES = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "HOUR",
    "MONTH",
    "DAY_OF_YEAR",
    "HOUR_SIN",
    "HOUR_COS"
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from DATE_TIME column."""
    df = df.copy()
    df["HOUR"]       = df["DATE_TIME"].dt.hour
    df["MONTH"]      = df["DATE_TIME"].dt.month
    df["DAY_OF_YEAR"]= df["DATE_TIME"].dt.dayofyear
    df["HOUR_SIN"]   = np.sin(2 * np.pi * df["HOUR"] / 24)
    df["HOUR_COS"]   = np.cos(2 * np.pi * df["HOUR"] / 24)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Returns only model input columns."""
    df = add_time_features(df)
    return df[FEATURES]