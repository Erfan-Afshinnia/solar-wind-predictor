import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Solar plant location (Gujarat, India)
LATITUDE = 22.98
LONGITUDE = 72.61
TIMEZONE = "Asia/Kolkata"

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "data" / "processed" / "new_weather_data.csv"

def fetch_weather(days_back: int = 34) -> pd.DataFrame:
    """
    Fetch recent weather data from Open-Meteo archive API.
    Returns DataFrame with same features as training data.
    """
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days_back)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": [
            "temperature_2m",
            "shortwave_radiation",
            "direct_radiation",
        ],
    }

    print(f"Fetching weather data: {start_date} → {end_date}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Parse response into DataFrame
    hourly = data["hourly"]
    df = pd.DataFrame({
        "DATE_TIME":                        pd.to_datetime(hourly["time"]),
        "AMBIENT_TEMPERATURE":              hourly["temperature_2m"],
        "IRRADIATION":                      hourly["shortwave_radiation"],
        "DIRECT_RADIATION":                 hourly["direct_radiation"],
    })


    # Normalise irradiation to 0-1 range
    # Training data irradiation was in 0-1 scale
    # Open-Meteo returns W/m² (0-1000 range)
    max_irr = df["IRRADIATION"].max()
    if max_irr > 0:
        df["IRRADIATION"] = (df["IRRADIATION"] / max_irr).clip(0, 1)
    
    # Estimate module temperature
    # Module temp ≈ ambient + solar heating effect
    df["MODULE_TEMPERATURE"] = (
        df["AMBIENT_TEMPERATURE"] + df["IRRADIATION"] * 25
    )

    #