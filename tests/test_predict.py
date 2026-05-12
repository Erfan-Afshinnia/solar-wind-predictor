import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from src.models.predict import predict_single

def test_sunny_noon_prediction():
    """High irradiation at noon should produce significant power."""
    result = predict_single(
        irradiation=0.8,
        module_temperature=45.0,
        ambient_temperature=32.0,
        date_time="2020-06-01 12:00:00"
    )
    assert result > 10_000, f"Expected >10000 KW, got {result}"

def test_zero_irradiation_returns_near_zero():
    """No sunlight should return ~0 power."""
    result = predict_single(
        irradiation=0.0,
        module_temperature=22.0,
        ambient_temperature=20.0,
        date_time="2020-06-01 00:00:00"
    )
    assert result < 500, f"Expected < 500 KW at night, got {result}"

def test_prediction_is_non_negative():
    """Predictions must never be negative."""
    result = predict_single(
        irradiation=0.0,
        module_temperature=15.0,
        ambient_temperature=10.0,
        date_time= "2020-06-01 03:00:00"
    )
    assert result >= 0, f"Prediction should not be negative, got {result}"

def test_predict_batch():
    """Batch prediction should return a df with predictions column."""
    import pandas as pd
    from src.models.predict import predict_batch

    df = pd.DataFrame([
        {
            "DATE_TIME": "2020-06-01  12:00:00",
            "IRRADIATION": 0.8,
            "MODULE_TEMPERATURE": 45.0,
            "AMBIENT_TEMPERATURE": 32.0
        },
        {
            "DATE_TIME": "2020-06-01  00:00:00",
            "IRRADIATION": 0.0,
            "MODULE_TEMPERATURE": 22.0,
            "AMBIENT_TEMPERATURE": 20.0
        }
    ])

    result = predict_batch(df)

    assert "PREDICTED_AC_POWER_KW" in result.columns
    assert len(result) == 2
    assert result["PREDICTED_AC_POWER_KW"].iloc[0] > 10_000  #daytime
    assert result["PREDICTED_AC_POWER_KW"].iloc[1] < 500     #nighttime

def test_predict_batch_missing_column():
    """Batch prediction should raise ValueError for missing columns."""
    import pandas as pd
    from src.models.predict import predict_batch

    df = pd.DataFrame([{
        "DATE_TIME": "2020-06-01  12:00:00",
        "IRRADIATION": 0.8,
        # MODULE_TEMPERATURE and AMBIENT_TEMPERATURE missing
    }])

    try:
        predict_batch(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing required columns" in str(e)