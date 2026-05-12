import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.pipeline.train import load_and_prepare, FEATURES

_ROOT      = Path(__file__).resolve().parents[2]
_CHAMPION  = _ROOT / "models" / "xgb_champion.json"
_CANDIDATE = _ROOT / "models" / "xgb_candidate.json"


def evaluate_models() -> dict:
    """
    Compare champion vs candidate on the same test set.
    Returns dict with both MAEs and weather candidate wins.
    """
    print("── Loading test data ───────────────────────────")
    df = load_and_prepare()
    x = df[FEATURES]
    y = df["AC_POWER"]

    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    mask = y_test > 0

    # Load both models
    champion = xgb.Booster()
    candidate = xgb.Booster()
    champion.load_model(str(_CHAMPION))
    candidate.load_model(str(_CANDIDATE))

    dtest = xgb.DMatrix(x_test)

    # Evaluate both
    champ_pred = champion.predict(dtest)
    cand_pred  = candidate.predict(dtest)

    champ_mae = mean_absolute_error(y_test[mask], champ_pred[mask])
    cand_mae = mean_absolute_error(y_test[mask], cand_pred[mask])

    candidate_wins = cand_mae < champ_mae
    improvement    = round(champ_mae - cand_mae, 2)

    print(f"   Champion  MAE: {champ_mae:,.1f} kW")
    print(f"   Candidate MAE: {cand_mae:,.1f} kW")
    print(f"   Improvement:   {improvement:+.1f} kW")
    print(f"   Decision:      "
          f"{'✅ PROMOTE candidate' if candidate_wins else '❌ KEEP champion'}")

    return {
        "champion_mae":   round(champ_mae, 2),
        "candidate_mae":  round(cand_mae, 2),
        "improvement_kw": improvement,
        "candidate_wins": candidate_wins,
    }


if __name__ == "__main__":
    result = evaluate_models()
