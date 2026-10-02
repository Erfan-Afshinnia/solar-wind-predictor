import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from src.feature_store.retrieve import get_training_features


_ROOT = Path(__file__).resolve().parents[2]
_CHAMPION = _ROOT / "models" / "xgb_champion.json"
_CANDIDATE = _ROOT / "models" / "xgb_candidate.json"


def evaluate_models() -> dict:
    """
    Compare champion and candidate models on the same chronological test set.
    Returns evaluation metrics and whether the candidate improves MAE.
    """
    print("── Loading test data ─────────────────────────────────────")
    X, y = get_training_features()

    split_index = int(len(X) * 0.8)
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    print(f"   Test set: {len(X_test)} rows")

    # Evaluate daytime samples only.
    mask = y_test > 0

    # Load both models.
    champion = xgb.Booster()
    candidate = xgb.Booster()

    champion.load_model(str(_CHAMPION))
    candidate.load_model(str(_CANDIDATE))

    dtest = xgb.DMatrix(X_test)

    # Generate predictions.
    champ_pred = champion.predict(dtest)
    cand_pred = candidate.predict(dtest)

    # Calculate MAE.
    champ_mae = mean_absolute_error(y_test[mask], champ_pred[mask])
    cand_mae = mean_absolute_error(y_test[mask], cand_pred[mask])

    candidate_wins = cand_mae < champ_mae
    improvement = round(champ_mae - cand_mae, 2)

    print(f"   Champion  MAE: {champ_mae:,.1f} kW")
    print(f"   Candidate MAE: {cand_mae:,.1f} kW")
    print(f"   Improvement:   {improvement:+.1f} kW")
    print(
        "   Decision:      "
        f"{'PROMOTE candidate' if candidate_wins else 'KEEP champion'}"
    )

    return {
        "champion_mae": round(champ_mae, 2),
        "candidate_mae": round(cand_mae, 2),
        "improvement": improvement,
        "candidate_wins": candidate_wins,
    }


if __name__ == "__main__":
    metrics = evaluate_models()
    print(f"\nFinal metrics: {metrics}")