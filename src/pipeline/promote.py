# src/pipeline/promote.py
import shutil
import json
from pathlib import Path
from datetime import datetime

_ROOT      = Path(__file__).resolve().parents[2]
_CHAMPION  = _ROOT / "models" / "xgb_champion.json"
_CANDIDATE = _ROOT / "models" / "xgb_candidate.json"
_HISTORY   = _ROOT / "models" / "promotion_history.json"


def promote_candidate(evaluation: dict):
    """
    Replace champion with candidate if evaluation says to.
    Logs promotion history.
    """
    if not evaluation["candidate_wins"]:
        print("❌ Candidate did not beat champion — keeping current model.")
        return False

    # ── Back up old champion ──────────────────────────────
    backup = _ROOT / "models" / f"xgb_champion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy(_CHAMPION, backup)
    print(f"📦 Champion backed up → {backup.name}")

    # ── Promote candidate to champion ─────────────────────
    shutil.copy(_CANDIDATE, _CHAMPION)
    print(f"🏆 Candidate promoted to champion!")
    print(f"   MAE improved: {evaluation['champion_mae']:,.1f}"
          f" → {evaluation['candidate_mae']:,.1f} kW"
          f" ({evaluation['improvement_kw']:+.1f} kW)")

    # ── Log promotion history ─────────────────────────────
    history = []
    if _HISTORY.exists():
        with open(_HISTORY) as f:
            history = json.load(f)

    history.append({
        "timestamp":      datetime.now().isoformat(),
        "champion_mae":   evaluation["champion_mae"],
        "candidate_mae":  evaluation["candidate_mae"],
        "improvement_kw": evaluation["improvement_kw"],
    })

    with open(_HISTORY, "w") as f:
        json.dump(history, f, indent=2)

    print(f"📝 Promotion logged → {_HISTORY}")
    return True


if __name__ == "__main__":
    # Test with dummy evaluation
    promote_candidate({
        "champion_mae": 500.0,
        "candidate_mae": 456.8,
        "improvement_kw": 43.2,
        "candidate_wins": True,
    })