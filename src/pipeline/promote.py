import json
import shutil
from datetime import datetime
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_CHAMPION = _ROOT / "models" / "xgb_champion.json"
_CANDIDATE = _ROOT / "models" / "xgb_candidate.json"
_HISTORY = _ROOT / "models" / "promotion_history.json"


def promote_candidate(evaluation: dict) -> bool:
    """
    Replace the champion model with the candidate when it has lower MAE.
    Logs promotion history locally.
    """
    if not evaluation["candidate_wins"]:
        print("❌ Candidate did not beat champion — keeping current model.")
        return False

    # Back up the current champion.
    backup = (
        _ROOT
        / "models"
        / f"xgb_champion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy(_CHAMPION, backup)
    print(f"📦 Champion backed up → {backup.name}")

    # Promote candidate to champion.
    shutil.copy(_CANDIDATE, _CHAMPION)
    print("🏆 Candidate promoted to champion!")
    print(
        f"   MAE improved: {evaluation['champion_mae']:,.1f}"
        f" → {evaluation['candidate_mae']:,.1f} kW"
        f" ({evaluation['improvement']:+.1f} kW)"
    )

    # Log promotion history locally.
    history = []
    if _HISTORY.exists():
        with open(_HISTORY) as f:
            history = json.load(f)

    history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "champion_mae": evaluation["champion_mae"],
            "candidate_mae": evaluation["candidate_mae"],
            "improvement": evaluation["improvement"],
        }
    )

    with open(_HISTORY, "w") as f:
        json.dump(history, f, indent=2)

    print(f"📝 Promotion logged → {_HISTORY}")
    return True