# src/pipeline/run_pipeline.py
import json
from pathlib import Path
from src.pipeline.train    import train_candidate
from src.pipeline.evaluate import evaluate_models
from src.pipeline.promote  import promote_candidate

_ROOT  = Path(__file__).resolve().parents[2]
_DRIFT = _ROOT / "data" / "processed" / "drift_report.json"


def check_drift_threshold(threshold: float = 0.5) -> bool:
    """Return True if drift % exceeds threshold."""
    if not _DRIFT.exists():
        print("⚠️  No drift report found — running retraining anyway.")
        return True

    with open(_DRIFT) as f:
        report = json.load(f)

    drift_pct = report["summary"]["drift_pct"] / 100
    print(f"   Drift detected: {report['summary']['drifted_features']}"
          f"/{report['summary']['total_features']} features"
          f" ({drift_pct:.0%})")

    if drift_pct >= threshold:
        print(f"   ⚠️  Drift {drift_pct:.0%} ≥ threshold {threshold:.0%}"
              f" → retraining triggered")
        return True
    else:
        print(f"   ✅ Drift {drift_pct:.0%} < threshold {threshold:.0%}"
              f" → no retraining needed")
        return False


def run_pipeline(force: bool = False):
    """
    Full retraining pipeline:
    1. Check drift
    2. Retrain candidate
    3. Evaluate vs champion
    4. Promote if better
    """
    print("=" * 55)
    print("  SOLAR POWER PREDICTOR — RETRAINING PIPELINE")
    print("=" * 55)

    # ── Step 1: Drift check ───────────────────────────────
    print("\n📊 Step 1: Drift Check")
    should_retrain = force or check_drift_threshold()
    if not should_retrain:
        print("\n✅ Pipeline complete — no action needed.")
        return

    # ── Step 2: Retrain ───────────────────────────────────
    print("\n🤖 Step 2: Training Candidate Model")
    train_metrics = train_candidate()

    # ── Step 3: Evaluate ──────────────────────────────────
    print("\n📏 Step 3: Evaluating Candidate vs Champion")
    evaluation = evaluate_models()

    # ── Step 4: Promote ───────────────────────────────────
    print("\n🏆 Step 4: Promotion Decision")
    promoted = promote_candidate(evaluation)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Retrained:  ✅")
    print(f"  Promoted:   {'✅ New champion deployed' if promoted else '❌ Champion unchanged'}")
    print(f"  Champion MAE after pipeline: "
          f"{evaluation['candidate_mae'] if promoted else evaluation['champion_mae']:,.1f} kW")


if __name__ == "__main__":
    run_pipeline(force=True)