"""Phase A.7 hard gate. Confirms regression_weight actually moves cell_auc_integrated.

  rw=0    -> integrated AUC must be ~ chance       (cell_auc_integrated <= RW_ENGAGEMENT_RW0_AUC_MAX)
  rw=15   -> integrated AUC must rise              (rw15 - rw0 >= RW_ENGAGEMENT_GAP_MIN)

cell_auc_posthoc does NOT count - a high posthoc under rw=0 is the dormant signature.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from .runner_drgp import run as run_drgp
from .dataset import dataset_path, write_dataset
from . import config


def run(truth_idx: int = 0, K: int = 8) -> dict:
    h2 = config.HEADLINE_CELL["h2"]
    r  = config.HEADLINE_CELL["r"]
    p = dataset_path(truth_idx, h2=h2, r=r, inner_seed=0)
    if not p.exists():
        p = write_dataset(truth_idx, h2, r, 0)
    base = config.SIM_ROOT / "gates" / "rw_engagement"
    base.mkdir(parents=True, exist_ok=True)
    results = {}
    for rw in (0.0, config.REGRESSION_WEIGHT):
        sub = base / f"rw{rw}"
        if sub.exists(): shutil.rmtree(sub)
        m = run_drgp(str(p), mode="unmasked", K=K, inner_seed=0,
                     out_dir=str(sub), regression_weight=rw, max_iter=2000,
                     early_stopping="none", verbose=True)
        # Gate on Theta-only train AUC — measures whether (β, υ) carry label-
        # discriminative signal without γ absorbing it. cell_auc_integrated_train
        # is blind to γ-memorization at high rw (γ alone can achieve high AUC);
        # cell_auc_theta_only_train measures the factor-level engagement directly.
        results[rw] = m["cell_auc_theta_only_train"]
        print(f"  rw={rw:>5}  theta-only train = {m['cell_auc_theta_only_train']:.4f}"
              f"   integrated train = {m['cell_auc_integrated_train']:.4f}"
              f"   theta-only held-out = {m['cell_auc_theta_only']:.4f}"
              f"   consistent held-out = {m['cell_auc_consistent']:.4f}")
    gap = results[config.REGRESSION_WEIGHT] - results[0.0]
    print(f"\n  gap (theta-only train AUC, rw=15 - rw=0) = {gap:+.4f}")
    assert results[0.0] <= config.RW_ENGAGEMENT_RW0_AUC_MAX, \
        f"rw=0 theta-only train AUC {results[0.0]:.3f} > {config.RW_ENGAGEMENT_RW0_AUC_MAX}"
    assert gap >= config.RW_ENGAGEMENT_GAP_MIN, \
        f"rw gap {gap:+.3f} < {config.RW_ENGAGEMENT_GAP_MIN}"
    (base / "gate.json").write_text(json.dumps(
        {"results": results, "gap": gap, "PASS": True}, indent=2))
    print("  PASS")
    return results


if __name__ == "__main__":
    run()
