"""Phase B hard gate. DRGP-unmasked vs best of {NMF+LR, scHPF+LR} at K=8,
3 truths x 3 r values x 1 seed x 1 fold. Asserts the crossover band shape."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
from . import config
from .dataset import write_dataset, dataset_path
from .runner_drgp import run as run_drgp
from .runner_unsup import run as run_unsup


def _ensure_dataset(truth_idx: int, r: float):
    p = dataset_path(truth_idx, h2=0.3, r=r, inner_seed=0)
    if not p.exists():
        write_dataset(truth_idx, 0.3, r, 0)


def run(K: int = 8, truths: list[int] | None = None) -> dict:
    truths = truths or list(range(config.GATE_N_TRUTHS))
    base = config.SIM_ROOT / "gates" / "r_crossover"
    if base.exists(): shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    gaps: dict[float, list[float]] = {r: [] for r in config.R_VALUES}
    for t in truths:
        for r in config.R_VALUES:
            _ensure_dataset(t, r)
            h5 = dataset_path(t, h2=0.3, r=r, inner_seed=0)
            sub = base / f"truth{t}_r{r}"; sub.mkdir(parents=True, exist_ok=True)
            drgp_m = run_drgp(str(h5), mode="unmasked", K=K, inner_seed=0,
                              out_dir=str(sub / "drgp"))
            nmf_m  = run_unsup(str(h5), method="nmf",  K=K, inner_seed=0,
                               out_dir=str(sub / "nmf"))
            schpf_m = run_unsup(str(h5), method="schpf", K=K, inner_seed=0,
                                out_dir=str(sub / "schpf"))
            # DRGP: regime-consistent head (Poisson-only theta_tr_pois + LR scored
            # on Poisson-only theta_te) — apples-to-apples with the unsup methods,
            # which fit and score their LR head in a single non-supervised regime.
            drgp_auc = drgp_m["cell_auc_consistent"]
            best_unsup = max(nmf_m["cell_auc_integrated"], schpf_m["cell_auc_integrated"])
            gap = drgp_auc - best_unsup
            gaps[r].append(gap)
            print(f"  truth={t} r={r}: drgp(consistent)={drgp_auc:.3f} "
                  f"best_unsup={best_unsup:.3f}  gap={gap:+.3f}  "
                  f"[drgp-integrated={drgp_m['cell_auc_integrated']:.3f}, "
                  f"drgp-theta-only={drgp_m['cell_auc_theta_only']:.3f}]")
    summary = {r: float(np.mean(gs)) for r, gs in gaps.items()}
    print(f"\n  mean gap per r: {summary}")
    assert summary[0.05] >= config.GATE_RLOW_MIN, \
        f"gap[r=0.05] = {summary[0.05]:.3f} < {config.GATE_RLOW_MIN}"
    assert config.GATE_RMID_LO <= summary[0.15] <= config.GATE_RMID_HI, \
        f"gap[r=0.15] = {summary[0.15]:.3f} outside [{config.GATE_RMID_LO}, {config.GATE_RMID_HI}]"
    assert summary[0.30] <= config.GATE_RHIGH_MAX, \
        f"gap[r=0.30] = {summary[0.30]:.3f} > {config.GATE_RHIGH_MAX}"
    (base / "gate.json").write_text(json.dumps(
        {"per_r_per_truth": gaps, "mean": summary, "PASS": True}, indent=2))
    print("  PASS")
    return summary


if __name__ == "__main__":
    run()
