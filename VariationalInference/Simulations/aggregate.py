"""Walks results/ for metrics.json, produces tidy parquet + bottleneck join."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config


def _flatten_metrics(m: dict) -> list[dict]:
    """Explode list-valued metrics into one row per index."""
    rows = []
    base = {k: v for k, v in m.items() if not isinstance(v, list)}
    rows.append(base | {"metric_kind": "scalar"})
    for key in ("matched_cosine_per_l", "jaccard_oracle_per_l",
                "jaccard_drgp_native_per_l", "splitting_concentration",
                "splitting_coverage"):
        if key in m and isinstance(m[key], list):
            for i, v in enumerate(m[key]):
                rows.append(base | {"metric_kind": key, "l_or_path_idx": i, "value": v})
    return rows


def collect(results_root: Path | None = None) -> pd.DataFrame:
    results_root = results_root or (config.SIM_ROOT / "results")
    rows: list[dict] = []
    for mj in results_root.rglob("metrics.json"):
        try:
            m = json.loads(mj.read_text())
        except Exception:
            continue
        rows.extend(_flatten_metrics(m))
    df = pd.DataFrame(rows)
    out = config.SIM_ROOT / "metrics.parquet"
    df.to_parquet(out)
    return df


def bottleneck_join(tau: float = 0.5) -> pd.DataFrame:
    """For each (method, mode, condition, truth, inner_seed), recovery-hit-rate x cell-AUC."""
    df = pd.read_parquet(config.SIM_ROOT / "metrics.parquet")
    cos = df[df["metric_kind"] == "matched_cosine_per_l"]
    grouped = cos.groupby(
        ["method", "mode", "truth_idx", "h2", "r", "K", "inner_seed"]
    )["value"].agg(lambda s: float((np.asarray(s) > tau).mean())).rename("hit_rate").reset_index()
    scalar = df[df["metric_kind"] == "scalar"]
    keys = ["method", "mode", "truth_idx", "h2", "r", "K", "inner_seed"]
    auc = scalar[keys + ["cell_auc_integrated"]].drop_duplicates(keys)
    out = grouped.merge(auc, on=keys, how="left")
    out["recovery_hit"] = out["hit_rate"] >= 0.5
    out.to_parquet(config.SIM_ROOT / "bottleneck.parquet")
    return out


def stability_summary(method_family_modes: list[tuple[str, str]],
                      K: int | None = None) -> pd.DataFrame:
    K = K or config.STABILITY_CELL["K"]
    from .evaluate import stability_run
    rows = []
    for fam, mode in method_family_modes:
        for t in range(config.G_TRUTH):
            try:
                rows.append(stability_run(t, fam, mode, K))
            except Exception as e:
                print(f"  skip {fam}/{mode} truth{t}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(config.SIM_ROOT / "stability.parquet")
    return df
