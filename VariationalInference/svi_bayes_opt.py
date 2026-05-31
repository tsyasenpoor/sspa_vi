#!/usr/bin/env python3
"""Bayesian Optimization for SVIPG hyperparameters.

Targets the stochastic VI implementation at svi.py (Hoffman & Blei 2013,
Algorithm 1; final-iterate point estimate; ρ_t = (τ_0 + t)^(-κ_lr)).

Parallels the structure of bayes_opt.py (CAVI variant) but optimizes
SVI-specific hyperparameters.  Search dimensions:

    Model:       n_factors, a, c, b_v, sigma_gamma, alpha_pi, beta_pi_scale
    SVI:         tau0, kappa_lr

Algorithmic dials NOT tuned (kept at sensible production values):
    batch_size, n_local_iter, n_pg_subsweeps, v_warmup, pg_ema_alpha

Two feature-type presets adjust the n_factors range:
    --feature-type gex          n_factors ∈ [100, 500] step 50
    --feature-type singscore    n_factors ∈ [30, 150]  step 10

Metric: macro-average validation AUC across all label columns.

Usage
-----
python -m VariationalInference.svi_bayes_opt \
    --data /labs/Aguiar/SSPA_BRAY/dataset/biorepository/Bcell_GEX_20251201.h5ad \
    --label-column t2dm cvda \
    --aux-columns Sex \
    --feature-type gex \
    --n-trials 40 \
    --max-iter 800 \
    --output-dir <out>
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
import warnings
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from VariationalInference.data_loader import DataLoader
from VariationalInference.svi import SVIPG, SVIConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


# ============================================================================
# Search space
# ============================================================================

SHARED_SEARCH_SPACE: Dict[str, tuple] = {
    "a":              ("float",     0.5, 4.0),
    "c":              ("float",     0.5, 4.0),
    "b_v":            ("log_float", 0.5, 5.0),   # wider than CAVI: tight values break SVI
    "sigma_gamma":    ("log_float", 0.3, 3.0),
    "alpha_pi":       ("log_float", 0.1, 5.0),
    "beta_pi_scale":  ("log_float", 1.0, 50.0),
    "tau0":           ("log_float", 16.0, 4096.0),
    "kappa_lr":       ("float",     0.51, 1.0),  # Robbins-Monro: κ ∈ (0.5, 1]
}

FEATURE_TYPE_RANGES = {
    "gex":       {"n_factors": ("int", 100, 500, 50)},
    "singscore": {"n_factors": ("int", 30,  150, 10)},
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple):
    t = spec[0]
    if t == "int":
        step = spec[3] if len(spec) > 3 else 1
        return trial.suggest_int(name, spec[1], spec[2], step=step)
    if t == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if t == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    raise ValueError(f"unknown param type {t}")


def build_search_space(feature_type: str) -> Dict[str, tuple]:
    space = dict(SHARED_SEARCH_SPACE)
    if feature_type not in FEATURE_TYPE_RANGES:
        raise ValueError(f"feature_type must be one of {list(FEATURE_TYPE_RANGES)}; got {feature_type}")
    space.update(FEATURE_TYPE_RANGES[feature_type])
    return space


# ============================================================================
# Trial objective
# ============================================================================

class SVITrialObjective:
    """One Optuna trial: train SVIPG with sampled hyperparams, return macro-AUC."""

    def __init__(
        self,
        X_train, y_train, X_aux_train,
        X_val,   y_val,   X_aux_val,
        search_space: Dict[str, tuple],
        params_to_tune: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        max_iter: int = 800,
        batch_size: int = 1024,
        n_local_iter: int = 5,
        n_pg_subsweeps: int = 3,
        v_warmup: int = 100,
        pg_ema_alpha: float = 0.3,
        check_freq: int = 50,
        random_state: Optional[int] = None,
        label_names: Optional[List[str]] = None,
        kappa: int = 1,
    ):
        self.X_train, self.y_train, self.X_aux_train = X_train, y_train, X_aux_train
        self.X_val,   self.y_val,   self.X_aux_val   = X_val,   y_val,   X_aux_val
        self.search_space   = search_space
        self.params_to_tune = params_to_tune
        self.fixed_params   = fixed_params or {}
        self.max_iter       = max_iter
        self.batch_size     = batch_size
        self.n_local_iter   = n_local_iter
        self.n_pg_subsweeps = n_pg_subsweeps
        self.v_warmup       = v_warmup
        self.pg_ema_alpha   = pg_ema_alpha
        self.check_freq     = check_freq
        self.random_state   = random_state
        self.label_names    = label_names
        self.kappa          = kappa

    def _sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            if self.params_to_tune is not None and name not in self.params_to_tune:
                continue
            if name in self.fixed_params:
                continue
            params[name] = _suggest(trial, name, spec)
        params.update(self.fixed_params)
        return params

    def _build_cfg(self, params: Dict[str, Any]) -> SVIConfig:
        # Fill SVIConfig from sampled params, defaulting everything else.
        return SVIConfig(
            n_factors      = int(params.get("n_factors", 50)),
            a              = float(params.get("a", 0.3)),
            c              = float(params.get("c", 0.3)),
            b_v            = float(params.get("b_v", 2.0)),
            sigma_gamma    = float(params.get("sigma_gamma", 1.0)),
            alpha_pi       = float(params.get("alpha_pi", 1.0)),
            beta_pi_scale  = float(params.get("beta_pi_scale", 5.0)),
            batch_size     = int(self.batch_size),
            tau0           = float(params.get("tau0", 100.0)),
            kappa_lr       = float(params.get("kappa_lr", 0.7)),
            n_local_iter   = int(self.n_local_iter),
            n_pg_subsweeps = int(self.n_pg_subsweeps),
            pg_ema_alpha   = float(self.pg_ema_alpha),
            v_warmup       = int(self.v_warmup),
            use_spike_slab = True,
        )

    @staticmethod
    def _macro_auc(y_true, y_proba) -> tuple:
        """Return (macro_auc, per_label_auc_list)."""
        y_true  = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        if y_true.ndim == 1:
            y_true  = y_true[:, None]
            y_proba = y_proba.reshape(-1, 1)
        elif y_proba.ndim == 1:
            y_proba = y_proba.reshape(-1, 1)
        aucs = []
        for k in range(y_true.shape[1]):
            yk, pk = y_true[:, k], y_proba[:, k]
            if len(np.unique(yk)) < 2:
                aucs.append(0.5)
            else:
                aucs.append(float(roc_auc_score(yk, pk)))
        return float(np.mean(aucs)), aucs

    def __call__(self, trial: optuna.Trial) -> float:
        try:
            params = self._sample(trial)
            cfg = self._build_cfg(params)
            model = SVIPG(cfg, random_state=self.random_state, kappa=self.kappa)
            t0 = time.time()
            model.fit(
                self.X_train, self.y_train, X_aux=self.X_aux_train,
                X_val=self.X_val, y_val=self.y_val, X_aux_val=self.X_aux_val,
                max_iter=self.max_iter, check_freq=self.check_freq,
                verbose=False, print_every=10**9,
            )
            train_secs = time.time() - t0

            val_proba = model.predict_proba(self.X_val, X_aux_new=self.X_aux_val, n_iter=20)
            macro_auc, aucs = self._macro_auc(self.y_val, val_proba)

            # Stash per-label + diagnostics for the trial record
            label_names = self.label_names or [f"label{k}" for k in range(len(aucs))]
            for k, ln in enumerate(label_names[: len(aucs)]):
                trial.set_user_attr(f"val_{ln}_auc", float(aucs[k]))
            trial.set_user_attr("val_macro_auc", float(macro_auc))
            trial.set_user_attr("train_seconds", float(train_secs))
            trial.set_user_attr("n_factors", int(cfg.n_factors))
            trial.set_user_attr("final_holl",
                                float(model.holl_history_[-1][1])
                                if model.holl_history_ else None)
            mu_v = np.asarray(model.mu_v if not hasattr(model.mu_v, "__array__")
                              else np.array(model.mu_v))
            trial.set_user_attr("v_norm",    float(np.linalg.norm(mu_v)))
            trial.set_user_attr("v_max_abs", float(np.abs(mu_v).max()))

            logger.info(
                "trial %d  macro_auc=%.4f  K=%d  tau0=%.1f  kappa_lr=%.2f  "
                "a=%.2f  c=%.2f  b_v=%.3f  sigma_g=%.3f  alpha_pi=%.2f  "
                "beta_pi_scale=%.2f  train=%.0fs",
                trial.number, macro_auc, cfg.n_factors, cfg.tau0, cfg.kappa_lr,
                cfg.a, cfg.c, cfg.b_v, cfg.sigma_gamma, cfg.alpha_pi, cfg.beta_pi_scale,
                train_secs,
            )
            return macro_auc

        except Exception as e:
            logger.exception("trial %d failed: %s", trial.number, e)
            return 0.0


# ============================================================================
# Orchestrator
# ============================================================================

class SVIBayesOpt:
    """Run an SVI hyperparameter search on a biorepo h5ad dataset."""

    def __init__(
        self,
        data_path: str,
        label_columns: List[str],
        aux_columns: Optional[List[str]],
        feature_type: str,
        n_trials: int = 40,
        max_iter: int = 800,
        batch_size: int = 1024,
        params_to_tune: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: Optional[int] = 42,
        output_dir: str = "./svi_bayes_opt_results",
        study_name: Optional[str] = None,
    ):
        self.data_path      = data_path
        self.label_columns  = list(label_columns)
        self.aux_columns    = list(aux_columns) if aux_columns else None
        self.feature_type   = feature_type
        self.n_trials       = n_trials
        self.max_iter       = max_iter
        self.batch_size     = batch_size
        self.params_to_tune = params_to_tune
        self.fixed_params   = fixed_params or {}
        self.train_ratio    = train_ratio
        self.val_ratio      = val_ratio
        self.random_state   = random_state
        self.output_dir     = Path(output_dir)
        self.study_name     = study_name or f"svi_bayes_opt_{datetime.now():%Y%m%d_%H%M%S}"

        self.data = None
        self.search_space: Optional[Dict[str, tuple]] = None

    def load(self):
        logger.info("loading %s", self.data_path)
        loader = DataLoader(data_path=self.data_path,
                            cache_dir="/labs/Aguiar/SSPA_BRAY/cache",
                            use_cache=True, verbose=False)
        label_arg = (self.label_columns if len(self.label_columns) > 1
                     else self.label_columns[0])
        self.data = loader.load_and_preprocess(
            label_column=label_arg,
            aux_columns=self.aux_columns,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            stratify_by=self.label_columns[0],
            min_cells_expressing=0.01,
            random_state=self.random_state,
            normalize=False,
            return_sparse=True,
        )
        Xt, _, _ = self.data["train"]
        logger.info("loaded: n_train=%d n_val=%d n_test=%d n_features=%d",
                    Xt.shape[0],
                    self.data["val"][0].shape[0],
                    self.data["test"][0].shape[0],
                    Xt.shape[1])

    def run(self) -> optuna.Study:
        if self.data is None:
            self.load()
        self.search_space = build_search_space(self.feature_type)

        X_train, X_aux_train, y_train = self.data["train"]
        X_val,   X_aux_val,   y_val   = self.data["val"]
        if y_train.ndim == 1:
            y_train = y_train[:, None]; y_val = y_val[:, None]
        kappa = y_train.shape[1]

        objective = SVITrialObjective(
            X_train=X_train, y_train=y_train, X_aux_train=X_aux_train,
            X_val=X_val, y_val=y_val, X_aux_val=X_aux_val,
            search_space=self.search_space,
            params_to_tune=self.params_to_tune,
            fixed_params=self.fixed_params,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            random_state=self.random_state,
            label_names=self.label_columns,
            kappa=kappa,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        sampler = TPESampler(seed=self.random_state, n_startup_trials=10, multivariate=True)
        pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=sampler, pruner=pruner,
        )

        logger.info("starting SVI BO: %d trials  max_iter=%d  batch=%d  feature_type=%s",
                    self.n_trials, self.max_iter, self.batch_size, self.feature_type)
        t0 = time.time()
        study.optimize(objective, n_trials=self.n_trials)
        elapsed = time.time() - t0

        self._report(study, elapsed)
        self._save(study)
        return study

    def _report(self, study: optuna.Study, elapsed: float):
        best = study.best_trial
        print("\n" + "=" * 72)
        print(f"  SVI BAYESIAN OPTIMIZATION COMPLETE  ({self.study_name})")
        print("=" * 72)
        print(f"  data:            {self.data_path}")
        print(f"  feature_type:    {self.feature_type}")
        print(f"  labels:          {self.label_columns}")
        print(f"  trials:          {len(study.trials)}")
        print(f"  best trial:      #{best.number}")
        print(f"  best macro AUC:  {best.value:.4f}")
        for k, v in best.user_attrs.items():
            if k.startswith("val_") or k in ("final_holl", "v_norm", "v_max_abs", "train_seconds"):
                vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {k:24s} {vstr}")
        print(f"  elapsed:         {elapsed:.0f}s ({elapsed/3600:.2f}h)")
        print(f"  best params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k:24s} {v if not isinstance(v, float) else f'{v:.6g}'}")
        print("=" * 72)

    def _save(self, study: optuna.Study):
        best = study.best_trial
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        record = {
            "method": "svi",
            "study_name": self.study_name,
            "data_path": str(self.data_path),
            "feature_type": self.feature_type,
            "label_columns": self.label_columns,
            "aux_columns": self.aux_columns,
            "n_trials": len(study.trials),
            "best_trial": best.number,
            "best_macro_auc": best.value,
            "best_params": best.params,
            "fixed_params": self.fixed_params,
            "best_attrs": {k: v for k, v in best.user_attrs.items()},
            "timestamp": ts,
            "config_template": {
                "batch_size": self.batch_size,
                "max_iter": self.max_iter,
                "v_warmup": 100,
                "n_local_iter": 5,
                "n_pg_subsweeps": 3,
                "pg_ema_alpha": 0.3,
            },
        }
        json_path = self.output_dir / f"best_params_svi_{self.feature_type}_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        logger.info("best params saved to %s", json_path)

        try:
            df = study.trials_dataframe()
            csv_path = self.output_dir / f"all_trials_svi_{self.feature_type}_{ts}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("trial table saved to %s", csv_path)
        except Exception as e:
            logger.warning("could not save trials CSV: %s", e)


# ============================================================================
# CLI
# ============================================================================

def _parse_fixed_params(raw: list) -> dict:
    out: dict = {}
    if not raw:
        return out
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--fixed-params expects KEY=VALUE; got '{item}'")
        k, v = item.split("=", 1)
        k = k.replace("-", "_")
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[k] = v
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bayesian opt for SVIPG hyperparameters")
    p.add_argument("--data", required=True)
    p.add_argument("--label-column", nargs="+", default=["t2dm", "cvda"])
    p.add_argument("--aux-columns", nargs="+", default=["Sex"])
    p.add_argument("--feature-type", choices=list(FEATURE_TYPE_RANGES), required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--max-iter", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--params-to-tune", nargs="+", default=None)
    p.add_argument("--fixed-params", nargs="+", default=None)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="./svi_bayes_opt_results")
    p.add_argument("--study-name", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bo = SVIBayesOpt(
        data_path=args.data,
        label_columns=args.label_column,
        aux_columns=args.aux_columns,
        feature_type=args.feature_type,
        n_trials=args.n_trials,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        params_to_tune=args.params_to_tune,
        fixed_params=_parse_fixed_params(args.fixed_params),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        output_dir=args.output_dir,
        study_name=args.study_name,
    )
    bo.run()


if __name__ == "__main__":
    main()
