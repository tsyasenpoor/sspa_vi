import numpy as np
import pandas as pd
import pytest
from VariationalInference.Simulations import config
from VariationalInference.Simulations.dataset import patient_composition

def test_patient_composition_independent_of_D():
    rng = np.random.default_rng(42)
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
    g_i, D = patient_composition(types, rng)
    assert g_i.shape == types.shape
    assert set(np.unique(g_i).tolist()) == set(range(config.N_PATIENTS))
    assert D.sum() == config.N_PATIENTS // 2
    cmp_case = (D[g_i] == 1)
    f_case = np.bincount(types[cmp_case], minlength=config.T) / cmp_case.sum()
    f_ctrl = np.bincount(types[~cmp_case], minlength=config.T) / (~cmp_case).sum()
    assert np.max(np.abs(f_case - f_ctrl)) < 0.05    # composition independent of D


def test_liability_label_decoy_excluded_and_prevalence_target():
    from VariationalInference.Simulations.truths import draw_truth
    from VariationalInference.Simulations import dataset
    rng = np.random.default_rng(7)
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
    t = draw_truth(0)
    g_i, D = dataset.patient_composition(types, rng)
    theta = dataset.activity(t, types, g_i, D, rng)
    out = dataset.liability_label(theta, t["v_star"], h2=0.5, T_ell=t["T_ell"],
                                  D=D, g_i=g_i, types=types, rng=rng)
    causal_resp = np.zeros(len(types), dtype=bool)
    for l in range(1, config.L_COLS):
        causal_resp |= (D[g_i] == 1) & np.isin(types, t["T_ell"][l])
    target = causal_resp.mean()
    assert abs(out["y"].mean() - target) < 0.05
    assert np.var(out["liability"]) == pytest.approx(1.0, abs=0.02)
