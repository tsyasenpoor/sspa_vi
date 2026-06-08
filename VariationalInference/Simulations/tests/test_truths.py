import numpy as np
from VariationalInference.Simulations.truths import draw_truth
from VariationalInference.Simulations import config

def test_draw_truth_shapes_and_disjoint():
    t = draw_truth(truth_idx=0)
    assert t["S"].shape == (config.L_COLS,)         # object array of index arrays
    assert t["u"].shape == (config.N_GENES, config.L_COLS)
    assert t["v_star"].shape == (config.L_COLS,)
    assert t["mask_M"].shape == (config.N_GENES, config.KAPPA_PATH)
    sizes = [len(s) for s in t["S"]]
    for sz in sizes:
        assert config.CARRIER_SIZE_LO <= sz <= config.CARRIER_SIZE_HI
    flat = np.concatenate([s for s in t["S"]])
    assert len(flat) == len(set(flat.tolist()))     # all carriers disjoint
    assert t["v_star"][0] == 0.0                    # decoy at l=0
    nonzero_signs = np.sign(t["v_star"][1:])
    assert nonzero_signs.sum() == 0                 # balanced signs across the 4 causal

def test_draw_truth_responder_types():
    t = draw_truth(truth_idx=3)
    for l, T_l in enumerate(t["T_ell"]):
        assert config.RESPONDER_SIZE_LO <= len(T_l) <= config.RESPONDER_SIZE_HI
        assert len(T_l) < config.T   # no program covers all 6 types

def test_draw_truth_deterministic():
    a = draw_truth(truth_idx=2); b = draw_truth(truth_idx=2)
    np.testing.assert_array_equal(a["u"], b["u"])
    for sa, sb in zip(a["S"], b["S"]):
        np.testing.assert_array_equal(sa, sb)
