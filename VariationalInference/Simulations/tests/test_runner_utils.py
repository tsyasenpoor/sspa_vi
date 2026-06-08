import numpy as np
from VariationalInference.Simulations._runner_utils import (
    derive_seeds, patient_grouped_split,
)

def test_derive_seeds_split_invariant_to_method_and_K():
    base = dict(truth_idx=2, h2=0.3, r=0.15, inner_seed=0)
    s1 = derive_seeds(**base, K=8,  method="drgp_unmasked")
    s2 = derive_seeds(**base, K=14, method="nmf_lr")
    assert s1["split_seed"] == s2["split_seed"]
    assert s1["fit_seed"]   != s2["fit_seed"]

def test_derive_seeds_dataset_change_breaks_split():
    a = derive_seeds(truth_idx=0, h2=0.3, r=0.15, inner_seed=0, K=8, method="drgp_unmasked")
    b = derive_seeds(truth_idx=1, h2=0.3, r=0.15, inner_seed=0, K=8, method="drgp_unmasked")
    assert a["split_seed"] != b["split_seed"]

def test_patient_grouped_split_no_leak():
    patient_ids = np.array([f"P{i:02d}" for i in range(40) for _ in range(200)])
    train_idx, test_idx = patient_grouped_split(patient_ids, n_test_patients=8, seed=42)
    train_p = set(patient_ids[train_idx]); test_p = set(patient_ids[test_idx])
    assert train_p.isdisjoint(test_p)
    assert len(train_p) == 32 and len(test_p) == 8
    assert len(train_idx) + len(test_idx) == len(patient_ids)
