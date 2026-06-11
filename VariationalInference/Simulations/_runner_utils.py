"""Seed derivation and patient-grouped split helpers shared by all runners."""
from __future__ import annotations
import hashlib
import numpy as np
from typing import Sequence


def _hash_to_int32(*parts) -> int:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=4).digest()
    return int.from_bytes(h, "big", signed=False) & 0x7FFFFFFF


def derive_seeds(*, truth_idx: int, h2: float, r: float, inner_seed: int,
                 K: int, method: str) -> dict[str, int]:
    """Two seeds per fit (design §5).

    split_seed depends on (truth, h2, r, inner_seed) only — same train/test
    patient partition for every method and Κ at a given dataset.
    fit_seed adds (K, method) — different init noise per method.
    """
    return {
        "split_seed": _hash_to_int32("split", truth_idx, h2, r, inner_seed),
        "fit_seed":   _hash_to_int32("fit",   truth_idx, h2, r, inner_seed, K, method),
    }


def patient_grouped_split(patient_ids: np.ndarray, *, n_test_patients: int,
                          seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-index arrays (train, test) keeping every cell of a patient on one side."""
    rng = np.random.default_rng(seed)
    unique = np.unique(patient_ids)
    perm = rng.permutation(unique)
    test_p = set(perm[:n_test_patients].tolist())
    mask = np.array([pid in test_p for pid in patient_ids])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def pseudobulk_mean(X, patient_ids: np.ndarray, idx: np.ndarray, y: np.ndarray):
    """v2 patient pseudo-bulk: mean RAW-count profile per patient over the cells in `idx`.

    Returns (Xpb, ypb, pid_order): Xpb is (n_patients_in_idx x p) float32 mean counts; ypb the
    patient label (y is constant within patient under inherited labels); pid_order the patient-id
    strings in row order so callers can align liability_patient via int(pid[1:])."""
    import numpy as _np
    import scipy.sparse as _sp
    sub = X[idx]
    pids = patient_ids[idx]
    uniq = _np.unique(pids)
    p = X.shape[1]
    Xpb = _np.zeros((len(uniq), p), dtype=_np.float32)
    ypb = _np.zeros(len(uniq), dtype=_np.float32)
    for j, pid in enumerate(uniq):
        sel = pids == pid
        rows = sub[sel]
        Xpb[j] = _np.asarray(rows.mean(axis=0)).ravel() if _sp.issparse(rows) else rows.mean(axis=0)
        ypb[j] = float(y[idx][sel][0])
    return Xpb, ypb, uniq


def patient_liability(pid_order: np.ndarray, liability_patient: np.ndarray) -> np.ndarray:
    """Align uns['liability_patient'] (indexed by patient int g) to pid_order ('P03' -> g=3)."""
    import numpy as _np
    return _np.array([liability_patient[int(str(pid)[1:])] for pid in pid_order],
                     dtype=_np.float32)
