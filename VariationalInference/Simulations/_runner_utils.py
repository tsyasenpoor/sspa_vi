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
