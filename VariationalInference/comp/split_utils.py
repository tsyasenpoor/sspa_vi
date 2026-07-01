"""Shared helper: apply a canonical patient->fold split (exported by export_split.py)
to a per-cell metadata frame, so scHPF / Spectra downstream classifiers evaluate on the
IDENTICAL train/val/test partition that the DRGP single-cell runs used.

The split file is JSON: {"patient_split": {patient_id: "train"|"val"|"test", ...}, ...}.
Cells whose patient is absent from the split (e.g. not in the subsampled portion) are
dropped from all folds.
"""
import json
import numpy as np


def load_patient_split(path):
    with open(path) as fh:
        d = json.load(fh)
    return d["patient_split"] if isinstance(d, dict) and "patient_split" in d else d


def indices_from_split(patient_ids, patient_split):
    """Map per-cell patient ids to fold indices using a {patient_id: fold} dict.

    Returns (train_idx, val_idx, test_idx, n_dropped) as arrays of row positions
    into ``patient_ids``. Patients not present in ``patient_split`` contribute to
    ``n_dropped`` and appear in no fold.
    """
    pid = np.asarray([str(p) for p in patient_ids])
    fold_of = np.array([patient_split.get(p, None) for p in pid], dtype=object)
    train_idx = np.where(fold_of == "train")[0]
    val_idx = np.where(fold_of == "val")[0]
    test_idx = np.where(fold_of == "test")[0]
    n_dropped = int(np.sum(fold_of == None))  # noqa: E711 (object-dtype compare)
    return train_idx, val_idx, test_idx, n_dropped
