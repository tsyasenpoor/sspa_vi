"""Per-outer-seed structural ground truth: carrier sets, unit pattern, T_l, v_star, mask."""
from __future__ import annotations
import hashlib
import numpy as np
from pathlib import Path
from . import config

L = config.L_COLS         # = 5 (decoy + 4 causal)
N_CAUSAL = config.IOTA    # = 4


def _seed(truth_idx: int) -> int:
    h = hashlib.blake2b(f"truth-{truth_idx}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF


def draw_truth(truth_idx: int) -> dict:
    rng = np.random.default_rng(_seed(truth_idx))

    sizes = rng.integers(config.CARRIER_SIZE_LO, config.CARRIER_SIZE_HI + 1, size=L)
    total = sizes.sum()
    pool = rng.permutation(config.N_GENES)[:total]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    S = np.empty(L, dtype=object)
    for l in range(L):
        S[l] = pool[offsets[l]:offsets[l + 1]].astype(np.int64)

    u = np.zeros((config.N_GENES, L), dtype=np.float32)
    for l in range(L):
        u[S[l], l] = rng.uniform(config.U_LO, config.U_HI, size=len(S[l])).astype(np.float32)

    T_ell = np.empty(L, dtype=object)
    for l in range(L):
        while True:
            k_resp = int(rng.integers(config.RESPONDER_SIZE_LO, config.RESPONDER_SIZE_HI + 1))
            picks = np.sort(rng.choice(config.T, size=k_resp, replace=False))
            if len(picks) < config.T:
                T_ell[l] = picks.astype(np.int8)
                break

    v_star = np.zeros(L, dtype=np.float32)
    v_star[0] = config.V_STAR_DECOY
    signs = np.array([+1, -1, +1, -1], dtype=np.float32)
    rng.shuffle(signs)
    v_star[1:1 + N_CAUSAL] = signs * np.array(config.V_STAR_MAGNITUDES, dtype=np.float32)

    mask_M = np.zeros((config.N_GENES, config.KAPPA_PATH), dtype=np.uint8)
    for k in range(config.KAPPA_PATH):
        mask_M[S[1 + k], k] = 1     # pathway columns are causal l = 1..K_path

    return dict(truth_idx=truth_idx, S=S, u=u, T_ell=T_ell,
                v_star=v_star, mask_M=mask_M, seed=_seed(truth_idx))


def save_truth(truth_idx: int, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (config.SIM_ROOT / "truths" / f"{truth_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t = draw_truth(truth_idx)
    np.savez(out_dir / "truth.npz",
             S=t["S"], u=t["u"], T_ell=t["T_ell"],
             v_star=t["v_star"], mask_M=t["mask_M"],
             truth_idx=truth_idx, seed=t["seed"])
    return out_dir / "truth.npz"


def load_truth(truth_idx: int) -> dict:
    p = config.SIM_ROOT / "truths" / f"{truth_idx}" / "truth.npz"
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
