"""Versioned, evidence-driven preprocessing for TopoAR.

Per Phase 3.0, we don't lock in Step 2's pipeline. We start minimal and add
transforms only when an observed pathology justifies one. Each version is a
strict superset of the previous:

    v0  RobustScaler                              (raw values)
    v1  RobustScaler over first-differences       (handles cross-session drift)
    v2  RobustScaler over arcsinh(diff(.))        (heavy-tail compression)
    v3  v2 then clip to ±CLIP_SCALED              (bounded LSTM gradients)

Block-isolation rule (always on, regardless of version): deltas never cross a
block_id boundary. The first row of each block is dropped from the delta stream.

Type-shared scalers (always on): du_scaler is fit on the vertical *stack* of
DU rows across all train topologies — single (W_DU, du_scaler) pair applies
to any DU instance. This is non-negotiable for topology agnosticism.

Zero-variance flooring (always on): features flagged in schema.json's
{cu,du}_zero_variance_indices have their RobustScaler IQR replaced with 1.0
post-fit so dividing by IQR doesn't blow up. The raw delta on those features
is preserved — that's the whole point: a flat-on-normal feature spiking under
stress is the strongest possible anomaly signal.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler


CLIP_SCALED = 5.0  # only used in v3


@dataclass
class PreprocessBundle:
    """Frozen, persistable preprocessing state.

    cu_scaler / du_scaler are fit-once on train; the same fitted scalers
    transform train, val, and test (no re-fit at eval time).

    cu_zero_variance_idx / du_zero_variance_idx are 0-based indices into the
    fitted scalers' feature axes. After scaler.fit(), we overwrite scale_[idx]
    with 1.0 so transform() leaves these features at their raw delta value.
    """
    version: str                          # "v0".."v3"
    cu_scaler: RobustScaler
    du_scaler: RobustScaler
    cu_dim: int
    du_dim: int
    use_delta: bool
    use_arcsinh: bool
    use_clip: bool
    cu_zero_variance_idx: List[int]
    du_zero_variance_idx: List[int]


VERSION_FLAGS: Dict[str, Tuple[bool, bool, bool]] = {
    # version: (use_delta, use_arcsinh, use_clip)
    "v0": (False, False, False),
    # v0_dunorm: same raw transform as v0; CU net normalization is learned inside
    # the model (W_DU softplus), not done here.
    "v0_dunorm": (False, False, False),
    "v1": (True,  False, False),
    "v2": (True,  True,  False),
    "v3": (True,  True,  True),
}


def block_diff(arr: np.ndarray, block_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First-difference along axis 0, dropping the first row of each block.

    arr:       (T, ...) any trailing shape
    block_id:  (T,) int

    returns (deltas, kept_mask) where:
        deltas    shape (sum(block_lens-1), ...) = (T - num_blocks, ...)
        kept_mask shape (T,) bool — True for rows whose delta was kept
                  (i.e. NOT the first row of any block).
    """
    if len(arr) == 0:
        return arr.copy(), np.zeros(0, dtype=bool)
    boundary = np.concatenate([[True], block_id[1:] != block_id[:-1]])
    raw_diff = np.diff(arr, axis=0)                     # (T-1, ...)
    keep_diff = ~boundary[1:]                           # drop diffs that bridge blocks
    deltas = raw_diff[keep_diff]
    kept_mask = np.concatenate([[False], keep_diff])    # row k pairs with diff_{k-1}
    return deltas, kept_mask


def apply_block_transform(
    cu: np.ndarray,
    du: np.ndarray,
    block_id: np.ndarray,
    use_delta: bool,
    use_arcsinh: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply delta + arcsinh in block-isolated fashion.

    cu:       (T, cu_dim)
    du:       (T, N, du_dim)
    block_id: (T,) int

    returns (cu_t, du_t, kept_mask) with kept_mask describing which original
    rows survived (so labels.csv / block_id can be subset to match).
    """
    if not use_delta:
        return cu, du, np.ones(len(cu), dtype=bool)
    cu_t, kept_mask = block_diff(cu, block_id)
    du_t, _ = block_diff(du, block_id)
    if use_arcsinh:
        cu_t = np.arcsinh(cu_t)
        du_t = np.arcsinh(du_t)
    return cu_t, du_t, kept_mask


def fit_bundle(
    train_streams: List[Dict[str, np.ndarray]],
    cu_zero_variance_idx: List[int],
    du_zero_variance_idx: List[int],
    version: str,
) -> PreprocessBundle:
    """Fit type-shared scalers across all train topologies for the given version.

    train_streams: list of {"cu": (T, cu_dim), "du": (T, N, du_dim), "block_id": (T,)}.
    For multi-topology training (Exp C), pass one stream per topology — the DU
    scaler is fit on the vertical stack of all DUs across all of them.
    """
    use_delta, use_arcsinh, use_clip = VERSION_FLAGS[version]

    cu_chunks, du_chunks = [], []
    for s in train_streams:
        cu_t, du_t, _ = apply_block_transform(
            s["cu"], s["du"], s["block_id"], use_delta, use_arcsinh
        )
        cu_chunks.append(cu_t)
        du_chunks.append(du_t.reshape(-1, du_t.shape[-1]))

    cu_all = np.concatenate(cu_chunks, axis=0)
    du_all = np.concatenate(du_chunks, axis=0)

    cu_scaler = RobustScaler().fit(cu_all)
    du_scaler = RobustScaler().fit(du_all)

    # Floor IQR for zero-variance features so they pass through unchanged.
    # RobustScaler stores per-feature IQR in scale_; setting it to 1.0 means
    # transformed = (raw - median) / 1 = raw - median ≈ raw (median is tiny).
    if cu_zero_variance_idx:
        cu_scaler.scale_[cu_zero_variance_idx] = 1.0
    if du_zero_variance_idx:
        du_scaler.scale_[du_zero_variance_idx] = 1.0

    return PreprocessBundle(
        version=version,
        cu_scaler=cu_scaler,
        du_scaler=du_scaler,
        cu_dim=cu_all.shape[1],
        du_dim=du_all.shape[1],
        use_delta=use_delta,
        use_arcsinh=use_arcsinh,
        use_clip=use_clip,
        cu_zero_variance_idx=list(cu_zero_variance_idx),
        du_zero_variance_idx=list(du_zero_variance_idx),
    )


def transform_stream(
    bundle: PreprocessBundle,
    cu: np.ndarray,
    du: np.ndarray,
    block_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the bundle's pipeline to one stream.

    returns (cu_s, du_s, kept_mask, kept_block_id)
        cu_s:          (T', cu_dim)   float32, scaled (and clipped if v3)
        du_s:          (T', N, du_dim)
        kept_mask:     (T,) bool — which rows of the input survived block-diff
        kept_block_id: (T',) int    — block_id for the surviving rows
    """
    cu_t, du_t, kept_mask = apply_block_transform(
        cu, du, block_id, bundle.use_delta, bundle.use_arcsinh
    )
    cu_s = bundle.cu_scaler.transform(cu_t)
    N = du_t.shape[1] if du_t.ndim == 3 else 1
    du_flat = du_t.reshape(-1, du_t.shape[-1])
    du_s = bundle.du_scaler.transform(du_flat).reshape(du_t.shape)

    if bundle.use_clip:
        cu_s = np.clip(cu_s, -CLIP_SCALED, CLIP_SCALED)
        du_s = np.clip(du_s, -CLIP_SCALED, CLIP_SCALED)

    cu_s = cu_s.astype(np.float32)
    du_s = du_s.astype(np.float32)
    kept_block_id = block_id[kept_mask]
    return cu_s, du_s, kept_mask, kept_block_id


def causal_rolling_normalize(
    arr: np.ndarray,
    window: int = 300,
    min_samples: int = 10,
    eps: float = 1e-3,
) -> np.ndarray:
    """Per-entity, per-feature causal rolling z-score.

    For each timestep t, computes mean and std over the strictly-past `window`
    timesteps and returns (x_t - mean) / (std + eps). Designed as a drop-in
    replacement for the global RobustScaler when cross-topology baseline shift
    is the dominant failure mode.

    Properties:
      * Causal — never reads future values, so deployable identically online.
      * Per-entity — for (T, N, dim), each of the N entities maintains its own
        independent rolling stats.
      * No fitted state — pure function of whatever stream flows through it.
        Identical behavior at train and test time; cannot encode topology bias.

    arr shape:
        (T, dim)     — single entity (e.g., CU)
        (T, N, dim)  — N entities (e.g., DUs); rolling stats kept per-entity
    Returns same shape, float32.
    """
    arr64 = arr.astype(np.float64)
    if arr64.ndim == 2:
        T, D = arr64.shape
        flat = arr64
        reshape_back = lambda x: x
    elif arr64.ndim == 3:
        T, N, D = arr64.shape
        flat = arr64.reshape(T, N * D)
        reshape_back = lambda x: x.reshape(T, N, D)
    else:
        raise ValueError(f"causal_rolling_normalize: unsupported shape {arr.shape}")

    csum  = np.cumsum(flat,        axis=0)
    csum2 = np.cumsum(flat ** 2,   axis=0)
    out   = np.zeros_like(flat)

    for t in range(min_samples, T):
        lo = max(0, t - window)
        n  = t - lo
        s  = csum[t - 1]  - (csum[lo - 1]  if lo > 0 else 0.0)
        s2 = csum2[t - 1] - (csum2[lo - 1] if lo > 0 else 0.0)
        mean = s / n
        var  = np.maximum(s2 / n - mean ** 2, 0.0)
        std  = np.sqrt(var) + eps
        out[t] = (flat[t] - mean) / std

    return reshape_back(out).astype(np.float32)


def cold_start_mask(block_id: np.ndarray, K: int) -> np.ndarray:
    """Mask out the first K timesteps of each block (LSTM hidden-state warmup).

    Returns a (T,) bool array; True = keep for scoring, False = discard.
    """
    if len(block_id) == 0:
        return np.zeros(0, dtype=bool)
    boundary = np.concatenate([[True], block_id[1:] != block_id[:-1]])
    boundary_idx = np.where(boundary)[0]
    keep = np.ones(len(block_id), dtype=bool)
    for start in boundary_idx:
        keep[start : start + K] = False
    return keep
