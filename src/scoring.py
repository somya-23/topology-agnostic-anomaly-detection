"""Per-entity scoring, threshold, and localization helpers.

The per-entity score is `max-pool lift over per-feature squared error`:

    sqerr_i(t) = (x̂_i(t) - x_i(t)) ** 2          shape (entity_dim,)
    lift_i(t)  = sqerr_i(t) / feat_norm_i        # train per-feature mean residual
    score_i(t) = max(lift_i(t))                  # max-pool over features

Scores are produced *per entity* (one for the CU, one for each DU). One
threshold (99-pct of val-fold scores) is frozen at training time and applied
uniformly to every entity at any topology — that's the no-recalibration claim.

`feat_norm` for CU and DU comes from the val fold of train (normal-only). For
multi-topology training we pool val scores across topologies before taking
the percentile, so the same single threshold sees both N regimes.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


FEAT_NORM_FLOOR = 1e-6


def per_entity_sqerr(
    pred: np.ndarray, true: np.ndarray
) -> np.ndarray:
    """Element-wise squared error.

    pred / true: (T, ..., dim)
    returns:    (T, ..., dim)
    """
    return (pred - true) ** 2


def feat_norm_from_val(sqerr_val: np.ndarray) -> np.ndarray:
    """Per-feature mean squared error on a normal-only val fold, with a small floor.

    sqerr_val: (M, dim)            M = number of (window, step) val datapoints
    returns:   (dim,)               feat_norm — divide test sqerr by this for "lift"
    """
    return sqerr_val.mean(axis=0) + FEAT_NORM_FLOOR


def lift_score(sqerr: np.ndarray, feat_norm: np.ndarray) -> np.ndarray:
    """Max-pool over features of (sqerr / feat_norm).

    sqerr:     (..., dim)
    feat_norm: (dim,)
    returns:   (...,) one score per entity-row
    """
    return (sqerr / feat_norm).max(axis=-1)


def threshold_from_val_scores(scores: np.ndarray, percentile: float = 99.0) -> float:
    """Frozen threshold: `percentile`-th percentile of val (normal) entity scores.

    scores: (M,) flat array of val entity scores (CU + every DU pooled).
    """
    return float(np.percentile(scores, percentile))


def top_features_per_row(
    sqerr: np.ndarray,
    feat_norm: np.ndarray,
    feature_names: List[str],
    k: int = 5,
) -> List[List[str]]:
    """For each row, list the k features with the highest lift (descending).

    sqerr:         (M, dim)
    feat_norm:     (dim,)
    feature_names: len = dim
    returns:       list of M lists of names (length k each)
    """
    lift = sqerr / feat_norm
    idx = np.argsort(-lift, axis=-1)[:, :k]
    return [[feature_names[j] for j in row] for row in idx]


def localization_metrics(
    flag_per_entity: np.ndarray,
    label_per_entity: np.ndarray,
) -> Dict[str, float]:
    """Localization-quality metrics computed from per-(timestep, entity) flags.

    flag_per_entity:  (T, E) bool — model's per-entity firing
    label_per_entity: (T, E) bool — ground-truth per-entity anomaly

    Returns:
        localization_accuracy: of timesteps where ANY entity has a true
            anomaly AND any entity flagged, the fraction where the flagged
            entity set ⊇ the true-anomaly entity set.
        false_co_firing_rate: of (timestep, entity) pairs where the entity
            is the only true anomaly, the rate that some OTHER entity also
            flagged at that timestep — i.e. cross-entity contamination.
    """
    flag = flag_per_entity.astype(bool)
    lab = label_per_entity.astype(bool)

    any_true = lab.any(axis=1)
    any_flag = flag.any(axis=1)
    correct = ((flag | ~lab).all(axis=1)) & (flag & lab).any(axis=1)
    denom = int((any_true & any_flag).sum())
    loc_acc = float(correct.sum() / denom) if denom else float("nan")

    co_fire_num, co_fire_den = 0, 0
    for t in range(lab.shape[0]):
        true_ents = np.where(lab[t])[0]
        for e in true_ents:
            co_fire_den += 1
            if flag[t][np.arange(flag.shape[1]) != e].any():
                co_fire_num += 1
    co_fire = float(co_fire_num / co_fire_den) if co_fire_den else float("nan")

    return {
        "localization_accuracy": loc_acc,
        "false_co_firing_rate": co_fire,
    }


def propagation_chains(
    flag_per_entity: np.ndarray,
    block_id: np.ndarray,
    timestamps: np.ndarray,
    entity_names: List[str],
) -> List[Dict]:
    """Per-block onset order: which entity flagged FIRST, second, etc.

    flag_per_entity: (T, E) bool
    block_id:        (T,) int
    timestamps:      (T,) any dtype (used only as a label in onset_at_timestamp)
    entity_names:    list of E entity ids in column order

    Returns one dict per block where at least one entity flagged. The "delta"
    field is in row-index units (each row = one scrape interval) — the dataset
    has uniform sampling, so this is equivalent to seconds.
    """
    out = []
    for bid in np.unique(block_id):
        mask = block_id == bid
        flags = flag_per_entity[mask]
        ts = timestamps[mask]
        if not flags.any():
            continue
        first_idx_per_entity: Dict[str, int] = {}
        for e in range(flags.shape[1]):
            hits = np.where(flags[:, e])[0]
            if len(hits):
                first_idx_per_entity[entity_names[e]] = int(hits[0])
        ordered = sorted(first_idx_per_entity.items(), key=lambda kv: kv[1])
        first_global = ordered[0][1]
        out.append({
            "block_id": int(bid),
            "onset_order": [name for name, _ in ordered],
            "delta_rows_from_first": [int(i - first_global) for _, i in ordered],
            "first_onset_at": str(ts[first_global]) if len(ts) else None,
        })
    return out
