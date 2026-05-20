"""Calibrated TopoAR — same architecture, fixed scoring + threshold.

Why this file exists. Phase 3 baseline (model.py + scoring.py + train.py)
hit Exp A F1 ≈ 0.04 even though the model learns normal correctly —
ROC-AUC = 0.68 with a 2,600× anomaly-vs-normal ranking gap. The damage is
upstream of the model:

  Bug 1 — FEAT_NORM_FLOOR = 1e-6 destroys precision on zero-variance
          features. Those features (pusch_ta_ns, ul_mcs, nof_pusch_invalid_*,
          ...) are constant on normal data, so val sqerr is ~0 → feat_norm
          stays at the floor. Tiny test-set noise of 1e-4 on those features
          divides by 1e-6 → lift = 100 → flags MOST normal rows.

  Bug 2 — Pooled (CU + DU) into one 99-pct threshold. DU has du_dim=37
          versus CU cu_dim=7, so DU's max-pool tail dominates. CU normal
          scores have median ~97 against threshold 14.8 → CU fires ~always.

This file fixes both without touching the architecture:

  feat_norm_calibrated()
      Floor each feature at FLOOR_FRAC × median of `raw_feat_norm` across
      features (NOT a fixed 1e-6). For zero-variance features this raises
      feat_norm ~200×, suppressing noise lift to <1, while genuine
      anomalies (which sit 1e6× above normal in raw sqerr) still produce
      lift in the thousands and cross threshold.

  dual_threshold_from_val()
      Compute CU and DU thresholds separately, each as the 99-pct of its
      own type's val scores. Topology agnosticism is preserved — the
      thresholds are per *entity type*, not per topology instance, so the
      same (cu_thr, du_thr) pair applies to any (CU, [DU]) configuration.

Architecturally this is the same TopoAR; CalibratedTopoAR is a thin
subclass for symmetry with MaskedTopoAR (so train_calibrated.py /
evaluate_calibrated.py have a parallel "model variant" entry point).
"""

from typing import Tuple

import numpy as np

from model import TopoAR
from scoring import FEAT_NORM_FLOOR


# Floor for feat_norm: per-feature value is never less than FLOOR_FRAC × median
# of the raw feat_norm vector (the typical feature's normal residual).
FLOOR_FRAC = 0.1


class CalibratedTopoAR(TopoAR):
    """Identical to TopoAR architecturally. Exists so that --variant calibrated
    has a class to instantiate and so checkpoints from calibrated runs are
    self-describing (manifest.json carries variant="calibrated")."""
    pass


def feat_norm_calibrated(
    sqerr_val: np.ndarray, floor_frac: float = FLOOR_FRAC
) -> np.ndarray:
    """Per-feature normal-residual mean, floored at floor_frac × global median.

    sqerr_val:   (M, dim)  per-row, per-feature val sqerr (normal-only)
    floor_frac:           how aggressive the floor is. 0.1 = floor any feature
                          at 10% of the typical feature's residual.
    returns:     (dim,)    never below max(floor_frac × median(raw), FEAT_NORM_FLOOR)

    The intuition: a feature whose val residual is meaningfully smaller than
    a typical feature's is one we shouldn't be over-confident about. We don't
    let any feature contribute >10× the typical feature's amplification to
    the max-pool. Genuine anomalies (huge sqerr) are unaffected — only noise
    on zero-variance features gets attenuated.
    """
    raw = sqerr_val.mean(axis=0)
    med = float(np.median(raw))
    floor = max(med * floor_frac, FEAT_NORM_FLOOR)
    return np.maximum(raw, floor).astype(np.float64)


def dual_threshold_from_val(
    cu_val_scores: np.ndarray,
    du_val_scores: np.ndarray,
    percentile: float = 99.0,
) -> Tuple[float, float]:
    """Per-entity-type val-fold thresholds.

    Returns (cu_threshold, du_threshold). Each is computed only from its own
    type's val scores so neither dominates the other. Both are frozen at
    train time and applied uniformly at any topology.
    """
    cu_thr = float(np.percentile(cu_val_scores, percentile))
    du_thr = float(np.percentile(du_val_scores, percentile))
    return cu_thr, du_thr
