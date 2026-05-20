"""diagnose_scores.py — Plot per-timestep anomaly scores vs stress labels.

Shows lift_score over time for CU and each DU with:
  - Red shading  = ground-truth cpu stress
  - Dashed line  = threshold
  - Score curve  = what the model actually produces

This exposes WHY detection fails (score never rises? rises then drops? threshold too high?)

Run from:  project_root/step3_topoar/clear_pipeline/
    python diagnose_scores.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import fit_bundle, transform_stream
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated, dual_threshold_from_val
from scoring import lift_score

# ── same settings as run_experiment.py ────────────────────────────────────────
TRAIN_TOPO    = "cu1_du2"
TEST_TOPO     = "cu0_du0du1"
BASE_DIR      = Path("output")
CU_FEAT_SLICE = slice(0, 2)
DU_FEAT_SLICE = slice(0, 1)
EMBED_DIM     = 32
CAL_FRAC      = 0.2
COLD_START_K  = 64
THRESHOLD_PCT = 99.9
MODEL_CKPT    = Path("model_ckpt.pt")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMPUTE        = False   # must match run_experiment.py


def load(topo, split):
    p = BASE_DIR / f"{topo}_stress1" / f"{split}.npz"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Forward-fill rows where any feature is 0.0 (Prometheus scrape artifact)."""
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            if (arr[t] < eps).any():
                arr[t] = arr[t - 1]
        else:
            glitch = (arr[t] < eps).any(axis=-1)
            if glitch.any():
                arr[t, glitch] = arr[t - 1, glitch]
    return arr


def slice_feat(z):
    cu  = z["cu"][:, CU_FEAT_SLICE].astype(np.float32)
    du  = z["du"][:, :, DU_FEAT_SLICE].astype(np.float32)
    if IMPUTE:
        cu = impute_cpu_glitch(cu)
        du = impute_cpu_glitch(du)
    bid = z["block_id"].astype(np.int64)
    return cu, du, bid


def shade(ax, t, mask, color="red", alpha=0.20, label="stress"):
    mask = np.asarray(mask, dtype=bool)
    in_b, t0, first = False, None, True
    for i in range(len(mask)):
        if mask[i] and not in_b:
            t0, in_b = t[i], True
        elif not mask[i] and in_b:
            ax.axvspan(t0, t[i], color=color, alpha=alpha,
                       label=(label if first else None))
            in_b, first = False, False
    if in_b:
        ax.axvspan(t0, t[-1]+1, color=color, alpha=alpha,
                   label=(label if first else None))


def main():
    # ── preprocessing ─────────────────────────────────────────────────────────
    z_tr = load(TRAIN_TOPO, "train")
    cu_tr, du_tr, bid_tr = slice_feat(z_tr)

    bundle = fit_bundle(
        train_streams=[{"cu": cu_tr, "du": du_tr, "block_id": bid_tr}],
        cu_zero_variance_idx=[], du_zero_variance_idx=[], version="v0",
    )
    cu_s_tr, du_s_tr, _, _ = transform_stream(bundle, cu_tr, du_tr, bid_tr)

    n_total = len(cu_s_tr)
    n_cal   = int(round(CAL_FRAC * n_total))
    n_fit   = n_total - n_cal
    cu_s_cal = cu_s_tr[n_fit:]
    du_s_cal = du_s_tr[n_fit:]

    # ── load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(MODEL_CKPT, map_location=DEVICE)
    model = CalibratedTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"],
                              embed_dim=ckpt["embed_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def infer(cu_s, du_s):
        cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
        du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cu_hat, du_hat = model(cu_t, du_t)
        cu_sqerr = (cu_hat[0,:-1] - cu_t[0,1:]).pow(2).cpu().numpy()
        du_sqerr = (du_hat[0,:-1] - du_t[0,1:]).pow(2).cpu().numpy()
        return cu_sqerr, du_sqerr

    # ── calibrate on train-cal stream ─────────────────────────────────────────
    cu_sqerr_cal, du_sqerr_cal = infer(cu_s_cal, du_s_cal)
    cu_sqerr_n = cu_sqerr_cal[COLD_START_K:]
    du_sqerr_n = du_sqerr_cal[COLD_START_K:]

    cu_fn = feat_norm_calibrated(cu_sqerr_n)
    du_fn = feat_norm_calibrated(du_sqerr_n.reshape(-1, du_sqerr_n.shape[-1]))

    cu_norm_scores = lift_score(cu_sqerr_n, cu_fn)
    du_norm_scores = lift_score(du_sqerr_n.reshape(-1, du_sqerr_n.shape[-1]), du_fn)
    cu_thr, du_thr = dual_threshold_from_val(cu_norm_scores, du_norm_scores, THRESHOLD_PCT)

    print(f"CU thr={cu_thr:.4f}   DU thr={du_thr:.4f}")

    # ── test inference ────────────────────────────────────────────────────────
    z_te = load(TEST_TOPO, "test")
    cu_te, du_te, bid_te = slice_feat(z_te)
    cu_s_te, du_s_te, kept, _ = transform_stream(bundle, cu_te, du_te, bid_te)

    cu_stress = z_te["cu_stress"][kept].astype(np.int32)
    du_stress = z_te["du_stress"][kept].astype(np.int32)
    n_du      = du_s_te.shape[1]

    cu_sqerr, du_sqerr = infer(cu_s_te, du_s_te)

    # scores (length T-1); eval covers COLD_START_K+1 .. T-1
    start   = COLD_START_K
    score_t = np.arange(start + 1, len(cu_s_te))   # timestep index for scores

    cu_scores = lift_score(cu_sqerr[start:], cu_fn)
    du_scores = np.stack(
        [lift_score(du_sqerr[start:, i, :], du_fn) for i in range(n_du)], axis=1
    )  # (T', n_du)

    # ── plot ──────────────────────────────────────────────────────────────────
    entities = [("CU", cu_scores, cu_thr, cu_stress)]
    for i in range(n_du):
        entities.append((f"DU_{i}", du_scores[:, i], du_thr, du_stress[:, i]))

    fig, axes = plt.subplots(len(entities), 1,
                             figsize=(16, 3.5 * len(entities)), sharex=True)
    if len(entities) == 1:
        axes = [axes]

    fig.suptitle(
        f"Model anomaly scores over time — {TEST_TOPO} test stream\n"
        f"red bg = ground-truth CPU stress   |   dashed = threshold",
        fontsize=11,
    )

    for ax, (name, scores, thr, stress_lbl) in zip(axes, entities):
        shade(ax, score_t, stress_lbl[start+1:] == 1, color="red", alpha=0.25,
              label="GT stress")

        ax.plot(score_t, scores, color="navy", lw=0.7, label="anomaly score")
        ax.axhline(thr, color="crimson", ls="--", lw=1.2, label=f"threshold={thr:.1f}")

        # mark detected points
        detected = scores > thr
        if detected.any():
            ax.scatter(score_t[detected], scores[detected],
                       color="red", s=8, zorder=5, label="detected")

        ax.set_yscale("log")
        pos_scores = scores[scores > 0]
        if len(pos_scores):
            ax.set_ylim(bottom=pos_scores.min() * 0.5)
        ax.set_ylabel(f"{name}\nlift score (log)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        ax.grid(True, lw=0.3, alpha=0.4)

        # print per-entity stats
        in_stress  = stress_lbl[start+1:] == 1
        out_stress = ~in_stress
        if in_stress.any():
            print(f"{name}  score DURING stress:  "
                  f"mean={scores[in_stress].mean():.2f}  "
                  f"max={scores[in_stress].max():.2f}  "
                  f"min={scores[in_stress].min():.2f}")
        if out_stress.any():
            print(f"{name}  score OUTSIDE stress: "
                  f"mean={scores[out_stress].mean():.2f}  "
                  f"max={scores[out_stress].max():.2f}  "
                  f"p99={np.percentile(scores[out_stress], 99):.2f}")
        print()

    axes[-1].set_xlabel("Timestep", fontsize=9)
    plt.tight_layout()
    out = Path("diagnose_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
