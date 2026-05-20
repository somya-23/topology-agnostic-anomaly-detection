"""diagnose_cal_scores.py — Compare anomaly scores on CAL (train) vs TEST stream.

Plots two columns side-by-side for each entity (CU, DU_0):
  LEFT  — held-out CAL stream (last 20% of train normal data, no stress)
  RIGHT — TEST stream (from diagnose_scores.py, with red stress shading)

Outlier glitches in the normal CAL stream are marked with orange triangles.
Both columns share the same threshold lines so you can compare directly.

Run from:  project_root/step3_topoar/clear_pipeline/
    python diagnose_cal_scores.py
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
from model_calibrated import CalibratedTopoAR, feat_norm_calibrated
from scoring import lift_score

# ── same settings as run_experiment.py ────────────────────────────────────────
TRAIN_TOPO       = "cu1_du2"
TEST_TOPO        = "cu0_du0du1"
BASE_DIR         = Path("output")
CU_FEAT_SLICE    = slice(0, 2)
DU_FEAT_SLICE    = slice(0, 1)
EMBED_DIM        = 32
CAL_FRAC         = 0.2
COLD_START_K     = 64
CU_THRESHOLD_PCT = 99.9
DU_THRESHOLD_PCT = 99.0
MODEL_CKPT       = Path("model_ckpt.pt")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
IMPUTE           = False   # must match run_experiment.py


def load_npz(topo, split):
    return dict(np.load(BASE_DIR / f"{topo}_stress1/{split}.npz"))


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
    return cu, du, z["block_id"].astype(np.int64)


def infer_open(model, cu_s, du_s):
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()
    du_sqerr = (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()
    return cu_sqerr, du_sqerr


def shade_stress(ax, t, mask, alpha=0.25, label="GT stress"):
    mask = np.asarray(mask, dtype=bool)
    in_b, t0, first = False, None, True
    for i in range(len(mask)):
        if mask[i] and not in_b:
            t0, in_b = t[i], True
        elif not mask[i] and in_b:
            ax.axvspan(t0, t[i], color="red", alpha=alpha,
                       label=(label if first else None))
            in_b, first = False, False
    if in_b:
        ax.axvspan(t0, t[-1] + 1, color="red", alpha=alpha,
                   label=(label if first else None))


def style_ax(ax, scores, t, thr_lo, thr_hi, label, side):
    pos = scores[scores > 0]
    if len(pos):
        ax.set_ylim(bottom=pos.min() * 0.3)
    ax.set_yscale("log")
    ax.set_ylabel(f"{label}\nlift score (log)", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlabel("Timestep", fontsize=8)
    lo = min(t); hi = max(t)
    ax.set_xlim(lo, hi)
    if side == "cal":
        ax.set_title(f"CAL stream (normal train tail)\n"
                     f"orange▲ = glitch rows  dashed = thresholds", fontsize=8)
    else:
        ax.set_title(f"TEST stream ({TEST_TOPO})\n"
                     f"red bg = GT stress  dashed = thresholds", fontsize=8)


def main():
    # ── preprocess ────────────────────────────────────────────────────────────
    z_tr = load_npz(TRAIN_TOPO, "train")
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

    # ── CAL stream inference ──────────────────────────────────────────────────
    cu_sqerr_cal, du_sqerr_cal = infer_open(model, cu_s_cal, du_s_cal)
    cu_sqerr_n = cu_sqerr_cal[COLD_START_K:]
    du_sqerr_n = du_sqerr_cal[COLD_START_K:]

    cu_fn = feat_norm_calibrated(cu_sqerr_n)
    du_fn = feat_norm_calibrated(du_sqerr_n.reshape(-1, du_sqerr_n.shape[-1]))

    cu_scores_cal = lift_score(cu_sqerr_n, cu_fn)              # (M,)
    du_scores_cal = lift_score(du_sqerr_n[:, 0, :], du_fn)    # (M,) — 1 DU in train

    cu_thr = float(np.percentile(cu_scores_cal, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_scores_cal, DU_THRESHOLD_PCT))
    cu_thr_hi = float(np.percentile(cu_scores_cal, 99.9))
    du_thr_hi = float(np.percentile(du_scores_cal, 99.9))

    print(f"CU  p{CU_THRESHOLD_PCT}={cu_thr:.4f}")
    print(f"DU  p{DU_THRESHOLD_PCT}={du_thr:.4f}   p99.9={du_thr_hi:.4f}")

    # ── TEST stream inference ─────────────────────────────────────────────────
    z_te = load_npz(TEST_TOPO, "test")
    cu_te, du_te, bid_te = slice_feat(z_te)
    cu_s_te, du_s_te, kept, _ = transform_stream(bundle, cu_te, du_te, bid_te)

    cu_stress = z_te["cu_stress"][kept].astype(np.int32)
    du_stress = z_te["du_stress"][kept].astype(np.int32)
    n_du_te   = du_s_te.shape[1]

    cu_sqerr_te, du_sqerr_te = infer_open(model, cu_s_te, du_s_te)

    start_te = COLD_START_K
    te_t     = np.arange(start_te + 1, len(cu_s_te))

    cu_scores_te = lift_score(cu_sqerr_te[start_te:], cu_fn)
    du_scores_te = np.stack(
        [lift_score(du_sqerr_te[start_te:, i, :], du_fn) for i in range(n_du_te)],
        axis=1,
    )  # (T', n_du_te)

    # ── time axes ─────────────────────────────────────────────────────────────
    cal_t = np.arange(COLD_START_K + 1, len(cu_s_cal))  # scores cover t+1

    # ── plot: 2 rows (CU, DU_0) × 2 cols (cal, test) ─────────────────────────
    n_rows = 1 + n_du_te   # CU + each DU in test (DU_0 and DU_1)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(18, 4 * n_rows),
                             sharex=False)

    fig.suptitle(
        "Anomaly score comparison: CAL (normal) vs TEST (stressed)\n"
        "Both use the same thresholds from the CAL stream.",
        fontsize=11,
    )

    # ── row 0: CU ─────────────────────────────────────────────────────────────
    # CAL side
    ax = axes[0, 0]
    ax.plot(cal_t, cu_scores_cal, color="navy", lw=0.6, label="score (cal)")
    ax.axhline(cu_thr, color="crimson", ls="--", lw=1.2,
               label=f"p{CU_THRESHOLD_PCT}={cu_thr:.1f}")
    glitch_cu = np.where(cu_scores_cal > cu_thr)[0]
    if len(glitch_cu):
        ax.scatter(cal_t[glitch_cu], cu_scores_cal[glitch_cu],
                   color="orange", marker="^", s=30, zorder=5, label="glitch")
    style_ax(ax, cu_scores_cal, cal_t, cu_thr, cu_thr_hi, "CU", "cal")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    # TEST side
    ax = axes[0, 1]
    shade_stress(ax, te_t, cu_stress[start_te + 1:] == 1)
    ax.plot(te_t, cu_scores_te, color="navy", lw=0.6, label="score (test)")
    ax.axhline(cu_thr, color="crimson", ls="--", lw=1.2,
               label=f"p{CU_THRESHOLD_PCT}={cu_thr:.1f}")
    det = cu_scores_te > cu_thr
    if det.any():
        ax.scatter(te_t[det], cu_scores_te[det],
                   color="red", s=8, zorder=5, label="detected")
    style_ax(ax, cu_scores_te, te_t, cu_thr, cu_thr_hi, "CU", "test")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    # ── rows 1..: DUs ─────────────────────────────────────────────────────────
    for i in range(n_du_te):
        row = i + 1
        name = f"DU_{i}"

        # CAL side (only 1 DU in train; use it for all test DUs as reference)
        ax = axes[row, 0]
        ax.plot(cal_t, du_scores_cal, color="navy", lw=0.6, label="DU_0 score (cal)")
        ax.axhline(du_thr, color="crimson", ls="--", lw=1.2,
                   label=f"p{DU_THRESHOLD_PCT}={du_thr:.2f}")
        ax.axhline(du_thr_hi, color="darkorange", ls=":", lw=1.2,
                   label=f"p99.9={du_thr_hi:.1f}  (old thr)")
        glitch_du = np.where(du_scores_cal > du_thr)[0]
        if len(glitch_du):
            ax.scatter(cal_t[glitch_du], du_scores_cal[glitch_du],
                       color="orange", marker="^", s=30, zorder=5,
                       label=f"glitch ({len(glitch_du)} rows)")
        style_ax(ax, du_scores_cal, cal_t, du_thr, du_thr_hi, f"{name} (ref: cal DU_0)", "cal")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        # Print glitch spacing
        if len(glitch_du) > 1:
            diffs = np.diff(glitch_du)
            print(f"\n{name} cal glitch intervals: {diffs[:10]}")
            print(f"  → typical spacing: ~{int(np.median(diffs[diffs < 400]))} steps "
                  f"(≈{int(np.median(diffs[diffs < 400]))} seconds @ 1s scrape = "
                  f"{int(np.median(diffs[diffs < 400]))//60} min window)")

        # TEST side
        ax = axes[row, 1]
        shade_stress(ax, te_t, du_stress[start_te + 1:, i] == 1)
        ax.plot(te_t, du_scores_te[:, i], color="navy", lw=0.6, label=f"{name} score (test)")
        ax.axhline(du_thr, color="crimson", ls="--", lw=1.2,
                   label=f"p{DU_THRESHOLD_PCT}={du_thr:.2f}")
        ax.axhline(du_thr_hi, color="darkorange", ls=":", lw=1.2,
                   label=f"p99.9={du_thr_hi:.1f}  (old thr)")
        det_i = du_scores_te[:, i] > du_thr
        if det_i.any():
            ax.scatter(te_t[det_i], du_scores_te[det_i, i],
                       color="red", s=8, zorder=5, label="detected")
        style_ax(ax, du_scores_te[:, i], te_t, du_thr, du_thr_hi, name, "test")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

        # Per-entity stats
        in_s  = du_stress[start_te + 1:, i] == 1
        out_s = ~in_s
        if in_s.any():
            s = du_scores_te[:, i]
            print(f"\n{name} TEST during stress:   "
                  f"mean={s[in_s].mean():.2f}  max={s[in_s].max():.2f}  "
                  f"above_thr={in_s[du_scores_te[:, i] > du_thr].sum()}/{in_s.sum()}")
        if out_s.any():
            s = du_scores_te[:, i]
            print(f"{name} TEST outside stress:  "
                  f"mean={s[out_s].mean():.2f}  "
                  f"p99={np.percentile(s[out_s], 99):.2f}  "
                  f"above_thr={out_s[du_scores_te[:, i] > du_thr].sum()}/{out_s.sum()}")

    plt.tight_layout()
    out = Path("diagnose_cal_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
