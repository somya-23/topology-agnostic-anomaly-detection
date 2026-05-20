"""plot_recon_comparison.py — Compare actual model reconstruction errors:
  cpu-only  (recon_errors_{topo}_f1.npz)  vs
  cpu+mem   (recon_errors_{topo}_f2.npz)

Layout — one block of rows per entity (CU, DU_0, DU_1, ...):
  Row A: per-feature squared error over time  (log y)
  Row B: lift score vs threshold              (log y)
Both columns: left = f1 (cpu-only), right = f2 (cpu+mem).
Stress periods are shaded in red.

Usage:
    cd clear_pipeline/
    python plot_recon_comparison.py                     # default test topo
    python plot_recon_comparison.py cu1_du2             # explicit topo
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
TEST_TOPO   = sys.argv[1] if len(sys.argv) > 1 else "cu0_du0du1"
COLD_START  = 64          # rows to grey-out at the start (LSTM warmup)

FEAT_NAMES  = {1: ["cpu"], 2: ["cpu", "mem_pct"]}
FEAT_COLOR  = {"cpu": "steelblue", "mem_pct": "darkorange"}
COL_TITLES  = {1: "cpu-only  (f1)", 2: "cpu + mem_pct  (f2)"}

# ── helpers ───────────────────────────────────────────────────────────────────

def load(topo, fdim):
    p = Path(f"recon_errors_{topo}_f{fdim}.npz")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run run_experiment.py with SAVE_ERRORS=True "
            f"and CU/DU_FEAT_SLICE=slice(0,{fdim}) first."
        )
    return dict(np.load(p))

def shade_stress(ax, lbl, cold_start=COLD_START):
    """Red bands for stress, grey band for LSTM cold-start."""
    if cold_start > 0:
        ax.axvspan(0, cold_start, color="grey", alpha=0.15, label="cold-start")
    in_stress, start = False, 0
    for i, v in enumerate(lbl):
        if v == 1 and not in_stress:
            start = i; in_stress = True
        elif v != 1 and in_stress:
            ax.axvspan(start, i, color="red", alpha=0.20)
            in_stress = False
    if in_stress:
        ax.axvspan(start, len(lbl), color="red", alpha=0.20)

def lift(sqerr, feat_norm):
    """Max-lift score over features: (T, dim) → (T,)"""
    return (sqerr / (feat_norm + 1e-9)).max(axis=-1)

# ── load both configs ─────────────────────────────────────────────────────────
d1 = load(TEST_TOPO, 1)
d2 = load(TEST_TOPO, 2)

cu_stress  = d1["cu_stress"].astype(int)   # (T,)
du_stress  = d1["du_stress"].astype(int)   # (T, N)
N_DU       = du_stress.shape[1]
n_entities = 1 + N_DU

configs = [
    (1, d1["cu_sqerr"], d1["du_sqerr"], d1["cu_feat_norm"], d1["du_feat_norm"],
        float(d1["cu_thr_adj"].flat[0]), float(d1["du_thr"].flat[0])),
    (2, d2["cu_sqerr"], d2["du_sqerr"], d2["cu_feat_norm"], d2["du_feat_norm"],
        float(d2["cu_thr_adj"].flat[0]), float(d2["du_thr"].flat[0])),
]

# ── figure layout: (2 rows per entity) × (2 feature configs) ─────────────────
n_rows = n_entities * 2   # sqerr row + lift row per entity
n_cols = 2                # f1 | f2
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(7 * n_cols, 3.5 * n_rows),
    squeeze=False,
)
fig.suptitle(
    f"Reconstruction error comparison  |  test={TEST_TOPO}\n"
    f"Left: cpu-only   Right: cpu + mem_pct   "
    f"(red shading = stress, grey = LSTM warmup)",
    fontsize=11, y=1.01,
)

# Column headers
for ci, (fdim, *_) in enumerate(configs):
    axes[0, ci].set_title(COL_TITLES[fdim], fontsize=11, fontweight="bold", pad=8)

for ei in range(n_entities):
    is_cu   = (ei == 0)
    name    = "CU" if is_cu else f"DU_{ei - 1}"
    du_i    = ei - 1
    lbl     = cu_stress if is_cu else du_stress[:, du_i]

    row_sqerr = ei * 2
    row_lift  = ei * 2 + 1

    # Left y-label for this entity block
    axes[row_sqerr, 0].set_ylabel(f"{name}\nsqerr (log)", fontsize=9)
    axes[row_lift,  0].set_ylabel(f"{name}\nlift (log)",  fontsize=9)

    for ci, (fdim, cu_sq, du_sq, cu_fn, du_fn, cu_thr, du_thr) in enumerate(configs):
        fn     = FEAT_NAMES[fdim]

        # pick the right sqerr array and feat_norm for this entity
        if is_cu:
            sq    = cu_sq                          # (T-1, fdim)
            fnorm = cu_fn                          # (fdim,)
            thr   = cu_thr
        else:
            sq    = du_sq[:, du_i, :]             # (T-1, fdim)
            fnorm = du_fn                          # (fdim,)
            thr   = du_thr

        T_score = sq.shape[0]
        ts      = np.arange(T_score)
        # stress label aligned to score length (sqerr has T-1 rows)
        lbl_sc  = lbl[1:T_score + 1]

        # ── Row A: per-feature sqerr ───────────────────────────────────────
        ax = axes[row_sqerr, ci]
        for fi, fname in enumerate(fn):
            ax.semilogy(ts, np.clip(sq[:, fi], 1e-8, None),
                        color=FEAT_COLOR[fname], lw=0.6, alpha=0.85,
                        label=fname)
        shade_stress(ax, lbl_sc, cold_start=COLD_START)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, T_score)

        # ── Row B: lift score vs threshold ────────────────────────────────
        ax = axes[row_lift, ci]
        ls = lift(sq, fnorm)
        ax.semilogy(ts, np.clip(ls, 1e-8, None),
                    color="navy", lw=0.6, alpha=0.85, label="lift score")
        ax.axhline(thr, color="red", ls="--", lw=1.2,
                   label=f"threshold={thr:.3f}")
        shade_stress(ax, lbl_sc, cold_start=COLD_START)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, T_score)

    # x-label only on the bottom row of each entity block
    for ci in range(n_cols):
        axes[row_lift, ci].set_xlabel("Timestep", fontsize=8)

plt.tight_layout()
out = Path(f"recon_comparison_{TEST_TOPO}.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved → {out.resolve()}")
