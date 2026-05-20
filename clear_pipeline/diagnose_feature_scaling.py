"""diagnose_feature_scaling.py — Plot raw + scaled time series for cpu and mem_pct.

Shows 4 columns side-by-side:
  COL 1: TRAIN raw values (cpu + mem_pct)
  COL 2: TEST  raw values (cpu + mem_pct)
  COL 3: TRAIN scaled WITHOUT imputation
  COL 4: TRAIN scaled WITH    imputation

  Plus a summary table of RobustScaler median/IQR and feat_norm per feature.

Glitch rows (raw == 0) are marked with red vertical lines.

Run from:  project_root/step3_topoar/clear_pipeline/
    python diagnose_feature_scaling.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from preprocess import fit_bundle, transform_stream
from model_calibrated import feat_norm_calibrated

BASE_DIR      = Path("output")
TRAIN_TOPO    = "cu1_du2"
TEST_TOPO     = "cu0_du0du1"
CU_FEAT_SLICE = slice(0, 2)
DU_FEAT_SLICE = slice(0, 2)
CAL_FRAC      = 0.2
COLD_START_K  = 64
FEAT_NAMES    = ["cpu", "mem_pct"]


def impute(arr, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            if (arr[t] < eps).any():
                arr[t] = arr[t - 1]
        else:
            g = (arr[t] < eps).any(axis=-1)
            if g.any():
                arr[t, g] = arr[t - 1, g]
    return arr


def load_raw(topo, split):
    z = dict(np.load(BASE_DIR / f"{topo}_stress1/{split}.npz"))
    cu  = z["cu"][:, CU_FEAT_SLICE].astype(np.float32)
    du  = z["du"][:, 0, DU_FEAT_SLICE].astype(np.float32)   # first DU only
    bid = z["block_id"].astype(np.int64)
    stress = z.get("cu_stress", np.zeros(len(cu), dtype=np.int32))
    return cu, du, bid, stress


def glitch_indices(arr):
    """Row indices where ANY feature is 0."""
    return np.where((arr < 1e-6).any(axis=1))[0]


def mark_glitches(ax, glitch_idx, t_max, color="red", alpha=0.6):
    for g in glitch_idx:
        if g < t_max:
            ax.axvline(g, color=color, lw=0.8, alpha=alpha)


def mark_stress(ax, stress, color="orange", alpha=0.2):
    stress = np.asarray(stress, dtype=bool)
    in_b, t0 = False, None
    for i in range(len(stress)):
        if stress[i] and not in_b:
            t0, in_b = i, True
        elif not stress[i] and in_b:
            ax.axvspan(t0, i, color=color, alpha=alpha)
            in_b = False
    if in_b:
        ax.axvspan(t0, len(stress), color=color, alpha=alpha)


def plot_series(ax, t, vals, color, lw=0.6, label=None):
    ax.plot(t, vals, color=color, lw=lw, label=label)


def main():
    # ── load raw data ─────────────────────────────────────────────────────────
    cu_tr_raw, du_tr_raw, bid_tr, _          = load_raw(TRAIN_TOPO, "train")
    cu_te_raw, du_te_raw, bid_te, stress_te  = load_raw(TEST_TOPO,  "test")

    t_tr = np.arange(len(cu_tr_raw))
    t_te = np.arange(len(cu_te_raw))

    glitch_tr_cu = glitch_indices(cu_tr_raw)
    glitch_tr_du = glitch_indices(du_tr_raw)
    glitch_te_cu = glitch_indices(cu_te_raw)
    glitch_te_du = glitch_indices(du_te_raw)

    print(f"Train glitch rows — CU: {len(glitch_tr_cu)}  DU: {len(glitch_tr_du)}")
    print(f"Test  glitch rows — CU: {len(glitch_te_cu)}  DU: {len(glitch_te_du)}")

    # ── fit scalers: with and without imputation ──────────────────────────────
    bundles = {}
    scaled  = {}

    for tag, do_imp in [("no_imp", False), ("imp", True)]:
        cu_tr = impute(cu_tr_raw) if do_imp else cu_tr_raw.copy()
        du_tr_3d = (impute(du_tr_raw[:, np.newaxis, :]) if do_imp
                    else du_tr_raw[:, np.newaxis, :].copy())

        b = fit_bundle(
            train_streams=[{"cu": cu_tr, "du": du_tr_3d, "block_id": bid_tr}],
            cu_zero_variance_idx=[], du_zero_variance_idx=[], version="v0",
        )
        bundles[tag] = b

        cu_s_tr, du_s_tr, _, _ = transform_stream(b, cu_tr, du_tr_3d, bid_tr)

        cu_te = impute(cu_te_raw) if do_imp else cu_te_raw.copy()
        du_te_3d = (impute(du_te_raw[:, np.newaxis, :]) if do_imp
                    else du_te_raw[:, np.newaxis, :].copy())
        cu_s_te, du_s_te, kept, _ = transform_stream(b, cu_te, du_te_3d, bid_te)

        # CAL tail feat_norm proxy (variance of cal stream, no model needed)
        n_cal = int(round(CAL_FRAC * len(cu_s_tr)))
        cu_cal = cu_s_tr[len(cu_s_tr) - n_cal + COLD_START_K:]
        du_cal = du_s_tr[len(du_s_tr) - n_cal + COLD_START_K:, 0, :]
        cu_fn_proxy = feat_norm_calibrated((cu_cal - cu_cal.mean(0))**2)
        du_fn_proxy = feat_norm_calibrated((du_cal - du_cal.mean(0))**2)

        scaled[tag] = {
            "cu_tr": cu_s_tr, "du_tr": du_s_tr[:, 0, :],
            "cu_te": cu_s_te, "du_te": du_s_te[:, 0, :],
            "cu_iqr": b.cu_scaler.scale_, "cu_med": b.cu_scaler.center_,
            "du_iqr": b.du_scaler.scale_, "du_med": b.du_scaler.center_,
            "cu_fn":  cu_fn_proxy, "du_fn": du_fn_proxy,
            "kept": kept,
        }

        print(f"\n[{tag}] CU RobustScaler — median={b.cu_scaler.center_}  IQR={b.cu_scaler.scale_}")
        print(f"[{tag}] DU RobustScaler — median={b.du_scaler.center_}  IQR={b.du_scaler.scale_}")
        print(f"[{tag}] feat_norm proxy — CU: {cu_fn_proxy}  DU: {du_fn_proxy}")
        print(f"[{tag}] TEST DU mem_pct scaled — mean={du_s_te[:,0,1].mean():.2f}  std={du_s_te[:,0,1].std():.2f}")

    # ── figure layout ─────────────────────────────────────────────────────────
    # 8 rows × 4 cols:
    #   rows 0-1: CU cpu / CU mem_pct
    #   rows 2-3: DU cpu / DU mem_pct
    #   rows 4-5: CU cpu scaled / CU mem_pct scaled (no_imp vs imp)
    #   rows 6-7: DU cpu scaled / DU mem_pct scaled (no_imp vs imp)
    # cols: TRAIN raw | TEST raw | scaled no_imp | scaled with_imp

    NROWS, NCOLS = 8, 4
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 22), sharex=False)
    fig.suptitle(
        "Feature time series: raw vs scaled, with vs without imputation\n"
        "Red lines = glitch rows (raw=0).  Orange shading = CPU stress.",
        fontsize=12, y=1.002,
    )

    col_titles = [
        f"TRAIN raw\n({TRAIN_TOPO})",
        f"TEST raw\n({TEST_TOPO})",
        "TRAIN scaled\n(WITHOUT imputation)",
        "TRAIN scaled\n(WITH imputation)",
    ]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    row_labels = [
        "CU  cpu", "CU  mem_pct",
        "DU₀ cpu", "DU₀ mem_pct",
        "CU  cpu\n(scaled)",  "CU  mem_pct\n(scaled)",
        "DU₀ cpu\n(scaled)",  "DU₀ mem_pct\n(scaled)",
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    # ── raw rows ──────────────────────────────────────────────────────────────
    raw_data = [
        (cu_tr_raw, cu_te_raw, glitch_tr_cu, glitch_te_cu, stress_te),   # row 0: CU cpu
        (None,      None,      None,          None,          stress_te),   # row 1: CU mem_pct (same arrays, different feature)
        (du_tr_raw, du_te_raw, glitch_tr_du, glitch_te_du, stress_te),   # row 2: DU cpu
        (None,      None,      None,          None,          stress_te),   # row 3: DU mem_pct
    ]

    for feat_idx in range(2):   # 0=cpu, 1=mem_pct
        for entity_idx in range(2):  # 0=CU, 1=DU
            row = entity_idx * 2 + feat_idx
            tr_arr = cu_tr_raw if entity_idx == 0 else du_tr_raw
            te_arr = cu_te_raw if entity_idx == 0 else du_te_raw
            gl_tr  = glitch_tr_cu if entity_idx == 0 else glitch_tr_du
            gl_te  = glitch_te_cu if entity_idx == 0 else glitch_te_du

            # TRAIN raw
            ax = axes[row, 0]
            ax.plot(t_tr, tr_arr[:, feat_idx], color="steelblue", lw=0.5)
            mark_glitches(ax, gl_tr, len(t_tr))
            ax.grid(True, lw=0.3, alpha=0.4)

            # TEST raw
            ax = axes[row, 1]
            ax.plot(t_te, te_arr[:, feat_idx], color="darkorange", lw=0.5)
            mark_glitches(ax, gl_te, len(t_te))
            mark_stress(ax, stress_te == 1)
            ax.grid(True, lw=0.3, alpha=0.4)

    # ── scaled rows ───────────────────────────────────────────────────────────
    for feat_idx in range(2):
        for entity_idx in range(2):
            row = 4 + entity_idx * 2 + feat_idx

            for col_idx, tag in enumerate(["no_imp", "imp"]):
                col = 2 + col_idx
                sd  = scaled[tag]
                tr_scaled = sd["cu_tr"] if entity_idx == 0 else sd["du_tr"]
                te_scaled = sd["cu_te"] if entity_idx == 0 else sd["du_te"]
                gl_tr     = glitch_tr_cu if entity_idx == 0 else glitch_tr_du
                gl_te     = glitch_te_cu if entity_idx == 0 else glitch_te_du
                kept      = sd["kept"]

                ax = axes[row, col]

                # TRAIN scaled
                ax.plot(t_tr, tr_scaled[:, feat_idx],
                        color="steelblue", lw=0.5, alpha=0.7, label="train")
                mark_glitches(ax, gl_tr, len(t_tr), color="red")

                # TEST scaled (on same axis, different color)
                t_te_kept = np.arange(len(te_scaled))
                ax.plot(t_te_kept, te_scaled[:, feat_idx],
                        color="darkorange", lw=0.5, alpha=0.7, label="test")
                mark_stress(ax, stress_te[kept] == 1)

                iqr = sd["cu_iqr"][feat_idx] if entity_idx == 0 else sd["du_iqr"][feat_idx]
                med = sd["cu_med"][feat_idx] if entity_idx == 0 else sd["du_med"][feat_idx]
                fn  = sd["cu_fn"][feat_idx]  if entity_idx == 0 else sd["du_fn"][feat_idx]
                ax.set_title(f"median={med:.4f}  IQR={iqr:.5f}\nfeat_norm≈{fn:.5f}",
                             fontsize=7)

                if row == 4:
                    ax.legend(loc="upper right", fontsize=6, framealpha=0.6)
                ax.grid(True, lw=0.3, alpha=0.4)

    # hide unused raw cols 2-3 for raw rows (only 2 cols used)
    for r in range(4):
        for c in range(2, 4):
            axes[r, c].set_visible(False)

    # x-labels on bottom row
    for c in range(NCOLS):
        axes[-1, c].set_xlabel("Timestep", fontsize=8)

    plt.tight_layout()
    out = Path("diagnose_feature_scaling.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {out.resolve()}")
    plt.close()

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — RobustScaler IQR and feat_norm comparison")
    print("=" * 70)
    print(f"{'':30s}  {'no imputation':>20s}  {'with imputation':>20s}")
    for entity, ek in [("CU", "cu"), ("DU", "du")]:
        for fi, fn in enumerate(FEAT_NAMES):
            ni = scaled["no_imp"][f"{ek}_iqr"][fi]
            wi = scaled["imp"][f"{ek}_iqr"][fi]
            nf = scaled["no_imp"][f"{ek}_fn"][fi]
            wf = scaled["imp"][f"{ek}_fn"][fi]
            print(f"  {entity} {fn:10s}  IQR={ni:.6f}  fn≈{nf:.5f}    "
                  f"IQR={wi:.6f}  fn≈{wf:.5f}")

    print("\nKEY:")
    print("  DU mem_pct IQR ≈ 0.001 in both cases → any 0.001 raw diff → 1.0 scaled unit")
    print("  feat_norm without imputation is huge (glitch sqerr dominates)")
    print("  feat_norm with imputation is tiny → cross-topology shift × 1/tiny = huge FP score")


if __name__ == "__main__":
    main()
