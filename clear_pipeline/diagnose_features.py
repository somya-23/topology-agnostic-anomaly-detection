"""diagnose_features.py — Step-by-step visual diagnosis of why adding mem_pct breaks DU recall.

Produces one figure per topology with panels showing:
  1. Raw CPU and mem_pct time series (train-normal vs test, stress shaded)
  2. Distribution (KDE) of raw values: train-normal vs test-normal vs test-stress
  3. After RobustScaler: same distribution in scaled space
  4. Per-feature squared error under stress vs normal — shows which feature dominates the score
  5. Lift score (max over features) vs threshold — the actual decision surface

Run from clear_pipeline/:
    python diagnose_features.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from scipy.stats import gaussian_kde

# ── config (mirror run_experiment.py) ────────────────────────────────────────
BASE_DIR      = Path("output")
ALL_TOPOS     = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
CU_FEAT_SLICE = slice(0, 2)
DU_FEAT_SLICE = slice(0, 2)
FEAT_NAMES    = ["cpu", "mem_pct"]
IMPUTE_EPS    = 1e-6
CAL_FRAC      = 0.2
COLD_START_K  = 64

# ── helpers ───────────────────────────────────────────────────────────────────

def load_npz(topo, split):
    p = BASE_DIR / f"{topo}_stress1" / f"{split}.npz"
    assert p.exists(), f"Missing: {p}"
    return dict(np.load(p))

def impute_cpu(arr):
    """Forward-fill rows where cpu (idx 0) or mem_pct (idx 1) is near-zero."""
    arr = arr.copy()
    glitch_idx = [i for i in [0, 1] if i < (arr.shape[1] if arr.ndim == 2 else arr.shape[2])]
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, glitch_idx] < IMPUTE_EPS
            arr[t, glitch_idx] = np.where(glitch, arr[t-1, glitch_idx], arr[t, glitch_idx])
        else:
            glitch = arr[t, :, glitch_idx] < IMPUTE_EPS
            arr[t, :, glitch_idx] = np.where(glitch, arr[t-1, :, glitch_idx], arr[t, :, glitch_idx])
    return arr

def slice_feat(z):
    cu = impute_cpu(z["cu"][:, CU_FEAT_SLICE].astype(np.float32))
    du = impute_cpu(z["du"][:, :, DU_FEAT_SLICE].astype(np.float32))
    return cu, du

def fit_scalers(train_zs):
    """Fit one RobustScaler on all train topologies pooled (mirrors fit_bundle v0)."""
    cu_chunks, du_chunks = [], []
    for z in train_zs:
        cu, du = slice_feat(z)
        cu_chunks.append(cu)
        du_chunks.append(du.reshape(-1, du.shape[-1]))
    cu_scaler = RobustScaler().fit(np.concatenate(cu_chunks))
    du_scaler = RobustScaler().fit(np.concatenate(du_chunks))
    return cu_scaler, du_scaler

def kde_plot(ax, data, label, color, ls="-", clip=None):
    data = data[np.isfinite(data)]
    if clip is not None:
        data = np.clip(data, *clip)
    if len(data) < 5:
        return
    try:
        kde = gaussian_kde(data, bw_method=0.3)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color=color, ls=ls, label=label, lw=1.8)
        ax.fill_between(xs, kde(xs), alpha=0.10, color=color)
    except Exception:
        pass

def stress_bands(ax, lbl, color="red", alpha=0.18):
    in_stress = False
    start = 0
    for i, v in enumerate(lbl):
        if v == 1 and not in_stress:
            start = i; in_stress = True
        elif v != 1 and in_stress:
            ax.axvspan(start, i, color=color, alpha=alpha)
            in_stress = False
    if in_stress:
        ax.axvspan(start, len(lbl), color=color, alpha=alpha)

# ── main diagnosis ────────────────────────────────────────────────────────────

for test_topo in ALL_TOPOS:
    train_topos = [t for t in ALL_TOPOS if t != test_topo]
    print(f"\n{'='*65}")
    print(f"  Diagnosing: train={train_topos}  test={test_topo}")
    print(f"{'='*65}")

    train_zs   = [load_npz(t, "train") for t in train_topos]
    test_z     = load_npz(test_topo, "test")

    cu_scaler, du_scaler = fit_scalers(train_zs)

    # ── raw train-normal data (pool all train topos, cal portion) ─────────────
    tn_cu_chunks, tn_du_chunks = [], []
    for z in train_zs:
        cu, du = slice_feat(z)
        n_cal  = int(round(CAL_FRAC * len(cu)))
        cal_cu = cu[-n_cal:]
        cal_du = du[-n_cal:]
        tn_cu_chunks.append(cal_cu)
        tn_du_chunks.append(cal_du.reshape(-1, cal_du.shape[-1]))
    cal_cu = np.concatenate(tn_cu_chunks)
    cal_du = np.concatenate(tn_du_chunks)

    # ── test data ─────────────────────────────────────────────────────────────
    te_cu_raw, te_du_raw = slice_feat(test_z)
    cu_stress = test_z.get("cu_stress", test_z.get("labels", None))
    du_stress = test_z.get("du_stress", None)

    N_DU = te_du_raw.shape[1]

    # ── scaled versions ───────────────────────────────────────────────────────
    cal_cu_s = cu_scaler.transform(cal_cu)
    cal_du_s = du_scaler.transform(cal_du)

    te_cu_s  = cu_scaler.transform(te_cu_raw)
    te_du_s  = du_scaler.transform(te_du_raw.reshape(-1, te_du_raw.shape[-1])).reshape(te_du_raw.shape)

    # ── per-feature squared errors on test ────────────────────────────────────
    # simple "predict mean" baseline to see signal strength per feature
    train_cu_mean = cu_scaler.transform(
        np.concatenate([slice_feat(z)[0] for z in train_zs])
    ).mean(axis=0, keepdims=True)
    train_du_mean = du_scaler.transform(
        np.concatenate([slice_feat(z)[1].reshape(-1, te_du_raw.shape[-1]) for z in train_zs])
    ).mean(axis=0, keepdims=True)

    te_cu_sqerr = (te_cu_s - train_cu_mean) ** 2           # (T, cu_dim)
    te_du_sqerr = (te_du_s.reshape(-1, te_du_raw.shape[-1]) - train_du_mean) ** 2  # (T*N, du_dim)
    te_du_sqerr = te_du_sqerr.reshape(te_du_raw.shape)     # (T, N, du_dim)

    # ── print summary stats ───────────────────────────────────────────────────
    print(f"\n  [RAW] CU features (train-cal vs test)")
    for fi, fn in enumerate(FEAT_NAMES):
        tnorm = cal_cu[:, fi]
        tstress = te_cu_raw[cu_stress == 1, fi] if cu_stress is not None else np.array([])
        tnormal = te_cu_raw[cu_stress == 0, fi] if cu_stress is not None else te_cu_raw[:, fi]
        print(f"    {fn:8s}  train-cal: {tnorm.mean():.4f}±{tnorm.std():.4f}"
              f"   test-normal: {tnormal.mean():.4f}±{tnormal.std():.4f}"
              f"   test-STRESS: {tstress.mean():.4f}±{tstress.std():.4f}"
              f"   lift={tstress.mean()/max(tnormal.mean(),1e-6):.2f}x")

    print(f"\n  [RAW] DU features (train-cal vs test, averaged over DU instances)")
    for fi, fn in enumerate(FEAT_NAMES):
        tnorm = cal_du[:, fi]
        for du_i in range(N_DU):
            du_lbl = du_stress[:, du_i] if du_stress is not None else None
            raw_du_i = te_du_raw[:, du_i, fi]
            tstress = raw_du_i[du_lbl == 1] if du_lbl is not None else np.array([])
            tnormal = raw_du_i[du_lbl == 0] if du_lbl is not None else raw_du_i
            lift = tstress.mean() / max(tnormal.mean(), 1e-6)
            print(f"    DU_{du_i} {fn:8s}  train-cal: {tnorm.mean():.4f}±{tnorm.std():.4f}"
                  f"   test-normal: {tnormal.mean():.4f}±{tnormal.std():.4f}"
                  f"   test-STRESS: {tstress.mean():.4f}±{tstress.std():.4f}"
                  f"   lift={lift:.2f}x")

    print(f"\n  [SCALED] CU: median cal={np.median(cal_cu_s):.3f}  median test={np.median(te_cu_s):.3f}"
          f"   shift={(np.median(te_cu_s)-np.median(cal_cu_s)):.3f}")
    print(f"  [SCALED] DU: median cal={np.median(cal_du_s):.3f}  median test={np.median(te_du_s):.3f}"
          f"   shift={(np.median(te_du_s)-np.median(cal_du_s)):.3f}")

    # ── figure: 5 rows × (1+N_DU) cols ───────────────────────────────────────
    n_entities = 1 + N_DU
    fig, axes = plt.subplots(5, n_entities,
                             figsize=(5 * n_entities, 18),
                             squeeze=False)
    fig.suptitle(f"Feature diagnosis | train={train_topos} → test={test_topo}",
                 fontsize=11, y=1.01)

    row_titles = [
        "Row 1: Raw time series (test stream)",
        "Row 2: Raw value distribution  (KDE)",
        "Row 3: Scaled value distribution (KDE)",
        "Row 4: Per-feature squared error (test, log)",
        "Row 5: Max-lift score vs threshold",
    ]
    for ri, rt in enumerate(row_titles):
        axes[ri, 0].set_ylabel(rt, fontsize=8, labelpad=4)

    # Color encodes CONDITION, linestyle encodes FEATURE
    # → can immediately tell train-cal vs test-normal vs test-stress,
    #   and solid=cpu vs dashed=mem_pct
    COND_COLOR = {"cal": "forestgreen", "normal": "steelblue", "stress": "crimson"}
    FEAT_LS    = {"cpu": "-", "mem_pct": "--"}
    FEAT_LW    = {"cpu": 0.6, "mem_pct": 0.6}      # for time-series rows
    FEAT_COLOR = {"cpu": "steelblue", "mem_pct": "darkorange"}   # kept for sqerr row only

    for ei in range(n_entities):
        is_cu  = (ei == 0)
        name   = "CU" if is_cu else f"DU_{ei-1}"
        du_i   = ei - 1

        raw    = te_cu_raw        if is_cu else te_du_raw[:, du_i]     # (T, dim)
        scaled = te_cu_s          if is_cu else te_du_s[:, du_i]       # (T, dim)
        sqerr  = te_cu_sqerr      if is_cu else te_du_sqerr[:, du_i]   # (T, dim)
        lbl    = cu_stress        if is_cu else (du_stress[:, du_i] if du_stress is not None else None)
        cal_raw_e = cal_cu        if is_cu else cal_du
        cal_s_e   = cal_cu_s     if is_cu else cal_du_s

        # ── Row 0: raw time series ────────────────────────────────────────────
        # color = condition (blue=normal, red=stress via bands), linestyle = feature
        ax = axes[0, ei]
        T = len(raw)
        ts = np.arange(T)
        for fi, fn in enumerate(FEAT_NAMES):
            ax.plot(ts, raw[:, fi], color=FEAT_COLOR[fn],
                    ls=FEAT_LS[fn], lw=0.7, alpha=0.85, label=fn)
        if lbl is not None:
            stress_bands(ax, lbl)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Timestep")

        # ── Row 1: raw KDE ────────────────────────────────────────────────────
        # Color = condition: green=train-cal, blue=test-normal, red=test-stress
        # Linestyle = feature: solid=cpu, dashed=mem_pct
        ax = axes[1, ei]
        for fi, fn in enumerate(FEAT_NAMES):
            ls = FEAT_LS[fn]
            kde_plot(ax, cal_raw_e[:, fi],  f"cal·{fn}",    COND_COLOR["cal"],    ls=ls)
            if lbl is not None:
                kde_plot(ax, raw[lbl == 0, fi], f"normal·{fn}", COND_COLOR["normal"], ls=ls)
                kde_plot(ax, raw[lbl == 1, fi], f"stress·{fn}", COND_COLOR["stress"], ls=ls)
        ax.set_xlabel("raw value")
        ax.legend(fontsize=6, loc="upper right",
                  title="color=cond  line=feat(─cpu /──mem)", title_fontsize=5)

        # ── Row 2: scaled KDE ─────────────────────────────────────────────────
        # Clip computed per-entity from actual data so CU's large baseline shift
        # doesn't push everything to one clip boundary (which kills KDE).
        ax = axes[2, ei]
        all_scaled = np.concatenate([
            cal_s_e.reshape(-1),
            scaled.reshape(-1),
        ])
        s_lo = float(np.percentile(all_scaled[np.isfinite(all_scaled)],  1))
        s_hi = float(np.percentile(all_scaled[np.isfinite(all_scaled)], 99))
        s_pad = max((s_hi - s_lo) * 0.1, 0.5)
        s_clip = (s_lo - s_pad, s_hi + s_pad)
        for fi, fn in enumerate(FEAT_NAMES):
            ls = FEAT_LS[fn]
            kde_plot(ax, cal_s_e[:, fi],        f"cal·{fn}",    COND_COLOR["cal"],    ls=ls, clip=s_clip)
            if lbl is not None:
                kde_plot(ax, scaled[lbl == 0, fi],  f"normal·{fn}", COND_COLOR["normal"], ls=ls, clip=s_clip)
                kde_plot(ax, scaled[lbl == 1, fi],  f"stress·{fn}", COND_COLOR["stress"], ls=ls, clip=s_clip)
        ax.set_xlabel("scaled value")
        ax.legend(fontsize=6, loc="upper right",
                  title="color=cond  line=feat(─cpu /──mem)", title_fontsize=5)

        # ── Row 3: per-feature sqerr time series (log) ───────────────────────
        # Here color = feature (blue=cpu, orange=mem_pct) so you see which one spikes
        ax = axes[3, ei]
        for fi, fn in enumerate(FEAT_NAMES):
            ax.semilogy(ts[:-1] if len(sqerr) < T else ts,
                        np.clip(sqerr[:, fi], 1e-6, None),
                        color=FEAT_COLOR[fn], lw=0.6, alpha=0.85, label=fn)
        if lbl is not None:
            lbl_for_score = lbl[1:] if len(sqerr) < T else lbl
            stress_bands(ax, lbl_for_score)
        ax.legend(fontsize=7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("sqerr (log)", fontsize=8)

        # ── Row 4: max-lift score ─────────────────────────────────────────────
        ax = axes[4, ei]
        feat_norm = cal_s_e.var(axis=0) + 1e-6      # proxy for expected normal error scale
        lift = (sqerr / feat_norm).max(axis=-1)
        thr  = np.percentile((cal_s_e.var(axis=0) / feat_norm + 1e-6), 99.9)  # rough proxy thr
        cal_lift = ((cal_s_e ** 2) / feat_norm).max(axis=-1)
        thr = np.percentile(cal_lift, 99.9)
        ts_score = np.arange(len(lift))
        ax.semilogy(ts_score, np.clip(lift, 1e-6, None), color="navy", lw=0.5, alpha=0.8)
        ax.axhline(thr, color="red", ls="--", lw=1.2, label=f"thr~{thr:.2f}")
        if lbl is not None:
            lbl_sc = lbl[1:] if len(lift) < T else lbl
            stress_bands(ax, lbl_sc)
            # print per-feature sqerr means for stress vs normal
            if lbl_sc.shape[0] == sqerr.shape[0]:
                for fi, fn in enumerate(FEAT_NAMES):
                    n_mean = sqerr[lbl_sc == 0, fi].mean()
                    s_mean = sqerr[lbl_sc == 1, fi].mean()
                    print(f"    {name} {fn:8s}  sqerr normal={n_mean:.4f}  stress={s_mean:.4f}"
                          f"  ratio={s_mean/max(n_mean,1e-9):.1f}x")
        ax.legend(fontsize=7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("max-lift (log)", fontsize=8)

    plt.tight_layout()
    out = Path(f"diagnose_{test_topo}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out.resolve()}")
