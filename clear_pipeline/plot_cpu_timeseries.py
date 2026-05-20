"""plot_cpu_timeseries.py — Raw vs Scaled CPU timeseries with anomaly shading.

For each topology plots two columns:
  LEFT  — raw (unscaled) CPU metric
  RIGHT — v0-scaled CPU metric (same RobustScaler used in run_experiment.py)

Red background = cpu_stress == 1 ground-truth label.
Grey = train normal reference (to see baseline shift).

Run from:  project_root/step3_topoar/clear_pipeline/
    python plot_cpu_timeseries.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import fit_bundle, transform_stream

BASE_DIR = Path("output")

TOPOS = [
    ("cu1_du2",    "cu1_du2"),
    ("cu0_du0du1", "cu0_du0du1"),
]

CU_FEAT_SLICE = slice(0, 2)   # cpu + mem_pct — same as run_experiment.py
DU_FEAT_SLICE = slice(0, 2)
CPU_IDX = 0   # within the sliced features, index 0 = cpu


def load(topo: str, split: str) -> dict:
    p = BASE_DIR / f"{topo}_stress1" / f"{split}.npz"
    assert p.exists(), f"Missing: {p}"
    return dict(np.load(p))


def scale_streams(z_tr: dict, z_te: dict):
    """Fit v0 RobustScaler on train, apply to both train and test."""
    cu_tr = z_tr["cu"][:, CU_FEAT_SLICE].astype(np.float32)
    du_tr = z_tr["du"][:, :, DU_FEAT_SLICE].astype(np.float32)
    bid_tr = z_tr["block_id"].astype(np.int64)

    bundle = fit_bundle(
        train_streams=[{"cu": cu_tr, "du": du_tr, "block_id": bid_tr}],
        cu_zero_variance_idx=[],
        du_zero_variance_idx=[],
        version="v0",
    )
    cu_s_tr, du_s_tr, _, _ = transform_stream(bundle, cu_tr, du_tr, bid_tr)

    cu_te = z_te["cu"][:, CU_FEAT_SLICE].astype(np.float32)
    du_te = z_te["du"][:, :, DU_FEAT_SLICE].astype(np.float32)
    bid_te = z_te["block_id"].astype(np.int64)
    cu_s_te, du_s_te, kept, _ = transform_stream(bundle, cu_te, du_te, bid_te)

    return cu_s_tr, du_s_tr, cu_s_te, du_s_te, kept


def shade_anomaly(ax, t, stress_mask, label="CPU stress"):
    mask = np.asarray(stress_mask, dtype=bool)
    in_block = False
    t0 = None
    first = True
    for i in range(len(mask)):
        if mask[i] and not in_block:
            t0 = t[i]
            in_block = True
        elif not mask[i] and in_block:
            ax.axvspan(t0, t[i], color="red", alpha=0.25,
                       label=(label if first else None))
            in_block = False
            first = False
    if in_block:
        ax.axvspan(t0, t[-1] + 1, color="red", alpha=0.25,
                   label=(label if first else None))


def plot_entity(ax, t_tr, cpu_tr, t_te, cpu_te, stress, title, ylabel, color):
    ax.plot(t_tr, cpu_tr, color="0.65", lw=0.6, alpha=0.6, label="train (normal ref)")
    ax.plot(t_te, cpu_te, color=color,  lw=0.8, label="test CPU")
    shade_anomaly(ax, t_te, stress == 1)

    lo = np.percentile(cpu_te, 1)
    hi = np.percentile(cpu_te, 99)
    pad = max((hi - lo) * 0.3, 0.05)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, max(len(t_tr), len(t_te)))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    ax.grid(True, lw=0.3, alpha=0.4)


def main():
    # ── gather data ───────────────────────────────────────────────────────────
    records = {}
    max_du = 0
    for topo, label in TOPOS:
        z_tr = load(topo, "train")
        z_te = load(topo, "test")
        cu_s_tr, du_s_tr, cu_s_te, du_s_te, kept = scale_streams(z_tr, z_te)

        cu_stress = z_te["cu_stress"][kept].astype(np.int32)
        du_stress = z_te["du_stress"][kept].astype(np.int32)
        n_du = du_s_te.shape[1]
        max_du = max(max_du, n_du)

        records[topo] = dict(
            label=label, n_du=n_du,
            # raw
            cu_raw_tr=z_tr["cu"][:, CPU_IDX],
            du_raw_tr=z_tr["du"][:, :, CPU_IDX],
            cu_raw_te=z_te["cu"][:, CPU_IDX][kept],
            du_raw_te=z_te["du"][:, :, CPU_IDX][kept],
            # scaled
            cu_s_tr=cu_s_tr[:, CPU_IDX],
            du_s_tr=du_s_tr[:, :, CPU_IDX],
            cu_s_te=cu_s_te[:, CPU_IDX],
            du_s_te=du_s_te[:, :, CPU_IDX],
            # labels
            cu_stress=cu_stress,
            du_stress=du_stress,
        )

    # ── layout: rows = entities (CU + max_du DUs), cols = topo × 2 (raw, scaled) ──
    n_entities = 1 + max_du
    n_cols = len(TOPOS) * 2          # 2 sub-columns per topo (raw | scaled)

    fig, axes = plt.subplots(
        n_entities, n_cols,
        figsize=(7 * n_cols, 3 * n_entities),
        sharex=False,
    )
    if n_entities == 1:
        axes = axes[np.newaxis, :]

    # Column header labels via text above top row
    col_headers = []
    for topo, label in TOPOS:
        col_headers += [f"{label}\nRAW cpu", f"{label}\nSCALED cpu (v0 RobustScaler)"]

    for col, hdr in enumerate(col_headers):
        axes[0, col].set_title(hdr, fontsize=9, fontweight="bold")

    for col_base, (topo, _) in enumerate(TOPOS):
        r = records[topo]
        n_du = r["n_du"]
        t_tr = np.arange(len(r["cu_raw_tr"]))
        t_te = np.arange(len(r["cu_raw_te"]))

        raw_col    = col_base * 2
        scaled_col = col_base * 2 + 1

        # ── CU row ───────────────────────────────────────────────────────────
        plot_entity(
            axes[0, raw_col],
            t_tr, r["cu_raw_tr"], t_te, r["cu_raw_te"], r["cu_stress"],
            title="", ylabel="CU  raw cpu", color="steelblue",
        )
        plot_entity(
            axes[0, scaled_col],
            t_tr, r["cu_s_tr"], t_te, r["cu_s_te"], r["cu_stress"],
            title="", ylabel="CU  scaled cpu", color="darkorange",
        )

        # ── DU rows ──────────────────────────────────────────────────────────
        for i in range(max_du):
            row = i + 1
            if i < n_du:
                plot_entity(
                    axes[row, raw_col],
                    t_tr, r["du_raw_tr"][:, min(i, r["du_raw_tr"].shape[1]-1)],
                    t_te, r["du_raw_te"][:, i], r["du_stress"][:, i],
                    title="", ylabel=f"DU_{i}  raw cpu", color="steelblue",
                )
                plot_entity(
                    axes[row, scaled_col],
                    t_tr, r["du_s_tr"][:, min(i, r["du_s_tr"].shape[1]-1)],
                    t_te, r["du_s_te"][:, i], r["du_stress"][:, i],
                    title="", ylabel=f"DU_{i}  scaled cpu", color="darkorange",
                )
            else:
                axes[row, raw_col].set_visible(False)
                axes[row, scaled_col].set_visible(False)

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Timestep", fontsize=8)

    fig.suptitle(
        "CPU metric — raw vs v0-scaled  |  red bg = cpu_stress label\n"
        "grey = train normal reference,  blue = raw test,  orange = scaled test",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = Path("cpu_timeseries.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
