"""Ablation heatmap (Full + A1..A3), CU and DU entities, no-net_ratio final model.
Re-uses make_paper_figs infrastructure. Evaluates each ablation checkpoint on the
CU_*_random and DU_*_random captures. Prints the F1 matrices and saves a 2-panel PNG.
"""
import io, contextlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_experiment_v0_baseline as v0
from model_calibrated import CalibratedTopoAR
from run_noratio_ablations import slice_noratio, ABLATIONS
from make_paper_figs import patch, evalone, FIG

TOPOS = ["cu1_du2", "cu0_du0du1", "cu2_du3du4du5"]      # N=1, N=2, N=3
NDU   = {"cu1_du2": 1, "cu0_du0du1": 2, "cu2_du3du4du5": 3}
STRESS = [("CPU", 1), ("MEM", 2), ("NET", 3)]
CU_DIR = {"CPU": "CU_CPU_random_STRESS", "MEM": "CU_MEM_random_STRESS", "NET": "CU_NET_random_STRESS"}
DU_DIR = {"CPU": "DU_CPU_random_STRESS", "MEM": "DU_MEM_random_STRESS", "NET": "DU_NET_random_STRESS"}

ROWS = ["Full Topaz", "A1 -topo norm", "A2 fully-shared", "A3 mean-pool"]
VARIANTS = [("full", CalibratedTopoAR, slice_noratio, "v0_noratio_random_")]
for v in ["A1", "A2", "A3"]:
    mc, sf = ABLATIONS[v]
    VARIANTS.append((v, mc, sf, f"abl_{v}_noratio_random_"))

COLS = [f"{s}\nN={n}" for s, _ in STRESS for n in [1, 2, 3]]


def cu_row(stype, dirname):
    return [evalone(dirname, stype, t)["CU"]["f1"] for t in TOPOS]


def du_row(stype, dirname):
    out = []
    for t in TOPOS:
        m = evalone(dirname, stype, t)
        f1s = [m[f"DU_{i}"]["f1"] for i in range(NDU[t]) if f"DU_{i}" in m]
        out.append(float(np.mean(f1s)))
    return out


def build(entity):
    M = []
    for name, mc, sf, ck in VARIANTS:
        patch(mc, sf, ck)
        r = []
        for st, stype in STRESS:
            dirname = (CU_DIR if entity == "CU" else DU_DIR)[st]
            r += (cu_row if entity == "CU" else du_row)(stype, dirname)
        M.append(r)
    return np.array(M)


def panel(ax, M, title):
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(9)); ax.set_xticklabels(COLS, fontsize=8)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS)
    ax.set_title(title, fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] >= 0.7 else "black")
    return im


if __name__ == "__main__":
    CU = build("CU"); DU = build("DU")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    panel(axes[0], CU, "CU detection F1")
    im = panel(axes[1], DU, "DU detection F1 (mean over DUs)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="F1")
    fig.savefig(FIG / "fig_ablation_A1A3_cudu.png", dpi=150, bbox_inches="tight"); plt.close()
    print("saved fig_ablation_A1A3_cudu.png")
    for tag, M in [("CU", CU), ("DU", DU)]:
        print(f"\n=== {tag} matrix (rows={ROWS}) cols={[c.replace(chr(10),' ') for c in COLS]} ===")
        for r, row in zip(ROWS, M):
            print(f"{r:18s} " + " ".join(f"{x:.2f}" for x in row))
