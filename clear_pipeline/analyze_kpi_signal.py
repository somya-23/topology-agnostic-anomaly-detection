"""analyze_kpi_signal.py — Per-entity stress breakdown: who reacts when each container is stressed?

USAGE
-----
Edit the USER INPUTS block below, then run:
    cd project_root/step3_topoar/clear_pipeline
    python analyze_kpi_signal.py

WHAT IT DOES
------------
For each stress scenario (CU stressed, DU0 stressed, DU1 stressed, ...):
    - Shows how CU KPIs react         (expected: high when CU is stressed, low otherwise)
    - Shows how each DU's KPIs react  (expected: high for the stressed DU, low for others)

This reveals two things:
    1. Which KPIs are the primary signals for each entity's stress
    2. Whether stress on one entity propagates to others (cross-entity coupling)

Masks used:
    normal          : cu_stress=0  AND  all du_stress=0
    CU stress       : cu_stress=STRESS_TYPE  AND  all du_stress=0
    DU[i] stress    : cu_stress=0  AND  du_stress[i]=STRESS_TYPE  AND  all other du_stress=0

effect_d = (mean_stressed − mean_normal) / std_normal
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# =============================================================================
# USER INPUTS — change these only
# =============================================================================

TOPOLOGY         = "cu1_du2"
STRESS_TYPE      = 1           # 1=CPU | 2=MEM | 3=NET

# Set to a list of topologies to compare at the end. Set to [] to skip.
COMPARE_TOPOLOGIES = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]

# =============================================================================
# FIXED
# =============================================================================

STRESS_NAMES = {1: "CPU", 2: "MEM", 3: "NET"}

CU_KPI_NAMES = ["cpu", "mem_pct", "mem_bytes", "fs_reads", "fs_writes", "net_tx", "net_rx"]

# PCI/radio KPI names in sorted order (indices 7-36 in the DU array).
# These are the PCI-{n}_RNTI-* columns sorted alphabetically — same order
# as get_du_columns() in build_dataset.py.
PCI_KPI_NAMES = [
    "avg_crc_delay",          # 7
    "avg_pucch_harq_delay",   # 8
    "avg_pusch_harq_delay",   # 9
    "bsr",                    # 10
    "cqi",                    # 11
    "dl_brate",               # 12
    "dl_bs",                  # 13
    "dl_mcs",                 # 14
    "dl_nof_nok",             # 15
    "dl_nof_ok",              # 16
    "dl_ri",                  # 17
    "max_crc_delay",          # 18
    "max_pucch_harq_delay",   # 19
    "max_pusch_harq_delay",   # 20
    "nof_pucch_f0f1_inv_harq",# 21
    "nof_pucch_f2f3f4_inv_csi",# 22
    "nof_pucch_f2f3f4_inv_harq",# 23
    "nof_pusch_inv_csi",      # 24
    "nof_pusch_inv_harq",     # 25
    "pucch_snr_db",           # 26
    "pucch_ta_ns",            # 27
    "pusch_rsrp_db",          # 28
    "pusch_snr_db",           # 29
    "pusch_ta_ns",            # 30
    "ta_ns",                  # 31
    "ul_brate",               # 32
    "ul_mcs",                 # 33
    "ul_nof_nok",             # 34
    "ul_nof_ok",              # 35
    "ul_ri",                  # 36
]

NPZ_DIR = Path("short_cpu_stress_experiment") / f"{TOPOLOGY}_stress{STRESS_TYPE}"


def stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over axis 0."""
    return arr.mean(axis=0), arr.std(axis=0)


def print_kpi_table(
    name: str,
    kpi_labels: list[str],
    train_n: np.ndarray,
    test_n: np.ndarray,
    test_a: np.ndarray,
) -> None:
    mn_tr, sd_tr = stats(train_n)
    mn_tn, sd_tn = stats(test_n)
    mn_ta, sd_ta = stats(test_a)

    print(f"\n  {'KPI':<14s}  {'mean_train':>11s}  {'mean_t_norm':>11s}  {'mean_t_anom':>11s}"
          f"  {'std_train':>9s}  {'std_t_norm':>9s}  {'std_t_anom':>9s}  {'effect_d':>9s}")
    print("  " + "-" * 108)

    for i, label in enumerate(kpi_labels):
        d_raw = sd_tn[i]
        effect = (mn_ta[i] - mn_tn[i]) / (d_raw + 1e-12)
        arrow  = "↑" if effect > 0.3 else ("↓" if effect < -0.3 else " ")
        flag   = "***" if abs(effect) > 3 else ("** " if abs(effect) > 1.5 else
                 ("*  " if abs(effect) > 0.8 else "   "))
        print(f"  {label:<14s}  {mn_tr[i]:>11.4g}  {mn_tn[i]:>11.4g}  {mn_ta[i]:>11.4g}"
              f"  {sd_tr[i]:>9.4g}  {d_raw:>9.4g}  {sd_ta[i]:>9.4g}"
              f"  {arrow}{effect:>+7.2f} {flag}")


def effect_d_top(normal: np.ndarray, stressed: np.ndarray, labels: list[str], k: int = 5):
    """Return top-k (label, effect_d) pairs sorted by |effect_d|."""
    mn_n = normal.mean(axis=0)
    sd_n = normal.std(axis=0)
    mn_s = stressed.mean(axis=0)
    effects = (mn_s - mn_n) / (sd_n + 1e-12)
    ranked = sorted(zip(labels, effects.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:k]


def print_scenario(
    scenario_label: str,
    n_rows: int,
    cu_normal: np.ndarray,
    cu_stressed: np.ndarray,
    du_normals: list[np.ndarray],   # one per DU instance
    du_stressed_list: list[np.ndarray],
    du_names: list[str],
    stressed_entity: str,
) -> None:
    """Print per-entity KPI reaction for one stress scenario."""
    W = 90
    print(f"\n{'═'*W}")
    print(f"  SCENARIO: {scenario_label}  ({n_rows} rows)")
    print(f"  Stressed entity: {stressed_entity}")
    print(f"{'═'*W}")

    all_labels: list[str] = []
    all_effects: list[float] = []

    # ── CU reaction ──────────────────────────────────────────────────────────
    tag = "<<< STRESSED" if stressed_entity.startswith("CU") else "   (propagation?)"
    print(f"\n  CU KPIs  {tag}")
    print(f"  {'KPI':<22s}  {'mean_normal':>12s}  {'mean_stressed':>13s}  {'effect_d':>9s}")
    print("  " + "-"*65)
    mn_n_cu = cu_normal.mean(axis=0)
    mn_s_cu = cu_stressed.mean(axis=0)
    sd_n_cu = cu_normal.std(axis=0)
    for i, name in enumerate(CU_KPI_NAMES):
        d = (mn_s_cu[i] - mn_n_cu[i]) / (sd_n_cu[i] + 1e-12)
        arrow = "↑" if d > 0.3 else ("↓" if d < -0.3 else " ")
        flag  = "***" if abs(d) > 3 else ("** " if abs(d) > 1.5 else ("*  " if abs(d) > 0.8 else "   "))
        print(f"  CU:{name:<19s}  {mn_n_cu[i]:>12.4g}  {mn_s_cu[i]:>13.4g}  {arrow}{d:>+7.2f} {flag}")
        all_labels.append(f"CU:{name}")
        all_effects.append(float(d))

    # ── Each DU reaction ──────────────────────────────────────────────────────
    for du_idx, (du_name, du_n, du_s_arr) in enumerate(zip(du_names, du_normals, du_stressed_list)):
        is_stressed = du_name == stressed_entity
        tag = "<<< STRESSED" if is_stressed else "   (propagation?)"
        print(f"\n  {du_name} KPIs  {tag}")
        print(f"  {'KPI':<22s}  {'mean_normal':>12s}  {'mean_stressed':>13s}  {'effect_d':>9s}")
        print("  " + "-"*65)

        mn_n_du = du_n.mean(axis=0)
        mn_s_du = du_s_arr.mean(axis=0)
        sd_n_du = du_n.std(axis=0)

        du_full_labels = CU_KPI_NAMES + PCI_KPI_NAMES
        for i, name in enumerate(du_full_labels):
            d = (mn_s_du[i] - mn_n_du[i]) / (sd_n_du[i] + 1e-12)
            arrow = "↑" if d > 0.3 else ("↓" if d < -0.3 else " ")
            flag  = "***" if abs(d) > 3 else ("** " if abs(d) > 1.5 else ("*  " if abs(d) > 0.8 else "   "))
            print(f"  {du_name}:{name:<18s}  {mn_n_du[i]:>12.4g}  {mn_s_du[i]:>13.4g}  {arrow}{d:>+7.2f} {flag}")
            all_labels.append(f"{du_name}:{name}")
            all_effects.append(float(d))

    # ── Top-5 signals summary ─────────────────────────────────────────────────
    print(f"\n  ── TOP 5 signals in this scenario:")
    ranked = sorted(zip(all_labels, all_effects), key=lambda x: abs(x[1]), reverse=True)
    for lbl, d in ranked[:5]:
        arrow = "↑" if d > 0 else "↓"
        print(f"     {arrow} {lbl:<30s}  effect_d = {d:>+.2f}")


def main():
    z_train = np.load(NPZ_DIR / "train.npz")
    z_test  = np.load(NPZ_DIR / "test.npz")

    cu_tr = z_train["cu"]       # (T, 7)
    cu_te = z_test["cu"]        # (T, 7)
    du_te = z_test["du"]        # (T, N, 37)
    cu_s  = z_test["cu_stress"] # (T,)
    du_s  = z_test["du_stress"] # (T, N)
    n_du  = du_te.shape[1]

    normal_mask = (cu_s == 0) & np.all(du_s == 0, axis=1)

    print(f"\n{'='*90}")
    print(f"  KPI SIGNAL ANALYSIS — PER ENTITY STRESS BREAKDOWN")
    print(f"  Topology: {TOPOLOGY}  |  Stress: {STRESS_TYPE}={STRESS_NAMES[STRESS_TYPE]}")
    print(f"  test-normal rows: {normal_mask.sum():,}")
    print(f"{'='*90}")

    cu_normal = cu_te[normal_mask]                     # (N_normal, 7)
    du_normals = [du_te[normal_mask, i, :] for i in range(n_du)]  # list of (N_normal, 37)

    # ── Scenario: CU stressed ─────────────────────────────────────────────────
    cu_stress_mask = (cu_s == STRESS_TYPE) & np.all(du_s == 0, axis=1)
    print_scenario(
        scenario_label  = f"CU stressed ({STRESS_NAMES[STRESS_TYPE]}), all DUs normal",
        n_rows          = int(cu_stress_mask.sum()),
        cu_normal       = cu_normal,
        cu_stressed     = cu_te[cu_stress_mask],
        du_normals      = du_normals,
        du_stressed_list= [du_te[cu_stress_mask, i, :] for i in range(n_du)],
        du_names        = [f"DU{i}" for i in range(n_du)],
        stressed_entity = "CU",
    )

    # ── Scenario: each DU stressed individually ───────────────────────────────
    for i in range(n_du):
        mask = (cu_s == 0) & (du_s[:, i] == STRESS_TYPE)
        for j in range(n_du):
            if j != i:
                mask &= (du_s[:, j] == 0)

        print_scenario(
            scenario_label  = f"DU{i} stressed ({STRESS_NAMES[STRESS_TYPE]}), CU + other DUs normal",
            n_rows          = int(mask.sum()),
            cu_normal       = cu_normal,
            cu_stressed     = cu_te[mask],
            du_normals      = du_normals,
            du_stressed_list= [du_te[mask, j, :] for j in range(n_du)],
            du_names        = [f"DU{j}" for j in range(n_du)],
            stressed_entity = f"DU{i}",
        )


def load_npz(npz_dir: Path):
    """Load test.npz and return (cu, du, normal_mask, cu_stress_mask, du_stress_masks, n_du)."""
    z    = np.load(npz_dir / "test.npz")
    cu   = z["cu"]
    du   = z["du"]
    cu_s = z["cu_stress"]
    du_s = z["du_stress"]
    n_du = du.shape[1]
    normal   = (cu_s == 0) & np.all(du_s == 0, axis=1)
    cu_stress = (cu_s == STRESS_TYPE) & np.all(du_s == 0, axis=1)
    du_masks = []
    for i in range(n_du):
        m = (cu_s == 0) & (du_s[:, i] == STRESS_TYPE)
        for j in range(n_du):
            if j != i:
                m &= (du_s[:, j] == 0)
        du_masks.append(m)
    return cu, du, normal, cu_stress, du_masks, n_du


def eff(norm_arr: np.ndarray, stress_arr: np.ndarray) -> np.ndarray:
    return (stress_arr.mean(axis=0) - norm_arr.mean(axis=0)) / (norm_arr.std(axis=0) + 1e-12)


def avg_du_eff(du_arr, normal_mask, masks):
    """Average effect_d over all DU-stress scenarios (each DU stressed individually)."""
    return np.stack([eff(du_arr[normal_mask, i, :], du_arr[m, i, :])
                     for i, m in enumerate(masks)]).mean(axis=0)


def compare_topologies(topos: list[str]) -> None:
    W = 112
    # Load all topologies
    data = {}
    for t in topos:
        cu, du, nm, cu_sm, du_masks, n_du = load_npz(Path("short_cpu_stress_experiment") / f"{t}_stress{STRESS_TYPE}")
        data[t] = dict(cu=cu, du=du, nm=nm, cu_sm=cu_sm, du_masks=du_masks, n_du=n_du)

    n_du_str = " | ".join(f"{t}({data[t]['n_du']} DU)" for t in topos)
    print(f"\n\n{'#'*W}")
    print(f"  CROSS-TOPOLOGY COMPARISON — {n_du_str}")
    print(f"  Stress: {STRESS_TYPE} = {STRESS_NAMES[STRESS_TYPE]}")
    print(f"{'#'*W}")

    col_w = 13   # width per topology column

    # ── Part A: Normal baseline ───────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  A) NORMAL BASELINE — CU traffic should scale with #DUs, per-DU load should be identical")
    print(f"{'─'*W}")

    hdr = f"  {'KPI':<18s}" + "".join(f"  {t:>{col_w}s}" for t in topos) + "  note"
    print(f"\n  CU KPIs (mean over normal rows):")
    print(hdr);  print("  " + "-"*(18 + (col_w+2)*len(topos) + 10))
    for i, name in enumerate(CU_KPI_NAMES):
        vals = [data[t]["cu"][data[t]["nm"]].mean(axis=0)[i] for t in topos]
        ratios = [v / (vals[0] + 1e-12) for v in vals]
        # label by whether it scales with n_du
        n_du_vals = [data[t]["n_du"] for t in topos]
        scales = all(abs(ratios[j] - n_du_vals[j]/n_du_vals[0]) < 0.15 for j in range(len(topos)))
        note = "scales with #DUs ✓" if scales else "not traffic-dependent"
        row = f"  CU:{name:<15s}" + "".join(f"  {v:>{col_w}.4g}" for v in vals) + f"  {note}"
        print(row)

    print(f"\n  DU KPIs (per-DU mean — should be identical across topologies):")
    print(hdr);  print("  " + "-"*(18 + (col_w+2)*len(topos) + 10))
    for i, name in enumerate(CU_KPI_NAMES):
        vals = [data[t]["du"][data[t]["nm"]].mean(axis=(0,1))[i] for t in topos]
        refs = vals[0]
        all_same = all(abs(v/(refs+1e-12) - 1) < 0.1 for v in vals)
        note = "identical per-DU ✓" if all_same else "differs across topos ✗"
        row = f"  DU:{name:<15s}" + "".join(f"  {v:>{col_w}.4g}" for v in vals) + f"  {note}"
        print(row)

    # ── Part B: CU stress effect_d ────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  B) CU STRESS BEHAVIOR — effect_d (ALL STRONG = same dir + |d|>1.5 everywhere)")
    print(f"{'─'*W}")
    cu_effs = {t: eff(data[t]["cu"][data[t]["nm"]], data[t]["cu"][data[t]["cu_sm"]]) for t in topos}
    print(f"\n  {'CU KPI':<18s}" + "".join(f"  {'d_'+t:>{col_w}s}" for t in topos) + "  consistent?")
    print("  " + "-"*(18 + (col_w+2)*len(topos) + 14))
    for i, name in enumerate(CU_KPI_NAMES):
        ds = [cu_effs[t][i] for t in topos]
        same = all(d * ds[0] > 0 for d in ds)
        strong = all(abs(d) > 1.5 for d in ds)
        flag = "ALL STRONG ✓" if (same and strong) else ("same dir  ~" if same else "OPPOSITE  ✗")
        arrows = "".join(("↑" if d > 0 else "↓") + f"{d:>+{col_w-1}.2f}" for d in ds)
        print(f"  CU:{name:<15s}  {arrows}  {flag}")

    # ── Part C: DU stress effect_d ────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  C) DU STRESS BEHAVIOR — effect_d per DU KPI (avg over DU instances per topology)")
    print(f"     Only KPIs with |d|>0.3 in at least one topology shown")
    print(f"{'─'*W}")
    du_effs = {t: avg_du_eff(data[t]["du"], data[t]["nm"], data[t]["du_masks"]) for t in topos}
    print(f"\n  {'DU KPI':<22s}" + "".join(f"  {'d_'+t:>{col_w}s}" for t in topos) + "  consistent?")
    print("  " + "-"*(22 + (col_w+2)*len(topos) + 14))
    for i, name in enumerate(CU_KPI_NAMES + PCI_KPI_NAMES):
        ds = [du_effs[t][i] for t in topos]
        if all(abs(d) < 0.3 for d in ds):
            continue
        same = all(d * ds[0] > 0 for d in ds)
        strong = all(abs(d) > 1.5 for d in ds)
        flag = "ALL STRONG ✓" if (same and strong) else ("same dir  ~" if same else "OPPOSITE  ✗")
        arrows = "".join(("↑" if d > 0 else "↓") + f"{d:>+{col_w-1}.2f}" for d in ds)
        print(f"  DU:{name:<19s}  {arrows}  {flag}")


if __name__ == "__main__":
    main()
    if COMPARE_TOPOLOGIES:
        compare_topologies(COMPARE_TOPOLOGIES)
