"""compare_test_cu_stress.py — KPI comparison during CU CPU stress across topologies.

Shows for each topology, side by side:
    normal rows  : cu_stress=0, all du_stress=0
    CU CPU stress: cu_stress=1, all du_stress=0

Columns per topology: mean_normal  mean_stressed  effect_d
effect_d = (mean_stressed - mean_normal) / std_normal

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python compare_test_cu_stress.py
"""

from pathlib import Path
import numpy as np

BASE   = Path("short_cpu_stress_experiment")
TOPOS  = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
STRESS = 1

CU_KPI_NAMES = ["cpu", "mem_pct", "mem_bytes", "fs_reads", "fs_writes", "net_tx", "net_rx"]
PCI_KPI_NAMES = [
    "avg_crc_delay", "avg_pucch_harq_delay", "avg_pusch_harq_delay",
    "bsr", "cqi", "dl_brate", "dl_bs", "dl_mcs", "dl_nof_nok", "dl_nof_ok",
    "dl_ri", "max_crc_delay", "max_pucch_harq_delay", "max_pusch_harq_delay",
    "nof_pucch_f0f1_inv_harq", "nof_pucch_f2f3f4_inv_csi", "nof_pucch_f2f3f4_inv_harq",
    "nof_pusch_inv_csi", "nof_pusch_inv_harq", "pucch_snr_db", "pucch_ta_ns",
    "pusch_rsrp_db", "pusch_snr_db", "pusch_ta_ns", "ta_ns",
    "ul_brate", "ul_mcs", "ul_nof_nok", "ul_nof_ok", "ul_ri",
]
DU_KPI_NAMES = CU_KPI_NAMES + PCI_KPI_NAMES   # 37 total


def load(topo: str):
    z        = np.load(BASE / f"{topo}_stress{STRESS}" / "test.npz")
    cu       = z["cu"]         # (T, 7)
    du       = z["du"]         # (T, N, 37)
    cu_s     = z["cu_stress"]  # (T,)
    du_s     = z["du_stress"]  # (T, N)

    normal_mask   = (cu_s == 0) & np.all(du_s == 0, axis=1)
    cu_stress_mask = (cu_s == STRESS) & np.all(du_s == 0, axis=1)

    return cu, du, normal_mask, cu_stress_mask


def section(title: str):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def print_comparison(kpi_names, entity_label,
                     normals: list, stressed: list, topo_labels: list):
    """
    normals   : list of (T_n, n_kpi) arrays — normal rows per topology
    stressed  : list of (T_s, n_kpi) arrays — stressed rows per topology
    """
    col = 28   # width per topology block
    header = f"  {'KPI':<22s}"
    for t in topo_labels:
        header += f"  {t + ' (norm|stress|d)':<{col}s}"
    print(header)
    print("  " + "-" * (22 + (col + 2) * len(topo_labels)))

    for i, name in enumerate(kpi_names):
        row = f"  {entity_label}:{name:<20s}"
        for n_arr, s_arr in zip(normals, stressed):
            mn = n_arr[:, i].mean()
            ms = s_arr[:, i].mean()
            sd = n_arr[:, i].std() + 1e-12
            d  = (ms - mn) / sd
            flag = "***" if abs(d) > 3 else ("** " if abs(d) > 1.5 else ("*  " if abs(d) > 0.8 else "   "))
            cell = f"{mn:.3g}|{ms:.3g}|{d:+.1f}{flag}"
            row += f"  {cell:<{col}s}"
        print(row)


def main():
    data = {}
    for topo in TOPOS:
        cu, du, nm, sm = load(topo)
        n_du = du.shape[1]
        data[topo] = {
            "cu_normal":   cu[nm],
            "cu_stressed": cu[sm],
            "du_normal":   du[nm].mean(axis=1),    # avg over DUs → (T_n, 37)
            "du_stressed": du[sm].mean(axis=1),    # avg over DUs → (T_s, 37)
            "n_normal":    nm.sum(),
            "n_stressed":  sm.sum(),
            "n_du":        n_du,
        }
        print(f"  {topo}: {nm.sum()} normal rows, {sm.sum()} CU-stressed rows  (N_DU={n_du})")

    short_labels = [f"{t.split('_')[0]}" for t in TOPOS]

    section("CU KPIs during CU CPU stress  [norm_mean | stress_mean | effect_d]")
    print_comparison(
        CU_KPI_NAMES, "CU",
        [data[t]["cu_normal"]   for t in TOPOS],
        [data[t]["cu_stressed"] for t in TOPOS],
        short_labels,
    )

    section("DU KPIs during CU CPU stress  (averaged over DUs)  [norm_mean | stress_mean | effect_d]")
    print_comparison(
        DU_KPI_NAMES, "DU",
        [data[t]["du_normal"]   for t in TOPOS],
        [data[t]["du_stressed"] for t in TOPOS],
        short_labels,
    )

    # ── Effect-d summary: top movers ─────────────────────────────────────────
    section("TOP 10 KPIs by |effect_d|  (averaged across topologies)")
    all_names = [f"CU:{n}" for n in CU_KPI_NAMES] + [f"DU:{n}" for n in DU_KPI_NAMES]
    effects_per_topo = []
    for topo in TOPOS:
        cu_n = data[topo]["cu_normal"];   cu_s = data[topo]["cu_stressed"]
        du_n = data[topo]["du_normal"];   du_s = data[topo]["du_stressed"]
        cu_d = (cu_s.mean(0) - cu_n.mean(0)) / (cu_n.std(0) + 1e-12)
        du_d = (du_s.mean(0) - du_n.mean(0)) / (du_n.std(0) + 1e-12)
        effects_per_topo.append(np.concatenate([cu_d, du_d]))

    avg_abs = np.mean(np.abs(effects_per_topo), axis=0)
    ranked  = sorted(zip(all_names, avg_abs.tolist()), key=lambda x: x[1], reverse=True)

    print(f"  {'KPI':<28s}  {'avg|d|':>8s}" +
          "".join(f"  {l:>10s}" for l in short_labels))
    print("  " + "-" * 70)
    for name, avg_d in ranked[:10]:
        idx = all_names.index(name)
        per_topo_d = [effects_per_topo[j][idx] for j in range(len(TOPOS))]
        cols = "".join(f"  {d:>+10.2f}" for d in per_topo_d)
        print(f"  {name:<28s}  {avg_d:>8.2f}{cols}")


if __name__ == "__main__":
    main()
