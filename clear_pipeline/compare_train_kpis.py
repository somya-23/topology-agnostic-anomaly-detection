"""compare_train_kpis.py — Mean & std of each KPI across topologies (train/normal data only).

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python compare_train_kpis.py
"""

from pathlib import Path
import numpy as np

BASE    = Path("short_cpu_stress_experiment")
TOPOS   = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
STRESS  = 1

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
    z = np.load(BASE / f"{topo}_stress{STRESS}" / "train.npz")
    return z["cu"], z["du"]   # cu: (T,7)  du: (T,N,37)


def section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def compare_kpis(kpi_names, arrays_per_topo: list, topo_labels: list):
    """arrays_per_topo: list of 2-D arrays (T, n_kpi), one per topology."""
    col_w = 16
    header = f"  {'KPI':<22s}" + "".join(
        f"  {t:<{col_w}s}" for t in topo_labels
    )
    print(header)
    print("  " + "-" * (22 + (col_w + 2) * len(topo_labels)))

    for i, name in enumerate(kpi_names):
        row = f"  {name:<22s}"
        for arr in arrays_per_topo:
            m = arr[:, i].mean()
            s = arr[:, i].std()
            cell = f"{m:.4g}±{s:.4g}"
            row += f"  {cell:<{col_w}s}"
        print(row)


def main():
    cu_arrays, du_avg_arrays = [], []
    topo_labels = []
    n_du_counts = []

    for topo in TOPOS:
        cu, du = load(topo)
        n_du = du.shape[1]
        du_avg = du.mean(axis=1)   # average over DU instances → (T, 37)
        cu_arrays.append(cu)
        du_avg_arrays.append(du_avg)
        topo_labels.append(f"{topo}\n(N={n_du})")
        n_du_counts.append(n_du)
        print(f"Loaded {topo}: cu{cu.shape}  du{du.shape}")

    short_labels = [f"{t.split('_')[0]}(N={n})" for t, n in zip(TOPOS, n_du_counts)]

    section("CU KPIs — mean ± std across topologies  (train/normal only)")
    compare_kpis(CU_KPI_NAMES, cu_arrays, short_labels)

    section("DU KPIs — mean ± std (averaged over all DUs per topology)  (train/normal only)")
    compare_kpis(DU_KPI_NAMES, du_avg_arrays, short_labels)


if __name__ == "__main__":
    main()
