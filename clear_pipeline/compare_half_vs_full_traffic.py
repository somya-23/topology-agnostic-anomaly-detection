"""compare_half_vs_full_traffic.py — Compare train (half traffic) vs test (full traffic).

Both files contain only normal data (no anomalies).
Shows mean ± std side by side so you can see how each KPI shifts with traffic level.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python compare_half_vs_full_traffic.py
"""

from pathlib import Path
import numpy as np

FOLDER = Path("half_traffic_experiment") / "cu1_du2_stress1"

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
DU_KPI_NAMES = CU_KPI_NAMES + PCI_KPI_NAMES


def print_table(entity_label, kpi_names, arr_half, arr_full):
    """arr_half, arr_full: (T, n_kpi)"""
    print(f"\n  {'KPI':<26s}  {'half_mean':>12s}  {'half_std':>10s}  "
          f"{'full_mean':>12s}  {'full_std':>10s}  {'ratio(full/half)':>16s}  {'shift_d':>8s}")
    print("  " + "-" * 110)
    for i, name in enumerate(kpi_names):
        hm, hs = arr_half[:, i].mean(), arr_half[:, i].std()
        fm, fs = arr_full[:, i].mean(), arr_full[:, i].std()
        ratio  = fm / (hm + 1e-12)
        # shift_d: how many half-traffic stds does the full-traffic mean differ?
        d = (fm - hm) / (hs + 1e-12)
        flag = "***" if abs(d) > 3 else ("** " if abs(d) > 1.5 else ("*  " if abs(d) > 0.8 else "   "))
        print(f"  {entity_label}:{name:<24s}  {hm:>12.4g}  {hs:>10.4g}  "
              f"{fm:>12.4g}  {fs:>10.4g}  {ratio:>16.3f}  {d:>+7.2f} {flag}")


def main():
    tr = np.load(FOLDER / "train.npz")
    te = np.load(FOLDER / "test.npz")

    cu_half = tr["cu"]          # (T, 7)
    du_half = tr["du"][:, 0, :] # (T, 37)  — cu1_du2 has only 1 DU
    cu_full = te["cu"]
    du_full = te["du"][:, 0, :]

    print(f"\nFolder : {FOLDER}")
    print(f"Train (half traffic): {cu_half.shape[0]} rows")
    print(f"Test  (full traffic): {cu_full.shape[0]} rows")

    print(f"\n{'='*80}")
    print("  CU KPIs — half vs full traffic (normal only, no anomalies)")
    print(f"{'='*80}")
    print_table("CU", CU_KPI_NAMES, cu_half, cu_full)

    print(f"\n{'='*80}")
    print("  DU KPIs — half vs full traffic (normal only, no anomalies)")
    print(f"{'='*80}")
    print_table("DU", DU_KPI_NAMES, du_half, du_full)

    # ── KPIs most affected by traffic change ──────────────────────────────────
    print(f"\n{'='*80}")
    print("  TOP 10 KPIs most shifted by traffic level  (by |shift_d|)")
    print(f"{'='*80}")
    all_names  = [f"CU:{n}" for n in CU_KPI_NAMES] + [f"DU:{n}" for n in DU_KPI_NAMES]
    all_half   = np.concatenate([cu_half, du_half], axis=1)
    all_full   = np.concatenate([cu_full, du_full], axis=1)
    shifts = (all_full.mean(0) - all_half.mean(0)) / (all_half.std(0) + 1e-12)
    ranked = sorted(zip(all_names, shifts.tolist()), key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'KPI':<30s}  {'shift_d':>8s}  note")
    print("  " + "-" * 60)
    for name, d in ranked[:10]:
        note = "↑ higher at full traffic" if d > 0 else "↓ lower at full traffic"
        print(f"  {name:<30s}  {d:>+8.2f}  {note}")


if __name__ == "__main__":
    main()
