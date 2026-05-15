"""
CU-DU Pair Transformation for Topology-Agnostic Anomaly Detection
=================================================================
Transforms wide CSV (359 cols, 1 row/sec) into fixed-size CU-DU pair vectors.

Each original row → 6 pair rows:
  CU0-DU0, CU0-DU1, CU1-DU2, CU2-DU3, CU2-DU4, CU2-DU5

Pair vector (14 features):
  [CU_metrics(5) | focal_DU_metrics(8) | num_siblings(1)]

Binary labeling:
  0 = Normal   (no fault affecting this pair)
  1 = Anomaly  (focal DU stressed OR parent CU stressed)

  Key: sibling DU stressed → this pair is NORMAL (only the stressed DU's pair is anomalous)

Design decisions:
  - NO sibling features (caused indirect false alarms: 77-90%)
  - NO diff(): data already has irate() from Prometheus
  - CU NET_TX normalized by num_DUs (proven linear: 1.3 MB/s per DU)
  - Curated 14 features (not 359 — 128 were zero-var, 78 near-duplicates)
  - Vectorized numpy (seconds, not row-by-row Python loops)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# ============================================================
# TOPOLOGY DEFINITION
# ============================================================
TOPOLOGY = {
    'srscu0': ['srsdu0', 'srsdu1'],       # 2-DU topology
    'srscu1': ['srsdu2'],                   # 1-DU topology (no siblings)
    'srscu2': ['srsdu3', 'srsdu4', 'srsdu5'],  # 3-DU topology
}

PCI_MAP = {
    'srsdu0': 'PCI-1', 'srsdu1': 'PCI-2', 'srsdu2': 'PCI-3',
    'srsdu3': 'PCI-4', 'srsdu4': 'PCI-5', 'srsdu5': 'PCI-6',
}

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
# We select 5 infra metrics per DU, 5 per CU, 3 radio per DU
# These are the non-redundant, non-zero-variance features that
# have proven discriminative power for T1/T2/T3 faults.

DU_INFRA_FEATS = ['du_cpu_pct', 'du_mem_bytes', 'du_net_tx', 'du_fs_write', 'du_fs_read']
CU_INFRA_FEATS = ['cu_cpu_pct', 'cu_mem_bytes', 'cu_net_tx_per_du', 'cu_net_rx', 'cu_fs_read']
RADIO_FEATS = ['radio_ul_brate', 'radio_bsr', 'radio_ul_nof_ok']

# Column matching rules: (keyword_in_column, startswith_filter)
DU_COL_RULES = {
    'du_cpu_pct':   ('cpu_user_seconds', 'sum  (irate'),
    'du_mem_bytes': ('memory_usage_bytes', 'sum(container_memory'),
    'du_net_tx':    ('network_transmit_bytes', None),
    'du_fs_write':  ('fs_writes_bytes', None),
    'du_fs_read':   ('fs_reads_bytes', None),
}

CU_COL_RULES = {
    'cu_cpu_pct':       ('cpu_user_seconds', 'sum  (irate'),
    'cu_mem_bytes':     ('memory_usage_bytes', 'sum(container_memory'),
    'cu_net_tx_per_du': ('network_transmit_bytes', None),  # will be divided by num_DUs
    'cu_net_rx':        ('network_receive_bytes', None),
    'cu_fs_read':       ('fs_reads_bytes', None),
}

RADIO_COL_NAMES = {
    'radio_ul_brate':  'ul_brate',
    'radio_bsr':       'bsr',
    'radio_ul_nof_ok': 'ul_nof_ok',
}


def _find_col(df, container, keyword, startswith=None):
    """Find a Prometheus column matching container name and keyword."""
    for c in df.columns:
        if f'name="{container}"' not in c:
            continue
        if keyword not in c:
            continue
        if startswith and not c.startswith(startswith):
            continue
        return c
    return None


def build_column_map(df):
    """
    Build mapping: (container, clean_metric_name) -> actual CSV column name.
    Validates that all expected columns exist.
    """
    col_map = {}
    missing = []

    for du_idx in range(6):
        du = f'srsdu{du_idx}'
        # DU infra
        for metric, (kw, sw) in DU_COL_RULES.items():
            col = _find_col(df, du, kw, sw)
            if col is None:
                missing.append(f"{du}/{metric}")
            else:
                col_map[(du, metric)] = col
        # PCI radio
        pci = PCI_MAP[du]
        for clean_name, raw_suffix in RADIO_COL_NAMES.items():
            col_name = f'{pci}_RNTI-4601_{raw_suffix}'
            if col_name not in df.columns:
                missing.append(f"{du}/{clean_name} ({col_name})")
            else:
                col_map[(du, clean_name)] = col_name

    for cu_idx in range(3):
        cu = f'srscu{cu_idx}'
        for metric, (kw, sw) in CU_COL_RULES.items():
            col = _find_col(df, cu, kw, sw)
            if col is None:
                missing.append(f"{cu}/{metric}")
            else:
                col_map[(cu, metric)] = col

    # Label columns
    for container in [f'srsdu{i}' for i in range(6)] + [f'srscu{i}' for i in range(3)]:
        for suffix in ['stressType', 'stepStress']:
            label_col = f'{container}_{suffix}'
            if label_col in df.columns:
                col_map[(container, suffix)] = label_col

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return col_map


def transform_to_pairs(df, col_map):
    """
    Vectorized transformation: wide CSV → CU-DU pair rows.

    For each CU-DU pair, constructs:
      [CU_feats(5) | focal_DU_feats(8) | sibling_mean_feats(8) | num_siblings(1)]

    Returns DataFrame with 22 feature columns + metadata columns.
    """
    n_rows = len(df)
    all_pair_dfs = []

    # Feature column names for the output
    feat_names = (
        CU_INFRA_FEATS
        + ['focal_' + f for f in DU_INFRA_FEATS]
        + ['focal_' + f for f in RADIO_FEATS]
        + ['sib_' + f for f in DU_INFRA_FEATS]
        + ['sib_' + f for f in RADIO_FEATS]
        + ['num_siblings']
    )

    for cu, dus in TOPOLOGY.items():
        num_dus = len(dus)

        # --- Extract CU features (vectorized) ---
        cu_matrix = np.column_stack([
            df[col_map[(cu, m)]].values for m in CU_COL_RULES
        ])  # shape: (n_rows, 5)

        # Normalize CU NET_TX: divide by num_DUs
        # cu_net_tx_per_du is at index 2 in CU_COL_RULES
        net_tx_idx = list(CU_COL_RULES.keys()).index('cu_net_tx_per_du')
        cu_matrix[:, net_tx_idx] = cu_matrix[:, net_tx_idx] / num_dus

        # CU stress
        cu_stress_key = (cu, 'stressType')
        cu_stress = df[col_map[cu_stress_key]].values.astype(int) if cu_stress_key in col_map else np.zeros(n_rows, dtype=int)

        # --- Extract ALL DU features for this CU group ---
        du_infra = {}   # du -> (n_rows, 5)
        du_radio = {}   # du -> (n_rows, 3)
        du_stress = {}  # du -> (n_rows,)

        for du in dus:
            du_infra[du] = np.column_stack([
                df[col_map[(du, m)]].values for m in DU_COL_RULES
            ])
            du_radio[du] = np.column_stack([
                df[col_map[(du, rm)]].values for rm in RADIO_COL_NAMES
            ])
            stress_key = (du, 'stressType')
            du_stress[du] = df[col_map[stress_key]].values.astype(int) if stress_key in col_map else np.zeros(n_rows, dtype=int)

        # --- Build pair for each focal DU ---
        for focal_du in dus:
            siblings = [d for d in dus if d != focal_du]
            num_siblings = len(siblings)

            # Focal DU features
            focal_infra = du_infra[focal_du]      # (n, 5)
            focal_radio = du_radio[focal_du]      # (n, 3)

            # Sibling mean features
            if num_siblings > 0:
                sib_infra = np.mean([du_infra[s] for s in siblings], axis=0)
                sib_radio = np.mean([du_radio[s] for s in siblings], axis=0)
            else:
                sib_infra = np.zeros_like(focal_infra)
                sib_radio = np.zeros_like(focal_radio)

            # num_siblings column
            ns_col = np.full((n_rows, 1), num_siblings, dtype=float)

            # Stack all features
            pair_matrix = np.hstack([
                cu_matrix,       # 5
                focal_infra,     # 5
                focal_radio,     # 3
                sib_infra,       # 5
                sib_radio,       # 3
                ns_col,          # 1
            ])  # Total: 22

            # --- Labels ---
            focal_st = du_stress[focal_du]
            sib_stresses = [du_stress[s] for s in siblings]

            # 4-level label
            labels = np.zeros(n_rows, dtype=int)
            # Indirect: any sibling stressed
            if sib_stresses:
                any_sib = np.any(np.column_stack(sib_stresses) > 0, axis=1)
                labels[any_sib] = 2
            # CU fault
            labels[cu_stress > 0] = 3
            # Direct fault (highest priority)
            labels[focal_st > 0] = 1

            # Stress detail (which T1/T2/T3)
            stress_detail = np.zeros(n_rows, dtype=int)
            if sib_stresses:
                sib_max = np.max(np.column_stack(sib_stresses), axis=1)
                stress_detail[sib_max > 0] = sib_max[sib_max > 0]
            stress_detail[cu_stress > 0] = cu_stress[cu_stress > 0]
            stress_detail[focal_st > 0] = focal_st[focal_st > 0]

            # Build DataFrame
            pair_df = pd.DataFrame(pair_matrix, columns=feat_names)
            pair_df['label'] = labels
            pair_df['stress_type'] = stress_detail
            pair_df['cu_id'] = cu
            pair_df['du_id'] = focal_du
            pair_df['topology'] = f'{cu}->{focal_du}'

            all_pair_dfs.append(pair_df)

    return pd.concat(all_pair_dfs, ignore_index=True)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    print("=" * 60)
    print("CU-DU PAIR TRANSFORMATION")
    print("=" * 60)

    # Load
    print("\nLoading CSVs...")
    df_train = pd.read_csv(os.path.join(data_dir, 'train_normal_final.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test_anomaly_final.csv'))
    print(f"  Train: {df_train.shape[0]} rows × {df_train.shape[1]} cols")
    print(f"  Test:  {df_test.shape[0]} rows × {df_test.shape[1]} cols")

    # Build column maps
    print("\nMapping Prometheus columns → clean feature names...")
    col_map_train = build_column_map(df_train)
    col_map_test = build_column_map(df_test)
    print(f"  Mapped {len(col_map_train)} columns")

    # Transform
    print("\nTransforming to CU-DU pairs (vectorized)...")
    train_pairs = transform_to_pairs(df_train, col_map_train)
    test_pairs = transform_to_pairs(df_test, col_map_test)
    print(f"  Train pairs: {train_pairs.shape}")
    print(f"  Test pairs:  {test_pairs.shape}")

    # Verify training labels
    assert train_pairs['label'].max() == 0, "Training data should be all normal!"
    print("  Training data: all label=0 (normal) ✓")

    # Label distribution
    print("\n  Test label distribution:")
    label_names = {0: 'Normal', 1: 'Direct', 2: 'Indirect', 3: 'CU_fault'}
    for lv, ln in label_names.items():
        count = (test_pairs['label'] == lv).sum()
        pct = count / len(test_pairs) * 100
        print(f"    {lv} ({ln:>10}): {count:>6} ({pct:.1f}%)")

    # Per-topology breakdown
    print("\n  Per-topology breakdown:")
    for topo in sorted(test_pairs['topology'].unique()):
        s = test_pairs[test_pairs['topology'] == topo]
        n0 = (s['label']==0).sum()
        n1 = (s['label']==1).sum()
        n2 = (s['label']==2).sum()
        n3 = (s['label']==3).sum()
        print(f"    {topo}: {len(s)} rows  "
              f"[normal={n0}, direct={n1}, indirect={n2}, cu_fault={n3}]")

    # === NORMALIZATION ===
    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    feat_cols = [c for c in train_pairs.columns
                 if c not in ['label', 'stress_type', 'cu_id', 'du_id', 'topology']]

    print(f"  Feature columns: {len(feat_cols)}")
    print(f"  Fitting StandardScaler on {len(train_pairs)} training pairs...")

    scaler = StandardScaler()
    train_scaled = train_pairs.copy()
    test_scaled = test_pairs.copy()

    train_scaled[feat_cols] = scaler.fit_transform(train_pairs[feat_cols])
    test_scaled[feat_cols] = scaler.transform(test_pairs[feat_cols])

    # === SAVE ===
    out_dir = os.path.join(data_dir, 'pairs')
    os.makedirs(out_dir, exist_ok=True)

    train_out = os.path.join(out_dir, 'train_pairs.csv')
    test_out = os.path.join(out_dir, 'test_pairs.csv')
    scaler_out = os.path.join(out_dir, 'scaler.pkl')

    train_scaled.to_csv(train_out, index=False)
    test_scaled.to_csv(test_out, index=False)
    with open(scaler_out, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n  Saved:")
    print(f"    {train_out}")
    print(f"    {test_out}")
    print(f"    {scaler_out}")

    # === SANITY CHECK ===
    print("\n" + "=" * 60)
    print("SANITY CHECK: Feature means (scaled) by label")
    print("=" * 60)

    key_feats = ['focal_du_cpu_pct', 'focal_du_mem_bytes', 'focal_du_net_tx',
                 'focal_du_fs_write', 'focal_du_fs_read',
                 'sib_du_cpu_pct', 'sib_du_net_tx',
                 'cu_net_tx_per_du', 'focal_radio_ul_brate']

    for feat in key_feats:
        if feat not in test_scaled.columns:
            continue
        print(f"\n  {feat}:")
        for lv, ln in label_names.items():
            s = test_scaled[test_scaled['label'] == lv]
            if len(s) > 0:
                print(f"    {ln:>10}: mean={s[feat].mean():+.3f}  std={s[feat].std():.3f}")

    print("\n" + "=" * 60)
    print("TRANSFORMATION COMPLETE")
    print("=" * 60)
    return train_scaled, test_scaled, scaler


if __name__ == '__main__':
    main()
