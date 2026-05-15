"""
Anomaly Detection: Isolation Forest + Autoencoder
==================================================
Train on CU0 topology (normal only) → Test on all topologies.

Binary labeling:
  anomaly=1 if focal DU is stressed (direct) OR parent CU is stressed (CU_fault)
  anomaly=0 if normal OR sibling DU stressed (sibling fault ≠ this pair's fault)

Evaluation: per topology × per stress type (T1/T2/T3)
Analysis:  IF score distributions, feature deviation, AE per-feature error
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_TOPOLOGY = 'srscu0'

AE_EPOCHS = 300
AE_BATCH = 128
AE_LR = 1e-3
AE_PATIENCE = 20

THRESHOLD_PERCS = [95, 99, 99.5]
STRESS_NAMES = {0: 'none', 1: 'T1_CPU', 2: 'T2_MEM', 3: 'T3_NET'}

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# DATA LOADING
# ============================================================
def load_pairs(data_dir):
    train = pd.read_csv(os.path.join(data_dir, 'train_pairs.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test_pairs.csv'))
    feat_cols = [c for c in train.columns
                 if c not in ['label', 'stress_type', 'cu_id', 'du_id', 'topology']]
    return train, test, feat_cols


def binary_label(labels):
    """Binary: anomaly if direct fault (1) or CU fault (3). NOT indirect (2)."""
    return ((labels == 1) | (labels == 3)).astype(int)

# ============================================================
# AUTOENCODER
# ============================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, X_val):
    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=AE_BATCH, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []

    for epoch in range(AE_EPOCHS):
        model.train()
        epoch_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = criterion(val_recon, val_tensor).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{AE_EPOCHS}  "
                  f"train={avg_loss:.6f}  val={val_loss:.6f}  "
                  f"patience={patience_counter}/{AE_PATIENCE}")

        if patience_counter >= AE_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor)
        val_mse = torch.mean((val_recon - val_tensor) ** 2, dim=1).cpu().numpy()

    return model, val_mse, train_losses


def ae_predict(model, X):
    tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()
        per_feat = (recon - tensor).pow(2).cpu().numpy()  # per-feature error
    return mse, per_feat

# ============================================================
# EVALUATION
# ============================================================
def evaluate_binary(scores, df_subset, threshold, model_name, topo_name, perc):
    """Binary evaluation: per topology × per stress type."""
    labels = df_subset['label'].values
    stress_types = df_subset['stress_type'].values
    y_true = binary_label(labels)
    y_pred = (scores > threshold).astype(int)

    results = []

    # Overall
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = 0.0

    results.append({
        'model': model_name, 'topology': topo_name,
        'stress': 'ALL', 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'TPR': tpr, 'FPR': fpr, 'AUROC': auroc, 'threshold_perc': perc,
    })

    # Per stress type (for anomaly samples only: direct + CU_fault)
    for st_val in [1, 2, 3]:
        # Anomaly samples of this stress type
        mask_anom = ((labels == 1) | (labels == 3)) & (stress_types == st_val)
        n_anom = mask_anom.sum()
        if n_anom == 0:
            continue
        n_detected = y_pred[mask_anom].sum()

        results.append({
            'model': model_name, 'topology': topo_name,
            'stress': STRESS_NAMES[st_val],
            'TP': n_detected, 'FP': '-', 'FN': n_anom - n_detected,
            'TN': '-', 'TPR': n_detected / n_anom, 'FPR': '-',
            'AUROC': '-', 'threshold_perc': perc, 'n_anom': n_anom,
        })

    # Indirect false alarm rate: when sibling is stressed, how often flagged?
    mask_indirect = (labels == 2)
    n_indirect = mask_indirect.sum()
    if n_indirect > 0:
        n_flagged = y_pred[mask_indirect].sum()
        results.append({
            'model': model_name, 'topology': topo_name,
            'stress': 'INDIRECT_FA', 'TP': '-', 'FP': n_flagged,
            'FN': '-', 'TN': n_indirect - n_flagged,
            'TPR': '-', 'FPR': n_flagged / n_indirect,
            'AUROC': '-', 'threshold_perc': perc, 'n_indirect': n_indirect,
        })

    return pd.DataFrame(results)

# ============================================================
# PLOTTING
# ============================================================
def plot_if_score_distributions(if_scores_dict, val_scores, out_dir):
    """Violin/box plots of IF scores: normal vs T1/T2/T3, per topology."""
    topos = sorted(if_scores_dict.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    thresh_99 = np.percentile(val_scores, 99)

    for idx, topo in enumerate(topos):
        ax = axes[idx]
        scores, df_sub = if_scores_dict[topo]
        labels = df_sub['label'].values
        stress = df_sub['stress_type'].values

        # Build categories
        cats = []
        vals = []

        # Normal (label=0)
        mask = labels == 0
        if mask.sum() > 0:
            # Sample for readability
            n_sample = min(500, mask.sum())
            idx_sample = np.random.choice(np.where(mask)[0], n_sample, replace=False)
            cats.extend(['Normal'] * n_sample)
            vals.extend(scores[idx_sample])

        # Direct faults by stress type
        for st, st_name in [(1, 'T1_CPU'), (2, 'T2_MEM'), (3, 'T3_NET')]:
            mask = (labels == 1) & (stress == st)
            if mask.sum() > 0:
                cats.extend([st_name] * mask.sum())
                vals.extend(scores[mask])

        # CU fault (combined)
        mask = labels == 3
        if mask.sum() > 0:
            cats.extend(['CU_fault'] * mask.sum())
            vals.extend(scores[mask])

        plot_df = pd.DataFrame({'Category': cats, 'IF Score': vals})

        color_map = {'Normal': '#4CAF50', 'T1_CPU': '#F44336',
                     'T2_MEM': '#FF9800', 'T3_NET': '#2196F3', 'CU_fault': '#9C27B0'}
        order = [c for c in ['Normal', 'T1_CPU', 'T2_MEM', 'T3_NET', 'CU_fault']
                 if c in plot_df['Category'].unique()]

        sns.boxplot(data=plot_df, x='Category', y='IF Score', order=order,
                    palette=color_map, ax=ax, fliersize=1)
        ax.axhline(thresh_99, color='red', linestyle='--', alpha=0.7, label='p99 threshold')

        is_seen = '(seen)' if TRAIN_TOPOLOGY in topo else '(UNSEEN)'
        ax.set_title(f'{topo} {is_seen}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Isolation Forest — Score Distributions by Fault Type',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'if_score_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_deviation(test, feat_cols, out_dir):
    """For each fault type, mean |z-score| of features → which features drive detection."""
    # Compute mean of normal data
    normal = test[test['label'] == 0]
    normal_mean = normal[feat_cols].mean().values
    normal_std = normal[feat_cols].std().values
    normal_std[normal_std < 1e-10] = 1  # avoid div by zero

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    stress_labels = {1: 'T1_CPU', 2: 'T2_MEM', 3: 'T3_NET'}
    colors = {1: '#F44336', 2: '#FF9800', 3: '#2196F3'}

    for ax_idx, (st_val, st_name) in enumerate(stress_labels.items()):
        ax = axes[ax_idx]
        # Direct fault samples of this stress type
        mask = (test['label'] == 1) & (test['stress_type'] == st_val)
        if mask.sum() == 0:
            continue
        anom_data = test[mask][feat_cols].values
        # Mean deviation in σ
        sigma_dev = np.abs((anom_data.mean(axis=0) - normal_mean) / normal_std)

        # Sort by deviation
        order = np.argsort(sigma_dev)[::-1]
        top_n = 15  # show top 15
        top_idx = order[:top_n]

        feat_names_short = [feat_cols[i].replace('focal_', 'f_').replace('sib_', 's_')
                            for i in top_idx]
        ax.barh(range(top_n), sigma_dev[top_idx], color=colors[st_val], alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_names_short, fontsize=9)
        ax.set_xlabel('Mean |σ-deviation| from normal')
        ax.set_title(f'{st_name} — Top Feature Deviations', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Feature Deviations During Faults (explains what IF detects)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'if_feature_deviation.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_if_detection_heatmap(results_df, out_dir):
    """Heatmap: all 6 topologies × 3 stress types, showing TPR @ p99."""
    mask = (
        (results_df['model'] == 'IsolationForest') &
        (results_df['threshold_perc'] == 99) &
        (~results_df['stress'].isin(['ALL', 'INDIRECT_FA']))
    )
    df = results_df[mask].copy()
    if len(df) == 0:
        return
    df['TPR'] = pd.to_numeric(df['TPR'], errors='coerce')

    pivot = df.pivot_table(values='TPR', index='stress', columns='topology', aggfunc='first')
    # Reorder columns
    col_order = ['srscu0->srsdu0', 'srscu0->srsdu1', 'srscu1->srsdu2',
                 'srscu2->srsdu3', 'srscu2->srsdu4', 'srscu2->srsdu5']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={'fontsize': 11})
    ax.set_title('Isolation Forest — Anomaly Detection Rate (TPR) @ 99th percentile\n'
                 '[Binary: anomaly = focal DU fault OR CU fault]',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Stress Type')
    ax.set_xlabel('Topology (CU→DU pair)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'if_detection_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_ae_diagnosis(model, X_train_cu0, X_val, test, feat_cols, val_mse, out_dir):
    """Diagnose WHY AE has high FPR: per-feature reconstruction error analysis."""
    # 1. MSE distribution: val normal vs test normal (CU0) vs test anomaly (CU0)
    test_cu0 = test[test['cu_id'] == TRAIN_TOPOLOGY]
    test_normal = test_cu0[test_cu0['label'] == 0]
    test_anom = test_cu0[binary_label(test_cu0['label'].values) == 1]

    X_test_normal = test_normal[feat_cols].values
    X_test_anom = test_anom[feat_cols].values

    mse_test_normal, pf_test_normal = ae_predict(model, X_test_normal)
    mse_test_anom, pf_test_anom = ae_predict(model, X_test_anom)

    thresh_99 = np.percentile(val_mse, 99)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: MSE distributions
    ax = axes[0]
    ax.hist(val_mse, bins=80, alpha=0.6, label=f'Val normal (n={len(val_mse)})',
            color='green', density=True)
    ax.hist(mse_test_normal, bins=80, alpha=0.6, label=f'Test normal (n={len(mse_test_normal)})',
            color='orange', density=True)
    ax.hist(mse_test_anom, bins=80, alpha=0.6, label=f'Test anomaly (n={len(mse_test_anom)})',
            color='red', density=True)
    ax.axvline(thresh_99, color='black', linestyle='--', linewidth=2, label=f'p99 threshold')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('AE: Why FPR is high on SAME topology\n'
                 'Test normal MSE >> Val normal MSE (distribution shift)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    val_above = (val_mse > thresh_99).mean() * 100
    test_n_above = (mse_test_normal > thresh_99).mean() * 100
    test_a_above = (mse_test_anom > thresh_99).mean() * 100
    ax.text(0.95, 0.95,
            f'Above threshold:\n  Val normal: {val_above:.1f}%\n  Test normal: {test_n_above:.1f}%\n  Test anomaly: {test_a_above:.1f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Per-feature reconstruction error (normal_val vs normal_test)
    ax = axes[1]
    _, pf_val = ae_predict(model, X_val)
    val_feat_err = pf_val.mean(axis=0)
    test_normal_feat_err = pf_test_normal.mean(axis=0)

    feat_names_short = [f.replace('focal_', 'f_').replace('sib_', 's_') for f in feat_cols]
    x_pos = np.arange(len(feat_cols))

    ax.bar(x_pos - 0.2, val_feat_err, 0.4, label='Val normal', color='green', alpha=0.7)
    ax.bar(x_pos + 0.2, test_normal_feat_err, 0.4, label='Test normal', color='orange', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feat_names_short, rotation=90, fontsize=7)
    ax.set_ylabel('Mean Reconstruction Error')
    ax.set_title('AE: Per-feature error (val normal vs test normal)\n'
                 'Features with higher test error → source of FPR')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('AutoEncoder Diagnosis — Same Topology (CU0)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ae_diagnosis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Print the features with biggest error increase
    print("\n  AE Diagnosis — features causing FPR (test/val error ratio):")
    ratio = test_normal_feat_err / (val_feat_err + 1e-10)
    order = np.argsort(ratio)[::-1]
    for i in order[:10]:
        print(f"    {feat_cols[i]:30s}  val={val_feat_err[i]:.6f}  test={test_normal_feat_err[i]:.6f}  ratio={ratio[i]:.2f}x")


def plot_ae_mse_per_topology(model, test, feat_cols, val_mse, out_dir):
    """MSE distribution per topology — normal vs anomaly."""
    thresh_99 = np.percentile(val_mse, 99)
    topos = sorted(test['topology'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, topo in enumerate(topos):
        ax = axes[idx]
        topo_df = test[test['topology'] == topo]
        labels = topo_df['label'].values
        X = topo_df[feat_cols].values
        mse, _ = ae_predict(model, X)

        y_bin = binary_label(labels)
        normal_mse = mse[y_bin == 0]
        anom_mse = mse[y_bin == 1]

        ax.hist(normal_mse, bins=80, alpha=0.6, label=f'Normal (n={len(normal_mse)})',
                color='steelblue', density=True)
        if len(anom_mse) > 0:
            ax.hist(anom_mse, bins=80, alpha=0.6, label=f'Anomaly (n={len(anom_mse)})',
                    color='salmon', density=True)
        ax.axvline(thresh_99, color='black', linestyle='--', linewidth=1.5, label='p99 threshold')

        fpr = (normal_mse > thresh_99).mean() * 100
        tpr = (anom_mse > thresh_99).mean() * 100 if len(anom_mse) > 0 else 0
        is_seen = '(seen)' if TRAIN_TOPOLOGY in topo else '(UNSEEN)'
        ax.set_title(f'{topo} {is_seen}\nFPR={fpr:.0f}%, TPR={tpr:.0f}%',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('AutoEncoder — MSE Distributions [Binary labeling]', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ae_mse_all_topos.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(if_scores_dict, ae_scores_dict, val_if, val_ae, out_dir):
    """ROC curves for both models, all topologies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (model_name, scores_dict) in zip(axes, [('IsolationForest', if_scores_dict),
                                                      ('AutoEncoder', ae_scores_dict)]):
        for topo in sorted(scores_dict.keys()):
            scores, df_sub = scores_dict[topo]
            y_true = binary_label(df_sub['label'].values)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
            roc_val = auc(fpr_arr, tpr_arr)
            is_seen = '(seen)' if TRAIN_TOPOLOGY in topo else '(UNSEEN)'
            ax.plot(fpr_arr, tpr_arr, linewidth=1.5,
                    label=f'{topo} {is_seen} AUC={roc_val:.3f}')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'{model_name} ROC Curves [Binary labeling]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curves_all.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_comparison(results_df, out_dir):
    """Side-by-side IF vs AE detection heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    col_order = ['srscu0->srsdu0', 'srscu0->srsdu1', 'srscu1->srsdu2',
                 'srscu2->srsdu3', 'srscu2->srsdu4', 'srscu2->srsdu5']

    for ax, model_name in zip(axes, ['IsolationForest', 'AutoEncoder']):
        mask = (
            (results_df['model'] == model_name) &
            (results_df['threshold_perc'] == 99) &
            (~results_df['stress'].isin(['ALL', 'INDIRECT_FA']))
        )
        df = results_df[mask].copy()
        if len(df) == 0:
            continue
        df['TPR'] = pd.to_numeric(df['TPR'], errors='coerce')
        pivot = df.pivot_table(values='TPR', index='stress', columns='topology', aggfunc='first')
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax, linewidths=0.5, annot_kws={'fontsize': 10})
        ax.set_title(f'{model_name} — Detection Rate @ p99', fontweight='bold')
        ax.set_ylabel('Stress Type')

    plt.suptitle('IF vs AE Comparison [Binary labeling]', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'if_vs_ae_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_fpr_comparison(results_df, out_dir):
    """Bar chart: FPR per topology for IF vs AE."""
    mask = (
        (results_df['stress'] == 'ALL') &
        (results_df['threshold_perc'] == 99)
    )
    df = results_df[mask].copy()
    df['FPR'] = pd.to_numeric(df['FPR'], errors='coerce')

    col_order = ['srscu0->srsdu0', 'srscu0->srsdu1', 'srscu1->srsdu2',
                 'srscu2->srsdu3', 'srscu2->srsdu4', 'srscu2->srsdu5']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(col_order))
    width = 0.35

    for i, model_name in enumerate(['IsolationForest', 'AutoEncoder']):
        fpr_vals = []
        for topo in col_order:
            row = df[(df['model'] == model_name) & (df['topology'] == topo)]
            val = pd.to_numeric(row['FPR'].values[0], errors='coerce') if len(row) > 0 else 0
            fpr_vals.append(val * 100)
        color = '#2196F3' if i == 0 else '#F44336'
        ax.bar(x + i * width - width/2, fpr_vals, width, label=model_name, color=color, alpha=0.8)

    ax.set_xticks(x)
    labels = [f'{t}\n{"(seen)" if TRAIN_TOPOLOGY in t else "(UNSEEN)"}' for t in col_order]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('False Positive Rate (%)')
    ax.set_title('FPR Comparison: IF vs AE @ 99th percentile [Binary labeling]', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar_group in ax.containers:
        ax.bar_label(bar_group, fmt='%.1f%%', fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fpr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pairs')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("TOPOLOGY-AGNOSTIC ANOMALY DETECTION")
    print("Binary labeling: anomaly = focal DU fault OR CU fault")
    print(f"Training on: {TRAIN_TOPOLOGY} (CU0→DU0, CU0→DU1)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load
    train, test, feat_cols = load_pairs(data_dir)
    print(f"\nData: train={len(train)}, test={len(test)}, features={len(feat_cols)}")

    # Training data: CU0 pairs only
    train_cu0 = train[train['cu_id'] == TRAIN_TOPOLOGY]
    X_train_all = train_cu0[feat_cols].values

    # 80/20 split for threshold calibration
    n_train = int(len(X_train_all) * 0.8)
    indices = np.random.permutation(len(X_train_all))
    X_train = X_train_all[indices[:n_train]]
    X_val = X_train_all[indices[n_train:]]
    print(f"CU0 training split: train={len(X_train)}, val={len(X_val)}")

    # Per-topology test sets
    test_by_topo = {}
    for topo in sorted(test['topology'].unique()):
        test_by_topo[topo] = test[test['topology'] == topo]

    all_results = []
    if_scores_dict = {}  # topo -> (scores, df)
    ae_scores_dict = {}

    # =========================================================
    # MODEL 1: ISOLATION FOREST
    # =========================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ISOLATION FOREST")
    print("=" * 70)

    iforest = IsolationForest(
        n_estimators=200, contamination=0.01,
        random_state=SEED, n_jobs=-1,
    )
    iforest.fit(X_train)
    if_val_scores = -iforest.score_samples(X_val)

    for topo, topo_df in test_by_topo.items():
        X_test = topo_df[feat_cols].values
        scores = -iforest.score_samples(X_test)
        if_scores_dict[topo] = (scores, topo_df)

        for perc in THRESHOLD_PERCS:
            thresh = np.percentile(if_val_scores, perc)
            res = evaluate_binary(scores, topo_df, thresh, 'IsolationForest', topo, perc)
            all_results.append(res)

    print("  IF training complete.")

    # =========================================================
    # MODEL 2: AUTOENCODER
    # =========================================================
    print("\n" + "=" * 70)
    print("MODEL 2: AUTOENCODER")
    print(f"  Architecture: {len(feat_cols)} → 64 → 32 → 16 → 32 → 64 → {len(feat_cols)}")
    print("=" * 70)

    ae_model, val_mse, train_losses = train_autoencoder(X_train, X_val)
    print(f"\n  Val MSE: mean={val_mse.mean():.6f}, p95={np.percentile(val_mse, 95):.6f}, "
          f"p99={np.percentile(val_mse, 99):.6f}")

    for topo, topo_df in test_by_topo.items():
        X_test = topo_df[feat_cols].values
        ae_mse, _ = ae_predict(ae_model, X_test)
        ae_scores_dict[topo] = (ae_mse, topo_df)

        for perc in THRESHOLD_PERCS:
            thresh = np.percentile(val_mse, perc)
            res = evaluate_binary(ae_mse, topo_df, thresh, 'AutoEncoder', topo, perc)
            all_results.append(res)

    # =========================================================
    # RESULTS
    # =========================================================
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(out_dir, 'detection_results.csv'), index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS @ 99th percentile (Binary labeling)")
    print("=" * 70)

    topo_order = ['srscu0->srsdu0', 'srscu0->srsdu1', 'srscu1->srsdu2',
                  'srscu2->srsdu3', 'srscu2->srsdu4', 'srscu2->srsdu5']

    for model_name in ['IsolationForest', 'AutoEncoder']:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        # Detection rates
        print(f"\n  {'Topology':<22} {'T1_CPU':>8} {'T2_MEM':>8} {'T3_NET':>8} {'FPR':>8} {'AUROC':>8}")
        print(f"  {'-'*64}")

        for topo in topo_order:
            is_seen = '*' if TRAIN_TOPOLOGY in topo else ' '

            # Get FPR and AUROC
            mask_all = (
                (results_df['model'] == model_name) &
                (results_df['threshold_perc'] == 99) &
                (results_df['topology'] == topo) &
                (results_df['stress'] == 'ALL')
            )
            all_row = results_df[mask_all]
            fpr = all_row['FPR'].values[0] if len(all_row) > 0 else -1
            auroc = all_row['AUROC'].values[0] if len(all_row) > 0 else -1

            vals = {}
            for st in ['T1_CPU', 'T2_MEM', 'T3_NET']:
                mask = (
                    (results_df['model'] == model_name) &
                    (results_df['threshold_perc'] == 99) &
                    (results_df['topology'] == topo) &
                    (results_df['stress'] == st)
                )
                rows = results_df[mask]
                vals[st] = f"{rows['TPR'].values[0]*100:>6.1f}%" if len(rows) > 0 else "    N/A"

            seen_label = "(seen)" if is_seen == '*' else "(unsn)"
            print(f"  {topo} {seen_label:<6} "
                  f"{vals['T1_CPU']:>8} {vals['T2_MEM']:>8} {vals['T3_NET']:>8} "
                  f"{fpr*100:>6.1f}% {auroc:>7.3f}")

        # Indirect false alarm rate
        print(f"\n  Indirect false alarm (sibling stressed → pair flagged):")
        for topo in topo_order:
            mask = (
                (results_df['model'] == model_name) &
                (results_df['threshold_perc'] == 99) &
                (results_df['topology'] == topo) &
                (results_df['stress'] == 'INDIRECT_FA')
            )
            rows = results_df[mask]
            if len(rows) > 0:
                fpr_ind = rows['FPR'].values[0]
                print(f"    {topo}: {fpr_ind*100:.1f}%")

    # =========================================================
    # PLOTS
    # =========================================================
    print("\n\nGenerating plots...")

    # 1. IF score distributions
    print("  1. IF score distributions...")
    plot_if_score_distributions(if_scores_dict, if_val_scores, out_dir)

    # 2. Feature deviation analysis
    print("  2. Feature deviation analysis...")
    plot_feature_deviation(test, feat_cols, out_dir)

    # 3. IF detection heatmap
    print("  3. IF detection heatmap...")
    plot_if_detection_heatmap(results_df, out_dir)

    # 4. AE training loss
    print("  4. AE training loss...")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('AutoEncoder Training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ae_training_loss.png'), dpi=150)
    plt.close()

    # 5. AE diagnosis
    print("  5. AE diagnosis (per-feature error)...")
    plot_ae_diagnosis(ae_model, X_train, X_val, test, feat_cols, val_mse, out_dir)

    # 6. AE MSE per topology
    print("  6. AE MSE distributions per topology...")
    plot_ae_mse_per_topology(ae_model, test, feat_cols, val_mse, out_dir)

    # 7. ROC curves
    print("  7. ROC curves...")
    plot_roc_curves(if_scores_dict, ae_scores_dict, if_val_scores, val_mse, out_dir)

    # 8. IF vs AE comparison heatmap
    print("  8. IF vs AE comparison...")
    plot_summary_comparison(results_df, out_dir)

    # 9. FPR comparison bar chart
    print("  9. FPR comparison...")
    plot_fpr_comparison(results_df, out_dir)

    print(f"\nAll results saved to: {out_dir}/")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
