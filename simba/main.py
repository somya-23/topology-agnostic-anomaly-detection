import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import json
import copy
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix, f1_score
import wandb
from tqdm.auto import tqdm

from model import Simba

# -----------------------------------------------------------------------------
# Data Utils (Fast Vectorized + View-based Windowing)
# -----------------------------------------------------------------------------
class WindowIndexDataset(Dataset):
    def __init__(self, S: int):
        self.S = int(S)
    def __len__(self):
        return self.S
    def __getitem__(self, i):
        return int(i)

class WindowCollate:
    def __init__(self, Xw: torch.Tensor, yw: torch.Tensor):
        self.Xw = Xw
        self.yw = yw
    def __call__(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return self.Xw.index_select(0, idx), self.yw.index_select(0, idx)

def make_split_views(X: torch.Tensor, y: torch.Tensor, t0: int, t1: int, L: int):
    # Create windows [S, L, N, F] without copying memory
    Xs = X[t0:t1].contiguous()
    ys = y[t0:t1].contiguous()
    Xw = Xs.unfold(0, L, 1).permute(0, 3, 1, 2).contiguous()
    yw = ys[L:].contiguous() # Next step labels
    return Xw, yw

def prepare_data(data_file_path, seq_length):
    print(f"Loading {data_file_path}...")
    df = pd.read_csv(data_file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(['Time', 'BS_ID']).reset_index(drop=True)

    # Dynamically detect feature columns (exclude metadata columns)
    meta_cols = ['Time', 'BS_ID', 'FaultType', 'BS_X', 'BS_Y', 'BS_Z']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Ensure we have at least some features
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in CSV. Expected columns other than Time, BS_ID, FaultType, BS_X, BS_Y, BS_Z")
    
    print(f"Using {len(feature_cols)} feature columns: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Using {len(feature_cols)} feature columns: {feature_cols}")
    
    # Dense Tensors
    times = df['Time'].unique()
    nodes = df['BS_ID'].unique()
    T, N, F = len(times), len(nodes), len(feature_cols)
    
    # Map to dense indices
    t_map = pd.Series(np.arange(T), index=times)
    n_map = pd.Series(np.arange(N), index=nodes)
    
    X = np.zeros((T, N, F), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.int64)
    
    ti = t_map.loc[df['Time']].values
    ni = n_map.loc[df['BS_ID']].values
    
    for k, col in enumerate(feature_cols):
        X[ti, ni, k] = df[col].values
    
    y[ti, ni] = df['FaultType'].fillna(0).values.astype(np.int64)
    
    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)
    
    # Split Boundaries (50% Train, 25% Val, 25% Test per Paper)
    train_end = int(T * 0.50)
    val_end = int(T * 0.75)
    
    # Scaling (Fit on Train only)
    scaler = StandardScaler()
    X_train_flat = X[:train_end].reshape(-1, F)
    scaler.fit(X_train_flat)
    
    X_all = scaler.transform(X.reshape(-1, F)).reshape(T, N, F)
    X_t = torch.from_numpy(X_all).float()
    y_t = torch.from_numpy(y).long()
    
    return X_t, y_t, train_end, val_end, scaler

# -----------------------------------------------------------------------------
# Training Utils
# -----------------------------------------------------------------------------
def set_seeds(seed, deterministic=True, benchmark=False):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

def compute_weights_eq2(y_tensor):
    """
    Implements Eq. 2 Weights: w_y = Class Ratio
    Paper implies weighted cross entropy to handle imbalance.
    """
    labels = y_tensor.numpy().flatten()
    counts = np.bincount(labels)
    total = len(labels)
    # Inverse frequency weights
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, loader, optimizer, loss_fn, device, num_classes):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Flatten for loss
        loss = loss_fn(logits.view(-1, num_classes), y.view(-1))
        loss.backward()
        
        # Gradient Clipping (Standard practice for Transformers/GNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

# ==========================================
# Event-Level Metrics (Section 10.2)
# ==========================================
def compute_event_metrics(logits, targets, threshold=0.5):
    """
    Compute event-level metrics:
    (a) Fault existence detection
    (b) Top-k localization recall
    (c) Type accuracy on detected nodes
    """
    # logits: (B, N, 3), targets: (B, N)
    probs = torch.softmax(logits, dim=-1)  # (B, N, 3)
    fault_scores = 1 - probs[:, :, 0]  # Per-node fault probability (B, N)
    
    # (a) Fault existence detection
    network_fault_score = fault_scores.max(dim=1)[0]  # (B,) - max fault score per window
    pred_fault_exists = (network_fault_score > threshold).cpu().numpy()
    true_fault_exists = (targets != 0).any(dim=1).cpu().numpy()
    fault_detection_acc = (pred_fault_exists == true_fault_exists).mean()
    
    # (b) Top-k localization recall
    top_k = 1
    B, N = targets.shape
    top_k_nodes = fault_scores.topk(k=min(top_k, N), dim=1)[1].cpu().numpy()  # (B, k) - node indices
    
    recall_at_k = 0.0
    fault_windows = np.where(true_fault_exists)[0]
    if len(fault_windows) > 0:
        hits = 0
        for b_idx in fault_windows:
            true_faulty_mask = (targets[b_idx] != 0).cpu().numpy()
            true_faulty_node_indices = np.where(true_faulty_mask)[0]
            if len(true_faulty_node_indices) > 0:
                if np.any(np.isin(true_faulty_node_indices, top_k_nodes[b_idx])):
                    hits += 1
        recall_at_k = hits / len(fault_windows) if len(fault_windows) > 0 else 0.0
    
    # (c) Type accuracy on detected nodes
    detected_fault_windows = np.where(pred_fault_exists & true_fault_exists)[0]
    type_acc = 0.0
    if len(detected_fault_windows) > 0:
        correct_types = 0
        total_detected = 0
        for b_idx in detected_fault_windows:
            pred_nodes = top_k_nodes[b_idx]
            true_faulty_mask = (targets[b_idx] != 0).cpu().numpy()
            true_faulty_node_indices = np.where(true_faulty_mask)[0]
            for pn in pred_nodes:
                if pn in true_faulty_node_indices:
                    pred_type = logits[b_idx, pn].argmax().item()
                    true_type = targets[b_idx, pn].item()
                    if pred_type == true_type:
                        correct_types += 1
                    total_detected += 1
        type_acc = correct_types / total_detected if total_detected > 0 else 0.0
    
    return {
        'fault_detection_acc': fault_detection_acc,
        'localization_recall@1': recall_at_k,
        'type_accuracy': type_acc
    }

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes, return_report=False, compute_event_metrics_flag=False):
    model.eval()
    total_loss = 0
    preds_all, truth_all = [], []
    all_logits = []
    all_targets_tensor = []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, num_classes), y.view(-1))
        total_loss += loss.item()
        
        preds = logits.argmax(dim=-1).cpu().numpy().flatten()
        truth = y.cpu().numpy().flatten()
        preds_all.extend(preds)
        truth_all.extend(truth)
        
        if compute_event_metrics_flag:
            all_logits.append(logits.cpu())
            all_targets_tensor.append(y.cpu())
        
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(truth_all, preds_all)
    
    # Weighted metrics (class-weighted by support)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(truth_all, preds_all, average='weighted', zero_division=0)
    
    # Macro metrics (unweighted mean across classes)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(truth_all, preds_all, average='macro', zero_division=0)
    
    # Fault-only macro F1
    fault_f1_m = f1_score(truth_all, preds_all, average='macro', labels=[1, 2], zero_division=0)
    
    # Per-class metrics
    per_class_metrics = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        truth_all, preds_all, labels=[0, 1, 2], average=None, zero_division=0
    )
    class_names = ['Normal', 'Interference', 'Power']
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Event-level metrics
    event_metrics = {}
    if compute_event_metrics_flag and len(all_logits) > 0:
        logits_concat = torch.cat(all_logits, dim=0)
        targets_concat = torch.cat(all_targets_tensor, dim=0)
        event_metrics = compute_event_metrics(logits_concat, targets_concat, threshold=0.5)
    
    if return_report:
        print("\n" + "="*40)
        print("       FINAL TEST REPORT       ")
        print("="*40)
        print(classification_report(truth_all, preds_all, target_names=class_names, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(truth_all, preds_all, labels=[0, 1, 2]))
        print("\nPer-Class Metrics:")
        for name, metrics in per_class_metrics.items():
            print(f"  {name:12s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Support={metrics['support']}")
        if event_metrics:
            print("\nEvent-Level Metrics:")
            print(f"  Fault Detection Acc: {event_metrics['fault_detection_acc']:.4f}")
            print(f"  Localization Recall@1: {event_metrics['localization_recall@1']:.4f}")
            print(f"  Type Accuracy: {event_metrics['type_accuracy']:.4f}")
        print("="*40)
    
    return {
        'loss': avg_loss, 'acc': acc, 'p_w': p_w, 'r_w': r_w, 'f1_w': f1_w,
        'p_m': p_m, 'r_m': r_m, 'f1_m': f1_m, 'fault_f1_m': fault_f1_m,
        'per_class': per_class_metrics, 'event_metrics': event_metrics
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Table III Hyperparameters
    parser.add_argument("--data-file", default="data.csv")
    parser.add_argument("--batch-size", type=int, default=2000, help="Table III")
    parser.add_argument("--lr", type=float, default=0.0003, help="Table III")
    parser.add_argument("--seq-len", type=int, default=5, help="Table III")
    parser.add_argument("--gc-filters", type=int, default=32, help="Table III")
    parser.add_argument("--gl-alpha", type=float, default=0.5, help="Table III")
    parser.add_argument("--epochs", type=int, default=50) # Not specified in table, 50 is reasonable
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_path", default="best_simba.pth")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument('--multi_seed', action='store_true', help='Run with multiple seeds for statistical significance')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds to run')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Specific seeds to use')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Use deterministic CUDA operations')
    parser.add_argument('--benchmark', action='store_true', default=False, help='Use CUDA benchmark mode for faster training')
    
    args = parser.parse_args()
    
    # Helper function for single run
    def train_and_test_single(seed=None, ablation_config=None):
        # Create a copy of args for this run
        run_args = copy.deepcopy(args)
        
        if seed is not None:
            set_seeds(seed, deterministic=run_args.deterministic, benchmark=run_args.benchmark)
            print(f"Using seed: {seed}, deterministic={run_args.deterministic}, benchmark={run_args.benchmark}")
        
        # Apply ablation config if provided
        if ablation_config:
            for key, value in ablation_config.items():
                setattr(run_args, key, value)
        
        run_name = "simba-normal"
        if seed is not None:
            run_name += f"-seed{seed}"
        if ablation_config:
            run_name += f"-{ablation_config.get('name', 'abl')}"
        
        wandb.init(project="simba-ran", config=vars(run_args), name=run_name, reinit=True)
    
        # 1. Prepare Data
        X_t, y_t, train_end, val_end, scaler = prepare_data(run_args.data_file, run_args.seq_len)
    
        # Create Views
        L = run_args.seq_len
        # Train: 0 -> 50%
        X_tr, y_tr = make_split_views(X_t, y_t, 0, train_end, L)
        # Val: 50% -> 75%
        X_va, y_va = make_split_views(X_t, y_t, train_end - L, val_end, L)
        # Test: 75% -> 100%
        X_te, y_te = make_split_views(X_t, y_t, val_end - L, X_t.shape[0], L)
        
        # Loaders
        train_loader = DataLoader(WindowIndexDataset(len(y_tr)), batch_size=run_args.batch_size, 
                                  shuffle=True, collate_fn=WindowCollate(X_tr, y_tr))
        val_loader = DataLoader(WindowIndexDataset(len(y_va)), batch_size=run_args.batch_size, 
                                collate_fn=WindowCollate(X_va, y_va))
        test_loader = DataLoader(WindowIndexDataset(len(y_te)), batch_size=run_args.batch_size, 
                                 collate_fn=WindowCollate(X_te, y_te))
        
        num_nodes = X_t.shape[1]
        num_features = X_t.shape[2]
        num_classes = int(y_t.max().item()) + 1
        
        print(f"Nodes: {num_nodes}, Features: {num_features}, Classes: {num_classes}")
        
        # 2. Init Model (Table III Params)
        model = Simba(
            num_nodes=num_nodes,
            in_features=num_features,
            out_classes=num_classes,
            gl_emb_dim=10, # Standard GNN embedding
            top_k=7,       # Table III: "Subgraph size for mix-hop: 7"
            gc_hops=2,
            gc_channels=run_args.gc_filters,
            tf_d_model=128,
            tf_heads=4,
            tf_layers=2
        ).to(run_args.device)
        
        # 3. Loss & Optimizer
        # Eq. 2 Weighted Cross Entropy
        class_weights = compute_weights_eq2(y_tr).to(run_args.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=run_args.lr)
        
        # 4. Training Loop
        trigger_times = 0
        best_f1 = 0
        save_path = run_args.model_path if seed is None else run_args.model_path.replace('.pth', f'_seed{seed}.pth')
        
        for epoch in range(run_args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, run_args.device, num_classes)
            val_results = evaluate(model, val_loader, loss_fn, run_args.device, num_classes, compute_event_metrics_flag=False)
            val_loss = val_results['loss']
            val_acc = val_results['acc']
            val_f1_w = val_results['f1_w']
            val_f1_m = val_results['f1_m']
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
                  f"Val F1 (weighted): {val_f1_w:.4f} | Val F1 (macro): {val_f1_m:.4f}")
            
            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_weighted": val_f1_w,
                "val_f1_macro": val_f1_m,
                "val_precision_weighted": val_results['p_w'],
                "val_precision_macro": val_results['p_m'],
                "val_recall_weighted": val_results['r_w'],
                "val_recall_macro": val_results['r_m'],
                "val_acc": val_acc
            }
            # Add per-class metrics
            for class_name, metrics in val_results['per_class'].items():
                log_dict[f"val_{class_name.lower()}_precision"] = metrics['precision']
                log_dict[f"val_{class_name.lower()}_recall"] = metrics['recall']
                log_dict[f"val_{class_name.lower()}_f1"] = metrics['f1']
            wandb.log(log_dict)
            
            # Checkpoint & Early Stopping Logic
            if val_f1_m > best_f1:
                best_f1 = val_f1_m
                torch.save(model.state_dict(), save_path)
                trigger_times = 0
                print("  -> Saved new best model")
            else:
                trigger_times += 1
                print(f"  -> No improvement (Patience: {trigger_times}/{run_args.patience})")
                
                if trigger_times >= run_args.patience:
                    print("Early stopping triggered!")
                    break
                
        # 5. Final Test
        model.load_state_dict(torch.load(save_path))
        test_results = evaluate(model, test_loader, loss_fn, run_args.device, num_classes, 
                                return_report=True, compute_event_metrics_flag=True)
        
        print("\n--- Test Results ---")
        print(f"Accuracy:  {test_results['acc']:.4f}")
        print(f"\nWeighted Metrics (class-weighted by support):")
        print(f"  F1 Score:  {test_results['f1_w']:.4f}")
        print(f"  Precision: {test_results['p_w']:.4f}")
        print(f"  Recall:    {test_results['r_w']:.4f}")
        print(f"\nMacro Metrics (unweighted mean across classes):")
        print(f"  F1 Score:  {test_results['f1_m']:.4f}")
        print(f"  Precision: {test_results['p_m']:.4f}")
        print(f"  Recall:    {test_results['r_m']:.4f}")
        
        log_dict = {
            "test_f1_weighted": test_results['f1_w'],
            "test_f1_macro": test_results['f1_m'],
            "test_precision_weighted": test_results['p_w'],
            "test_precision_macro": test_results['p_m'],
            "test_recall_weighted": test_results['r_w'],
            "test_recall_macro": test_results['r_m'],
            "test_acc": test_results['acc']
        }
        # Per-class metrics
        for class_name, metrics in test_results['per_class'].items():
            log_dict[f"test_{class_name.lower()}_precision"] = metrics['precision']
            log_dict[f"test_{class_name.lower()}_recall"] = metrics['recall']
            log_dict[f"test_{class_name.lower()}_f1"] = metrics['f1']
            log_dict[f"test_{class_name.lower()}_support"] = metrics['support']
        # Event-level metrics
        if test_results['event_metrics']:
            for key, value in test_results['event_metrics'].items():
                log_dict[f"test_{key}"] = value
        wandb.log(log_dict)
        wandb.finish()
        
        return test_results
    
    # Main execution logic
    if args.ablation:
        # Run ablation study
        print(f"\n{'='*60}")
        print("ABLATION STUDY")
        print(f"{'='*60}\n")
        
        ablations = [
            {'name': 'Baseline', 'config': {}},
            {'name': 'SeqLen_3', 'config': {'seq_len': 3}},
            {'name': 'SeqLen_10', 'config': {'seq_len': 10}},
            {'name': 'GC_Channels_16', 'config': {'gc_filters': 16}},
            {'name': 'GC_Channels_64', 'config': {'gc_filters': 64}},
        ]
        
        results = []
        for ablation in ablations:
            print(f"\n{'='*60}")
            print(f"Running: {ablation['name']}")
            print(f"{'='*60}\n")
            result = train_and_test_single(seed=42, ablation_config=ablation['config'])
            if result:
                results.append({
                    'name': ablation['name'],
                    'config': ablation['config'],
                    'macro_f1': result['f1_m'],
                    'fault_f1_m': result.get('fault_f1_m', 0.0)
                })
        
        print(f"\n{'='*60}")
        print("ABLATION RESULTS")
        print(f"{'='*60}\n")
        print(f"{'Configuration':<25s} | {'Macro-F1':<10s} | {'Fault-F1':<10s}")
        print("-" * 50)
        for r in results:
            print(f"{r['name']:<25s} | {r['macro_f1']:<10.4f} | {r['fault_f1_m']:<10.4f}")
        
        ablation_file = args.model_path.replace('.pth', '_ablation.json')
        with open(ablation_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAblation results saved to: {ablation_file}")
        
    elif args.multi_seed:
        # Run multiple seeds
        seeds = args.seeds if args.seeds else [42, 123, 456, 789, 999][:args.num_seeds]
        print(f"\n{'='*60}")
        print(f"Running {len(seeds)} seeds for statistical significance")
        print(f"{'='*60}\n")
        
        all_results = []
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Seed: {seed}")
            print(f"{'='*60}\n")
            result = train_and_test_single(seed=seed)
            if result:
                all_results.append(result)
        
        if len(all_results) >= 2:
            print(f"\n{'='*60}")
            print("STATISTICAL SUMMARY (Mean ± Std)")
            print(f"{'='*60}\n")
            
            metrics_to_aggregate = ['f1_m', 'f1_w', 'acc', 'fault_f1_m']
            summary = {}
            for metric in metrics_to_aggregate:
                values = [r[metric] for r in all_results if metric in r]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    summary[metric] = {'mean': mean_val, 'std': std_val, 'values': values}
                    print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
            
            print("\nPer-Class Metrics (Mean ± Std):")
            class_names = ['Normal', 'Interference', 'Power']
            for class_name in class_names:
                for metric in ['precision', 'recall', 'f1']:
                    values = [r['per_class'][class_name][metric] for r in all_results]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {class_name:12s} {metric:10s}: {mean_val:.4f} ± {std_val:.4f}")
            
            if all_results[0]['event_metrics']:
                print("\nEvent-Level Metrics (Mean ± Std):")
                for key in all_results[0]['event_metrics'].keys():
                    values = [r['event_metrics'][key] for r in all_results]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {key:25s}: {mean_val:.4f} ± {std_val:.4f}")
            
            summary_file = args.model_path.replace('.pth', '_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")
    else:
        # Single run
        train_and_test_single(seed=args.seed)