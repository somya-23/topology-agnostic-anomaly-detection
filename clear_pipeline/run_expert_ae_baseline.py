#!/usr/bin/env python3
"""
Expert_Context_AE (ICC_workshop_2026) as a ZERO-SHOT baseline on TopoAR data.

This runs the ICC repo's `ExpertLSTMAutoEncoder` on OUR data, using the *exact*
feature engineering from clear_pipeline/run_experiment.py:

  * CU net traffic normalised by topology size:  cu net_tx, net_rx  /=  N_DU
  * the same CU_FEAT_SLICE / DU_FEAT_SLICE column selections,
  * the same Prometheus irate-glitch forward-fill imputation,
  * the same derived relational features (net_diff, net_ratio),
  * the same type-shared RobustScaler v0 (fit on pooled train topologies).

The only difference from run_experiment.py is the *model*: instead of TopoAR's
type-shared attention predictor, each (CU, DU_i) pair at each timestep is flattened
to a single feature row  [cu(7) || du_i(30)] = 37 dims  and fed to the ICC LSTM
autoencoder. Detection = reconstruction MSE over a window vs a percentile threshold
calibrated on held-out normal windows (the AE's native scheme).

ZERO-SHOT leave-one-out: train on the NORMAL streams of two topologies, test on the
held-out topology's labelled test stream. Repeated for all three holdouts so the
output is directly comparable to run_experiment.py's LOO summary.

Configurable for ANY stress type exactly like run_all_experiments.py:
    python run_expert_ae_baseline.py --base-dir CU_CPU_bidir_STRESS --stress 1   # CPU (default)
    python run_expert_ae_baseline.py --base-dir CU_MEM_bidir_STRESS --stress 2   # MEM
    python run_expert_ae_baseline.py --base-dir DU_NET_bidir_STRESS --stress 3   # NET on DU
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- Make the ICC model and our preprocessing importable -----------------------
THIS_DIR = Path(__file__).resolve().parent                 # .../clear_pipeline
TOPOAR_ROOT = THIS_DIR.parent                              # .../topoar_gpu_run
ICC_AE_DIR = Path("/home/somya/workspace/ICC_workshop_2026/anomalyDetection")
sys.path.insert(0, str(TOPOAR_ROOT))                      # for `src.preprocess`
sys.path.insert(0, str(ICC_AE_DIR))                       # for `models.expert_Context_AE`

from src.preprocess import fit_bundle, transform_stream            # noqa: E402
from models.expert_Context_AE import ExpertLSTMAutoEncoder         # noqa: E402

# ===============================================================================
# CONFIG  — feature recipe copied verbatim from run_experiment.py
# ===============================================================================
ALL_TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
STRESS_NAMES = {1: "CPU", 2: "MEM", 3: "NET"}

# Feature slices (run_experiment.py lines 92-96). Raw cu has 7 cols, raw du has 37.
CU_FEAT_SLICE = [0, 1, 2, 5, 6]                       # cpu, mem_pct, mem_bytes, net_tx, net_rx
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6,
                 7, 8, 9, 10, 11, 12, 13, 14,
                 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]   # 28 cols
# Irate/counter indices in the POST-SLICE arrays (for glitch imputation).
CU_IRATE_IDX = [0, 3, 4]            # cpu, net_tx, net_rx
DU_IRATE_IDX = [0, 3, 4, 5]         # cpu, fs_writes, net_tx, net_rx

PREPROCESS_VERSION = "v0"           # RobustScaler on raw values (no delta/arcsinh)
CU_ZV_IDX, DU_ZV_IDX = [], []       # no zero-variance features remain after slicing
IMPUTE = True                       # forward-fill Prometheus irate glitch zeros
USE_NDIV = False
# divide CU net_tx/net_rx by N_DU (topology-size norm).
                                    # Toggle with --no-ndiv; mirrors run_experiment vs
                                    # run_experiment_no_ndiv.py.

# AE / detection hyperparameters
WINDOW_LEN = 64                     # temporal context (matches run_experiment.WINDOW_LEN)
BATCH_SIZE = 256
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 10
LR = 1e-3
THRESHOLD_PCT = 99.9                # percentile of normal-val MSE (ICC AE default)
VAL_FRAC = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================================================================
# DATA  — load + slice_features identical to run_experiment.py
# ===============================================================================
def load_npz(base_dir: Path, topo: str, stress_type: int, split: str) -> dict:
    p = base_dir / f"{topo}_stress{stress_type}" / f"{split}.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx: list, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:                       # CU: (T, dim)
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:                                   # DU: (T, N, dim)
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(z: dict):
    """CU/DU slicing + topology-size normalisation + derived features.

    Mirrors run_experiment.slice_features exactly. Returns:
        cu  (T, 7), du (T, N, 30), block_id (T,)
    """
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    # --- TOPOLOGY NORMALISATION: CU net traffic scales ~linearly with DU count.
    if USE_NDIV:
        cu[:, 5] = cu[:, 5] / N_DU      # net_tx / N
        cu[:, 6] = cu[:, 6] / N_DU      # net_rx / N

    cu = cu[:, CU_FEAT_SLICE]           # (T, 5)
    du = du[:, :, DU_FEAT_SLICE]        # (T, N, 28)

    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
        du = impute_cpu_glitch(du, DU_IRATE_IDX)

    # Derived relational features (post-imputation).
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]                       # post-slice CU: 3=net_tx, 4=net_rx
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)   # (T, 7)

    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]           # post-slice DU: 4=net_tx, 5=net_rx
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)  # (T, N, 30)

    return cu, du, z["block_id"].astype(np.int64)


# ===============================================================================
# PAIR FLATTENING + WINDOWING  (the AE-specific bit)
# ===============================================================================
def pair_rows(cu_s, du_s, block_id, cu_stress=None, du_stress=None):
    """Flatten a scaled stream into one feature row per (CU, DU_i) pair per timestep.

    cu_s (T, 7), du_s (T, N, 30). For each DU instance i, emit a stream
        feat = [cu_s || du_s[:, i, :]]      (T, 37)
        lab  = (cu_stress | du_stress[:, i])  (T,)   [zeros if labels are None]
    Yields (feat, lab, block_id, du_idx). block_id is shared, so windowing stays
    block-isolated within each pair.
    """
    T, N = cu_s.shape[0], du_s.shape[1]
    if cu_stress is None:
        cu_stress = np.zeros(T, dtype=np.int64)
    for i in range(N):
        feat = np.concatenate([cu_s, du_s[:, i, :]], axis=1).astype(np.float32)   # (T, 37)
        du_lab_i = du_stress[:, i] if du_stress is not None else np.zeros(T, dtype=np.int64)
        lab = np.maximum(cu_stress, du_lab_i).astype(np.int64)
        yield feat, lab, block_id, i


@torch.no_grad()
def recon_mse_batched(model, X, batch_size=512):
    """Per-window reconstruction MSE, computed in chunks to bound GPU memory.

    X: (n, window, n_features) numpy or CPU tensor. Returns (n,) numpy.
    """
    model.eval()
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(DEVICE)
        rb = model(xb)
        out.append(torch.mean((rb - xb) ** 2, dim=(1, 2)).cpu())
    return torch.cat(out).numpy() if out else np.empty(0, np.float32)


def make_windows(feat, lab, block_id, window):
    """Block-isolated sliding windows (stride 1). Window label = last-step label.

    Also returns the *raw* stress-type code at the last step (for per-type recall).
    """
    X, y, ytype = [], [], []
    for b in np.unique(block_id):
        m = block_id == b
        f, l = feat[m], lab[m]
        if len(f) <= window:
            continue
        for k in range(len(f) - window + 1):
            X.append(f[k:k + window])
            y.append(int(l[k + window - 1] > 0))     # binary: anomalous or not
            ytype.append(int(l[k + window - 1]))      # raw stress-type code 0/1/2/3
    if not X:
        return (np.empty((0, window, feat.shape[1]), np.float32),
                np.empty(0, np.int64), np.empty(0, np.int64))
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(ytype, np.int64)


# ===============================================================================
# ONE LEAVE-ONE-OUT RUN
# ===============================================================================
def run_one(base_dir: Path, stress_type: int, test_topo: str):
    train_topos = [t for t in ALL_TOPOS if t != test_topo]
    print("\n" + "=" * 70)
    print(f"ZERO-SHOT LOO  |  stress={STRESS_NAMES.get(stress_type, stress_type)}  "
          f"|  test=[{test_topo}]  train={train_topos}")
    print("=" * 70)

    # --- 1. Load + slice all streams ------------------------------------------
    train_raw = []
    for t in train_topos:
        cu, du, bid = slice_features(load_npz(base_dir, t, stress_type, "train"))
        train_raw.append({"cu": cu, "du": du, "block_id": bid})
        print(f"  train {t:16s} cu{cu.shape} du{du.shape} (N={du.shape[1]})")

    ztest = load_npz(base_dir, test_topo, stress_type, "test")
    cu_te, du_te, bid_te = slice_features(ztest)
    cu_stress_te = ztest["cu_stress"].astype(np.int64)
    du_stress_te = ztest["du_stress"].astype(np.int64)

    # --- 2. Type-shared RobustScaler v0, fit on pooled train (run_experiment) --
    bundle = fit_bundle(train_raw, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    # --- 3. Build NORMAL training windows from train topos --------------------
    Xtr = []
    for raw in train_raw:
        cu_s, du_s, _, kept_bid = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        for feat, lab, bid, _ in pair_rows(cu_s, du_s, kept_bid):     # train = all normal
            Xw, _, _ = make_windows(feat, lab, bid, WINDOW_LEN)
            if len(Xw):
                Xtr.append(Xw)
    Xtr = np.concatenate(Xtr, axis=0)
    print(f"  -> normal train windows: {Xtr.shape}")

    # --- 4. Build TEST windows (labelled) from held-out topo ------------------
    cu_s_te, du_s_te, kept_mask_te, kept_bid_te = transform_stream(bundle, cu_te, du_te, bid_te)
    # Subset labels to rows that survived preprocessing (no-op for v0; matters if delta is on).
    cu_stress_te = cu_stress_te[kept_mask_te]
    du_stress_te = du_stress_te[kept_mask_te]
    Xte, yte, ytype = [], [], []
    for feat, lab, bid, _ in pair_rows(cu_s_te, du_s_te, kept_bid_te,
                                       cu_stress=cu_stress_te,
                                       du_stress=du_stress_te):
        Xw, yw, ytw = make_windows(feat, lab, bid, WINDOW_LEN)
        if len(Xw):
            Xte.append(Xw); yte.append(yw); ytype.append(ytw)
    Xte = np.concatenate(Xte, axis=0)
    yte = np.concatenate(yte, axis=0)
    ytype = np.concatenate(ytype, axis=0)
    print(f"  -> test windows: {Xte.shape}  (normal={int((yte==0).sum())}, anomaly={int((yte>0).sum())})")

    # --- 5. Train AE on normal windows ----------------------------------------
    n_features = Xtr.shape[2]            # 37
    Xtr_fit, Xtr_val = train_test_split(Xtr, test_size=VAL_FRAC, random_state=SEED)
    tr_t = torch.tensor(Xtr_fit, dtype=torch.float32)   # kept on CPU; batches moved to GPU
    val_t = torch.tensor(Xtr_val, dtype=torch.float32)  # inference is batched (recon_mse_batched)
    te_t = torch.tensor(Xte, dtype=torch.float32)

    set_seed(SEED)
    model = ExpertLSTMAutoEncoder(WINDOW_LEN, n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loader = DataLoader(TensorDataset(tr_t), batch_size=BATCH_SIZE, shuffle=True)
    print(f"  model: input_features={n_features} window={WINDOW_LEN} "
          f"params={sum(p.numel() for p in model.parameters())} device={DEVICE}")

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            opt.step()
        vloss = float(recon_mse_batched(model, val_t, batch_size=BATCH_SIZE).mean())
        sched.step(vloss)
        if vloss < best_val:
            best_val, patience = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if (epoch + 1) % 25 == 0:
            print(f"    epoch {epoch+1:3d}/{NUM_EPOCHS}  val_mse={vloss:.6f}  patience={patience}/{EARLY_STOPPING_PATIENCE}")
        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"    early stop @ epoch {epoch+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 6. Threshold on normal-val MSE, evaluate on test ---------------------
    val_mse = recon_mse_batched(model, val_t, batch_size=BATCH_SIZE)
    te_mse = recon_mse_batched(model, te_t, batch_size=BATCH_SIZE)
    threshold = float(np.percentile(val_mse, THRESHOLD_PCT))
    y_pred = (te_mse > threshold).astype(int)

    print(f"\n  threshold (p{THRESHOLD_PCT} of normal-val MSE) = {threshold:.6f}")
    print(classification_report(yte, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))

    tn, fp, fn, tp = confusion_matrix(yte, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = f1_score(yte, y_pred, zero_division=0)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn} | Recall={recall:.4f} "
          f"Specificity={spec:.4f} Precision={prec:.4f} F1={f1:.4f}")

    # Per stress-type recall (only the injected type will have samples here)
    print("  Per-stress-type recall:")
    for code, name in STRESS_NAMES.items():
        mask = ytype == code
        n = int(mask.sum())
        if n:
            det = int(y_pred[mask].sum())
            print(f"    {name:3s}: {det}/{n} = {det / n:.4f}")

    return {"test_topo": test_topo, "recall": recall, "specificity": spec,
            "precision": prec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold": threshold}


def main():
    ap = argparse.ArgumentParser(description="Expert_Context_AE zero-shot LOO baseline on TopoAR data")
    ap.add_argument("--base-dir", default="CU_CPU_bidir_STRESS",
                    help="stress dataset dir (e.g. CU_CPU_bidir_STRESS, DU_NET_bidir_STRESS)")
    ap.add_argument("--stress", type=int, default=1, choices=[1, 2, 3],
                    help="stress type: 1=CPU 2=MEM 3=NET (must match --base-dir)")
    ap.add_argument("--test-topo", default=None,
                    help="run a single holdout instead of all 3 LOO splits")
    ap.add_argument("--no-ndiv", action="store_true",
                    help="disable CU net_tx/net_rx ÷ N_DU normalisation (ablation)")
    args = ap.parse_args()

    global USE_NDIV
    USE_NDIV = not args.no_ndiv

    base_dir = (THIS_DIR / args.base_dir).resolve()
    assert base_dir.exists(), f"base dir not found: {base_dir}"
    print(f"BASE_DIR={base_dir.name}  STRESS={STRESS_NAMES[args.stress]}  "
          f"WINDOW={WINDOW_LEN}  DEVICE={DEVICE}  CU_ndiv={'ON' if USE_NDIV else 'OFF'}")

    topos = [args.test_topo] if args.test_topo else ALL_TOPOS
    results = [run_one(base_dir, args.stress, t) for t in topos]

    print("\n" + "=" * 70)
    print(f"SUMMARY  —  Expert_Context_AE zero-shot LOO  ({base_dir.name}, "
          f"{STRESS_NAMES[args.stress]})")
    print("=" * 70)
    print(f"{'test_topo':18s} {'Recall':>8s} {'Spec':>8s} {'Prec':>8s} {'F1':>8s}  "
          f"{'TP':>5s}{'FP':>5s}{'FN':>5s}{'TN':>6s}")
    for r in results:
        print(f"{r['test_topo']:18s} {r['recall']:8.4f} {r['specificity']:8.4f} "
              f"{r['precision']:8.4f} {r['f1']:8.4f}  "
              f"{r['tp']:5d}{r['fp']:5d}{r['fn']:5d}{r['tn']:6d}")
    if len(results) > 1:
        import statistics as st
        print(f"{'MEAN':18s} {st.mean(r['recall'] for r in results):8.4f} "
              f"{st.mean(r['specificity'] for r in results):8.4f} "
              f"{st.mean(r['precision'] for r in results):8.4f} "
              f"{st.mean(r['f1'] for r in results):8.4f}")


if __name__ == "__main__":
    main()
