"""run_simba_oran.py — Supervised Simba baseline, cross-topology LOO.

WHAT IT DOES (parallel to clear_pipeline/run_experiment.py)
-----------------------------------------------------------
  Leave-one-out across topologies: hold out one topology for test, train on the
  others, loop all splits, print a summary table and append per-entity metrics to
  a CSV. Same preprocessing as TopoAR (RobustScaler v0 + feature slicing + /N
  topology normalization + Prometheus glitch imputation) so the ONLY differences
  between this baseline and TopoAR are (a) the model and (b) supervised vs
  unsupervised. That isolates "is the spatio-temporal classifier better than the
  reconstruction autoencoder, given the same data and preprocessing?".

KEY DIFFERENCE FROM TopoAR — and why it is unavoidable
------------------------------------------------------
  TopoAR is UNSUPERVISED: it trains on train.npz (NORMAL only) and flags anomalies
  by reconstruction error. Simba is a SUPERVISED classifier — it must SEE labeled
  stress examples to learn the stress class. train.npz has no stress and no labels,
  so a supervised model cannot learn from it.

  Therefore, for held-out topology X, Simba trains on the LABELED streams
  (test.npz, which contains normal+stress rows with cu_stress/du_stress) of the
  OTHER topologies, plus their train.npz as extra normal examples. It is tested on
  X's test.npz. X is still never seen in training, so cross-topology generalization
  is the property under test — but note in the thesis that Simba is given labels
  TopoAR never receives. That asymmetry favors the supervised baseline; if TopoAR
  still wins, the result is strong.

PROTOCOL PER LOO SPLIT
----------------------
  [1] Fit RobustScaler (v0) on the TRAIN topologies' train.npz normal streams
      (identical to TopoAR's bundle fit).
  [2] Build supervised training windows from each train topology:
        - train.npz  → all-normal windows (label 0)
        - test.npz   → labeled windows (label = stress == STRESS_TYPE)
      Windows are block-pure (never bridge a stress-state change), so each window
      has one unambiguous per-node label.
  [3] Train SimbaORAN with class-weighted cross-entropy; early-stop on val F1.
  [4] Build test windows from X's test.npz, predict per node.
  [5] Report per-entity (CU, each DU, ANY) precision/recall/F1; append to CSV.

USAGE
-----
    cd topoar_gpu_run/simba
    python run_simba_oran.py
  Change BASE_DIR / STRESS_TYPE for a different stress family (CPU/MEM/NET),
  exactly like run_experiment.py.
"""

import os
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from sklearn.metrics import precision_recall_fscore_support

# Reuse TopoAR's preprocessing + N-homogeneous batch sampler (identical pipeline).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import fit_bundle, transform_stream          # noqa: E402
from dataset import MultiTopologyBatchSampler                 # noqa: E402

from simba_model_for_ORAN import SimbaORAN                     # noqa: E402

# =============================================================================
# USER INPUTS — mirror run_experiment.py
# =============================================================================
ALL_TOPOS    = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
RUN_ALL_LOO  = True

# Point BASE_DIR / STRESS_TYPE at the stress family you want (same dirs as TopoAR).
BASE_DIR     = Path("../clear_pipeline/DU_CPU_bidir_STRESS")
STRESS_TYPE  = 1                              # 1=CPU | 2=MEM | 3=NET (must match dir)
STRESS_NAMES = {1: "CPU", 2: "MEM", 3: "NET"}

# Feature slices — IDENTICAL to run_experiment.py for a fair comparison.
CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6,
                 7, 8, 9, 10, 11, 12, 13, 14,
                 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]
CU_IRATE_IDX  = [0, 3, 4]
DU_IRATE_IDX  = [0, 3, 4, 5]
PREPROCESS_VERSION = "v0"
CU_ZV_IDX, DU_ZV_IDX = [], []
IMPUTE = True

# Model / training hyperparameters
WINDOW_LEN = 32
STRIDE     = 2
D_MODEL    = 64
GC_CHANNELS = 32
GC_HOPS    = 2
TF_HEADS   = 4
TF_LAYERS  = 2
TF_HIDDEN  = 128
NUM_CLASSES = 2          # binary: normal vs this stress type
BATCH_SIZE = 256
EPOCHS     = 60
PATIENCE   = 8
LR         = 5e-4
VAL_FRAC   = 0.1
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DATA HELPERS  (slice_features / impute copied verbatim from run_experiment.py)
# =============================================================================

def load_npz(topo: str, split: str) -> dict:
    p = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / f"{split}.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(z: dict):
    """Identical to run_experiment.py: slice, /N topology-normalize, impute, derive."""
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
        du = impute_cpu_glitch(du, DU_IRATE_IDX)

    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)

    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)

    block_id = z["block_id"].astype(np.int64)
    return cu, du, block_id


# =============================================================================
# WINDOWING — block-pure windows with per-node labels
# =============================================================================

def make_labeled_windows(cu_s, du_s, block_id, cu_lbl, du_lbl, L, stride):
    """Slice block-pure windows. Each window's label = state at its last timestep.

    Returns dict of np arrays:
      cu_w : (S, L, cu_dim)   du_w : (S, L, N, du_dim)
      cu_y : (S,)             du_y : (S, N)
    """
    T = len(cu_s)
    starts = []
    s = 0
    while s + L <= T:
        seg = block_id[s:s + L]
        if (seg == seg[0]).all():
            starts.append(s)
            s += stride
        else:
            s += int(np.argmax(seg != seg[0]))
    if not starts:
        return None
    starts = np.array(starts, dtype=np.int64)
    ends = starts + L
    cu_w = np.stack([cu_s[a:b] for a, b in zip(starts, ends)]).astype(np.float32)
    du_w = np.stack([du_s[a:b] for a, b in zip(starts, ends)]).astype(np.float32)
    cu_y = cu_lbl[ends - 1].astype(np.int64)
    du_y = du_lbl[ends - 1].astype(np.int64)
    return {"cu": cu_w, "du": du_w, "cu_y": cu_y, "du_y": du_y}


class LabeledWindowDataset(Dataset):
    def __init__(self, w: dict):
        self.cu = torch.from_numpy(w["cu"])
        self.du = torch.from_numpy(w["du"])
        self.cu_y = torch.from_numpy(w["cu_y"])
        self.du_y = torch.from_numpy(w["du_y"])

    def __len__(self):
        return self.cu.shape[0]

    def __getitem__(self, i):
        return {"cu": self.cu[i], "du": self.du[i],
                "cu_y": self.cu_y[i], "du_y": self.du_y[i]}


def collate(items):
    return {
        "cu":   torch.stack([it["cu"] for it in items], 0),     # (B, L, cu_dim)
        "du":   torch.stack([it["du"] for it in items], 0),     # (B, L, N, du_dim)
        "cu_y": torch.stack([it["cu_y"] for it in items], 0),   # (B,)
        "du_y": torch.stack([it["du_y"] for it in items], 0),   # (B, N)
    }


def node_logits_and_labels(logits, cu_y, du_y):
    """Flatten (CU node 0) + (DU nodes 1..N) into pooled (rows, C) logits + (rows,) labels."""
    B, M, C = logits.shape
    lab = torch.cat([cu_y.unsqueeze(1), du_y], dim=1)   # (B, M)
    return logits.reshape(B * M, C), lab.reshape(B * M)


# =============================================================================
# STREAM ASSEMBLY
# =============================================================================

def build_train_windows(bundle, topo):
    """Windows for one TRAIN topology: normal (train.npz) + labeled (test.npz)."""
    out = []
    # normal stream → label 0 everywhere
    zt = load_npz(topo, "train")
    cu, du, bid = slice_features(zt)
    cu_s, du_s, kept, kbid = transform_stream(bundle, cu, du, bid)
    cu_lbl = np.zeros(len(cu_s), dtype=np.int64)
    du_lbl = np.zeros((len(cu_s), du_s.shape[1]), dtype=np.int64)
    w = make_labeled_windows(cu_s, du_s, kbid, cu_lbl, du_lbl, WINDOW_LEN, STRIDE)
    if w:
        out.append(w)
    # labeled stream → real stress labels
    zx = load_npz(topo, "test")
    cu, du, bid = slice_features(zx)
    cu_s, du_s, kept, kbid = transform_stream(bundle, cu, du, bid)
    cu_lbl = (zx["cu_stress"][kept] == STRESS_TYPE).astype(np.int64)
    du_lbl = (zx["du_stress"][kept] == STRESS_TYPE).astype(np.int64)
    w = make_labeled_windows(cu_s, du_s, kbid, cu_lbl, du_lbl, WINDOW_LEN, STRIDE)
    if w:
        out.append(w)
    return out


def build_test_windows(bundle, topo):
    zx = load_npz(topo, "test")
    cu, du, bid = slice_features(zx)
    cu_s, du_s, kept, kbid = transform_stream(bundle, cu, du, bid)
    cu_lbl = (zx["cu_stress"][kept] == STRESS_TYPE).astype(np.int64)
    du_lbl = (zx["du_stress"][kept] == STRESS_TYPE).astype(np.int64)
    return make_labeled_windows(cu_s, du_s, kbid, cu_lbl, du_lbl, WINDOW_LEN, 1)


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_model(train_window_sets, cu_dim, du_dim):
    """train_window_sets: list of window dicts (one per stream, possibly mixed N)."""
    rng = np.random.RandomState(SEED)
    train_subsets, val_subsets, train_lens, val_lens = [], [], [], []
    pooled_labels = []
    for w in train_window_sets:
        ds = LabeledWindowDataset(w)
        n = len(ds)
        perm = rng.permutation(n)
        n_val = max(1, int(round(VAL_FRAC * n)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        train_subsets.append(Subset(ds, tr_idx))
        val_subsets.append(Subset(ds, val_idx))
        train_lens.append(len(tr_idx))
        val_lens.append(len(val_idx))
        pooled_labels.append(w["cu_y"])
        pooled_labels.append(w["du_y"].reshape(-1))

    train_loader = DataLoader(
        ConcatDataset(train_subsets),
        batch_sampler=MultiTopologyBatchSampler(train_lens, BATCH_SIZE, shuffle=True, seed=SEED),
        collate_fn=collate)
    val_loader = DataLoader(
        ConcatDataset(val_subsets),
        batch_sampler=MultiTopologyBatchSampler(val_lens, BATCH_SIZE, shuffle=False, seed=SEED),
        collate_fn=collate)

    # Inverse-frequency class weights (pooled over all nodes).
    labels = np.concatenate(pooled_labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = labels.size / (NUM_CLASSES * np.clip(counts, 1, None))
    class_w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print(f"  class counts={counts.tolist()}  weights={weights.round(3).tolist()}")

    torch.manual_seed(SEED)
    model = SimbaORAN(cu_dim, du_dim, num_classes=NUM_CLASSES, d_model=D_MODEL,
                      gc_channels=GC_CHANNELS, gc_hops=GC_HOPS,
                      tf_heads=TF_HEADS, tf_layers=TF_LAYERS, tf_hidden=TF_HIDDEN).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in train_loader:
            logits = model(b["cu"].to(DEVICE), b["du"].to(DEVICE))
            lg, lab = node_logits_and_labels(logits, b["cu_y"].to(DEVICE), b["du_y"].to(DEVICE))
            loss = loss_fn(lg, lab)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # val: stress-class F1 pooled over nodes
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for b in val_loader:
                logits = model(b["cu"].to(DEVICE), b["du"].to(DEVICE))
                lg, lab = node_logits_and_labels(logits, b["cu_y"], b["du_y"])
                vp.append(lg.argmax(-1).cpu().numpy()); vt.append(lab.numpy())
        vp, vt = np.concatenate(vp), np.concatenate(vt)
        _, _, f1, _ = precision_recall_fscore_support(vt, vp, labels=[1], average="macro", zero_division=0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  val stress-F1={f1:.4f}")
        if f1 > best_f1 + 1e-5:
            best_f1, patience = f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best val F1={best_f1:.4f})")
                break
    model.load_state_dict(best_state)
    return model


def evaluate_topology(model, w):
    """Per-entity predictions on the held-out topology's windows."""
    ds = LabeledWindowDataset(w)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    preds, labs = [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            logits = model(b["cu"].to(DEVICE), b["du"].to(DEVICE))  # (B, M, C)
            cu_y, du_y = b["cu_y"], b["du_y"]
            lab = torch.cat([cu_y.unsqueeze(1), du_y], dim=1)        # (B, M)
            preds.append(logits.argmax(-1).cpu().numpy())
            labs.append(lab.numpy())
    pred = np.concatenate(preds, 0)   # (S, M)
    lab = np.concatenate(labs, 0)     # (S, M)
    M = pred.shape[1]

    def m(name, p, y):
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        anom = int((y == 1).sum())
        print(f"  {name:<10s}  anom={anom:>6d}  TP={tp:>6d}  FP={fp:>6d}  FN={fn:>6d}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
        return {"tp": tp, "fp": fp, "fn": fn, "p": prec, "r": rec, "f1": f1, "anom": anom}

    print(f"\n  {'Entity':<10s}  {'anom':>6s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  {'P':>5s}  {'R':>5s}  {'F1':>5s}")
    print(f"  {'-'*68}")
    out = {"CU": m("CU", pred[:, 0], lab[:, 0])}
    for i in range(1, M):
        out[f"DU_{i-1}"] = m(f"DU_{i-1}", pred[:, i], lab[:, i])
    any_p = (pred == 1).any(1); any_y = (lab == 1).any(1)
    print(f"  {'-'*68}")
    out["ANY"] = m("ANY", any_p.astype(int), any_y.astype(int))
    return out


# =============================================================================
# LOO ORCHESTRATION
# =============================================================================

def run_one(train_topos, test_topo):
    print(f"\n{'='*70}")
    print(f"  Supervised Simba — {STRESS_NAMES[STRESS_TYPE]} stress, cross-topology LOO")
    print(f"  Train : {train_topos}  (labeled test.npz + normal train.npz)")
    print(f"  Test  : {test_topo}    (unseen topology)")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}")

    # [1] Fit scaler on TRAIN topologies' normal streams (same as TopoAR).
    raw_streams = []
    for t in train_topos:
        cu, du, bid = slice_features(load_npz(t, "train"))
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})
    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    # [2] Build supervised training windows.
    print("\n[2] Building supervised training windows ...")
    train_window_sets = []
    for t in train_topos:
        ws = build_train_windows(bundle, t)
        for w in ws:
            print(f"  {t:18s} windows={len(w['cu_y']):>6d}  N_DU={w['du'].shape[2]}  "
                  f"stress_frac={(w['du_y'] == 1).mean():.3f}")
        train_window_sets += ws
    cu_dim = train_window_sets[0]["cu"].shape[-1]
    du_dim = train_window_sets[0]["du"].shape[-1]

    # [3] Train.
    print(f"\n[3] Training SimbaORAN (cu_dim={cu_dim}, du_dim={du_dim}) ...")
    model = train_model(train_window_sets, cu_dim, du_dim)

    # [4-5] Evaluate held-out topology.
    print(f"\n[5] Evaluating held-out topology {test_topo} ...")
    w_te = build_test_windows(bundle, test_topo)
    return evaluate_topology(model, w_te)


def main():
    splits = ALL_TOPOS if RUN_ALL_LOO else [ALL_TOPOS[0]]
    all_results = []
    for test_t in splits:
        train_ts = [t for t in ALL_TOPOS if t != test_t]
        all_results.append({"test_topo": test_t, "metrics": run_one(train_ts, test_t)})

    entity_keys = []
    for r in all_results:
        for k in r["metrics"]:
            if k not in entity_keys:
                entity_keys.append(k)

    print(f"\n\n{'='*70}\n  SIMBA LEAVE-ONE-OUT SUMMARY ({STRESS_NAMES[STRESS_TYPE]})\n{'='*70}")
    print("  " + f"{'Test topology':<22s}" + "".join(f"  {e+' F1':>10s}" for e in entity_keys))
    print(f"  {'-'*68}")
    for r in all_results:
        row = f"  {r['test_topo']:<22s}"
        for e in entity_keys:
            row += f"  {r['metrics'][e]['f1']:>10.3f}" if e in r["metrics"] else f"  {'N/A':>10s}"
        print(row)

    csv_path = Path("simba_loo_results.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(["stress_type", "test_topo", "entity", "anom", "tp", "fp", "fn",
                         "precision", "recall", "f1"])
        for r in all_results:
            for ent, mt in r["metrics"].items():
                wr.writerow([STRESS_NAMES[STRESS_TYPE], r["test_topo"], ent,
                             mt["anom"], mt["tp"], mt["fp"], mt["fn"],
                             f"{mt['p']:.4f}", f"{mt['r']:.4f}", f"{mt['f1']:.4f}"])
    print(f"\n  Results appended → {csv_path.resolve()}")


if __name__ == "__main__":
    main()
