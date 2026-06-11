"""run_experiment_deepsets.py — Cross-topology detection with DeepSets DU aggregation.

WHAT DIFFERS FROM run_experiment.py
------------------------------------
1. Model: DeepSetsTopoAR (src/model_deepsets.py) replaces CalibratedTopoAR.
   - Softmax attention over N DU keys → mean-pooling of DU value embeddings.
   - LSTM input = concat(CU_context, mean_DU_context) → 2 × embed_dim.
   - Structurally N-invariant by construction (DeepSets, Zaheer et al. 2017).

2. Preprocessing: adds linear baseline(N) for CU cpu / mem_pct / mem_bytes.
   - These three channels have median linear in N_DU, IQR flat (res/IQR ≈ 0).
   - Fit slope+intercept from training topologies → subtract topology-level
     idle consumption before the RobustScaler sees the data.
   - net_tx / net_rx keep /N_DU (physically: per-DU forwarding rate).

3. Checkpoint prefix: "deepsets_" to avoid collisions with the baseline runs.

WHAT IS UNCHANGED
-----------------
  RobustScaler v0, LSTM, attention Q/K/V projections, decoders, training loop,
  calibration, closed-loop inference, cold-start probe, evaluation metrics.

RUN ALL LOO SPLITS
------------------
    cd .../clear_pipeline
    python run_experiment_deepsets.py
  (RUN_ALL_LOO = True by default — prints a 3-row comparison table at the end)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess       import fit_bundle, transform_stream
from model_deepsets   import DeepSetsTopoAR
from model_calibrated import feat_norm_calibrated
from dataset          import (TopologySequenceDataset, MultiTopologyBatchSampler,
                               collate_windows)
from scoring          import lift_score

# =============================================================================
# USER INPUTS
# =============================================================================

ALL_TOPOS   = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
TEST_TOPO   = "cu2_du3du4du5"
TRAIN_TOPOS = [t for t in ALL_TOPOS if t != TEST_TOPO]
RUN_ALL_LOO = True    # set False to run one split only

BASE_DIR    = Path("DU_MEM_bidir_STRESS")
STRESS_TYPE = 2
STRESS_NAMES = {1: "CPU", 2: "MEM", 3: "NET"}

CU_FEAT_SLICE = [0, 1, 2, 5, 6]
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6,
                 7, 8, 9, 10, 11, 12, 13, 14,
                 16, 17, 18, 19, 20,
                 26, 27, 28, 29, 30, 31, 32, 33, 35]

CU_IRATE_IDX = [0, 3, 4]
DU_IRATE_IDX = [0, 3, 4, 5]

PREPROCESS_VERSION = "v0"
CU_ZV_IDX = []
DU_ZV_IDX = []

# ── NEW: baseline(N) channels ──────────────────────────────────────────────
# Post-slice CU indices whose median scales linearly with N_DU (IQR stays flat).
# Confirmed by analyze_channel_shifts.py: res/IQR ≈ 0 for all three.
# We subtract the fitted linear baseline before RobustScaler sees the data,
# removing the cross-topology level shift for these channels.
CU_BASELINE_IDX = [0, 1, 2]   # cpu, mem_pct, mem_bytes

N_PROBE_ROWS  = 300
EMBED_DIM     = 32
WINDOW_LEN    = 64
BATCH_SIZE    = 256
EPOCHS        = 150
PATIENCE      = 5
LR            = 5e-4
VAL_FRAC      = 0.1
CAL_FRAC      = 0.2
SEED          = 42
COLD_START_K  = WINDOW_LEN
CU_THRESHOLD_PCT = 99.9
DU_THRESHOLD_PCT = 99.9
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CLOSED_LOOP   = True
IMPUTE        = True
BIDIR         = "bidir" in BASE_DIR.name
SAVE_ERRORS   = True

# =============================================================================
# HELPERS — identical to run_experiment.py
# =============================================================================

def load_npz(topo: str, split: str) -> dict:
    p = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / f"{split}.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx: list, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t-1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t-1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(z: dict):
    """Apply feature slice, /N_DU for net, imputation, and derived features.

    Identical to run_experiment.py — net_tx/rx are divided by N_DU here.
    The baseline(N) correction for cpu/mem is applied separately in
    phase_preprocess (after all training streams are available for fitting).
    """
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    cu[:, 5] = cu[:, 5] / N_DU      # net_tx → per-DU forwarding rate
    cu[:, 6] = cu[:, 6] / N_DU      # net_rx → per-DU receive rate
    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]

    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
        du = impute_cpu_glitch(du, DU_IRATE_IDX)

    _tx = cu[:, 3:4]
    _rx = cu[:, 4:5]
    cu  = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)

    _du_tx = du[:, :, 4:5]
    _du_rx = du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)

    block_id = z["block_id"].astype(np.int64)
    return cu, du, block_id

# =============================================================================
# BASELINE(N) HELPERS  ← NEW in this file
# =============================================================================

def fit_cu_baseline(raw_streams: list, n_du_list: list):
    """Fit linear median(N) = alpha*N + beta for cpu/mem/mem_bytes CU channels.

    raw_streams : list of {"cu": (T, cu_dim), ...}  pre-scaled, post-slice arrays
    n_du_list   : list of int N_DU per topology (same order as raw_streams)

    Returns alpha, beta — float64 arrays of shape (len(CU_BASELINE_IDX),).

    With two training topologies (N=2 and N=3), two points uniquely determine a
    line → the fitted baseline extrapolates exactly to N=1 because the linear
    model is confirmed correct (res/IQR ≈ 0 from analyze_channel_shifts.py).
    """
    N    = np.array(n_du_list, dtype=np.float64)
    meds = np.array(
        [[np.median(s["cu"][:, c]) for c in CU_BASELINE_IDX] for s in raw_streams]
    )                                                  # (n_topos, 3)
    alpha = np.zeros(len(CU_BASELINE_IDX))
    beta  = np.zeros(len(CU_BASELINE_IDX))
    for i in range(len(CU_BASELINE_IDX)):
        alpha[i], beta[i] = np.polyfit(N, meds[:, i], 1)
    return alpha, beta


def apply_cu_baseline(cu: np.ndarray, n_du: int, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Subtract fitted topology-level idle baseline from cpu/mem CU channels."""
    cu = cu.copy()
    for i, c in enumerate(CU_BASELINE_IDX):
        cu[:, c] -= alpha[i] * n_du + beta[i]
    return cu

# =============================================================================
# PHASE 1: PREPROCESSING  ← modified to fit and apply baseline(N)
# =============================================================================

def phase_preprocess(train_zs, train_topos, train_n_dus):
    """Fit RobustScaler on all train topologies (baseline-centered, pooled).

    Steps:
      1. slice_features for each topology.
      2. Fit linear baseline(N) for CU cpu/mem channels from training data.
      3. Subtract baseline from each training stream.
      4. Fit RobustScaler on the centered, pooled data.
      5. Transform each training stream.

    Returns (bundle, streams, alpha_bl, beta_bl).
    """
    raw_streams = []
    for z in train_zs:
        cu, du, bid = slice_features(z)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})

    # Fit linear baseline
    alpha_bl, beta_bl = fit_cu_baseline(raw_streams, train_n_dus)
    ch_names = ["cpu", "mem_pct", "mem_bytes"]
    print("  baseline(N) fit:")
    for i, nm in enumerate(ch_names):
        print(f"    {nm}: slope={alpha_bl[i]:.4g}  intercept={beta_bl[i]:.4g}")

    # Apply baseline centering to all training streams
    for raw, n in zip(raw_streams, train_n_dus):
        raw["cu"] = apply_cu_baseline(raw["cu"], n, alpha_bl, beta_bl)

    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    streams = []
    for i, raw in enumerate(raw_streams):
        cu_s, du_s, _, kept_bid = transform_stream(
            bundle, raw["cu"], raw["du"], raw["block_id"]
        )
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_s {cu_s.shape}  du_s {du_s.shape}  "
              f"(μ={cu_s.mean():+.3f}, σ={cu_s.std():.3f})")
        streams.append((cu_s, du_s, kept_bid))

    return bundle, streams, alpha_bl, beta_bl

# =============================================================================
# PHASE 2: TRAIN  — uses DeepSetsTopoAR
# =============================================================================

def phase_train(fit_streams, train_topos, model_ckpt) -> DeepSetsTopoAR:
    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]

    train_subsets, val_subsets, train_lens, val_lens = [], [], [], []
    rng = np.random.RandomState(SEED)
    for i, (cu_s, du_s, bid) in enumerate(fit_streams):
        ds  = TopologySequenceDataset(cu_s, du_s, bid, window_len=WINDOW_LEN, stride=1)
        n   = len(ds)
        perm = rng.permutation(n)
        n_val = max(1, int(round(VAL_FRAC * n)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_subsets.append(torch.utils.data.Subset(ds, train_idx))
        val_subsets.append(torch.utils.data.Subset(ds, val_idx))
        train_lens.append(len(train_idx))
        val_lens.append(len(val_idx))
        print(f"  topo[{i}] {train_topos[i]:18s} N_DU={du_s.shape[1]}  "
              f"windows={n}  (train={len(train_idx)}, val={len(val_idx)})")

    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_subsets),
        batch_sampler=MultiTopologyBatchSampler(train_lens, BATCH_SIZE, shuffle=True,  seed=SEED),
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_subsets),
        batch_sampler=MultiTopologyBatchSampler(val_lens,   BATCH_SIZE, shuffle=False, seed=SEED),
        collate_fn=collate_windows,
    )

    torch.manual_seed(SEED)
    model = DeepSetsTopoAR(cu_dim=cu_dim, du_dim=du_dim, embed_dim=EMBED_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, patience_count, best_state = float("inf"), 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            cu_b = batch["cu"].to(DEVICE)
            du_b = batch["du"].to(DEVICE)
            cu_hat, du_hat = model(cu_b, du_b)
            loss = (((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean() +
                    ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean())
            optim.zero_grad(); loss.backward(); optim.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cu_b = batch["cu"].to(DEVICE); du_b = batch["du"].to(DEVICE)
                cu_hat, du_hat = model(cu_b, du_b)
                val_loss += (((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean() +
                             ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean()).item()
        val_loss /= max(len(val_loader), 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  tr={tr_loss/len(train_loader):.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss; patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop epoch {epoch}  (best_val={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    n_train_rows = sum(len(f[0]) for f in fit_streams)
    torch.save({
        "state_dict": best_state,
        "cu_dim": cu_dim, "du_dim": du_dim, "embed_dim": EMBED_DIM,
        "variant": "deepsets",
        "cal_frac": CAL_FRAC, "n_train_rows": n_train_rows,
        "topos": list(train_topos), "preprocess": PREPROCESS_VERSION,
    }, model_ckpt)
    print(f"  Model saved → {model_ckpt}")
    return model

# =============================================================================
# PHASE 3: INFERENCE (open-loop)
# =============================================================================

def phase_infer(model: DeepSetsTopoAR, cu_s: np.ndarray, du_s: np.ndarray):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()
    du_sqerr = (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()
    return cu_sqerr, du_sqerr

# =============================================================================
# PHASE 3b: CLOSED-LOOP INFERENCE
# =============================================================================

def phase_infer_closed_loop(model, cu_s, du_s, cu_feat_norm, du_feat_norm,
                             cu_thr, du_thr):
    model.eval()
    T, N = len(cu_s), du_s.shape[1]
    cu_sqerrs = np.zeros((T - 1, cu_s.shape[1]),     dtype=np.float32)
    du_sqerrs = np.zeros((T - 1, N, du_s.shape[2]),  dtype=np.float32)

    h, c = model.init_state(1, DEVICE)
    DU_HYSTERESIS, CU_HYSTERESIS = 5, 5
    du_anom_count = np.zeros(N, dtype=np.int32)
    cu_anom_count = 0

    cu_in = torch.tensor(cu_s[[0]], dtype=torch.float32).to(DEVICE)
    du_in = torch.tensor(du_s[[0]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        for t in range(T - 1):
            cu_tok, du_tok = model.project_tokens(cu_in, du_in)
            cu_hat, du_hat, h, c, _ = model.step(cu_tok, du_tok, h, c)

            cu_next = torch.tensor(cu_s[[t + 1]], dtype=torch.float32).to(DEVICE)
            du_next = torch.tensor(du_s[[t + 1]], dtype=torch.float32).to(DEVICE)

            cu_err = (cu_hat - cu_next).pow(2).cpu().numpy()[0]
            du_err = (du_hat - du_next).pow(2).cpu().numpy()[0]
            cu_sqerrs[t] = cu_err
            du_sqerrs[t] = du_err

            cu_score = float((cu_err / cu_feat_norm).max())
            cu_anom_count = cu_anom_count + 1 if cu_score > cu_thr else 0
            cu_in = cu_hat if cu_anom_count >= CU_HYSTERESIS else cu_next

            du_in = du_next.clone()
            for i in range(N):
                du_score_i = float((du_err[i] / du_feat_norm).max())
                du_anom_count[i] = du_anom_count[i] + 1 if du_score_i > du_thr else 0
                if du_anom_count[i] >= DU_HYSTERESIS:
                    du_in[0, i] = du_hat[0, i]

    return cu_sqerrs, du_sqerrs

# =============================================================================
# PHASE 4: EVALUATE
# =============================================================================

def phase_evaluate(cu_sqerr, du_sqerr, cu_feat_norm, du_feat_norm,
                   cu_thr, du_thr, cu_stress, du_stress):
    N     = du_sqerr.shape[1]
    start = COLD_START_K
    cu_sqerr_ev = cu_sqerr[start:]
    du_sqerr_ev = du_sqerr[start:]
    cu_lbl = (cu_stress[start + 1:] == STRESS_TYPE).astype(int)
    du_lbl = (du_stress[start + 1:] == STRESS_TYPE)

    cu_scores = lift_score(cu_sqerr_ev, cu_feat_norm)
    cu_pred   = (cu_scores > cu_thr).astype(int)
    du_scores = np.stack([lift_score(du_sqerr_ev[:, i, :], du_feat_norm) for i in range(N)], axis=1)
    du_pred   = (du_scores > du_thr).astype(int)

    all_metrics = {}

    def metrics(name, pred, lbl):
        tp = int(((pred == 1) & (lbl == 1)).sum())
        fp = int(((pred == 1) & (lbl == 0)).sum())
        fn = int(((pred == 0) & (lbl == 1)).sum())
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        total_anom = int((lbl == 1).sum())
        print(f"  {name:<12s}  anom={total_anom:>6d}  TP={tp:>6d}  FP={fp:>6d}  "
              f"FN={fn:>6d}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        all_metrics[name] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1, "anom": total_anom}

    print(f"\n  {'Entity':<12s}  {'anom':>6s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  "
          f"{'P':>5s}  {'R':>5s}  {'F1':>5s}")
    print(f"  {'-'*72}")
    metrics("CU", cu_pred, cu_lbl)
    for i in range(N):
        metrics(f"DU_{i}", du_pred[:, i], du_lbl[:, i].astype(int))
    any_pred = (cu_pred == 1) | du_pred.any(axis=1)
    any_lbl  = (cu_lbl  == 1) | du_lbl.any(axis=1)
    print(f"  {'-'*72}")
    metrics("ANY", any_pred.astype(int), any_lbl.astype(int))

    return cu_scores, du_scores, cu_pred, du_pred, all_metrics

# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run_one(train_topos, test_topo):
    model_ckpt = Path(f"deepsets_model_ckpt_test_{test_topo}.pt")
    cu_dim_info = len(np.arange(7)[CU_FEAT_SLICE])
    du_dim_info = len(np.arange(37)[DU_FEAT_SLICE])

    print(f"\n{'='*70}")
    print(f"  [DeepSets] {STRESS_NAMES[STRESS_TYPE]} stress detection")
    print(f"  Train : {train_topos}   Test : {test_topo}")
    print(f"  CU features: {cu_dim_info}   DU features: {du_dim_info}   Device: {DEVICE}")
    print(f"{'='*70}")

    # [1] Load + preprocess (with baseline(N))
    print(f"\n[1] Loading {len(train_topos)} train topologies, fitting baseline(N) + {PREPROCESS_VERSION} scaler ...")
    train_zs    = [load_npz(t, "train") for t in train_topos]
    train_n_dus = [int(z["du"].shape[1]) for z in train_zs]
    bundle, train_streams, alpha_bl, beta_bl = phase_preprocess(
        train_zs, train_topos, train_n_dus
    )

    # [2] Fit / cal split
    print(f"\n[2] Fit/cal split (cal_frac={CAL_FRAC}) ...")
    fit_streams, cal_streams = [], []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_cal = int(round(CAL_FRAC * len(cu_s)))
        n_fit = len(cu_s) - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        cal_streams.append((cu_s[n_fit:], du_s[n_fit:]))
        print(f"  topo[{i}] {train_topos[i]:18s}  fit={n_fit}  cal={n_cal}")
    n_fit_total = sum(len(f[0]) for f in fit_streams)

    # [3] Train or load
    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]
    if model_ckpt.exists():
        print(f"\n[3] Loading checkpoint {model_ckpt} — delete to retrain ...")
        ckpt = torch.load(model_ckpt, map_location=DEVICE)
        mismatches = []
        if ckpt.get("variant") != "deepsets":
            mismatches.append(f"variant: expected deepsets, got {ckpt.get('variant')}")
        if ckpt.get("topos") != list(train_topos):
            mismatches.append(f"topos: {ckpt.get('topos')} vs {list(train_topos)}")
        if ckpt.get("n_train_rows") != n_fit_total:
            mismatches.append(f"n_train_rows: {ckpt.get('n_train_rows')} vs {n_fit_total}")
        if ckpt.get("cu_dim") != cu_dim:
            mismatches.append(f"cu_dim: {ckpt.get('cu_dim')} vs {cu_dim}")
        if mismatches:
            raise SystemExit(
                f"\n  Incompatible checkpoint {model_ckpt}:\n    "
                + "\n    ".join(mismatches)
                + f"\n  Delete {model_ckpt} and rerun to retrain."
            )
        model = DeepSetsTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"],
                               embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"\n[3] Training DeepSetsTopoAR on {len(train_topos)} topologies ...")
        model = phase_train(fit_streams, train_topos, model_ckpt)

    # [4] Cal inference + calibrate
    print(f"\n[4] CAL inference (pooled across {len(train_topos)} topologies) ...")
    cu_sqerr_pool, du_sqerr_pool = [], []
    for i, (cu_s_cal, du_s_cal) in enumerate(cal_streams):
        cu_sq, du_sq = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sqerr_pool.append(cu_sq[COLD_START_K:])
        du_sqerr_pool.append(du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1]))
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_sqerr {cu_sq.shape}")
    cu_sqerr_n   = np.concatenate(cu_sqerr_pool, axis=0)
    du_sqerr_flt = np.concatenate(du_sqerr_pool, axis=0)

    print(f"\n[5] Calibrating thresholds ...")
    cu_fn = feat_norm_calibrated(cu_sqerr_n)
    du_fn = feat_norm_calibrated(du_sqerr_flt)
    cu_norm_scores = lift_score(cu_sqerr_n,   cu_fn)
    du_norm_scores = lift_score(du_sqerr_flt, du_fn)
    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    print(f"  CU thr (p{CU_THRESHOLD_PCT}): {cu_thr:.4f}   DU thr: {du_thr:.4f}")

    # [6] Transform test topology — apply baseline first
    print(f"\n[6] Transforming TEST topology ({test_topo}) ...")
    test_z = load_npz(test_topo, "test")
    cu_te, du_te, bid_te = slice_features(test_z)
    n_du_te = du_te.shape[1]

    # Apply the train-fitted baseline to the test topology's N_DU
    cu_te = apply_cu_baseline(cu_te, n_du_te, alpha_bl, beta_bl)

    cu_s_te, du_s_te, kept_mask, _ = transform_stream(bundle, cu_te, du_te, bid_te)
    cu_stress = test_z["cu_stress"][kept_mask].astype(np.int32)
    du_stress = test_z["du_stress"][kept_mask].astype(np.int32)
    print(f"  cu_s_te {cu_s_te.shape}  du_s_te {du_s_te.shape}  "
          f"(μ={cu_s_te.mean():+.3f}, σ={cu_s_te.std():.3f})")

    # [6b] Cold-start probe
    print(f"\n[6b] Cold-start probe ({N_PROBE_ROWS} rows) ...")
    n_probe = min(N_PROBE_ROWS + 1, len(cu_s_te))
    cu_sq_probe, du_sq_probe = phase_infer(model, cu_s_te[:n_probe], du_s_te[:n_probe])

    cu_probe_scores = lift_score(cu_sq_probe[COLD_START_K:], cu_fn)
    cu_test_p50     = float(np.percentile(cu_probe_scores, 50))
    cu_cal_p50      = float(np.percentile(cu_norm_scores,  50))
    cu_shift        = cu_test_p50 / max(cu_cal_p50, 1e-9)
    cu_thr_adj      = cu_thr * max(1.0, cu_shift)
    print(f"  CU probe p50: test={cu_test_p50:.4f}  cal={cu_cal_p50:.4f}  "
          f"shift={cu_shift:.2f}x  thr {cu_thr:.4f}→{cu_thr_adj:.4f}")

    du_sq_probe_flat = du_sq_probe[COLD_START_K:].reshape(-1, du_sq_probe.shape[-1])
    du_probe_scores  = lift_score(du_sq_probe_flat, du_fn)
    du_test_p50      = float(np.percentile(du_probe_scores, 50))
    du_cal_p50       = float(np.percentile(du_norm_scores,  50))
    du_shift         = du_test_p50 / max(du_cal_p50, 1e-9)
    du_thr_adj       = du_thr * np.sqrt(max(1.0, du_shift))
    print(f"  DU probe p50: test={du_test_p50:.4f}  cal={du_cal_p50:.4f}  "
          f"shift={du_shift:.2f}x  thr {du_thr:.4f}→{du_thr_adj:.4f}")

    # [7] Full test inference
    if CLOSED_LOOP:
        print("\n[7] Closed-loop inference on test stream ...")
        cu_sqerr, du_sqerr = phase_infer_closed_loop(
            model, cu_s_te, du_s_te, cu_fn, du_fn, cu_thr_adj, du_thr_adj
        )
    else:
        print("\n[7] Open-loop inference on test stream ...")
        cu_sqerr, du_sqerr = phase_infer(model, cu_s_te, du_s_te)

    if SAVE_ERRORS:
        feat_tag = f"f{cu_dim}"
        err_path = Path(f"recon_errors_deepsets_{test_topo}_{feat_tag}.npz")
        np.savez(err_path,
                 cu_sqerr=cu_sqerr, du_sqerr=du_sqerr,
                 cu_stress=cu_stress, du_stress=du_stress,
                 cu_feat_norm=cu_fn,  du_feat_norm=du_fn,
                 cu_thr=np.array([cu_thr]), cu_thr_adj=np.array([cu_thr_adj]),
                 du_thr=np.array([du_thr]), du_thr_adj=np.array([du_thr_adj]))
        print(f"  Errors saved → {err_path}")

    # [7b] Ablation: same errors, raw (unadjusted) thresholds
    # Difference vs [8] below isolates the cold-start probe's contribution.
    print("\n[7b] ABLATION — raw threshold (no cold-start probe adjustment) ...")
    _, _, _, _, metrics_raw = phase_evaluate(
        cu_sqerr, du_sqerr, cu_fn, du_fn,
        cu_thr, du_thr,
        cu_stress, du_stress,
    )
    print("  [ABLATION raw-thr] " + "  ".join(
        f"{k}={v['f1']:.3f}" for k, v in metrics_raw.items()
    ))

    # [8] Evaluate
    print("\n[8] Evaluation ...")
    cu_scores, du_scores, cu_pred, du_pred, eval_metrics = phase_evaluate(
        cu_sqerr, du_sqerr, cu_fn, du_fn, cu_thr_adj, du_thr_adj, cu_stress, du_stress
    )

    print("\n[8b] Score distribution: CAL vs TEST")
    pcts = [50, 90, 99, 99.9]
    print(f"  {'':6s}  " + "  ".join(f"p{p:4.1f}" for p in pcts))
    cu_te_scores = lift_score(cu_sqerr, cu_fn)
    cu_scores_cal = lift_score(cu_sqerr_n, cu_fn)
    print(f"  {'CU cal':6s}  " + "  ".join(f"{np.percentile(cu_scores_cal, p):6.3f}" for p in pcts))
    print(f"  {'CU te ':6s}  " + "  ".join(f"{np.percentile(cu_te_scores,  p):6.3f}" for p in pcts))

    print("\nDone.")
    return eval_metrics


def main():
    if RUN_ALL_LOO:
        all_results = []
        for test_t in ALL_TOPOS:
            train_ts = [t for t in ALL_TOPOS if t != test_t]
            result   = run_one(train_ts, test_t)
            all_results.append({"test_topo": test_t, "metrics": result})

        entity_keys: list = []
        for r in all_results:
            for k in r["metrics"]:
                if k not in entity_keys:
                    entity_keys.append(k)

        print(f"\n\n{'='*70}")
        print(f"  [DeepSets + baseline(N)]  LOO SUMMARY  ({len(ALL_TOPOS)} configs)")
        print(f"{'='*70}")
        header = f"  {'Test topology':<22s}" + "".join(f"  {e+' F1':>10s}" for e in entity_keys)
        print(header)
        print(f"  {'-'*68}")
        for r in all_results:
            row = f"  {r['test_topo']:<22s}"
            for e in entity_keys:
                row += f"  {r['metrics'][e]['f1']:>10.3f}" if e in r["metrics"] else f"  {'N/A':>10s}"
            print(row)
        print(f"  {'-'*68}")
        print(f"\n  Compare with baseline (run_experiment.py RUN_ALL_LOO=True)")
    else:
        run_one(TRAIN_TOPOS, TEST_TOPO)


if __name__ == "__main__":
    main()
