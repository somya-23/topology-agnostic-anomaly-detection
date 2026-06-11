"""run_experiment.py — Cross-topology CPU stress detection (Step 3, v0 preprocessing).

WHAT IT DOES
------------
  Train on one topology's normal data (train.npz from TRAIN_TOPO).
  Test  on a DIFFERENT topology's mixed data (test.npz from TEST_TOPO).
  Report F1 / precision / recall for CU and each DU.

KEY DESIGN DECISIONS
--------------------
  Preprocessing v0: raw → RobustScaler, NO block_diff.
    Why no block_diff: stress is constant (sustained 80% CPU). After diff, both
    normal-delta≈0 and stress-delta≈0 look identical — the level shift disappears.
    RobustScaler alone preserves the level shift as a large positive scaled value.

  Traffic-invariant features only (DU_FEAT_SLICE = slice(0, 2) = cpu + mem_pct):
    net_tx and dl_brate scale ~2× when traffic doubles — unreliable across topologies.
    cpu and mem_pct are fractions of hardware capacity, comparable across machines.

  Prometheus irate glitch imputation (applied BEFORE RobustScaler):
    irate(container_cpu_user_seconds_total[5m]) returns exactly 0.0 at 5-minute
    clock boundaries when the sliding window crosses a counter reset. These appear
    every 300 seconds in both train and test. The raw 0.0 is physically impossible
    (a process cannot use zero CPU for an entire 1-second scrape interval) and is a
    pure measurement artifact. After RobustScaler, 0.0 maps to (0-median)/IQR ≈ -27
    for DU and ≈ -5 for CU — the model fails to predict these, spiking lift scores
    to ~286 (DU) and ~304 (CU). This inflates the p99.9 threshold to values set
    entirely by artifacts rather than real normal data. Fix: forward-fill (replace
    raw 0.0 with the previous timestep's value) before scaling. The imputation is
    applied identically to train and test so the scaler never sees the outlier.

  Train-side held-out calibration (NO test label leakage):
    The last CAL_FRAC of the TRAIN stream is held out before training. The model
    is fit on the remaining train-train portion only. After training, that held-out
    cal stream (which the model has NEVER seen) is used to compute feat_norm and
    the lift-score threshold. This matches deployment: at inference time we have
    no labels on the live stream, so we cannot use test-normal rows for the
    threshold. Cross-topology baseline shift is now a real failure mode for the
    model, not something the threshold is allowed to hide.

FLOW
----
  [1] Fit preprocessing on TRAIN topology normal data (with CPU glitch imputation).
  [2] Split train stream into (train_fit, cal) by row index — last CAL_FRAC is cal.
  [3] Train DUNormTopoAR on train_fit only.
  [4] Run inference on the held-out cal stream → calibrate feat_norm + threshold.
  [5] Apply the SAME scalers to TEST topology (no refit; imputation also applied).
  [6] Run sequential LSTM inference on the full TEST stream.
  [7] Score all TEST rows, compare vs ground-truth labels, print metrics.

USAGE
-----
    cd project_root/step3_topoar/clear_pipeline
    python run_experiment.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import fit_bundle, transform_stream, causal_rolling_normalize
from model_calibrated import feat_norm_calibrated, dual_threshold_from_val
from model_dunorm    import DUNormTopoAR
from dataset import TopologySequenceDataset, MultiTopologyBatchSampler, collate_windows
from scoring import lift_score

# =============================================================================
# USER INPUTS — change these to try different cross-topology pairs
# =============================================================================

ALL_TOPOS     = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]   # all available topologies
TEST_TOPO     = "cu1_du2"                    # held-out topology; change to try a different split
TRAIN_TOPOS   = [t for t in ALL_TOPOS if t != TEST_TOPO]      # auto-derived: all except TEST_TOPO
RUN_ALL_LOO   = False   # True → run all 3 leave-one-out splits sequentially and print a summary table

BASE_DIR      = Path("CU_NET_bidir_STRESS")
STRESS_TYPE   = 3           # 1=CPU | 2=MEM | 3=NET  — must match the test dataset
STRESS_NAMES  = {1: "CPU", 2: "MEM", 3: "NET"}

# Feature slices — all KPIs minus permanently-zero features.
# Dropped (confirmed 0 in both normal AND stress across all topologies — zero discriminative value):
#   CU raw 3 (fs_reads), CU raw 4 (fs_writes)
#   DU raw 3 (fs_reads), DU raw 15,21,22,23,24,25,34,36 (PCI columns always 0)
CU_FEAT_SLICE = [0, 1, 2, 5, 6]           # cpu, mem_pct, mem_bytes, net_tx, net_rx  (5 features)
DU_FEAT_SLICE = [0, 1, 2, 4, 5, 6,        # cpu, mem_pct, mem_bytes, fs_writes, net_tx, net_rx
                 7, 8, 9, 10, 11, 12, 13, 14,           # PCI 0-7
                 16, 17, 18, 19, 20,                    # PCI 9-13
                 26, 27, 28, 29, 30, 31, 32, 33, 35]    # PCI 19-26, 28  (28 features total)

# Irate/counter indices in the POST-SLICE arrays (imputation runs after slicing).
# CU sliced: 0=cpu, 1=mem_pct, 2=mem_bytes, 3=net_tx (raw), 4=net_rx (raw) | derived: 5=net_ratio
# DU sliced: 0=cpu, 1=mem_pct, 2=mem_bytes, 3=fs_writes, 4=net_tx, 5=net_rx, 6+=PCI | derived: -2=net_diff, -1=net_ratio
CU_IRATE_IDX = [0, 3, 4]    # cpu, net_tx, net_rx (mem_pct zeros co-occur with cpu zeros but imputing shrinks feat_norm → amplifies cross-topology mem shift → FPs)
DU_IRATE_IDX = [0, 3, 4, 5] # cpu, fs_writes, net_tx, net_rx

# Preprocessing: RobustScaler v0_dunorm (no /N_DU on CU net; no CU net_diff —
# the model's W_DU produces a per-DU softplus scalar whose sum normalizes CU
# net_tx/net_rx inside the model, as a learned data-driven substitute for /N).
PREPROCESS_VERSION = "v0_dunorm"
# No zero-variance features remain after dropping the always-0 columns above.
CU_ZV_IDX = []
DU_ZV_IDX = []

# Model hyperparameters
EMBED_DIM     = 32
WINDOW_LEN    = 64
BATCH_SIZE    = 256             # bumped from 64 to amortize per-batch CUDA kernel-launch overhead
EPOCHS        = 150
PATIENCE      = 5
LR            = 5e-4
VAL_FRAC      = 0.1             # window-level val for early stopping inside training
CAL_FRAC      = 0.2             # tail fraction of train stream held out for threshold calibration
SEED          = 42
# MODEL_CKPT is computed per-run inside run_one() from the test topology name — delete to retrain.

# First COLD_START_K rows of the test stream are skipped: the LSTM hidden
# state is initialised at zero and takes ~WINDOW_LEN steps to warm up.
COLD_START_K  = WINDOW_LEN
# Separate thresholds per entity type (CU and DU have different error scales).
# CU: large cross-topology baseline shift elevates CU errors throughout the test
#   stream; p99.9 cleanly separates that elevated normal baseline from stress.
# DU: DU normal errors are small; p99.0 gives a threshold safely below the stress
#   signal mean (~2.3).  After the Prometheus glitch imputation removes the ~286
#   spike from the cal stream, p99.9 may also work — try raising after retraining.
CU_THRESHOLD_PCT = 99.9         # percentile for CU threshold (normal scores)
DU_THRESHOLD_PCT = 99.9         # same as CU — safe now that glitch spikes are removed by imputation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Closed-loop inference: when an entity is detected as anomalous, its actual
# input is replaced by the model's own prediction for the next step.  This
# prevents the LSTM hidden state from adapting to a sustained anomaly level,
# keeping the prediction anchored to normal behaviour throughout the stress
# window.  Set to False to revert to the original open-loop inference.
CLOSED_LOOP = True

BIDIR = "bidir" in BASE_DIR.name   # derived from BASE_DIR name; True when directory contains "bidir"

# Prometheus irate glitch imputation: forward-fill rows where raw cpu/mem == 0.0.
# WARNING: enabling this currently breaks results because feat_norm shrinks ~177×
# after the glitch spikes are removed, which amplifies the cross-topology baseline
# shift and causes massive FPs.  A proper fix requires calibrating feat_norm on the
# test cold-start too (not yet implemented).  Leave False until that is fixed.
IMPUTE = True

# Save reconstruction errors after closed-loop inference so plot_recon_comparison.py
# can overlay cpu-only vs cpu+mem results in a single figure.
# Files are named  recon_errors_{test_topo}_f{cu_dim}.npz  (f1=cpu-only, f2=cpu+mem).
SAVE_ERRORS = True

# =============================================================================
# HELPERS
# =============================================================================

def load_npz(topo: str, split: str) -> dict:
    p = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / f"{split}.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr: np.ndarray, irate_idx: list, eps: float = 1e-6) -> np.ndarray:
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:  # CU: (T, dim)
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t - 1, irate_idx], arr[t, irate_idx])
        else:              # DU: (T, N, dim)
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t - 1, :, irate_idx], arr[t, :, irate_idx])
    return arr

def slice_features(z: dict):
    """Apply CU/DU feature slices and return (cu, du, block_id).

    DUNorm variant changes vs run_experiment.py:
      * NO /N_DU division on CU net_tx, net_rx — raw values feed the model.
        The model computes its own sum_extra (softplus of W_DU's extra dim,
        summed across DUs) and divides net_tx, net_rx by it INSIDE the model.
      * NO CU net_diff derived feature. Only CU net_ratio kept.
    DU-side features and derived (net_diff, net_ratio per-DU) are unchanged.

    Resulting CU feature order (cu_dim = 6):
       0=cpu  1=mem_pct  2=mem_bytes  3=net_tx (raw)  4=net_rx (raw)  5=net_ratio
    (Indices 3, 4 must match DUNormTopoAR.NET_TX_IDX / NET_RX_IDX.)
    """
    cu       = z["cu"].astype(np.float32)
    du       = z["du"].astype(np.float32)

    # /N_DU on CU net intentionally removed — replaced by the model's learned
    # sum_extra normalization inside DUNormTopoAR.project_tokens().

    cu = cu[:, CU_FEAT_SLICE]
    du = du[:, :, DU_FEAT_SLICE]
    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX)
        du = impute_cpu_glitch(du, DU_IRATE_IDX)

    # CU derived: keep ONLY net_ratio (drop net_diff). CU post-slice: 3=net_tx, 4=net_rx.
    _tx = cu[:, 3:4]
    _rx = cu[:, 4:5]
    cu = np.concatenate([cu,
                         _tx / (_rx + 1e-6)],       # net_ratio → position 5
                        axis=1)

    # DU derived (unchanged from run_experiment.py). DU post-slice: 4=net_tx, 5=net_rx.
    _du_tx = du[:, :, 4:5]
    _du_rx = du[:, :, 5:6]
    du = np.concatenate([du,
                         _du_tx - _du_rx,             # net_diff  → last-1 position
                         _du_tx / (_du_rx + 1e-6)],   # net_ratio → last position
                        axis=2)

    block_id = z["block_id"].astype(np.int64)                   # (T,)
    return cu, du, block_id

# =============================================================================
# PHASE 1: PREPROCESSING — fit RobustScaler on all train topologies (pooled),
# then transform each stream. The fitted bundle is applied unchanged to the test
# stream; cross-topology baseline shift is absorbed by the pooled-train scaler and
# the type-shared /N normalization (no inference-time threshold adjustment).
# =============================================================================

def phase_preprocess(train_zs, train_topos):
    """Fit RobustScaler on all train topologies (pooled), transform each stream.

    Args:
        train_zs:    list of loaded train.npz dicts (one per topology).
        train_topos: list of topology names (for display only).
    Returns:
        bundle:  PreprocessBundle (fitted scalers; passed to transform_stream for test)
        streams: list of (cu_s, du_s, kept_bid) tuples per topology
    """
    raw_streams = []
    for z in train_zs:
        cu, du, bid = slice_features(z)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})

    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)

    streams = []
    for i, raw in enumerate(raw_streams):
        cu_s, du_s, _, kept_bid = transform_stream(
            bundle, raw["cu"], raw["du"], raw["block_id"]
        )
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_s {cu_s.shape}  du_s {du_s.shape}  "
              f"(μ={cu_s.mean():+.3f}, σ={cu_s.std():.3f} after {PREPROCESS_VERSION})")
        streams.append((cu_s, du_s, kept_bid))
    return bundle, streams

# =============================================================================
# PHASE 2: TRAIN MODEL
# Next-step prediction: at each timestep t, predict x_{t+1} from x_t.
# Loss = MSE(predicted t+1, actual t+1) for both CU and DU.
# Early stopping on validation loss.
# =============================================================================

def phase_train(fit_streams, train_topos, model_ckpt) -> DUNormTopoAR:
    """Train on multiple topologies with N-homogeneous batches.

    Args:
        fit_streams: list of (cu_s, du_s, block_id) — train_fit portions per topology.
        train_topos: list of topology names (for display and checkpoint metadata).
        model_ckpt:  Path to save the checkpoint after training.
    """
    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]

    # One dataset per topology so the sampler can build homogeneous-N batches.
    train_subsets, val_subsets = [], []
    train_lens, val_lens = [], []
    rng = np.random.RandomState(SEED)
    for i, (cu_s, du_s, bid) in enumerate(fit_streams):
        ds = TopologySequenceDataset(cu_s, du_s, bid, window_len=WINDOW_LEN, stride=1)
        n = len(ds)
        perm = rng.permutation(n)
        n_val = max(1, int(round(VAL_FRAC * n)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_subsets.append(torch.utils.data.Subset(ds, train_idx))
        val_subsets.append(torch.utils.data.Subset(ds, val_idx))
        train_lens.append(len(train_idx))
        val_lens.append(len(val_idx))
        print(f"  topo[{i}] {train_topos[i]:18s} N_DU={du_s.shape[1]}  "
              f"windows={n}  (train={len(train_idx)}, val={len(val_idx)})")

    train_concat = torch.utils.data.ConcatDataset(train_subsets)
    val_concat   = torch.utils.data.ConcatDataset(val_subsets)

    train_loader = DataLoader(
        train_concat,
        batch_sampler=MultiTopologyBatchSampler(train_lens, BATCH_SIZE, shuffle=True,  seed=SEED),
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        val_concat,
        batch_sampler=MultiTopologyBatchSampler(val_lens,   BATCH_SIZE, shuffle=False, seed=SEED),
        collate_fn=collate_windows,
    )

    torch.manual_seed(SEED)
    model = DUNormTopoAR(cu_dim=cu_dim, du_dim=du_dim, embed_dim=EMBED_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            cu_b = batch["cu"].to(DEVICE)    # (B, L, cu_dim)
            du_b = batch["du"].to(DEVICE)    # (B, L, N, du_dim)
            cu_hat, du_hat = model(cu_b, du_b)
            # cu_hat[:, t] = prediction for timestep t+1
            # Compare hat[t] with actual[t+1] for t = 0..L-2
            loss = (
                ((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean() +
                ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cu_b = batch["cu"].to(DEVICE)
                du_b = batch["du"].to(DEVICE)
                cu_hat, du_hat = model(cu_b, du_b)
                val_loss += (
                    ((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean() +
                    ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean()
                ).item()
        val_loss /= max(len(val_loader), 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  tr={tr_loss/len(train_loader):.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best_val={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    n_train_rows = sum(len(f[0]) for f in fit_streams)
    torch.save({"state_dict": best_state,
                "cu_dim": cu_dim, "du_dim": du_dim, "embed_dim": EMBED_DIM,
                "cal_frac": CAL_FRAC, "n_train_rows": n_train_rows,
                "topos": list(train_topos),
                "preprocess": PREPROCESS_VERSION}, model_ckpt)
    print(f"  Model saved → {model_ckpt}")
    return model

# =============================================================================
# PHASE 3: SEQUENTIAL INFERENCE
# Pass the full test sequence through the model in one shot, preserving
# LSTM hidden state continuity across the whole stream.
#
# Output:
#   cu_sqerr[t'] = squared error at timestep t'+1  (prediction FROM t')
#   du_sqerr[t'] = same, per DU
# Both arrays have shape (T-1, ...) covering error at timesteps 1..T-1.
# =============================================================================

def phase_infer(model: DUNormTopoAR, cu_s: np.ndarray, du_s: np.ndarray):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)   # (1, T, cu_dim)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)   # (1, T, N, du_dim)

    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)

    # cu_hat[0, t] predicts timestep t+1 → error at t+1 is (hat[t] - actual[t+1])^2
    cu_sqerr = (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy()   # (T-1, cu_dim)
    du_sqerr = (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()   # (T-1, N, du_dim)
    return cu_sqerr, du_sqerr


# =============================================================================
# PHASE 3b: CLOSED-LOOP SEQUENTIAL INFERENCE
#
# Root cause of DU miss: LSTM is a next-step predictor. When DU stress is
# sustained for ~177 steps, after the first 1-2 steps the LSTM hidden state
# tracks the elevated CPU value and starts predicting it correctly → sqerr≈0
# → score drops below threshold → all subsequent stress timesteps are missed.
#
# Fix: when an entity's score exceeds the threshold at step t, replace its
# actual input at t+1 with the model's own prediction (what normal looks like).
# The LSTM hidden state then stays anchored to normal behaviour, so actual
# (still-high) CPU values keep diverging from the prediction throughout the
# entire stress window → score stays elevated → detections continue.
#
# Important: error is always (prediction - ACTUAL), not vs the substituted
# input, so evaluation metrics are not artificially inflated.
#
# Calibration inference (phase_infer above) is unchanged — the cal stream is
# all-normal, so no replacement ever fires there.
# =============================================================================

def phase_infer_closed_loop(
    model: DUNormTopoAR,
    cu_s: np.ndarray,
    du_s: np.ndarray,
    cu_feat_norm: np.ndarray,
    du_feat_norm: np.ndarray,
    cu_thr: float,
    du_thr: float,
):
    model.eval()
    T  = len(cu_s)
    N  = du_s.shape[1]

    cu_sqerrs = np.zeros((T - 1, cu_s.shape[1]),    dtype=np.float32)
    du_sqerrs = np.zeros((T - 1, N, du_s.shape[2]), dtype=np.float32)

    h, c = model.init_state(1, DEVICE)
    # Require K consecutive anomalous timesteps before closed-loop replacement.
    DU_HYSTERESIS = 5
    du_anom_count = np.zeros(N, dtype=np.int32)

    CU_HYSTERESIS = 5
    cu_anom_count = 0

    # Feed actual values at t=0 to warm the LSTM
    cu_in = torch.tensor(cu_s[[0]], dtype=torch.float32).to(DEVICE)  # (1, cu_dim)
    du_in = torch.tensor(du_s[[0]], dtype=torch.float32).to(DEVICE)  # (1, N, du_dim)

    with torch.no_grad():
        for t in range(T - 1):
            cu_tok, du_tok = model.project_tokens(cu_in, du_in)
            cu_hat, du_hat, h, c, _ = model.step(cu_tok, du_tok, h, c)

            # Actual values at the NEXT timestep
            cu_next = torch.tensor(cu_s[[t + 1]], dtype=torch.float32).to(DEVICE)
            du_next = torch.tensor(du_s[[t + 1]], dtype=torch.float32).to(DEVICE)

            # Error always measured against actual (not against substituted input)
            cu_err = (cu_hat - cu_next).pow(2).cpu().numpy()[0]  # (cu_dim,)
            du_err = (du_hat - du_next).pow(2).cpu().numpy()[0]  # (N, du_dim)
            cu_sqerrs[t] = cu_err
            du_sqerrs[t] = du_err

            # Score each entity and decide the input for step t+1
            cu_score = float((cu_err / cu_feat_norm).max())

            if cu_score > cu_thr:
                cu_anom_count += 1
            else:
                cu_anom_count = 0

            if cu_anom_count >= CU_HYSTERESIS:
                cu_in = cu_hat
            else:
                cu_in = cu_next

            du_in = du_next.clone()

            for i in range(N):
                du_score_i = float((du_err[i] / du_feat_norm).max())

                if du_score_i > du_thr:
                    du_anom_count[i] += 1
                else:
                    du_anom_count[i] = 0

                # Enter closed-loop only after K consecutive anomaly steps
                if du_anom_count[i] >= DU_HYSTERESIS:
                    du_in[0, i] = du_hat[0, i]

    return cu_sqerrs, du_sqerrs

# =============================================================================
# PHASE 4: CALIBRATE THRESHOLDS ON HELD-OUT TRAIN-CAL STREAM
#
# Input is the squared-error arrays from inference on the held-out CAL portion
# of the TRAIN stream. By construction every row in this stream is normal
# (train.npz is normal-only), so no stress mask is needed.
#
# No test labels are used here — this matches deployment, where the live
# stream is unlabelled. Whatever shift exists between train and test (e.g.,
# cross-topology baseline shift) is now a real, visible failure mode of the
# model rather than something the threshold absorbs.
#
# feat_norm_calibrated:
#   Per-feature mean squared error on normal rows, floored at 10% of median.
#   Prevents zero-variance features from causing false explosions in lift_score.
#
# dual_threshold_from_val:
#   Separate percentile thresholds for CU and DU so neither type dominates.
# =============================================================================

def phase_calibrate(cu_sqerr, du_sqerr):
    # Skip first COLD_START_K rows: LSTM hidden state still warming on this
    # fresh stream (we did not preserve hidden state from training).
    cu_sqerr_n = cu_sqerr[COLD_START_K:]
    du_sqerr_n = du_sqerr[COLD_START_K:]

    n_normal = len(cu_sqerr_n)
    print(f"  Normal-calibration rows (held-out train tail): {n_normal}")
    if n_normal < 100:
        raise RuntimeError(f"Too few cal rows for threshold calibration: {n_normal}")

    # feat_norm_calibrated:  (dim,) — per-feature mean sqerr on normal rows, floored
    cu_feat_norm  = feat_norm_calibrated(cu_sqerr_n)                       # (cu_dim,)
    du_sqerr_flat = du_sqerr_n.reshape(-1, du_sqerr_n.shape[-1])
    du_feat_norm  = feat_norm_calibrated(du_sqerr_flat)                    # (du_dim,)

    # Lift scores on the cal stream → threshold at THRESHOLD_PCT
    cu_norm_scores = lift_score(cu_sqerr_n, cu_feat_norm)                  # (M,)
    du_norm_scores = lift_score(du_sqerr_flat, du_feat_norm)               # (M*N,)

    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    print(f"  CU threshold (p{CU_THRESHOLD_PCT:.1f}): {cu_thr:.4f}")
    print(f"  DU threshold (p{DU_THRESHOLD_PCT:.1f}): {du_thr:.4f}")
    return cu_feat_norm, du_feat_norm, cu_thr, du_thr

# =============================================================================
# PHASE 5: EVALUATE
#
# Score alignment:
#   cu_sqerr[t'] = error at timestep t'+1  →  label is cu_stress[t'+1]
#   We skip the first COLD_START_K score positions (= errors at timesteps 1..K).
#   Remaining eval range: errors at timesteps K+1 .. T-1.
# =============================================================================

def phase_evaluate(cu_sqerr, du_sqerr, cu_feat_norm, du_feat_norm,
                   cu_thr, du_thr, cu_stress, du_stress):
    N = du_sqerr.shape[1]

    # scores at index t' → timestep t'+1; skip first COLD_START_K positions
    start = COLD_START_K
    cu_sqerr_ev = cu_sqerr[start:]            # (T', cu_dim)
    du_sqerr_ev = du_sqerr[start:]            # (T', N, du_dim)

    # Labels: stress at the SAME timesteps (t'+1 shifted to 0-based eval array)
    # cu_stress has length T.  Eval covers timesteps start+1 .. T-1.
    cu_lbl = (cu_stress[start + 1:] == STRESS_TYPE).astype(int)   # (T',)
    du_lbl = (du_stress[start + 1:] == STRESS_TYPE)               # (T', N)

    assert len(cu_sqerr_ev) == len(cu_lbl), \
        f"Length mismatch: sqerr {len(cu_sqerr_ev)} vs label {len(cu_lbl)}"

    # CU: one score per timestep
    cu_scores = lift_score(cu_sqerr_ev, cu_feat_norm)    # (T',)
    cu_pred   = (cu_scores > cu_thr).astype(int)

    # DU: one score per (timestep, DU instance)
    du_scores = np.stack(
        [lift_score(du_sqerr_ev[:, i, :], du_feat_norm) for i in range(N)],
        axis=1,
    )  # (T', N)
    du_pred = (du_scores > du_thr).astype(int)

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

    print(f"\n  {'Entity':<12s}  {'anom':>6s}  {'TP':>6s}  {'FP':>6s}  "
          f"{'FN':>6s}  {'P':>5s}  {'R':>5s}  {'F1':>5s}")
    print(f"  {'-'*72}")
    metrics("CU", cu_pred, cu_lbl)
    for i in range(N):
        metrics(f"DU_{i}", du_pred[:, i], du_lbl[:, i].astype(int))

    # Combined: flag if ANY entity flagged, anomaly if CU or ANY DU stressed
    any_pred  = (cu_pred == 1) | du_pred.any(axis=1)
    any_lbl   = (cu_lbl  == 1) | du_lbl.any(axis=1)
    print(f"  {'-'*72}")
    metrics("ANY", any_pred.astype(int), any_lbl.astype(int))

    return cu_scores, du_scores, cu_pred, du_pred, all_metrics

# =============================================================================
# PLOTTING
# =============================================================================

def _shade(ax, t_array, mask, color, alpha, label=None):
    """Shade contiguous True regions of mask along t_array."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    in_block = False
    block_start = None
    first = True
    for i in range(len(mask)):
        if mask[i] and not in_block:
            block_start = t_array[i]
            in_block = True
        elif not mask[i] and in_block:
            ax.axvspan(block_start, t_array[i], color=color, alpha=alpha,
                       label=(label if first else None))
            in_block = False
            first = False
    if in_block:
        ax.axvspan(block_start, t_array[-1] + 1, color=color, alpha=alpha,
                   label=(label if first else None))


def phase_plot(cu_s_te, du_s_te, cu_stress, du_stress,
               cu_scores, du_scores, cu_pred, du_pred, cu_thr, du_thr,
               train_topos, test_topo):
    T      = len(cu_s_te)
    t_full = np.arange(T)
    # scores/preds cover timesteps COLD_START_K+1 .. T-1
    score_t = np.arange(COLD_START_K + 1, T)

    n_du = du_s_te.shape[1]
    entities = [("CU",   cu_s_te,           cu_stress,
                          cu_scores,          cu_pred,          cu_thr)]
    for i in range(n_du):
        entities.append((f"DU_{i}", du_s_te[:, i, :], du_stress[:, i],
                         du_scores[:, i],    du_pred[:, i],    du_thr))

    fig, axes = plt.subplots(len(entities), 2, figsize=(18, 4 * len(entities)),
                             sharex=False)
    fig.suptitle(
        f"Cross-topology {STRESS_NAMES[STRESS_TYPE]} stress detection\n"
        f"Train: {'+'.join(train_topos)}  →  Test: {test_topo}  "
        f"(v0 preprocessing, CU={len(np.arange(7)[CU_FEAT_SLICE])}feat DU={len(np.arange(37)[DU_FEAT_SLICE])}feat)",
        fontsize=12, y=1.01,
    )

    for row, (name, feat, stress_lbl, scores, pred, thr) in enumerate(entities):
        ax_f  = axes[row, 0]
        ax_sc = axes[row, 1]

               # ── feature panel ────────────────────────────────────────────────────
        _cu_labels = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_ratio"]
        _du_labels = (["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
                      + [f"pci_{i}" for i in range(22)]
                      + ["net_diff", "net_ratio"])
        feat_labels = _cu_labels if name == "CU" else _du_labels
        _colors_base = ["steelblue", "darkorange", "green", "purple", "brown", "crimson", "teal"]
        feat_colors = (_colors_base + [f"C{i}" for i in range(feat.shape[1] - len(_colors_base))])
        for fi in range(feat.shape[1]):
            ax_f.plot(t_full, feat[:, fi], color=feat_colors[fi], lw=0.7,
                      label=feat_labels[fi])
        _shade(ax_f, t_full,  stress_lbl == STRESS_TYPE, "red",    0.20, "GT anomaly")
        _shade(ax_f, score_t, pred.astype(bool),        "yellow", 0.40, "Detected")
        # clip y-axis to 5th–95th pct of non-outlier range so startup spikes don't squash the view
        lo = np.percentile(feat[COLD_START_K:], 1)
        hi = np.percentile(feat[COLD_START_K:], 99)
        pad = max((hi - lo) * 0.3, 0.5)
        ax_f.set_ylim(lo - pad, hi + pad)
        ax_f.set_ylabel(name, fontsize=10)
        ax_f.set_title(f"{name} — scaled features", fontsize=10)
        ax_f.legend(loc="upper right", fontsize=7, framealpha=0.7)

        # ── score panel ───────────────────────────────────────────────────────
        ax_sc.plot(score_t, scores, color="navy", lw=0.7, label="lift score")
        ax_sc.axhline(thr, color="red", ls="--", lw=1.2, label=f"threshold={thr:.4f}")
        _shade(ax_sc, score_t, stress_lbl[COLD_START_K + 1:] == STRESS_TYPE, "red", 0.20, "GT anomaly")
        ax_sc.set_yscale("log")
        ax_sc.set_ylim(bottom=max(scores[scores > 0].min() * 0.5, 1e-4))
        ax_sc.set_ylabel("lift score (log)", fontsize=9)
        ax_sc.set_title(f"{name} — anomaly score", fontsize=10)
        ax_sc.legend(loc="upper right", fontsize=7, framealpha=0.7)

    for col in range(2):
        axes[-1, col].set_xlabel("Timestep", fontsize=9)

    plt.tight_layout()
    out = Path(f"cross_anomaly_plot_{test_topo}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out.resolve()}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def run_one(train_topos, test_topo):
    """Run one leave-one-out configuration and return per-entity evaluation metrics.

    train_topos: list of topology names used for training (pooled).
    test_topo:   topology name held out for testing (must not be in train_topos).
    Returns:     dict mapping entity name → {tp, fp, fn, p, r, f1, anom}
                 e.g. {"CU": {...}, "DU_0": {...}, "ANY": {...}}
    """
    model_ckpt = Path(f"{'bidr_' if BIDIR else ''}dunorm_model_ckpt_test_{test_topo}.pt")   # per-split; delete to retrain
    cu_dim_info = len(np.arange(7)[CU_FEAT_SLICE])
    du_dim_info = len(np.arange(37)[DU_FEAT_SLICE])

    print(f"\n{'='*70}")
    print(f"  Cross-topology {STRESS_NAMES[STRESS_TYPE]} stress detection (multi-train + RobustScaler {PREPROCESS_VERSION})")
    print(f"  Train : {train_topos}  (normal only, pooled)")
    print(f"  Test  : {test_topo}    (unseen topology)")
    print(f"  Preprocessing : RobustScaler {PREPROCESS_VERSION}  (raw thresholds, no cold-start probe)")
    print(f"  CU features   : {cu_dim_info}   DU features: {du_dim_info}")
    print(f"  Device        : {DEVICE}")
    print(f"{'='*70}")

    # [1] Load + preprocess ALL train topologies ──────────────────────────────
    print(f"\n[1] Loading {len(train_topos)} train topologies, fitting {PREPROCESS_VERSION} scaler ...")
    train_zs = [load_npz(t, "train") for t in train_topos]
    bundle, train_streams = phase_preprocess(train_zs, train_topos)

    # [2] Per-topology train_fit / cal split ──────────────────────────────────
    print(f"\n[2] Per-topology fit/cal split (cal_frac={CAL_FRAC}) ...")
    fit_streams, cal_streams = [], []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_total = len(cu_s)
        n_cal   = int(round(CAL_FRAC * n_total))
        n_fit   = n_total - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        cal_streams.append((cu_s[n_fit:], du_s[n_fit:]))
        print(f"  topo[{i}] {train_topos[i]:18s} total={n_total}  fit={n_fit}  cal={n_cal}")
    n_fit_total = sum(len(f[0]) for f in fit_streams)

    # [3] Train (or load) with strict checkpoint safety check ─────────────────
    cu_dim = fit_streams[0][0].shape[1]
    du_dim = fit_streams[0][1].shape[2]
    if model_ckpt.exists():
        print(f"\n[3] Loading model from checkpoint ({model_ckpt}) — delete to retrain ...")
        ckpt = torch.load(model_ckpt, map_location=DEVICE)
        mismatches = []
        if ckpt.get("topos") != list(train_topos):
            mismatches.append(f"topos: ckpt={ckpt.get('topos')} vs current={list(train_topos)}")
        if ckpt.get("n_train_rows") != n_fit_total:
            mismatches.append(f"n_train_rows: ckpt={ckpt.get('n_train_rows')} vs current={n_fit_total}")
        if ckpt.get("cu_dim") != cu_dim:
            mismatches.append(f"cu_dim: ckpt={ckpt.get('cu_dim')} vs current={cu_dim}")
        if ckpt.get("du_dim") != du_dim:
            mismatches.append(f"du_dim: ckpt={ckpt.get('du_dim')} vs current={du_dim}")
        if ckpt.get("preprocess") != PREPROCESS_VERSION:
            mismatches.append(f"preprocess: ckpt={ckpt.get('preprocess')} vs current={PREPROCESS_VERSION}")
        if mismatches:
            raise SystemExit(
                f"\n  Refusing to load incompatible checkpoint {model_ckpt}:\n    "
                + "\n    ".join(mismatches)
                + f"\n  Delete {model_ckpt} and rerun to retrain from scratch."
            )
        model = DUNormTopoAR(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"],
                                 embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"\n[3] Training DUNormTopoAR on {len(train_topos)} topologies "
              f"(cu_dim={cu_dim}, du_dim={du_dim}, embed={EMBED_DIM}) ...")
        model = phase_train(fit_streams, train_topos, model_ckpt)

    # [4] Inference on each held-out CAL stream → pool → calibrate ────────────
    print(f"\n[4] Inference on held-out CAL streams (per topology, pooled for calibration) ...")
    cu_sqerr_pool, du_sqerr_pool = [], []
    for i, (cu_s_cal, du_s_cal) in enumerate(cal_streams):
        cu_sq, du_sq = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sqerr_pool.append(cu_sq[COLD_START_K:])
        du_sqerr_pool.append(du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1]))
        print(f"  topo[{i}] {train_topos[i]:18s} cu_sqerr {cu_sq.shape}  du_sqerr {du_sq.shape}")
    cu_sqerr_n   = np.concatenate(cu_sqerr_pool, axis=0)
    du_sqerr_flt = np.concatenate(du_sqerr_pool, axis=0)
    print(f"  Pooled cal rows: cu={cu_sqerr_n.shape[0]}  du(flat)={du_sqerr_flt.shape[0]}")

    print(f"\n[5] Calibrating thresholds on POOLED cal stream (no test labels) ...")
    cu_fn = feat_norm_calibrated(cu_sqerr_n)
    du_fn = feat_norm_calibrated(du_sqerr_flt)
    cu_norm_scores = lift_score(cu_sqerr_n,   cu_fn)
    du_norm_scores = lift_score(du_sqerr_flt, du_fn)
    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    print(f"  CU threshold (p{CU_THRESHOLD_PCT:.1f}): {cu_thr:.4f}")
    print(f"  DU threshold (p{DU_THRESHOLD_PCT:.1f}): {du_thr:.4f}")

    # [6] Transform test topology with the train-fitted scaler ────────────────
    print(f"\n[6] Transforming TEST topology ({test_topo}) with train-fitted scaler ...")
    test_z  = load_npz(test_topo, "test")
    cu_te, du_te, bid_te = slice_features(test_z)
    cu_s_te, du_s_te, kept_mask, _ = transform_stream(bundle, cu_te, du_te, bid_te)

    cu_stress = test_z["cu_stress"][kept_mask].astype(np.int32)   # (T,)
    du_stress = test_z["du_stress"][kept_mask].astype(np.int32)   # (T, N)
    n_du_te   = du_s_te.shape[1]
    print(f"  cu_s_te {cu_s_te.shape}  du_s_te {du_s_te.shape}  "
          f"(μ={cu_s_te.mean():+.3f}, σ={cu_s_te.std():.3f})")
    print(f"  CU {STRESS_NAMES[STRESS_TYPE]}-stress rows : {(cu_stress == STRESS_TYPE).sum()}")
    for i in range(n_du_te):
        print(f"  DU_{i} {STRESS_NAMES[STRESS_TYPE]}-stress rows: {(du_stress[:, i] == STRESS_TYPE).sum()}")
    train_N_set = sorted({s[1].shape[1] for s in train_streams})
    if n_du_te not in train_N_set:
        print(f"  NOTE: test N_DU={n_du_te} not in train N_DU set {train_N_set} — "
              "type-shared weights generalise by design")

    # [7] Sequential inference on full test stream ────────────────────────────
    # Thresholds are the raw pooled-cal percentiles (cu_thr, du_thr). The
    # cold-start probe that previously rescaled them was removed: for NET stress
    # it inflated the CU threshold ~20× and zeroed real detections. Cross-topology
    # baseline shift is handled by the pooled-train RobustScaler + type-shared /N
    # normalization, not by an inference-time threshold band-aid.
    if CLOSED_LOOP:
        print("\n[7] Running CLOSED-LOOP inference on test stream "
              "(anomalous inputs replaced with model predictions) ...")
        cu_sqerr, du_sqerr = phase_infer_closed_loop(
            model, cu_s_te, du_s_te, cu_fn, du_fn, cu_thr, du_thr
        )
    else:
        print("\n[7] Running open-loop inference on test stream ...")
        cu_sqerr, du_sqerr = phase_infer(model, cu_s_te, du_s_te)
    print(f"  cu_sqerr {cu_sqerr.shape}  du_sqerr {du_sqerr.shape}")

    if SAVE_ERRORS:
        feat_tag = f"f{cu_dim}"
        err_path = Path(f"recon_errors_{test_topo}_{feat_tag}.npz")
        np.savez(err_path,
                 cu_sqerr=cu_sqerr, du_sqerr=du_sqerr,
                 cu_stress=cu_stress, du_stress=du_stress,
                 cu_feat_norm=cu_fn, du_feat_norm=du_fn,
                 cu_thr=np.array([cu_thr]),
                 du_thr=np.array([du_thr]))
        print(f"  Errors saved → {err_path}")

    # [8] Evaluate ─────────────────────────────────────────────────────────────
    print("\n[8] Evaluation results ...")
    cu_scores, du_scores, cu_pred, du_pred, eval_metrics = phase_evaluate(
        cu_sqerr, du_sqerr, cu_fn, du_fn, cu_thr, du_thr, cu_stress, du_stress
    )

    # [8b] Diagnostics ────────────────────────────────────────────────────────
    cu_scores_cal = lift_score(cu_sqerr_n, cu_fn)
    du_scores_cal = lift_score(du_sqerr_flt, du_fn)
    pcts = [50, 90, 99, 99.9]
    print("\n[8b] Score distribution: CAL vs TEST")
    print(f"  {'':6s}  " + "  ".join(f"p{p:4.1f}" for p in pcts))
    cu_te_scores = lift_score(cu_sqerr, cu_fn)
    print(f"  {'CU cal':6s}  " + "  ".join(f"{np.percentile(cu_scores_cal, p):6.3f}" for p in pcts))
    print(f"  {'CU te ':6s}  " + "  ".join(f"{np.percentile(cu_te_scores, p):6.3f}" for p in pcts))
    du_sqerr_flat_te = du_sqerr.reshape(-1, du_sqerr.shape[-1])
    du_te_scores = lift_score(du_sqerr_flat_te, du_fn)
    print(f"  {'DU cal':6s}  " + "  ".join(f"{np.percentile(du_scores_cal, p):6.3f}" for p in pcts))
    print(f"  {'DU te ':6s}  " + "  ".join(f"{np.percentile(du_te_scores, p):6.3f}" for p in pcts))
    print(f"  CU thr (cal={cu_thr:.4f})  →  "
          f"fraction test above thr: {(cu_te_scores > cu_thr).mean():.3f}")
    print(f"  DU thr (cal={du_thr:.4f})  →  "
          f"fraction test above thr: {(du_te_scores > du_thr).mean():.3f}")

    print("\n[8c] Per-channel CU mean sq-error: CAL vs TEST (normal rows only)")
    cu_sqerr_normal_te = cu_sqerr[COLD_START_K:][cu_stress[COLD_START_K + 1:] == 0]
    print(f"  {'channel':>10s}  {'cal_mean':>10s}  {'te_normal_mean':>14s}  {'ratio':>6s}")
    feat_names = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_ratio"]
    for c in range(cu_sqerr.shape[1]):
        cal_m = float(np.mean(cu_sqerr_n[:, c]))
        te_m  = float(np.mean(cu_sqerr_normal_te[:, c])) if len(cu_sqerr_normal_te) else float("nan")
        ratio = te_m / cal_m if cal_m > 0 else float("nan")
        print(f"  {feat_names[c]:>10s}  {cal_m:>10.4f}  {te_m:>14.4f}  {ratio:>6.2f}x")

    # [9] Plot ─────────────────────────────────────────────────────────────────
    print("\n[9] Generating plot ...")
    phase_plot(cu_s_te, du_s_te, cu_stress, du_stress,
               cu_scores, du_scores, cu_pred, du_pred, cu_thr, du_thr,
               train_topos, test_topo)

    print("\nDone.")
    return eval_metrics


def main():
    if RUN_ALL_LOO:
        all_results = []
        for test_t in ALL_TOPOS:
            train_ts = [t for t in ALL_TOPOS if t != test_t]
            result = run_one(train_ts, test_t)
            all_results.append({"test_topo": test_t, "metrics": result})

        # Collect entity keys in consistent order (CU first, DUs, then ANY last)
        entity_keys: list = []
        for r in all_results:
            for k in r["metrics"]:
                if k not in entity_keys:
                    entity_keys.append(k)

        print(f"\n\n{'='*70}")
        print(f"  LEAVE-ONE-OUT SUMMARY  ({len(ALL_TOPOS)} configurations)")
        print(f"{'='*70}")
        header = f"  {'Test topology':<22s}" + "".join(f"  {e+' F1':>10s}" for e in entity_keys)
        print(header)
        print(f"  {'-'*68}")
        for r in all_results:
            row = f"  {r['test_topo']:<22s}"
            for e in entity_keys:
                if e in r["metrics"]:
                    row += f"  {r['metrics'][e]['f1']:>10.3f}"
                else:
                    row += f"  {'N/A':>10s}"
            print(row)
        print(f"  {'-'*68}")
    else:
        run_one(TRAIN_TOPOS, TEST_TOPO)


if __name__ == "__main__":
    main()
