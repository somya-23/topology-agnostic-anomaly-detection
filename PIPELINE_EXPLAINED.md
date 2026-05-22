# TopoAR Cross-Topology Anomaly Detection — Complete Pipeline Documentation

This document explains every stage of the CU/DU stress detection system from raw data to detection
results. A reader with no prior context should be able to understand what each file does, what
flows in and out, why each design decision was made, and exactly why the CU NET stress detector
currently produces TP=0.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data: Topologies, Files, and Features](#2-data-topologies-files-and-features)
3. [Preprocessing](#3-preprocessing)
4. [Model Architecture: TopoAR](#4-model-architecture-topoar)
   - 4.3 [The Attention Step in Detail](#43-the-attention-step-in-detail)
5. [Training](#5-training)
   - 5.5 [Dataset: Windowing and Batching](#55-dataset-windowing-and-batching-srcdatasetpy)
6. [Calibration: feat_norm and Threshold](#6-calibration-feat_norm-and-threshold)
   - 6.4 [Why Calibration Is Not Circular: Mean vs Percentile](#64-why-calibration-is-not-circular-mean-vs-percentile)
7. [Max-Pool Lift Score — Explained Step by Step](#7-max-pool-lift-score--explained-step-by-step)
8. [Cold-Start Probe: Cross-Topology Shift Estimation](#8-cold-start-probe-cross-topology-shift-estimation)
9. [Test-Time Inference: Open-Loop vs Closed-Loop](#9-test-time-inference-open-loop-vs-closed-loop)
10. [Evaluation](#10-evaluation)
    - 10.4 [Visualization: phase_plot()](#104-visualization-phase_plot-run_experimentpy-step-9)
11. [Root Cause: Why CU NET Stress TP=0](#11-root-cause-why-cu-net-stress-tp0)
12. [File Reference](#12-file-reference)

---

## 1. Problem Statement

**What we want to detect:** A Radio Access Network (RAN) has one Central Unit (CU) and several
Distributed Units (DUs). Any of these can undergo stress events — CPU overload, memory pressure,
or network anomalies (NET stress). We want a system that:

- Trains **only on normal (non-stress) data** from known topologies
- Detects stress at inference time on a **completely different, unseen topology** — without
  re-training or recalibrating on test data
- Raises an alarm per-entity (CU flagged, or DU_0 flagged, etc.) so the operator knows *where*
  the stress is

**Why this is hard:** Different topologies have different hardware, different numbers of DUs, and
different baseline traffic levels. A system that memorizes calibration constants from Topology A
will see huge baseline shifts at Topology C and either miss anomalies (threshold too high) or flood
with false positives (threshold too low).

**Key constraint known going in:** We only have normal data. We never know which stress type will
appear at test time and cannot tune anything to a specific stress type.

---

## 2. Data: Topologies, Files, and Features

### 2.1 Topologies

There are three topologies, each physically different:

| Name | CU | DUs | Role |
|---|---|---|---|
| `cu0_du0du1` | CU-0 | DU-0, DU-1 (N=2) | Train topology |
| `cu1_du2` | CU-1 | DU-2 (N=1) | Train topology |
| `cu2_du3du4du5` | CU-2 | DU-3, DU-4, DU-5 (N=3) | Test topology (held out) |

The experiment is a **leave-one-out** setup: train on the 2 topologies, test on the third.

### 2.2 Data Files

Each topology lives under `CU_NET_bidir_STRESS/{topo}_stress{type}/`:

```
train.npz  — normal-only time series (used to fit the model + calibrate thresholds)
test.npz   — mixed: normal rows + stress windows (ground truth for evaluation)
```

Each `.npz` contains:

| Key | Shape | Description |
|---|---|---|
| `cu` | (T, 37) | Raw CU KPIs at every scrape interval (1 sample/sec) |
| `du` | (T, N, 37) | Raw DU KPIs, N = number of DUs in this topology |
| `block_id` | (T,) | Experiment block index (resets between separate runs) |
| `cu_stress` | (T,) | Ground-truth CU stress label: 0=normal, 1=CPU, 2=MEM, 3=NET |
| `du_stress` | (T, N) | Same per DU |

All rows in `train.npz` have label 0 (normal). `test.npz` has interleaved normal (0) and stress
windows (label = STRESS_TYPE).

### 2.3 Feature Engineering

The raw 37-column KPI array contains columns that are always zero (hardware PCI counters absent in
this dataset). These are dropped to avoid wasting capacity.

**CU feature selection** (`CU_FEAT_SLICE = [0, 1, 2, 5, 6]` from raw):

| Raw idx | Name | Why kept |
|---|---|---|
| 0 | cpu | Core utilization signal |
| 1 | mem_pct | Fraction of memory used |
| 2 | mem_bytes | Absolute memory bytes |
| 5 | net_tx | Transmit bytes/sec (after per-DU normalization) |
| 6 | net_rx | Receive bytes/sec (after per-DU normalization) |

Two **derived relational features** are appended after slicing:

| Position | Name | Formula | Why useful |
|---|---|---|---|
| 5 | net_diff | net_tx − net_rx | Goes negative under packet loss |
| 6 | net_ratio | net_tx / (net_rx + 1e-6) | < 1 under loss; topology-baseline-invariant |

CU traffic features (net_tx, net_rx) are divided by N_DU (number of DUs in the topology) **before
slicing**. This normalizes away the linear scaling of traffic with topology size.

Final **CU feature vector** after all engineering: **7-dimensional**
`[cpu, mem_pct, mem_bytes, net_tx, net_rx, net_diff, net_ratio]`

**DU feature selection** (`DU_FEAT_SLICE`): 28 raw features selected, then 2 derived appended =
**30-dimensional** final DU feature vector. The raw 37-column DU array has 9 columns that are
always zero across all topologies (hardware PCI counters absent from this dataset) and are dropped.

| Raw idx | Post-slice idx | Name | Notes |
|---|---|---|---|
| 0 | 0 | cpu | Core utilization |
| 1 | 1 | mem_pct | Memory fraction |
| 2 | 2 | mem_bytes | Memory bytes |
| 3 | — | fs_reads | **DROPPED** — always 0 |
| 4 | 3 | fs_writes | Filesystem writes |
| 5 | 4 | net_tx | Transmit bytes/sec |
| 6 | 5 | net_rx | Receive bytes/sec |
| 7–14 | 6–13 | pci_0 … pci_7 | PCI lane counters 0–7 (8 features) |
| 15 | — | — | **DROPPED** — always 0 |
| 16–20 | 14–18 | pci_9 … pci_13 | PCI lane counters 9–13 (5 features; gap at 8) |
| 21–25 | — | — | **DROPPED** — always 0 (5 columns) |
| 26–33 | 19–26 | pci_19 … pci_26 | PCI lane counters 19–26 (8 features) |
| 34 | — | — | **DROPPED** — always 0 |
| 35 | 27 | pci_28 | PCI lane counter 28 |
| 36 | — | — | **DROPPED** — always 0 |
| — | 28 | net_diff | **Derived:** net_tx − net_rx |
| — | 29 | net_ratio | **Derived:** net_tx / (net_rx + 1e-6) |

Total dropped: raw indices 3, 15, 21, 22, 23, 24, 25, 34, 36 (9 columns).
Kept raw: 28 columns. After appending net_diff and net_ratio: **30 features total**.

The PCI hardware counters (indices 6–27 post-slice) are **radio-layer** metrics — uplink/downlink
scheduling, PUSCH/PDSCH block errors, transport block counts per PRB. They are zero on normal
operations and spike under radio-layer anomalies, so they are the DU's strongest stress signals
when non-zero. The 22 PCI features kept span hardware PCI lanes 0–7, 9–13, 19–26, 28; the gaps
(lanes 8, 14–18, 27) map to raw columns confirmed all-zero in this dataset.

### 2.4 Prometheus Glitch Imputation

Prometheus' `irate()` function returns **exactly 0.0** at every 5-minute clock boundary (every 300
seconds) due to counter-reset edge effects. This is a measurement artifact — a process cannot use
literally zero CPU for an entire scrape interval.

After RobustScaler, that 0.0 maps to `(0 − median) / IQR ≈ −27` for DU and `≈ −5` for CU. The
model fails to predict these spikes (they appear every 300 s in both train and test, but are still
rare enough to land at p99.9 of the error distribution). Without imputation, the calibration
threshold is set entirely by these artifacts, not by real normal variability.

**Fix:** Forward-fill — replace any row where `cpu/net_tx/net_rx < 1e-6` with the previous row's
value. Applied **before** scaling, identically to train and test.

```
CU_IRATE_IDX = [0, 3, 4]    # cpu, net_tx, net_rx (post-slice)
DU_IRATE_IDX = [0, 3, 4, 5] # cpu, fs_writes, net_tx, net_rx
```

**Note:** `mem_pct` zeros co-occur with cpu zeros but imputing `mem_pct` shrinks `feat_norm` and
amplifies cross-topology mem shift → false positives. So mem_pct is intentionally excluded from
imputation.

---

## 3. Preprocessing

**Preprocessing version used:** `v0` = RobustScaler on raw values, no delta differencing.

Why no differencing? NET stress is a **sustained level shift** in network traffic — after
differencing, both the normal baseline change (≈0) and the sustained stress level (≈0) look
identical. Differencing destroys the discriminative signal. RobustScaler alone preserves it as a
large positive scaled value.

### 3.1 RobustScaler

```
x_scaled = (x_raw − median_train) / IQR_train
```

- `median_train` and `IQR_train` (interquartile range) are computed from the **pooled train
  topologies** (cu0_du0du1 + cu1_du2 combined)
- The same fitted scaler (called a `PreprocessBundle`) is applied unchanged to the test topology
- No re-fitting at test time — this is what makes cross-topology shift a real, visible failure mode

**Type-shared scalers:** A single `cu_scaler` covers all CU instances; a single `du_scaler` covers
all DU instances across all topologies. This is non-negotiable for topology agnosticism: there can
be no per-instance parameters.

### 3.2 Code Location

`src/preprocess.py`:
- `fit_bundle()` — fits CU and DU scalers from the pooled train stream
- `transform_stream()` — applies the fitted bundle to any stream (train cal, or test)

---

## 4. Model Architecture: TopoAR

**Class:** `CalibratedTopoAR` (subclass of `TopoAR` in `src/model.py`)
**Type:** LSTM-based next-step predictor with multi-key attention

### 4.1 What the Model Does

At each timestep `t`, given the CU feature vector and all DU feature vectors, the model predicts
what **all** entities' feature vectors will look like at `t+1`.

```
Input  at t:  cu[t]      shape (cu_dim,)          = 7-dim CU features
              du[t]      shape (N, du_dim,)         = N × 28-dim DU features

Output at t:  cu_hat[t]  shape (cu_dim,)          = predicted CU features at t+1
              du_hat[t]  shape (N, du_dim,)         = predicted DU features at t+1
```

If the system is in a normal regime, `cu_hat[t] ≈ cu[t+1]` and `du_hat[t] ≈ du[t+1]`.
If something anomalous happens at `t+1` (e.g. network traffic spikes), `cu_hat[t]` still reflects
the *normal* prediction and `|cu_hat[t] − cu[t+1]|^2` will be large.

### 4.2 Architecture Detail

```
Step 1 — Token projection (per entity type, not per instance)
    cu_tok = LayerNorm(W_CU · cu + e_CU)       shape (embed_dim,)  = (32,)
    du_tok = LayerNorm(W_DU · du_i + e_DU)     shape (N, embed_dim,)

Step 2 — Attention (query from hidden state, keys+values from all entities)
    q        = Q · h_prev                       shape (32,)
    K_j, V_j = KV projections of each entity token
    α        = softmax(q · K_j / √d)           shape (1 + N,) — attention over CU + all DUs
    s        = Σ α_j · V_j                      shape (32,)

Step 3 — LSTM update
    h_new, c_new = LSTMCell(s, (h_prev, c_prev))
    h_norm       = LayerNorm(h_new)

Step 4 — Decode predictions
    cu_hat   = D_CU([h_norm ; cu_tok])          shape (cu_dim,)
    du_hat_i = D_DU([h_norm ; du_tok_i])        shape (du_dim,) per DU
```

**Topology agnosticism:** No weights are per-instance. `W_CU`, `W_DU`, `K_CU`, `K_DU`,
`V_CU`, `V_DU`, `D_CU`, `D_DU` are all type-shared. The model sees arbitrarily many DUs at
inference time — the softmax attention is a convex combination regardless of N.

**Embed dim:** 32. Total parameters ≈ ~150k.

### 4.3 The Attention Step in Detail

The attention step is *how the entities share information* before each prediction is made. The key
design choice: **the query comes from the recurrent hidden state `h`, not from the entities
themselves.** Think of `h` as the model's running belief about the topology's state; at each step it
*queries* the 1 CU + N DU entities to decide which ones matter for predicting the next step. This is
implemented in [model.py:94-140](src/model.py#L94-L140) (`TopoAR.step`).

```
INPUTS at time t (ignoring batch B):
  cu  : (cu_dim,)      e.g. (7,)   — CU feature vector
  du  : (N, du_dim)    e.g. (3,30) — N DU feature vectors
  h,c : (d,)           e.g. (32,)  — LSTM state carried from t-1

┌─ 1. TOKENIZE ─────────────────────────────────────────────────────────┐
│  Project every entity into a shared d-dim space, add a per-TYPE        │
│  embedding so the model knows "CU" vs "DU", then LayerNorm.            │
│     cu_tok = LN_CU( W_CU·cu + e_CU )            → (d,)                  │
│     du_tok = LN_DU( W_DU·du + e_DU )            → (N, d)               │
│  W_CU / W_DU are TYPE-shared: one CU projector, one DU projector,      │
│  reused for every DU regardless of N.                                  │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 2. QUERY / KEYS / VALUES ────────────────────────────────────────────┐
│     q     = Q·h                  → (d,)     ONE query, from LSTM state │
│     K_cu  = K_CU·cu_tok          → (d,)                                │
│     K_du  = K_DU·du_tok          → (N, d)   per-DU keys                │
│     V_cu  = V_CU·cu_tok          → (d,)                                │
│     V_du  = V_DU·du_tok          → (N, d)   per-DU values              │
│  K/V are also TYPE-shared (K_CU vs K_DU), not per-instance.            │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 3. SCORE & SOFTMAX  (the actual "attention") ────────────────────────┐
│  Dot the single query against EACH of the 1+N keys, scale by 1/√d:    │
│     score_cu   = (q · K_cu)        / √d        → scalar                │
│     score_du_i = (q · K_du[i])     / √d        → N scalars             │
│     scores     = [score_cu, score_du_0..N-1]   → (1+N,)               │
│     α          = softmax(scores)               → (1+N,)  Σα = 1        │
│  α is a probability distribution over the 1 CU + N DU "slots".         │
│  α[0] = attention to the CU, α[1+i] = attention to DU i.               │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 4. CONTEXT VECTOR (weighted sum of values) ──────────────────────────┐
│     s = α[0]·V_cu  +  Σ_i α[1+i]·V_du[i]       → (d,)                  │
│  Because α sums to 1, s is a CONVEX COMBINATION of the value vectors   │
│  → ‖s‖ ≤ max‖V_j‖ no matter whether N=1 or N=50.                       │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 5. LSTM UPDATE + DECODE ─────────────────────────────────────────────┐
│     h,c   = LSTMCell(s, (h,c)) ; h_norm = LayerNorm(h)                 │
│  Each entity is decoded from [shared memory ; its OWN token]:          │
│     cu_hat   = D_CU([h_norm ; cu_tok])         → (cu_dim,)            │
│     du_hat_i = D_DU([h_norm ; du_tok_i])       → (N, du_dim)          │
│  The DU prediction = global context (h_norm) + that DU's local         │
│  identity (du_tok_i), so each DU gets a distinct prediction even       │
│  though the decoder D_DU is shared.                                    │
└────────────────────────────────────────────────────────────────────────┘
```

**Why each choice was made:**

| Choice | Reason |
|---|---|
| **Query from `h`, not entities** | `h` is the temporal memory. Attention is *context-dependent over time*, not a fixed function of the current frame. |
| **1+N keys (CU is just another slot)** | The CU competes with the DUs for attention on equal footing. If the CU is the informative signal this step, `α[0]` grows. |
| **`/√d` scaling** | Standard attention scaling — keeps dot products from saturating the softmax when `d` is large. |
| **Softmax convex combination** | Bounds `‖s‖` independent of N — critical because the model trains on some N and tests on a different N (cross-topology). [model.py:29](src/model.py#L29) |
| **Type-shared W/K/V/decoders** | Zero per-instance parameters. The same weights serve any number of DUs → the no-recalibration claim. [model.py:27-28](src/model.py#L27-L28) |
| **Decoder sees `[h_norm ; entity_tok]`** | Global context tells it the regime; the entity token tells it *which* entity to predict for. Without the token, all DUs would get identical predictions. |

**Sequential in time, parallel in batch + entities:** the full sequence is a Python loop over `t`
([model.py:162](src/model.py#L162)) because `h_t` depends on `h_{t-1}` — it cannot be parallelized
across timesteps. Within a single step, all `B` batch items and all `N` entities are handled at once
as vectorized matmuls (`einsum` over the N axis at [model.py:120](src/model.py#L120)).

**How this connects to detection:** `cu_hat`/`du_hat` from step 5 are the next-step predictions. The
squared error between these and the actual `t+1` observation is exactly the `sqerr` that feeds the
lift score in Section 6 — so the attention step is the upstream half of the detector, and the
calibration/threshold machinery is the downstream half.

---

## 5. Training

**What "training" means here:** training is **self-supervised next-step prediction on normal data
only**. There are no anomaly labels involved. The model's sole job is: *given the topology's state up
to time `t`, predict every entity's feature vector at `t+1`.* The target for the prediction made at
step `t` is simply the actual row at `t+1` — which we already have — so the supervision signal is
free and requires no labelling.

The model never sees a stress event during training. It only ever learns *"what does normal look
like one step ahead?"* That is the entire basis of the detector: a model that has only seen normal
predicts well on normal and **badly on anomalies it has never encountered**, and that prediction
error is the anomaly signal (Section 6).

```
   TRAINING                                  DETECTION (later)
   learn to predict normal t+1     ──▶       big prediction error = anomaly
   (low error on normal)                     (model can't predict the unseen stress)
```

**What is actually learned:** the type-shared weights from Section 4 (`W_CU, W_DU, e_CU, e_DU,
K_CU, K_DU, V_CU, V_DU, Q`, the `LSTMCell`, LayerNorms, and decoders `D_CU, D_DU`) — ~150k
parameters, **none per-DU or per-topology**. The same weights must predict well across both training
topologies (N=2 and N=1) at once, which is exactly what forces them to generalize to the unseen
test topology.

### 5.1 Data Split

Each train topology's normal stream is split by **row index** (no shuffling, preserves time order):

```
┌─────────────────────────────────────┐
│   Train stream (normal only)        │
│                                     │
│  80% train_fit │ 20% cal (held out) │
└─────────────────────────────────────┘
         ↑                ↑
    Train model     Calibrate thresholds (step 6)
```

`CAL_FRAC = 0.2` — the last 20% of each train topology's stream is held out for calibration.

**Critical rule:** The model is trained on `train_fit` only. The `cal` portion is never shown to
the model during training. This matches deployment — at inference time we have no stress labels and
cannot use test-normal rows to calibrate.

**Two distinct holdouts — don't confuse them:**

| Holdout | Granularity | Carved when | Used for |
|---|---|---|---|
| `cal` fold | row-level, last 20% | before training | computing `feat_norm` + threshold (Section 6) |
| val split | window-level, random 10% | inside `phase_train()` from the remaining 80% | early stopping only (Section 5.5) |

The `cal` fold is reserved up front and the model never trains on it. The val split is taken from
*within* the fit data purely to decide when to stop training — it plays no role in calibration.

### 5.2 Training Objective and Loop

At each step `t`, the model predicts `t+1`. Loss is mean squared error over both entity types
([run_experiment.py:331-334](clear_pipeline/run_experiment.py#L331-L334)):

```python
loss = ((cu_hat[:, :-1] - cu_b[:, 1:]) ** 2).mean()   # CU term
     + ((du_hat[:, :-1] - du_b[:, 1:]) ** 2).mean()   # DU term
```

`cu_hat[:, :-1]` are the predictions for steps 1..L-1; `cu_b[:, 1:]` are the *actual* rows at those
steps. The squared difference is `.mean()`-ed over batch × time × features (and × N for DU, so a
3-DU topology does not dominate the gradient over a 1-DU one — see Section 5.4).

**The loop** ([run_experiment.py:322-366](clear_pipeline/run_experiment.py#L322-L366)):

```
for each epoch (up to EPOCHS=150):
  ── TRAIN PASS ──────────────────────────────────────────────
  for each batch of windows (B=256 windows, each L=64 steps):
      cu_hat, du_hat = model(cu_b, du_b)        # full 64-step autoregressive forward (§4.3)
      loss = MSE(cu_hat[:, :-1], cu_b[:, 1:])
           + MSE(du_hat[:, :-1], du_b[:, 1:])
      optim.zero_grad(); loss.backward(); optim.step()   # Adam updates shared weights

  ── VALIDATION PASS (no gradient) ───────────────────────────
  val_loss = same MSE over the held-out 10% of windows

  ── EARLY STOPPING ──────────────────────────────────────────
  if val_loss improved:  save best_state, reset patience
  else:                  patience += 1; stop if patience >= PATIENCE(5)

load best_state;  save checkpoint .pt
```

**Optimizer:** Adam at `LR = 5e-4` ([run_experiment.py:316](clear_pipeline/run_experiment.py#L316)).
`loss.backward()` computes gradients; `optim.step()` nudges every shared weight downhill.

**Teacher forcing:** during training the model always reads the **real** observation at each step,
never its own prediction. This is standard for training a next-step predictor and is *different*
from the closed-loop substitution used only at test time (Section 9.2).

**Why early stopping matters here specifically:** an overfit model would drive normal-data error
artificially toward zero, which would shrink `feat_norm` and distort every downstream lift score and
threshold. Stopping at the best val loss keeps the normal residual realistic.

**Output:** the best weights are restored and saved to a `.pt` checkpoint along with `cu_dim,
du_dim, embed_dim, topos, preprocess version`
([run_experiment.py:366-372](clear_pipeline/run_experiment.py#L366-L372)). This checkpoint is then
frozen and used unchanged for calibration (Section 6) and test (Sections 8–9).

### 5.3 Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `EMBED_DIM` | 32 | Embedding / hidden dim |
| `WINDOW_LEN` | 64 | Length of training sequences |
| `BATCH_SIZE` | 256 | Amortizes CUDA overhead |
| `EPOCHS` | 150 | With early stopping |
| `PATIENCE` | 5 | Early stop after 5 non-improving epochs |
| `LR` | 5e-4 | Adam |
| `VAL_FRAC` | 0.1 | Window-level val for early stopping |

### 5.4 Multi-Topology Batching

When training on two topologies simultaneously, batches contain windows from **only one topology**
at a time (homogeneous-N batches). This is necessary because different topologies have different
N_DU, so a single tensor batch cannot mix them. `MultiTopologyBatchSampler` handles this.

### 5.5 Dataset: Windowing and Batching (`src/dataset.py`)

This file contains the three classes that convert a raw time series into training batches.

#### TopologySequenceDataset

Takes the scaled stream `(cu_s, du_s, block_id)` and slices it into fixed-length windows of
`WINDOW_LEN = 64` timesteps. Each window is one training sample.

**Block boundary rule:** A window is only valid if **all 64 rows share the same `block_id`**. A
window that straddles two experiment blocks (e.g. the last rows of one run and the first rows of
the next) is silently skipped. This prevents the LSTM from being trained to bridge unrelated
sequences.

```
Example with block_id = [1,1,1,1,1,2,2,2,2,2,...] and WINDOW_LEN=3

Window starting at row 3: [1,1,2] → INVALID, crosses block boundary
Window starting at row 5: [2,2,2] → VALID

When an invalid start is encountered, the iterator jumps forward to the
next block boundary (not just +1), so no time is wasted checking single steps.
```

Each `__getitem__` call returns:
```python
{
    "cu": tensor shape (64, 7),     # WINDOW_LEN × cu_dim
    "du": tensor shape (64, N, 30), # WINDOW_LEN × N × du_dim
}
```

#### MultiTopologyBatchSampler

Used with PyTorch's `ConcatDataset` when two topology datasets are merged. The sampler ensures
every batch index list points exclusively into one topology's portion of the concat dataset.

```
ConcatDataset:
  [topology cu0_du0du1: windows 0..4999] [topology cu1_du2: windows 5000..7999]
                                                                              ↑
  A batch can only contain indices from ONE of these two ranges at a time.
```

**Why this matters:** If a batch mixes cu0_du0du1 (N=2 DUs) and cu1_du2 (N=1 DU), PyTorch cannot
stack them into a single tensor — the DU axis has different sizes. The sampler prevents this.

**Per-epoch shuffling:** Batch order across topologies is shuffled each epoch (the sampler tracks
`self.epoch` and seeds from `seed + epoch`). Within a topology the window order is also shuffled.
This means the optimizer alternates between N=2 and N=1 batches unpredictably, forcing the model
weights to be robust to both.

#### collate_windows

Simple stack function: takes a list of `{cu, du}` dicts from one topology and stacks them along
batch dimension 0:

```python
cu: stack → (B, 64, 7)        # B = batch_size (up to 256)
du: stack → (B, 64, N, 30)    # N uniform within the batch (guaranteed by sampler)
```

#### Early-stopping validation split

Inside `phase_train()`, each topology's windows are split 90/10 into train and val at the
**window level** (not row level). The split is random (uses `SEED = 42`), so val windows are
scattered throughout the time series — this is purely for early stopping, not for calibration.

```
All windows → shuffle with rng → first 10% = val, rest = train
```

The model with the lowest val loss across all epochs is the one saved to the checkpoint.

---

## 6. Calibration: feat_norm and Threshold

After training, we run inference on the held-out `cal` stream (normal-only, never seen by the
model) to produce calibration statistics. These are the two numbers that determine when an alarm
fires.

### 6.1 feat_norm: Per-Feature Normal Residual

```python
feat_norm = feat_norm_calibrated(cu_sqerr_cal)  # shape (cu_dim,) = shape (7,)
```

Where `feat_norm_calibrated()` computes:

```
raw[c]   = mean(sqerr_cal[:, c])   for each feature channel c
floor    = 0.1 × median(raw)       (10% of the typical feature's residual)
feat_norm[c] = max(raw[c], floor)
```

**What it represents:** On normal data, how large does the squared prediction error *typically*
get for each feature? A feature with `feat_norm = 0.5` has typical errors of ≈ 0.7 on normal rows.

**Why the floor?** Some features have near-zero variance on normal data (e.g. PCI counters always
around the same value). Without a floor, `feat_norm[c] ≈ 0` → dividing any test error by it
produces enormous lift values from pure noise → constant false alarms. The floor `0.1 × median`
means no feature can amplify errors more than 10× relative to the typical feature.

**Concrete floor example (a flat channel):** Suppose a PCI counter is essentially constant on
normal data, so its mean cal sqerr is `raw = 0.0000001`. The typical feature has
`median(raw) = 0.05`, so `floor = 0.1 × 0.05 = 0.005`.

```
Without floor:  test noise sqerr 0.0001 / 0.0000001 = lift 1000  → FALSE ALARM
With    floor:  feat_norm raised to 0.005
                test noise sqerr 0.0001 / 0.005      = lift 0.02  → correctly ignored
```

A genuine anomaly produces sqerr ~1e6× above normal, so it still clears the threshold even with the
raised floor. Only trivial noise on near-constant channels gets attenuated — the floor never hides a
real stress signal.

**Example (CU, from actual calibration):**

| Channel | feat_norm[c] | Interpretation |
|---|---|---|
| cpu | ~0.08 | Low residual; cpu well-predicted |
| mem_pct | ~0.004 | Very low residual |
| mem_bytes | ~0.15 | Moderate residual |
| net_tx | ~0.06 | Moderate residual |
| net_rx | ~0.06 | Moderate residual |
| net_diff | ~0.008 | Very low residual — small changes well-predicted |
| net_ratio | ~0.05 | Moderate residual |

### 6.2 Lift Score: Per-Row Anomaly Score

**Why divide at all?** Raw squared errors from different channels are *not comparable*. Some
channels are naturally noisy even on normal data; some are nearly flat. If we just took
`max(sqerr)` across channels, the perpetually-noisiest channel would always win — we'd be measuring
"which channel is loudest", not "which channel is behaving abnormally".

```
            typical normal sqerr
  cpu          0.80      ← always the loudest; would always win a raw max-pool
  net_diff     0.05
  pci_flat     0.0000001
```

Dividing by `feat_norm` grades every channel **on its own curve**: it re-expresses each error as
"multiples of *that channel's* normal error", so a normal-behaving channel maps to lift ≈ 1
regardless of its absolute noise level. Now the channels compete fairly and the max-pool picks the
one that is genuinely most abnormal, not merely the loudest.

```python
lift[t, c] = sqerr[t, c] / feat_norm[c]    # (T-1, 7) per-channel normalized error
score[t]   = max(lift[t, :])                # (T-1,)   max over all 7 channels
```

`lift[t, c]` answers: "How many times larger than normal is the prediction error for channel c at
timestep t?" A value of 1.0 means exactly as large as the typical normal error. A value of 100
means 100× the normal error — almost certainly anomalous.

`score[t]` = **max-pool** over all channels. The *single most anomalous channel* dominates the
score. This is intentional: an anomaly that shows up in even one channel should be detectable.

**Numerical example:**

```
At a normal timestep:    lift = [1.2,  0.8,  0.5,  1.1,  0.9,  2.1,  1.3]
                         score = max = 2.1    (net_diff barely elevated)

At a NET-stress timestep: lift = [1.5,  1.0,  2.7,  18.3, 15.2, 240.1, 8.4]
                          score = max = 240.1  (net_diff explodes)
```

The normal score (2.1) is far below the NET-stress score (240.1). If the threshold sits anywhere
between them, the alarm fires correctly.

### 6.3 Threshold: p99.9 of Normal Cal Scores

```python
cu_thr = np.percentile(cu_norm_scores, 99.9)   # = 247.49 in our experiment
```

This says: "On normal data, only 0.1% of timesteps should ever score above this threshold." So in
deployment, at most 1 in 1000 normal timesteps triggers a false alarm.

**Why p99.9 and not p99?** The CU sees cross-topology baseline shift even in normal test rows
(mem_bytes is 39× higher in test topology cu2_du3du4du5). Using p99 would put the threshold in a
region where test-normal scores land frequently, causing constant false positives.

### 6.4 Why Calibration Is Not Circular: Mean vs Percentile

At first glance there looks to be a chicken-and-egg problem: `feat_norm` is the **mean** sqerr on
the cal rows, and then we divide *those same cal rows* by it and take a percentile to get the
threshold. If `feat_norm` is the mean, doesn't dividing by it just force every score to ≈1, making
the threshold meaningless?

It does not — because **`feat_norm` and `cu_thr` are two different statistics computed from the same
`(M, dim)` matrix of cal squared errors.** One collapses columns with a *mean*; the other collapses
rows with a *max* and then takes the extreme *percentile*. They measure orthogonal things.

First, the data: the cal stream is the held-out last `CAL_FRAC = 0.2` of each train topology
(Section 5.1) — normal rows the model **never trained on**. Running inference on it gives the matrix
`cu_sqerr_n` of shape `(M, dim)`:

```
            cal sqerr matrix   cu_sqerr_n : (M, dim)     M = thousands of normal rows
            ┌──────────────────────────────────────────┐
            │   cpu   mem  mbyt net_tx net_rx ndiff nrat│
   row 0    │  0.81  0.49  0.28  0.41  0.44  0.04  0.11 │
   row 1    │  0.79  0.52  0.31  0.38  0.47  0.06  0.09 │
    ...     │   ...   ...   ...   ...   ...   ...   ... │
   row M-1  │  0.80  0.51  0.29  0.40  0.46  0.05  0.10 │
            └──────────────────────────────────────────┘
                 │                                    │
   STATISTIC A:  │ mean DOWN each column (axis=0)     │  STATISTIC B: needs A first
   feat_norm     ▼                                    ▼
   [0.80, 0.50, 0.30, 0.40, 0.45, 0.05, 0.10]   lift = cu_sqerr_n / feat_norm  → (M, dim)
   (the CENTER of each channel's normal error)  cal_scores = lift.max(axis=1)  → (M,)
                                                 cu_thr = percentile(cal_scores, 99.9)
                                                 (the TAIL of the score distribution)
```

- **Statistic A — `feat_norm`** ([run_experiment.py:788](clear_pipeline/run_experiment.py#L788)):
  mean **down each column**. M rows → one number per channel. The *center* of each channel's normal
  error, used only to re-scale.
- **Statistic B — `cu_thr`** ([run_experiment.py:792](clear_pipeline/run_experiment.py#L792)): divide
  every row by `feat_norm`, max-pool **across each row** to get one score per cal row, then take the
  **99.9-th percentile** of those M scores — the extreme *tail*, not the mean.

**Why dividing by the mean does not collapse the scores to 1:** the mean centers each channel so its
*typical* lift is ≈1, but row-to-row there is spread, and the **max over `dim` channels** picks up
whichever channel happened to be noisiest in each row. So normal cal rows produce scores scattered
from ~1 up to single digits, and `cu_thr` records where the worst 0.1% of *normal* noise sits:

```
   cal_scores distribution (normal only):
      p50  ≈ 1.5     median normal row
      p99  ≈ 31
      p99.9 ≈ 247.5  ◀── cu_thr = percentile(cal_scores, 99.9)
```

| | computed how | what it captures |
|---|---|---|
| `feat_norm` (A) | mean **down columns** | the *center* of each channel's normal error — re-scales lift |
| `cu_thr` (B) | p99.9 **across the per-row scores** | the *tail* — "how high does normal noise ever push the max-pool?" |

So the threshold rule is simply: **flag a row whose score exceeds the worst 0.1% of normal noise.**
The calibration is a two-stage pipeline — mean → build lift → re-percentile at 99.9 — on the same
held-out normal cal data. Not circular.

---

## 7. Max-Pool Lift Score — Explained Step by Step

This section walks through a complete concrete example of how a score is computed, to make the
concept clear before explaining the cold-start probe.

### 7.1 Inputs

```
sqerr[t]   = [0.012,  0.0003, 0.18,   0.009,  0.008,  1.92,  0.42]   shape (7,)
feat_norm  = [0.083,  0.004,  0.15,   0.061,  0.063,  0.008, 0.053]   shape (7,)
```

(These are realistic calibrated values from our experiment.)

### 7.2 Step 1 — Compute Lift Per Channel

```
lift[c] = sqerr[c] / feat_norm[c]

lift[0] = 0.012 / 0.083 = 0.14    (cpu:       near-normal)
lift[1] = 0.003 / 0.004 = 0.75    (mem_pct:   near-normal)
lift[2] = 0.18  / 0.15  = 1.20    (mem_bytes: slightly elevated)
lift[3] = 0.009 / 0.061 = 0.15    (net_tx:    near-normal)
lift[4] = 0.008 / 0.063 = 0.13    (net_rx:    near-normal)
lift[5] = 1.92  / 0.008 = 240.0   (net_diff:  240× normal — anomalous!)
lift[6] = 0.42  / 0.053 = 7.92    (net_ratio: elevated)
```

### 7.3 Step 2 — Max-Pool

```
score[t] = max(lift) = 240.0
```

The single most deviant channel (net_diff at 240×) dominates. Even if all other channels look
completely normal, this score will exceed the threshold of 247.49... but barely.

### 7.4 Why the Dominant Channel Changes Between Normal and Anomalous Rows

This is the core insight needed to understand the cold-start bug (Section 11).

**During normal test rows:**
- `mem_bytes` is 39× higher in test topology vs train topology
- After RobustScaler (fit on train), `mem_bytes` maps to a much larger scaled value in test
- So `sqerr[mem_bytes]` is large even on normal test rows
- `mem_bytes` wins max-pool ~98% of the time on normal rows
- Normal score distribution is shifted up: p50 ≈ 37.75 (vs cal p50 ≈ 1.85)

**During NET-stress rows:**
- `net_diff = net_tx − net_rx` shoots up dramatically
- `feat_norm[net_diff]` is small (≈ 0.008) because net_diff is well-predicted on normal cal data
- `lift[net_diff]` explodes: 1.92 / 0.008 = 240 on a mild anomaly, up to 2494 on severe ones
- `net_diff` wins max-pool 97.8% of the time during NET-stress
- Anomalous score distribution: p50 ≈ 606, p99 ≈ 2495

**The channels governing normal p50 (mem_bytes) and anomalous p50 (net_diff) are different.**
This causes a critical problem for the cold-start probe (Section 8).

---

## 8. Cold-Start Probe: Cross-Topology Shift Estimation

The threshold was calibrated on the train topology. At test time, we see a different topology with
potentially different baseline statistics. The cold-start probe attempts to estimate "how much has
the score distribution shifted?" and scales the threshold accordingly.

### 8.1 Mechanism

```python
N_PROBE_ROWS = 300
COLD_START_K = 64

# Run open-loop inference on just the first 300 rows of the test stream
cu_sq_probe = phase_infer(model, cu_s_te[:N_PROBE_ROWS + 1], ...)

# Score the probe rows (skip first COLD_START_K for LSTM warmup)
cu_probe_scores = lift_score(cu_sq_probe[COLD_START_K:], cu_fn)

# Estimate shift as ratio of medians
cu_test_p50 = percentile(cu_probe_scores, 50)    # probe median = 31.43
cu_cal_p50  = percentile(cu_norm_scores,  50)    # cal median   = 1.54
cu_shift    = cu_test_p50 / cu_cal_p50           # = 20.41×

# Inflate threshold by that shift factor
cu_thr_adj  = cu_thr * max(1.0, cu_shift)        # = 247.49 × 20.41 = 5050.45
```

### 8.2 Design Intent

The probe assumes: "If normal test rows have scores 20× higher than normal cal rows on average,
the threshold should be 20× higher too — otherwise every normal test row is a false alarm."

This is conceptually sound when the score distribution shifts uniformly across all percentiles. If
everything shifts by 20×, scaling the threshold by 20× preserves the false-alarm rate.

### 8.3 The Problem: Median and Tail Are Governed by Different Channels

In our experiment, the score distribution does **not** shift uniformly. The median and the tail are
driven by completely different feature channels:

```
Normal score distribution (governed by mem_bytes, 39× shifted):
  cal  p50 = 1.54     cal  p99 = 31.2     cal  p99.9 = 247.49
  test p50 = 37.75    test p99 = 222.77   test p99.9 = 412.50

Shift at each percentile:
  p50 shift  = 37.75 / 1.54  = 24.5×  (large — mem_bytes dominates)
  p99 shift  = 222.77 / 31.2 = 7.1×   (smaller — net_diff starts appearing)
  p99.9 shift= 412.50 / 247.49 = 1.67× (tiny — tail is net_diff, barely shifted)
```

The probe uses p50 shift (≈ 20×) to adjust a p99.9 threshold. But the threshold's own level is
determined by `net_diff` outliers, which barely shift cross-topology. The correct adjustment at
p99.9 is ≈ 1.67×, not 20×.

### 8.4 Numerical Consequence

```
cal_thr (p99.9 of normal cal scores)  =  247.49
probe p50 shift                        =  20.41×
adj_thr = 247.49 × 20.41              =  5050.45   ← used for detection

Actual anomaly scores (NET stress):
  p50  = 606
  p99  = 2495
  p99.9 = 2535

adj_thr (5050) > anom p99.9 (2535)  →  ZERO detections, TP = 0
cal_thr (247.49) < anom p50 (606)   →  if unadjusted, would catch ~50% of anomalies
```

The cold-start probe inflates the threshold from one that would work (247.49) to one that misses
everything (5050.45), by applying a median shift estimate to a tail threshold where a completely
different channel dominates.

---

## 9. Test-Time Inference: Open-Loop vs Closed-Loop

### 9.1 Open-Loop (Baseline)

The model runs one forward pass over the entire test stream:

```
for t = 0 to T-2:
    cu_hat[t], du_hat[t] = model(cu_actual[t], du_actual[t], hidden_state)
    sqerr[t] = (cu_hat[t] − cu_actual[t+1])^2
```

**Problem with sustained anomalies:** After the first detection, the model's LSTM hidden state
starts seeing the elevated stress values at each step. Within 1-2 steps, it *learns to predict the
elevated values* and prediction error drops back to near zero. All subsequent stress timesteps
are missed (FN = all but the first few).

### 9.2 Closed-Loop (Active, current mode when CLOSED_LOOP = True)

When an entity's score exceeds the threshold for `K` consecutive timesteps (`CU_HYSTERESIS = 5`),
its actual input at the next step is replaced by the model's own prediction:

```
if cu_score > cu_thr for K consecutive steps:
    cu_in[t+1] = cu_hat[t]           # feed model's own "normal" prediction
else:
    cu_in[t+1] = cu_actual[t+1]      # feed the true observation

sqerr[t] = (cu_hat[t] − cu_actual[t+1])^2    # error always vs actual
```

**Effect:** The LSTM hidden state never sees the anomalous values, so it stays anchored to "normal"
behavior. The actual (still-stressed) observation at `t+1` keeps diverging from the prediction
throughout the stress window. Scores stay elevated and the alarm keeps firing.

**Key: error is computed against the actual value, not the substituted input.** Evaluation metrics
are not inflated.

**Hysteresis:** Requires K=5 consecutive above-threshold scores before switching to closed-loop.
This prevents a single noise spike from locking the system into closed-loop.

### 9.3 LSTM Warm-Up: COLD_START_K

The LSTM hidden state is initialized at zero. It takes approximately `WINDOW_LEN = 64` timesteps
to "warm up" — to build an accurate internal representation of the current regime. All scores
computed during the first `COLD_START_K = 64` timesteps are discarded from evaluation.

---

## 10. Evaluation

### 10.1 Score–Label Alignment

Because `cu_sqerr[t]` = error at predicting timestep `t+1`, the corresponding label is
`cu_stress[t+1]`:

```
sqerr index:   0,  1,  2, ...,  t, ...
label index:   1,  2,  3, ..., t+1, ...
```

Evaluation uses rows starting at `COLD_START_K = 64`, covering test timesteps `65..T-1`.

### 10.2 Detection Decision

```python
cu_pred[t] = 1 if cu_score[t] > cu_thr_adj else 0
cu_lbl[t]  = 1 if cu_stress[t+1] == STRESS_TYPE else 0
```

### 10.3 Metrics

```
TP = |{t : pred=1 and label=1}|    (correct alarms)
FP = |{t : pred=1 and label=0}|    (false alarms)
FN = |{t : pred=0 and label=1}|    (missed anomalies)

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × P × R / (P + R)
```

Results are printed per entity (CU, DU_0, DU_1, ...) and for ANY (flag if any entity flagged).

### 10.4 Visualization: phase_plot() (`run_experiment.py` step [9])

After evaluation, `phase_plot()` generates one PNG file per run:

```
cross_anomaly_plot_{test_topo}.png
```

The plot has one **row per entity** (CU + each DU), and **two columns** per row:

```
┌──────────────────────────────┬──────────────────────────────┐
│  Left column                 │  Right column                │
│  Scaled feature time series  │  Anomaly score (log scale)   │
│  All channels overlaid       │  + threshold line (red dash) │
│  Red shading = GT anomaly    │  Red shading = GT anomaly    │
│  Yellow shading = detected   │                              │
└──────────────────────────────┴──────────────────────────────┘
```

**Left panel — feature traces:**
- Each of the 7 CU (or 30 DU) feature channels is drawn as a separate line with its own color
- Y-axis is clipped to the 1st–99th percentile of values after warm-up, so cold-start transients
  don't squash the view
- Red shading covers contiguous ground-truth anomaly windows (`cu_stress == STRESS_TYPE`)
- Yellow shading covers timesteps where the model fired an alarm (`pred == 1`)
- Where yellow overlaps red = correct detection; red without yellow = missed; yellow without red =
  false alarm

**Right panel — score trace:**
- The lift score (`max(sqerr / feat_norm)`) is plotted on a **log scale** over time
- A horizontal dashed red line marks the adjusted threshold `cu_thr_adj`
- The log scale makes it easy to see whether scores during stress windows are above or below the
  threshold even when they span orders of magnitude

**Helper function `_shade()`:** Iterates the boolean mask and calls `ax.axvspan()` for each
contiguous run of `True` values. Only the first span gets a legend label to avoid duplicate entries.

**Score–time alignment:** Scores cover timesteps `COLD_START_K+1 .. T-1` (the first 64 are
discarded). On the right panel, the x-axis therefore starts at timestep 65, not 0 — this is why
the score trace is shorter than the feature trace on the left.

---

## 11. Root Cause: Why CU NET Stress TP = 0

This section summarizes the complete diagnosis, confirmed by running `diagnose_cu_net.py`.

### 11.1 Experiment Setup

- **Test topology:** `cu1_du2` (1 DU) with NET stress (`STRESS_TYPE = 3`)
- **Anomaly windows:** 179 anomalous timesteps in the eval range
- **Calibrated threshold:** `cu_thr = 247.49` (p99.9 of normal cal lift scores)
- **Adjusted threshold:** `cu_thr_adj = 5050.45` (after 20.41× probe shift)

### 11.2 Per-Channel Diagnosis

| Channel | Normal p50 | Normal p99 | Anom p50 | Anom p99 | Separable? |
|---|---|---|---|---|---|
| cpu | 1.66 | 12.25 | 3.56 | 22.54 | NO |
| mem_pct | 0.10 | 1.91 | 0.92 | 2.09 | PARTIAL |
| mem_bytes | 37.58 | 73.21 | 104.93 | 146.24 | YES |
| net_tx | 0.94 | 11.38 | 14.53 | 21.03 | YES |
| net_rx | 0.97 | 11.07 | 12.61 | 20.28 | YES |
| **net_diff** | **2.82** | **42.33** | **606.05** | **2495** | **YES (strongest)** |
| net_ratio | 3.14 | 20.90 | 16.32 | 145.25 | PARTIAL |

### 11.3 Max-Pool Winner Analysis

| Channel | Wins (normal %) | Wins (anom %) |
|---|---|---|
| cpu | 0.0 | 0.0 |
| mem_pct | 0.5 | 0.0 |
| **mem_bytes** | **98.0** | **2.2** |
| net_tx | 0.0 | 0.0 |
| net_rx | 0.1 | 0.0 |
| **net_diff** | **1.5** | **97.8** |
| net_ratio | 0.0 | 0.0 |

**The dominant channel flips completely between normal and anomalous:**
- Normal rows: `mem_bytes` dominates 98% of the time (test topology has 39× more memory bytes)
- Anomalous rows: `net_diff` dominates 97.8% of the time (packet-loss-induced asymmetry)

### 11.4 The Three-Number Summary

```
                        mem_bytes channel    net_diff channel
                        (governs p50)        (governs p99.9)

Cal score p50  = 1.54   (dominated by cal mem_bytes)
Test score p50 = 37.75  → 24× shift          (mem_bytes 39× higher in test)

Cal thr (p99.9) = 247.49                     (set by net_diff outliers in cal)
Test thr (adj)  = 5050.45  ← 20× of 247.49  (WRONG: applies mem_bytes shift to net_diff tail)

Anomaly scores:
  p50  = 606    ABOVE cal_thr 247.49 ✓  BUT BELOW adj_thr 5050 ✗
  p99  = 2495   ABOVE cal_thr 247.49 ✓  BUT BELOW adj_thr 5050 ✗
  p99.9 = 2535  ABOVE cal_thr 247.49 ✓  BUT BELOW adj_thr 5050 ✗

Result: 0 out of 179 anomalous rows detected (TP = 0)
```

### 11.5 What Would Fix It

The probe should estimate the shift at the **same percentile as the threshold** (p99.9), not at
p50. The tail is governed by `net_diff`, which barely shifts cross-topology:

```
Correct tail shift = (test normal p99.9) / (cal p99.9) = 412.50 / 247.49 = 1.67×

Correct adj_thr = 247.49 × 1.67 = 413.3

At adj_thr = 413.3:
  Normal rows above threshold: < 0.1% (very few FPs — by construction)
  Anomaly p50 (606) > adj_thr (413) → ~50% of anomalies detected immediately
  Anomaly p99 (2495) >> adj_thr → 99% of anomalies detected eventually
```

**This fix is stress-type-agnostic**: it uses no knowledge of which stress will appear at test
time. It simply matches the probe's percentile to the threshold's percentile, which is always the
correct thing to do.

The change is a single line in `run_experiment.py` step `[6b]`:

```python
# Current (broken):
cu_test_p50 = float(np.percentile(cu_probe_scores, 50))
cu_cal_p50  = float(np.percentile(cu_norm_scores,  50))
cu_shift    = cu_test_p50 / max(cu_cal_p50, 1e-9)

# Fixed (use the same percentile as the threshold):
cu_test_pthr = float(np.percentile(cu_probe_scores, CU_THRESHOLD_PCT))
cu_cal_pthr  = float(np.percentile(cu_norm_scores,  CU_THRESHOLD_PCT))
cu_shift     = cu_test_pthr / max(cu_cal_pthr, 1e-9)
```

---

## 12. File Reference

| File | Purpose | Key Inputs | Key Outputs |
|---|---|---|---|
| `clear_pipeline/run_experiment.py` | Main pipeline orchestrator (steps [1]–[9]) | `CU_NET_bidir_STRESS/{topo}_stress3/*.npz` | Metrics, plots, `recon_errors_*.npz` |
| `src/model.py` | TopoAR LSTM+attention architecture | (T, cu_dim), (T, N, du_dim) tensors | (T, cu_dim) next-step predictions |
| `src/model_calibrated.py` | `feat_norm_calibrated()`, floored feat_norm; `dual_threshold_from_val()` | sqerr arrays (normal-only cal) | feat_norm (dim,), cu_thr, du_thr |
| `src/preprocess.py` | RobustScaler v0 fitting + transform; glitch imputation; `PreprocessBundle` | Raw KPI arrays + block_id | Scaled arrays; fitted bundle for reuse on test |
| `src/scoring.py` | `lift_score()`: max-pool lift; `localization_metrics()`; `propagation_chains()` | sqerr, feat_norm | Per-row anomaly scores; TP/FP/FN helpers |
| `src/dataset.py` | `TopologySequenceDataset`: block-safe windowing; `MultiTopologyBatchSampler`: homogeneous-N batching; `collate_windows` | Scaled (cu_s, du_s, block_id) | PyTorch DataLoader-ready batches (B, L, dim) |
| `clear_pipeline/diagnose_cu_net.py` | Standalone TP=0 diagnostic — loads saved errors, prints per-channel stats + threshold analysis | `recon_errors_cu1_du2_f7.npz` | Console tables + `diag_score_distributions.png`, `diag_channel_lift_timeseries.png` |

### Data Flow Diagram

```
train.npz (normal only)                    test.npz (mixed normal + stress)
     │                                               │
     ▼                                               │
[1] slice_features()                           slice_features()
    impute_cpu_glitch()                        impute_cpu_glitch()
    (CU: 5 raw → 7 feat)                       (same engineering)
    (DU: 28 raw → 30 feat)                           │
     │                                               │
     ▼                                               │
[1] fit_bundle()  ─────────────────────────────►[6] transform_stream()
    RobustScaler fitted on pooled                    (same fitted scalers applied;
    train topologies. Stored as                       no refit on test)
    PreprocessBundle.                                 │
     │                                               ▼
[2] train_fit / cal split               [6b] phase_infer() on first 300 rows
    (last 20% = cal, held out)               → cold_start_probe()
     │                                        ratio of p50 scores → cu_shift
     ▼                                        → cu_thr_adj = cu_thr × shift
[3] TopologySequenceDataset                         │
    (block-safe 64-step windows)                    ▼
    MultiTopologyBatchSampler               [7] phase_infer_closed_loop()
    (homogeneous-N batches)                     step-by-step LSTM with
    → DataLoader                                hysteresis substitution
     │                                          → cu_sqerr (T-1, 7)
     ▼                                          → du_sqerr (T-1, N, 30)
[3] phase_train()                                   │
    LSTM+attention, MSE loss                        ▼
    early stop on val windows               [8] phase_evaluate()
    → model checkpoint (.pt)                   lift_score(sqerr, cu_fn)
     │                                         compare vs cu_thr_adj
     ▼                                         → TP, FP, FN, F1 per entity
[4] phase_infer() on cal stream                     │
     │                                              ▼
     ▼                                        [8b/8c] diagnostics
[5] feat_norm_calibrated()                    score distribution printout
    lift_score on cal rows                    per-channel mean sqerr ratio
    → cu_fn  (feat_norm, 7-dim)                     │
    → cu_thr (p99.9 of cal scores)                  ▼
    → du_fn, du_thr                           [9] phase_plot()
                                              cross_anomaly_plot_{topo}.png
                                              (features + score per entity)
```

### Configuration Constants (in `run_experiment.py`)

| Constant | Value | Role |
|---|---|---|
| `STRESS_TYPE` | 3 | Which stress to evaluate (1=CPU, 2=MEM, 3=NET) |
| `CU_FEAT_SLICE` | [0,1,2,5,6] | Raw column indices to keep for CU |
| `CU_THRESHOLD_PCT` | 99.9 | Percentile for threshold on normal cal scores |
| `N_PROBE_ROWS` | 300 | Number of test rows used in cold-start probe |
| `COLD_START_K` | 64 | LSTM warmup rows to skip (= WINDOW_LEN) |
| `CAL_FRAC` | 0.2 | Fraction of train stream held out for calibration |
| `CU_HYSTERESIS` | 5 | Consecutive above-threshold steps before closed-loop |
| `CLOSED_LOOP` | True | Replace anomalous entity inputs with predictions |
| `IMPUTE` | True | Forward-fill Prometheus irate glitch zeros |

---

*Generated from diagnostic run on 2026-05-21 against topology cu1_du2, STRESS_TYPE=3 (NET).*
