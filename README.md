# TopoAR — Cross-Topology CPU Stress Detection (Step 3)

Trains on one 5G network topology's normal data, detects CPU stress anomalies on a **different** topology without retraining.

---

## Directory layout (do not move files around)

```
topoar_gpu_run/
├── clear_pipeline/
│   ├── run_experiment.py        ← main script — run this
│   └── output/
│       ├── cu1_du2_stress1/
│       │   └── train.npz        ← TRAIN topology (normal data only)
│       └── cu0_du0du1_stress1/
│           └── test.npz         ← TEST topology (anomaly labels included)
├── src/
│   ├── model.py                 ← TopoAR architecture
│   ├── model_calibrated.py      ← CalibratedTopoAR + feat_norm helpers
│   ├── preprocess.py            ← RobustScaler pipeline (v0)
│   ├── dataset.py               ← windowed sequence dataset
│   └── scoring.py               ← lift_score, thresholds
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
# Install PyTorch with CUDA (check your CUDA version with: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# or cu118 for CUDA 11.8, cu124 for CUDA 12.4

# Other dependencies
pip install numpy scikit-learn matplotlib
```

### 2. Verify GPU is visible

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True  <your GPU name>
```

---

## Run

```bash
cd clear_pipeline
python run_experiment.py
```

That's it. On first run the script trains (~100 epochs with early stopping) and saves `model_ckpt.pt`. On subsequent runs it loads the checkpoint and skips training.

Expected runtime:
- CPU: ~10–20 min
- GPU: ~1–3 min

---

## Expected output

```
[epoch 1/100]  train=0.3214  val=0.2891
...
[epoch 12/100]  train=0.0821  val=0.0934  ← early stop
Model saved → model_ckpt.pt

CU   precision=0.xx  recall=0.xx  F1=0.xx
DU_0 precision=0.xx  recall=0.xx  F1=0.xx
DU_1 precision=0.xx  recall=0.xx  F1=0.xx

Saved → f1_curve.png
```

---

## Key settings (in `run_experiment.py`)

| Constant | Value | Meaning |
|---|---|---|
| `TRAIN_TOPO` | `cu1_du2` | topology used for training |
| `TEST_TOPO` | `cu0_du0du1` | topology used for anomaly detection |
| `CU_FEAT_SLICE` | `slice(0,2)` | CU: cpu + mem_pct |
| `DU_FEAT_SLICE` | `slice(0,1)` | DU: cpu only (mem_pct dropped — near-zero IQR amplifies cross-topology shift) |
| `EMBED_DIM` | `32` | LSTM / attention hidden size |
| `EPOCHS` | `100` | max epochs (early stopping kicks in earlier) |
| `CAL_FRAC` | `0.2` | last 20% of train stream used for threshold calibration |
| `CU_THRESHOLD_PCT` | `99.9` | CU anomaly threshold percentile |
| `DU_THRESHOLD_PCT` | `99.0` | DU anomaly threshold percentile |
| `CLOSED_LOOP` | `True` | replace anomalous inputs with model's own prediction during inference |
| `IMPUTE` | `False` | Prometheus glitch forward-fill (disabled — safe to enable once DU mem_pct is dropped) |

To delete the checkpoint and retrain from scratch:
```bash
rm clear_pipeline/model_ckpt.pt
python run_experiment.py
```

---

## What the model does

**TopoAR** is a topology-agnostic autoregressive anomaly detector:

1. **Input per timestep**: one CU feature vector + N DU feature vectors (N can vary)
2. **Architecture**: multi-key softmax attention over (1+N) entity tokens → LSTMCell → per-entity decoders predict next timestep
3. **Scoring**: per-entity squared error, normalized by calibration-stream residual (`feat_norm`), max-pooled across features → `lift_score`
4. **Threshold**: set at percentile of normal (calibration) stream scores, frozen — no test labels used

**Topology agnosticism** is achieved via type-shared weights (same W_CU/W_DU/K/V/Q/LSTM for any topology) and softmax convex combination (bounded output regardless of N).
