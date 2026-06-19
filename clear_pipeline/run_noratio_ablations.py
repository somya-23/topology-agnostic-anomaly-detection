"""run_noratio_ablations.py — Topoz (v0, net_ratio DROPPED) ablation suite.

Reproduces the architecture ablation on the MAIN data (72h normal training,
_37 CU test, random DU test), all variants built as v0 WITHOUT net_ratio
(CU 6 / DU 29 features), frozen p99.9 threshold.

Variants (each removes exactly one component of full Topoz):
  A1  -topology norm.   CalibratedTopoAR, slice WITHOUT cu[:,5:7]/=N_DU
  A2  fully-shared      FullySharedTopoAR (one shared encoder/decoder, no CU-DU
                        type separation, keeps query attention + LSTM)
  A3  mean-pool         MeanPoolTopoAR (attention -> uniform mean)
  A4  stateless MLP     StatelessMLPTopoAR (LSTM -> feedforward, no recurrence)
  A5  -hidden LayerNorm NoHiddenLNTopoAR (drop LN on hidden state)

One checkpoint set is trained per variant on the 72h normal (LOO over the three
topologies) and reused across all stress configs. CU stress is read from the _37
dirs, DU stress from the random dirs.

Usage (from clear_pipeline/):
  python run_noratio_ablations.py [A1,A2,A3,A4,A5]   # default: all
"""
import csv
import io
import contextlib
import sys
from pathlib import Path

import numpy as np

import run_experiment_v0_baseline as v0
from model_calibrated import CalibratedTopoAR
from model_ablation_A2_fully_shared import FullySharedTopoAR
from model_ablation_A3_meanpool_no_attention import MeanPoolTopoAR
from model_ablation_A4_stateless_mlp import StatelessMLPTopoAR
from model_ablation_A5_no_hidden_layernorm import NoHiddenLNTopoAR


def _slice_noratio(z, topology_norm=True):
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]
    if topology_norm:
        cu[:, 5] = cu[:, 5] / N_DU
        cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, v0.CU_FEAT_SLICE]
    du = du[:, :, v0.DU_FEAT_SLICE]
    if v0.IMPUTE:
        cu = v0.impute_cpu_glitch(cu, v0.CU_IRATE_IDX)
        du = v0.impute_cpu_glitch(du, v0.DU_IRATE_IDX)
    cu = np.concatenate([cu, cu[:, 3:4] - cu[:, 4:5]], axis=1)                  # +net_diff -> 6
    du = np.concatenate([du, du[:, :, 4:5] - du[:, :, 5:6]], axis=2)            # +net_diff -> 29
    return cu, du, z["block_id"].astype(np.int64)


def slice_noratio(z):            # A2-A5 (full preprocessing)
    return _slice_noratio(z, topology_norm=True)


def slice_noratio_noNnorm(z):    # A1 (no 1/N CU normalization)
    return _slice_noratio(z, topology_norm=False)


# variant -> (model class, slice fn)
ABLATIONS = {
    "A1": (CalibratedTopoAR, slice_noratio_noNnorm),
    "A2": (FullySharedTopoAR, slice_noratio),
    "A3": (MeanPoolTopoAR, slice_noratio),
    "A4": (StatelessMLPTopoAR, slice_noratio),
    "A5": (NoHiddenLNTopoAR, slice_noratio),
}

# (entity, stress label, base dir, stress code, metric key)
CONFIGS = [
    ("CU", "CPU", "CU_CPU_random37_STRESS", 1, "CU"),
    ("CU", "MEM", "CU_MEM_random37_STRESS", 2, "CU"),
    ("CU", "NET", "CU_NET_random37_STRESS", 3, "CU"),
    ("DU", "CPU", "DU_CPU_random_STRESS", 1, "ANY"),
    ("DU", "MEM", "DU_MEM_random_STRESS", 2, "ANY"),
    ("DU", "NET", "DU_NET_random_STRESS", 3, "ANY"),
]
TOPOS = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]


def patch_common():
    v0._CU_FEAT_NAMES = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff"]
    v0._DU_FEAT_NAMES = (["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
                         + [f"pci_{i}" for i in range(22)] + ["net_diff"])
    v0._RC_FEAT_GROUPS = {1: {"CU": {0}, "DU": {0}},
                          2: {"CU": {2}, "DU": {2}},
                          3: {"CU": {3, 4, 5}, "DU": {4, 5, 28}}}
    v0.SAVE_ERRORS = False
    v0.CU_THRESHOLD_PCT = 99.9
    v0.DU_THRESHOLD_PCT = 99.9


def run_variant(name):
    model_cls, slice_fn = ABLATIONS[name]
    patch_common()
    v0.slice_features = slice_fn
    v0.CalibratedTopoAR = model_cls
    v0.CKPT_PREFIX = f"abl_{name}_noratio_random_"

    rows = {}   # (stress, metrickey-entity) -> {topo: (p,r,f1)}
    for ent, st, bd, stype, key in CONFIGS:
        v0.BASE_DIR = Path(bd)
        v0.STRESS_TYPE = stype
        for t in TOPOS:
            train = [x for x in v0.ALL_TOPOS if x != t]
            with contextlib.redirect_stdout(io.StringIO()):
                m = v0.run_one(train, t)
            mm = m[key]
            rows.setdefault((ent, st), {})[t] = (mm["p"], mm["r"], mm["f1"])
            print(f"  [{name}] {ent} {st} {t:14s}  P={mm['p']:.3f} R={mm['r']:.3f} F1={mm['f1']:.3f}",
                  flush=True)
    # persist
    out = Path(f"ablation_noratio_{name}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "entity", "stress", "topo", "precision", "recall", "f1"])
        for (ent, st), d in rows.items():
            for t, (p, r, f1) in d.items():
                w.writerow([name, ent, st, t, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
    print(f"  [{name}] -> {out.resolve()}", flush=True)
    return rows


def main():
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ABLATIONS)
    for v in variants:
        assert v in ABLATIONS, f"unknown variant {v}; choose {list(ABLATIONS)}"
        print(f"\n##### ABLATION {v} #####", flush=True)
        run_variant(v)


if __name__ == "__main__":
    main()
