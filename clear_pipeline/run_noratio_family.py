"""run_noratio_family.py — retrain v0 / deepsets / settrans with net_ratio DROPPED.

Why: net_ratio is near-constant on normal data (tiny IQR -> tiny feat_norm ->
unstable cross-topology lift, 402 FPs on cu0) AND redundant with net_diff, which
carries the same NET-stress signature (AUC >= net_ratio on every topology) and is
reconstruction-stable. So we remove the net_ratio derived feature from both CU and
DU and retrain each model from scratch through the v0 pipeline.

After dropping net_ratio:  CU 7->6 features,  DU 30->29 features.

Reuses run_experiment_v0_baseline.py unchanged except for monkeypatching:
  - slice_features      -> a version that concatenates net_diff but NOT net_ratio
  - _CU/_DU_FEAT_NAMES  -> drop the trailing "net_ratio"
  - _RC_FEAT_GROUPS[3]  -> NET root-cause indices without net_ratio
  - CalibratedTopoAR    -> the chosen model class (v0 / deepsets / settrans)

Usage:
    python run_noratio_family.py <BASE_DIR> <STRESS_TYPE> [models]
    e.g. python run_noratio_family.py CU_NET_random_STRESS 3
         python run_noratio_family.py CU_NET_random_STRESS 3 v0,deepsets

Checkpoints: <model>_noratio_random_model_ckpt_test_<topo>.pt
CSV:         loo_results_<model>_noratio_<BASE_DIR>.csv  (same schema as v0 CSVs)
"""
import csv
import sys
from pathlib import Path

import numpy as np

import run_experiment_v0_baseline as v0
from model_setbaselines_noratio import DeepSetsBaselineNoRatio, SetTransformerBaselineNoRatio


def slice_features_noratio(z: dict):
    """slice_features with net_diff kept but net_ratio removed (CU 6, DU 29)."""
    cu = z["cu"].astype(np.float32)
    du = z["du"].astype(np.float32)
    N_DU = du.shape[1]

    # topology normalization of CU net traffic (unchanged)
    cu[:, 5] = cu[:, 5] / N_DU
    cu[:, 6] = cu[:, 6] / N_DU
    cu = cu[:, v0.CU_FEAT_SLICE]
    du = du[:, :, v0.DU_FEAT_SLICE]
    if v0.IMPUTE:
        cu = v0.impute_cpu_glitch(cu, v0.CU_IRATE_IDX)
        du = v0.impute_cpu_glitch(du, v0.DU_IRATE_IDX)

    # derived: net_diff ONLY (no net_ratio)
    _tx = cu[:, 3:4]
    _rx = cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx], axis=1)              # CU -> 6 features

    _du_tx = du[:, :, 4:5]
    _du_rx = du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx], axis=2)        # DU -> 29 features

    block_id = z["block_id"].astype(np.int64)
    return cu, du, block_id


MODELS = {
    "v0": v0.CalibratedTopoAR,                # captured at import = original TopoAR
    "deepsets": DeepSetsBaselineNoRatio,
    "settrans": SetTransformerBaselineNoRatio,
}


def run_model(model: str, base_dir: str, stress: int):
    assert model in MODELS, f"unknown model {model}; choose {list(MODELS)}"

    # --- patch the feature pipeline to drop net_ratio (CU 6, DU 29) ---
    v0.slice_features = slice_features_noratio
    v0._CU_FEAT_NAMES = ["cpu", "mem_pct", "mem_bytes", "net_tx", "net_rx", "net_diff"]
    v0._DU_FEAT_NAMES = (["cpu", "mem_pct", "mem_bytes", "fs_writes", "net_tx", "net_rx"]
                         + [f"pci_{i}" for i in range(22)] + ["net_diff"])
    # net_diff index: CU=5, DU=28 (du_dim-1). cpu/mem groups unchanged.
    v0._RC_FEAT_GROUPS = {
        1: {"CU": {0}, "DU": {0}},
        2: {"CU": {2}, "DU": {2}},
        3: {"CU": {3, 4, 5}, "DU": {4, 5, 28}},
    }

    # --- swap the model + checkpoint namespace ---
    v0.CalibratedTopoAR = MODELS[model]
    v0.CKPT_PREFIX = f"{model}_noratio_random_"
    v0.BASE_DIR = Path(base_dir)
    v0.STRESS_TYPE = stress

    all_results = []
    for test_t in v0.ALL_TOPOS:
        train_ts = [t for t in v0.ALL_TOPOS if t != test_t]
        all_results.append({"test_topo": test_t, "metrics": v0.run_one(train_ts, test_t)})

    csv_path = Path(f"loo_results_{model}_noratio_{v0.BASE_DIR.name}.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["stress_type", "test_topo", "entity", "anom", "tp", "fp", "fn",
                        "precision", "recall", "f1", "rc_correct", "rc_total",
                        "rc_pct", "dominant_rc"])
        for r in all_results:
            for entity, m in r["metrics"].items():
                w.writerow([
                    v0.STRESS_NAMES[v0.STRESS_TYPE], r["test_topo"], entity,
                    m["anom"], m["tp"], m["fp"], m["fn"],
                    f"{m['p']:.4f}", f"{m['r']:.4f}", f"{m['f1']:.4f}",
                    m.get("rc_correct", ""), m.get("rc_total", ""),
                    f"{m['rc_pct']:.1f}" if isinstance(m.get("rc_pct"), float)
                    and not (m["rc_pct"] != m["rc_pct"]) else "",
                    m.get("dominant_rc", ""),
                ])
    print(f"\n  [{model} noratio] results -> {csv_path.resolve()}")


def main():
    base_dir, stress = sys.argv[1], int(sys.argv[2])
    models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["v0", "deepsets", "settrans"]
    for mdl in models:
        run_model(mdl, base_dir, stress)


if __name__ == "__main__":
    main()
