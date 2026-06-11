#!/usr/bin/env python3
"""Ablation A5: No hidden-state LayerNorm.

Modification: self.LN_h removed; step() uses raw h_new directly in decoders
instead of LN_h(h_new). Everything else is unchanged.

Run:
    cd clear_pipeline/
    python run_ablation_A5_no_hidden_layernorm.py
"""

ABLATION_NAME = "A5_no_hidden_layernorm"
ABLATION_DESC = "self.LN_h removed; decoders receive raw h_new instead of LN_h(h_new)"

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

from preprocess import fit_bundle, transform_stream
from model_calibrated import feat_norm_calibrated
from model_ablation_A5_no_hidden_layernorm import NoHiddenLNTopoAR
from dataset import TopologySequenceDataset, MultiTopologyBatchSampler, collate_windows
from scoring import lift_score

MODEL_CLS = NoHiddenLNTopoAR

# =============================================================================
# CONSTANTS — identical to run_experiment.py
# =============================================================================

ALL_TOPOS     = ["cu0_du0du1", "cu1_du2", "cu2_du3du4du5"]
TEST_TOPO     = "cu2_du3du4du5"
TRAIN_TOPOS   = [t for t in ALL_TOPOS if t != TEST_TOPO]
RUN_ALL_LOO   = True

BASE_DIR      = Path("CU_NET_bidir_STRESS")
STRESS_TYPE   = 3
STRESS_NAMES  = {1: "CPU", 2: "MEM", 3: "NET"}

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

COLD_START_K     = WINDOW_LEN
CU_THRESHOLD_PCT = 99.9
DU_THRESHOLD_PCT = 99.9

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLOSED_LOOP = False
IMPUTE      = True

# =============================================================================
# CSV OUTPUT
# =============================================================================

CSV_PATH   = Path(f"results_{ABLATION_NAME}.csv")
CSV_HEADER = "ablation,test_topology,train_topologies,entity,precision,recall,f1,tp,fp,fn,threshold,stress_type\n"


def write_csv(all_metrics, test_topo, train_topos, cu_thr, du_thr):
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, "a", newline="") as f:
        if write_header:
            f.write(CSV_HEADER)
        for entity, m in all_metrics.items():
            thr = f"{cu_thr:.6f}" if entity == "CU" else (f"{du_thr:.6f}" if entity.startswith("DU") else "")
            f.write(f"{ABLATION_NAME},{test_topo},{'+'.join(train_topos)},"
                    f"{entity},{m['p']:.6f},{m['r']:.6f},{m['f1']:.6f},"
                    f"{m['tp']},{m['fp']},{m['fn']},{thr},{STRESS_TYPE}\n")
    print(f"  CSV appended → {CSV_PATH}")

# =============================================================================
# HELPERS
# =============================================================================

def load_npz(topo, split):
    p = BASE_DIR / f"{topo}_stress{STRESS_TYPE}" / f"{split}.npz"
    assert p.exists(), f"File not found: {p}"
    return dict(np.load(p))


def impute_cpu_glitch(arr, irate_idx, eps=1e-6):
    arr = arr.copy()
    for t in range(1, len(arr)):
        if arr.ndim == 2:
            glitch = arr[t, irate_idx] < eps
            arr[t, irate_idx] = np.where(glitch, arr[t-1, irate_idx], arr[t, irate_idx])
        else:
            glitch = arr[t, :, irate_idx] < eps
            arr[t, :, irate_idx] = np.where(glitch, arr[t-1, :, irate_idx], arr[t, :, irate_idx])
    return arr


def slice_features(z):
    cu = z["cu"].astype(np.float32); du = z["du"].astype(np.float32)
    N_DU = du.shape[1]
    cu[:, 5] /= N_DU; cu[:, 6] /= N_DU
    cu = cu[:, CU_FEAT_SLICE]; du = du[:, :, DU_FEAT_SLICE]
    if IMPUTE:
        cu = impute_cpu_glitch(cu, CU_IRATE_IDX); du = impute_cpu_glitch(du, DU_IRATE_IDX)
    _tx, _rx = cu[:, 3:4], cu[:, 4:5]
    cu = np.concatenate([cu, _tx - _rx, _tx / (_rx + 1e-6)], axis=1)
    _du_tx, _du_rx = du[:, :, 4:5], du[:, :, 5:6]
    du = np.concatenate([du, _du_tx - _du_rx, _du_tx / (_du_rx + 1e-6)], axis=2)
    return cu, du, z["block_id"].astype(np.int64)

# =============================================================================
# PHASES (identical to A3/A4 — only MODEL_CLS differs)
# =============================================================================

def phase_preprocess(train_zs, train_topos):
    raw_streams = []
    for z in train_zs:
        cu, du, bid = slice_features(z)
        raw_streams.append({"cu": cu, "du": du, "block_id": bid})
    bundle = fit_bundle(raw_streams, CU_ZV_IDX, DU_ZV_IDX, version=PREPROCESS_VERSION)
    streams = []
    for i, raw in enumerate(raw_streams):
        cu_s, du_s, _, kept_bid = transform_stream(bundle, raw["cu"], raw["du"], raw["block_id"])
        print(f"  topo[{i}] {train_topos[i]:18s}  cu_s {cu_s.shape}  du_s {du_s.shape}")
        streams.append((cu_s, du_s, kept_bid))
    return bundle, streams


def phase_train(fit_streams, train_topos, model_ckpt, model_cls=None):
    if model_cls is None:
        model_cls = MODEL_CLS
    cu_dim = fit_streams[0][0].shape[1]; du_dim = fit_streams[0][1].shape[2]
    train_subsets, val_subsets, train_lens, val_lens = [], [], [], []
    rng = np.random.RandomState(SEED)
    for i, (cu_s, du_s, bid) in enumerate(fit_streams):
        ds = TopologySequenceDataset(cu_s, du_s, bid, window_len=WINDOW_LEN, stride=1)
        n = len(ds); perm = rng.permutation(n); n_val = max(1, int(round(VAL_FRAC * n)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_subsets.append(torch.utils.data.Subset(ds, train_idx))
        val_subsets.append(torch.utils.data.Subset(ds, val_idx))
        train_lens.append(len(train_idx)); val_lens.append(len(val_idx))
        print(f"  topo[{i}] {train_topos[i]:18s} N_DU={du_s.shape[1]}  windows={n}")
    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_subsets),
        batch_sampler=MultiTopologyBatchSampler(train_lens, BATCH_SIZE, shuffle=True, seed=SEED),
        collate_fn=collate_windows)
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_subsets),
        batch_sampler=MultiTopologyBatchSampler(val_lens, BATCH_SIZE, shuffle=False, seed=SEED),
        collate_fn=collate_windows)
    torch.manual_seed(SEED)
    model = model_cls(cu_dim=cu_dim, du_dim=du_dim, embed_dim=EMBED_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss, patience_count, best_state = float("inf"), 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0.0
        for batch in train_loader:
            cu_b = batch["cu"].to(DEVICE); du_b = batch["du"].to(DEVICE)
            cu_hat, du_hat = model(cu_b, du_b)
            loss = (((cu_hat[:, :-1] - cu_b[:, 1:])**2).mean() +
                    ((du_hat[:, :-1] - du_b[:, 1:])**2).mean())
            optim.zero_grad(); loss.backward(); optim.step(); tr_loss += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cu_b = batch["cu"].to(DEVICE); du_b = batch["du"].to(DEVICE)
                cu_hat, du_hat = model(cu_b, du_b)
                val_loss += (((cu_hat[:, :-1] - cu_b[:, 1:])**2).mean() +
                             ((du_hat[:, :-1] - du_b[:, 1:])**2).mean()).item()
        val_loss /= max(len(val_loader), 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  tr={tr_loss/len(train_loader):.5f}  val={val_loss:.5f}")
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, patience_count = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best_val={best_val_loss:.5f})"); break
    model.load_state_dict(best_state)
    n_train_rows = sum(len(f[0]) for f in fit_streams)
    torch.save({"state_dict": best_state, "cu_dim": cu_dim, "du_dim": du_dim,
                "embed_dim": EMBED_DIM, "cal_frac": CAL_FRAC, "n_train_rows": n_train_rows,
                "topos": list(train_topos), "preprocess": PREPROCESS_VERSION,
                "ablation": ABLATION_NAME, "model_cls": model_cls.__name__}, model_ckpt)
    print(f"  Model saved → {model_ckpt}")
    return model


def phase_infer(model, cu_s, du_s):
    model.eval()
    cu_t = torch.tensor(cu_s).unsqueeze(0).to(DEVICE)
    du_t = torch.tensor(du_s).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cu_hat, du_hat = model(cu_t, du_t)
    return (cu_hat[0, :-1] - cu_t[0, 1:]).pow(2).cpu().numpy(), \
           (du_hat[0, :-1] - du_t[0, 1:]).pow(2).cpu().numpy()


def phase_infer_closed_loop(model, cu_s, du_s, cu_feat_norm, du_feat_norm, cu_thr, du_thr):
    model.eval()
    T, N = len(cu_s), du_s.shape[1]
    cu_sqerrs = np.zeros((T-1, cu_s.shape[1]), dtype=np.float32)
    du_sqerrs = np.zeros((T-1, N, du_s.shape[2]), dtype=np.float32)
    h, c = model.init_state(1, DEVICE)
    DU_HYSTERESIS, CU_HYSTERESIS = 5, 5
    du_anom_count = np.zeros(N, dtype=np.int32); cu_anom_count = 0
    cu_in = torch.tensor(cu_s[[0]], dtype=torch.float32).to(DEVICE)
    du_in = torch.tensor(du_s[[0]], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        for t in range(T - 1):
            cu_tok, du_tok = model.project_tokens(cu_in, du_in)
            cu_hat, du_hat, h, c, _ = model.step(cu_tok, du_tok, h, c)
            cu_next = torch.tensor(cu_s[[t+1]], dtype=torch.float32).to(DEVICE)
            du_next = torch.tensor(du_s[[t+1]], dtype=torch.float32).to(DEVICE)
            cu_err = (cu_hat - cu_next).pow(2).cpu().numpy()[0]
            du_err = (du_hat - du_next).pow(2).cpu().numpy()[0]
            cu_sqerrs[t], du_sqerrs[t] = cu_err, du_err
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


def phase_evaluate(cu_sqerr, du_sqerr, cu_feat_norm, du_feat_norm,
                   cu_thr, du_thr, cu_stress, du_stress):
    N = du_sqerr.shape[1]; start = COLD_START_K
    cu_lbl = (cu_stress[start+1:] == STRESS_TYPE).astype(int)
    du_lbl = (du_stress[start+1:] == STRESS_TYPE)
    cu_scores = lift_score(cu_sqerr[start:], cu_feat_norm)
    du_scores  = np.stack([lift_score(du_sqerr[start:, i, :], du_feat_norm) for i in range(N)], axis=1)
    cu_pred = (cu_scores > cu_thr).astype(int); du_pred = (du_scores > du_thr).astype(int)
    all_metrics = {}
    def metrics(name, pred, lbl):
        tp = int(((pred==1)&(lbl==1)).sum()); fp = int(((pred==1)&(lbl==0)).sum())
        fn = int(((pred==0)&(lbl==1)).sum())
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); f1 = 2*p*r/(p+r+1e-9)
        anom = int((lbl==1).sum())
        print(f"  {name:<12s}  anom={anom:>6d}  TP={tp:>6d}  FP={fp:>6d}  FN={fn:>6d}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        all_metrics[name] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1, "anom": anom}
    print(f"\n  {'Entity':<12s}  {'anom':>6s}  {'TP':>6s}  {'FP':>6s}  {'FN':>6s}  {'P':>5s}  {'R':>5s}  {'F1':>5s}")
    print(f"  {'-'*72}")
    metrics("CU", cu_pred, cu_lbl)
    for i in range(N):
        metrics(f"DU_{i}", du_pred[:, i], du_lbl[:, i].astype(int))
    any_pred = (cu_pred==1)|du_pred.any(axis=1); any_lbl = (cu_lbl==1)|du_lbl.any(axis=1)
    print(f"  {'-'*72}")
    metrics("ANY", any_pred.astype(int), any_lbl.astype(int))
    return cu_scores, du_scores, cu_pred, du_pred, all_metrics


def _shade(ax, t_array, mask, color, alpha, label=None):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any(): return
    in_block = False; block_start = None; first = True
    for i in range(len(mask)):
        if mask[i] and not in_block: block_start = t_array[i]; in_block = True
        elif not mask[i] and in_block:
            ax.axvspan(block_start, t_array[i], color=color, alpha=alpha,
                       label=(label if first else None))
            in_block = False; first = False
    if in_block:
        ax.axvspan(block_start, t_array[-1]+1, color=color, alpha=alpha,
                   label=(label if first else None))


def phase_plot(cu_s_te, du_s_te, cu_stress, du_stress,
               cu_scores, du_scores, cu_pred, du_pred, cu_thr, du_thr,
               train_topos, test_topo):
    T = len(cu_s_te); t_full = np.arange(T); score_t = np.arange(COLD_START_K+1, T)
    n_du = du_s_te.shape[1]
    entities = [("CU", cu_s_te, cu_stress, cu_scores, cu_pred, cu_thr)]
    for i in range(n_du):
        entities.append((f"DU_{i}", du_s_te[:, i, :], du_stress[:, i],
                         du_scores[:, i], du_pred[:, i], du_thr))
    fig, axes = plt.subplots(len(entities), 2, figsize=(18, 4*len(entities)), sharex=False)
    fig.suptitle(f"{ABLATION_NAME} — {STRESS_NAMES[STRESS_TYPE]} stress\n"
                 f"Train: {'+'.join(train_topos)}  →  Test: {test_topo}", fontsize=12, y=1.01)
    _cu_labels = ["cpu","mem_pct","mem_bytes","net_tx","net_rx","net_diff","net_ratio"]
    _du_labels = (["cpu","mem_pct","mem_bytes","fs_writes","net_tx","net_rx"] +
                  [f"pci_{i}" for i in range(22)] + ["net_diff","net_ratio"])
    for row, (name, feat, stress_lbl, scores, pred, thr) in enumerate(entities):
        ax_f, ax_sc = axes[row, 0], axes[row, 1]
        feat_labels = _cu_labels if name == "CU" else _du_labels
        for fi in range(feat.shape[1]):
            ax_f.plot(t_full, feat[:, fi], lw=0.7, label=feat_labels[fi] if fi < 8 else None)
        _shade(ax_f, t_full, stress_lbl==STRESS_TYPE, "red", 0.20, "GT anomaly")
        _shade(ax_f, score_t, pred.astype(bool), "yellow", 0.40, "Detected")
        lo = np.percentile(feat[COLD_START_K:], 1); hi = np.percentile(feat[COLD_START_K:], 99)
        ax_f.set_ylim(lo - max((hi-lo)*0.3, 0.5), hi + max((hi-lo)*0.3, 0.5))
        ax_f.set_ylabel(name, fontsize=10); ax_f.legend(loc="upper right", fontsize=7, framealpha=0.7)
        ax_sc.plot(score_t, scores, color="navy", lw=0.7, label="lift score")
        ax_sc.axhline(thr, color="red", ls="--", lw=1.2, label=f"thr={thr:.4f}")
        _shade(ax_sc, score_t, stress_lbl[COLD_START_K+1:]==STRESS_TYPE, "red", 0.20, "GT anomaly")
        ax_sc.set_yscale("log"); ax_sc.set_ylim(bottom=max(scores[scores>0].min()*0.5, 1e-4))
        ax_sc.legend(loc="upper right", fontsize=7, framealpha=0.7)
    for col in range(2):
        axes[-1, col].set_xlabel("Timestep", fontsize=9)
    plt.tight_layout()
    out = Path(f"{ABLATION_NAME}_plot_{test_topo}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot saved → {out.resolve()}")

# =============================================================================
# RUN ONE LOO FOLD
# =============================================================================

def run_one(train_topos, test_topo):
    print(f"\n{'='*72}")
    print(f"  ABLATION : {ABLATION_NAME}")
    print(f"  CHANGE   : {ABLATION_DESC}")
    print(f"  LOO      : train={train_topos}  test={test_topo}")
    print(f"  Stress   : {STRESS_NAMES[STRESS_TYPE]}  Device: {DEVICE}")
    print(f"{'='*72}")
    model_ckpt = Path(f"{ABLATION_NAME}_ckpt_test_{test_topo}.pt")

    print(f"\n[1] Loading {len(train_topos)} train topologies ...")
    train_zs = [load_npz(t, "train") for t in train_topos]
    bundle, train_streams = phase_preprocess(train_zs, train_topos)

    print(f"\n[2] Per-topology fit/cal split (cal_frac={CAL_FRAC}) ...")
    fit_streams, cal_streams = [], []
    for i, (cu_s, du_s, kept_bid) in enumerate(train_streams):
        n_total = len(cu_s); n_cal = int(round(CAL_FRAC * n_total)); n_fit = n_total - n_cal
        fit_streams.append((cu_s[:n_fit], du_s[:n_fit], kept_bid[:n_fit]))
        cal_streams.append((cu_s[n_fit:], du_s[n_fit:]))
        print(f"  topo[{i}] {train_topos[i]:18s} total={n_total}  fit={n_fit}  cal={n_cal}")
    n_fit_total = sum(len(f[0]) for f in fit_streams)
    cu_dim = fit_streams[0][0].shape[1]; du_dim = fit_streams[0][1].shape[2]

    if model_ckpt.exists():
        print(f"\n[3] Loading checkpoint ({model_ckpt}) ...")
        ckpt = torch.load(model_ckpt, map_location=DEVICE)
        mismatches = []
        if ckpt.get("topos") != list(train_topos): mismatches.append("topos mismatch")
        if ckpt.get("n_train_rows") != n_fit_total: mismatches.append("n_train_rows mismatch")
        if mismatches:
            raise SystemExit(f"Incompatible checkpoint {model_ckpt}: {mismatches}\nDelete it and rerun.")
        model = MODEL_CLS(cu_dim=ckpt["cu_dim"], du_dim=ckpt["du_dim"],
                          embed_dim=ckpt["embed_dim"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"\n[3] Training {MODEL_CLS.__name__} ...")
        model = phase_train(fit_streams, train_topos, model_ckpt, model_cls=MODEL_CLS)

    print(f"\n[4] Inference on held-out CAL streams ...")
    cu_sqerr_pool, du_sqerr_pool = [], []
    for cu_s_cal, du_s_cal in cal_streams:
        cu_sq, du_sq = phase_infer(model, cu_s_cal, du_s_cal)
        cu_sqerr_pool.append(cu_sq[COLD_START_K:])
        du_sqerr_pool.append(du_sq[COLD_START_K:].reshape(-1, du_sq.shape[-1]))
    cu_sqerr_n = np.concatenate(cu_sqerr_pool, axis=0)
    du_sqerr_flt = np.concatenate(du_sqerr_pool, axis=0)

    print(f"\n[5] Calibrating thresholds ...")
    cu_fn = feat_norm_calibrated(cu_sqerr_n); du_fn = feat_norm_calibrated(du_sqerr_flt)
    cu_norm_scores = lift_score(cu_sqerr_n, cu_fn); du_norm_scores = lift_score(du_sqerr_flt, du_fn)
    cu_thr = float(np.percentile(cu_norm_scores, CU_THRESHOLD_PCT))
    du_thr = float(np.percentile(du_norm_scores, DU_THRESHOLD_PCT))
    print(f"  CU thr (p{CU_THRESHOLD_PCT}): {cu_thr:.4f}   DU thr (p{DU_THRESHOLD_PCT}): {du_thr:.4f}")

    print(f"\n[6] Transforming test topology ({test_topo}) ...")
    test_z = load_npz(test_topo, "test")
    cu_te, du_te, bid_te = slice_features(test_z)
    cu_s_te, du_s_te, kept_mask, _ = transform_stream(bundle, cu_te, du_te, bid_te)
    cu_stress = test_z["cu_stress"][kept_mask].astype(np.int32)
    du_stress = test_z["du_stress"][kept_mask].astype(np.int32)
    print(f"  cu_s_te {cu_s_te.shape}  du_s_te {du_s_te.shape}")

    if CLOSED_LOOP:
        print("\n[7] CLOSED-LOOP inference ...")
        cu_sqerr, du_sqerr = phase_infer_closed_loop(
            model, cu_s_te, du_s_te, cu_fn, du_fn, cu_thr, du_thr)
    else:
        print("\n[7] Open-loop inference ...")
        cu_sqerr, du_sqerr = phase_infer(model, cu_s_te, du_s_te)

    print("\n[8] Evaluation ...")
    cu_scores, du_scores, cu_pred, du_pred, eval_metrics = phase_evaluate(
        cu_sqerr, du_sqerr, cu_fn, du_fn, cu_thr, du_thr, cu_stress, du_stress)

    write_csv(eval_metrics, test_topo, train_topos, cu_thr, du_thr)

    print("\n[9] Plotting ...")
    phase_plot(cu_s_te, du_s_te, cu_stress, du_stress,
               cu_scores, du_scores, cu_pred, du_pred, cu_thr, du_thr,
               train_topos, test_topo)
    return eval_metrics


def main():
    if RUN_ALL_LOO:
        all_results = []
        for test_t in ALL_TOPOS:
            train_ts = [t for t in ALL_TOPOS if t != test_t]
            result = run_one(train_ts, test_t)
            all_results.append({"test_topo": test_t, "metrics": result})
        entity_keys = []
        for r in all_results:
            for k in r["metrics"]:
                if k not in entity_keys: entity_keys.append(k)
        print(f"\n\n{'='*72}")
        print(f"  LOO SUMMARY — {ABLATION_NAME}")
        print(f"{'='*72}")
        header = f"  {'Test topology':<22s}" + "".join(f"  {e+' F1':>10s}" for e in entity_keys)
        print(header); print(f"  {'-'*70}")
        for r in all_results:
            row = f"  {r['test_topo']:<22s}"
            for e in entity_keys:
                row += f"  {r['metrics'][e]['f1']:>10.3f}" if e in r["metrics"] else f"  {'N/A':>10s}"
            print(row)
    else:
        run_one(TRAIN_TOPOS, TEST_TOPO)


if __name__ == "__main__":
    main()
