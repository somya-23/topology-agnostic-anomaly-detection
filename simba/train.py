import os
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from model import Simba
from utils import (
    RANAnomalyDataset,
    prepare_data,
    evaluate,
    compute_class_weights,
    class_balanced_weights,   # NEW
)

# --------------------
# small helpers
# --------------------
def none_or_int(v: str | int | None):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nil", "-1"):
        return None
    return int(v)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_tau_list(s: str | None):
    if not s:
        return None
    arr = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        arr.append(float(tok))
    return arr if arr else None

# --------------------
# Losses
# --------------------
class FocalLoss(nn.Module):
    """Standard focal loss with optional per-class alpha (tensor [C])."""
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            raise TypeError("alpha must be a torch.Tensor or None")
        self.register_buffer('alpha', alpha if isinstance(alpha, torch.Tensor) else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)                     # prob of true class
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al., 2019). Alpha should be CB weights."""
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.register_buffer('alpha', alpha)   # tensor [C]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --------------------
# Train loop
# --------------------
def train_one_epoch(model, data_loader, loss_fn, optimizer, device, num_classes,
                    scaler, use_amp: bool, max_norm: float = 1.0):
    """Single training epoch with optional AMP + grad clipping."""
    model.train()
    total_loss = 0.0
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type):
                outputs = model(inputs)  # [B, N, C]
                loss = loss_fn(outputs.view(-1, num_classes), labels.view(-1))
            # backward with GradScaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm)
            # grad-norm monitor
            with torch.no_grad():
                grads = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
                total_norm = torch.norm(torch.stack(grads), 2).item() if len(grads) else 0.0
            wandb.log({"train/grad_norm": total_norm, "train/lr": optimizer.param_groups[0]["lr"]})
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, num_classes), labels.view(-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm)
            with torch.no_grad():
                grads = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
                total_norm = torch.norm(torch.stack(grads), 2).item() if len(grads) else 0.0
            wandb.log({"train/grad_norm": total_norm, "train/lr": optimizer.param_groups[0]["lr"]})
            optimizer.step()

        total_loss += loss.item()

        # very light batch exposure monitor (rare)
        if random.random() < 0.02:
            with torch.no_grad():
                c1 = (labels == 1).sum().item()
                c2 = (labels == 2).sum().item()
            wandb.log({"train/batch_c1": c1, "train/batch_c2": c2})

    return total_loss / max(1, len(data_loader))

def log_confmat_wandb(model, data_loader, device, num_classes, split_name="val"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)              # [B, N]
            out = model(x)                # [B, N, C]
            preds = out.argmax(dim=2)     # [B, N]
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(y.view(-1).cpu().numpy())

    _ = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    wandb.log({
        f"{split_name}/conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=[f"class_{i}" for i in range(num_classes)]
        )
    })
    # also log predicted distribution
    pred_hist = np.bincount(np.array(all_preds), minlength=num_classes)
    wandb.log({f"{split_name}/pred_dist_{i}": int(pred_hist[i]) for i in range(num_classes)})

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIMBA training config")

    # Data & training
    parser.add_argument("--data-file", default="data/radio_metrics_lte_ues30_time3600_motion_static_seed67.csv")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--label-smooth", type=float, default=0.0)
    try:
        import argparse as _ap
        parser.add_argument("--amp", action=_ap.BooleanOptionalAction, default=True, help="Use torch.amp autocast")
    except Exception:
        parser.add_argument("--amp", type=int, choices=[0, 1], default=1, help="1 to enable, 0 to disable")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # Oversampling
    parser.add_argument("--oversample", type=float, default=10.0)
    parser.add_argument("--os-w1", type=float, default=0.0, help="per-seq oversample weight if class-1 present (0 disables)")
    parser.add_argument("--os-w2", type=float, default=0.0, help="per-seq oversample weight if class-2 present (0 disables)")

    # Logit adjustment
    parser.add_argument("--logit-adj-tau", type=float, default=0.0, help="0 disables during val/test each epoch")
    parser.add_argument("--tau-scan", type=str, default="0,0.25,0.5,1.0,1.5,2.0",
                        help="comma-separated τ values to scan on VAL after training; set empty to disable")

    # Loss config
    parser.add_argument("--loss", type=str, default="cb_focal",
                        choices=["ce", "focal", "cb_focal"], help="loss type")
    parser.add_argument("--gamma", type=float, default=2.0, help="focal gamma")
    parser.add_argument("--cb-beta", type=float, default=0.999, help="class-balanced beta (Cui)")

    # Model hyperparameters
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--trans-hidden", type=int, default=256)  # placeholder if you wire it later
    parser.add_argument("--gc-out", type=int, default=32)
    parser.add_argument("--gc-hops", type=int, default=2)
    parser.add_argument("--gl-emb", type=int, default=10)
    parser.add_argument("--topk", type=none_or_int, default=5, help="set to None to disable")
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    class Config:
        DATA_FILE_PATH = args.data_file
        MODEL_DIR = args.model_dir
        SEQ_LENGTH = args.seq_len
        NUM_EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        LEARNING_RATE = args.lr
        TEST_SIZE = args.test_size
        VAL_SIZE = args.val_size
        WEIGHT_DECAY = args.wd
        LABEL_SMOOTH = args.label_smooth
        USE_AMP = args.amp if isinstance(args.amp, bool) else (args.amp == 1)
        MAX_GRAD_NORM = args.max_grad_norm
        SEED = args.seed
        PATIENCE = args.patience
        MIN_DELTA = args.min_delta
        OVERSAMPLE_FACTOR = args.oversample
        OS_W1 = args.os_w1
        OS_W2 = args.os_w2
        LOGIT_ADJ_TAU = args.logit_adj_tau

        TRANSFORMER_HEADS = args.heads
        MODEL_DIM = args.dim
        TRANSFORMER_LAYERS = args.layers
        TRANSFORMER_HIDDEN_UNUSED = args.trans_hidden
        GC_OUT_CHANNELS = args.gc_out
        GC_NUM_HOPS = args.gc_hops
        GL_EMBEDDING_DIM = args.gl_emb
        TOP_K = args.topk
        FINAL_FF_DIM = args.ff_dim
        DROPOUT = args.dropout

        LOSS = args.loss
        GAMMA = args.gamma
        CB_BETA = args.cb_beta
        TAU_SCAN = parse_tau_list(args.tau_scan)

    cfg = Config()

    wandb.init(
        project="simba-ran",
        name=f"Simba_{cfg.SEQ_LENGTH}sl_{cfg.MODEL_DIM}d_{cfg.TRANSFORMER_HEADS}h_{cfg.LOSS}_seed{cfg.SEED}_data870hr_ue3_epoch60_static",
        config={
            "seq_length": cfg.SEQ_LENGTH,
            "batch_size": cfg.BATCH_SIZE,
            "lr": cfg.LEARNING_RATE,
            "epochs": cfg.NUM_EPOCHS,
            "heads": cfg.TRANSFORMER_HEADS,
            "model_dim": cfg.MODEL_DIM,
            "transformer_layers": cfg.TRANSFORMER_LAYERS,
            "gc_out": cfg.GC_OUT_CHANNELS,
            "gc_hops": cfg.GC_NUM_HOPS,
            "dropout": cfg.DROPOUT,
            "loss": cfg.LOSS,
            "gamma": cfg.GAMMA,
            "cb_beta": cfg.CB_BETA,
        },
    )

    # Seeds
    set_all_seeds(cfg.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_data = prepare_data(
        cfg.DATA_FILE_PATH, cfg.SEQ_LENGTH, test_size=cfg.TEST_SIZE, val_size=cfg.VAL_SIZE
    )
    if len(X_train) == 0:
        raise ValueError("Training data is empty after sequencing.")

    train_dataset = RANAnomalyDataset(X_train, y_train)
    val_dataset   = RANAnomalyDataset(X_val, y_val)
    test_dataset  = RANAnomalyDataset(X_test, y_test)

    # Oversampling (sequence-level)
    try:
        seq_weights = np.ones((y_train.shape[0],), dtype=np.float32)
        if cfg.OS_W1 > 0.0 or cfg.OS_W2 > 0.0:
            has_c1 = (y_train == 1).any(axis=1)
            has_c2 = (y_train == 2).any(axis=1)
            seq_weights = np.where(has_c1, cfg.OS_W1 if cfg.OS_W1 > 0 else 1.0, seq_weights)
            seq_weights = np.where(has_c2, cfg.OS_W2 if cfg.OS_W2 > 0 else 1.0, seq_weights)
        else:
            has_fault = (y_train != 0).any(axis=1).astype(np.float32)
            seq_weights = np.where(has_fault > 0, cfg.OVERSAMPLE_FACTOR, 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(weights=torch.tensor(seq_weights), num_samples=len(seq_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler,
            pin_memory=(device.type == 'cuda'), num_workers=2 if device.type == 'cuda' else 0
        )
    except Exception:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
            pin_memory=(device.type == 'cuda'), num_workers=2 if device.type == 'cuda' else 0
        )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=2 if device.type == 'cuda' else 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=2 if device.type == 'cuda' else 0
    )

    print("Sequences with class-1:", int(((y_train==1).any(axis=1)).sum()))
    print("Sequences with class-2:", int(((y_train==2).any(axis=1)).sum()))
    print("Any-fault sequences:", int(((y_train!=0).any(axis=1)).sum()))

    num_nodes = X_train.shape[2]
    num_features = X_train.shape[3]
    num_classes = len(np.unique(y_train.flatten()))
    print(f"Detected classes: {num_classes}")

    # Model
    model = Simba(
        num_nodes=num_nodes,
        in_features=num_features,
        out_features=num_classes,
        seq_len=cfg.SEQ_LENGTH,
        gl_embedding_dim=cfg.GL_EMBEDDING_DIM,
        top_k=cfg.TOP_K,
        gc_num_hops=cfg.GC_NUM_HOPS,
        gc_out_channels=cfg.GC_OUT_CHANNELS,
        transformer_heads=cfg.TRANSFORMER_HEADS,
        model_dim=cfg.MODEL_DIM,
        transformer_layers=cfg.TRANSFORMER_LAYERS,
        transformer_hidden_unused=cfg.TRANSFORMER_HIDDEN_UNUSED,
        final_ff_dim=cfg.FINAL_FF_DIM,
        dropout=cfg.DROPOUT
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)

    # Priors for logit adjustment
    class_counts = np.bincount(y_train.flatten(), minlength=num_classes).astype(np.float64)
    total_labels = max(1.0, float(class_counts.sum()))
    priors = np.clip(class_counts / total_labels, 1e-9, 1.0)
    prior_log = torch.tensor(np.log(priors), dtype=torch.float32, device=device)

    # Loss / Optimizer / Scheduler
    if cfg.LOSS == "ce":
        # safer CE weights (not too aggressive)
        weights_np = compute_class_weights(y_train, min_weight=1.0, max_weight=10.0, pow=0.5)
        alpha = torch.tensor(weights_np, dtype=torch.float32, device=device)
        try:
            loss_fn = nn.CrossEntropyLoss(weight=alpha, label_smoothing=cfg.LABEL_SMOOTH)
        except TypeError:
            loss_fn = nn.CrossEntropyLoss(weight=alpha)

    elif cfg.LOSS == "focal":
        # classic focal; alpha from inverse-freq with a bit more strength
        weights_np = compute_class_weights(y_train, min_weight=1.0, max_weight=50.0, pow=1.0)
        alpha = torch.tensor(weights_np, dtype=torch.float32, device=device)
        loss_fn = FocalLoss(alpha=alpha, gamma=cfg.GAMMA)

    else:  # "cb_focal" (default)
        cb_np = class_balanced_weights(y_train, beta=cfg.CB_BETA)   # normalized so majority ~1
        cb_alpha = torch.tensor(cb_np, dtype=torch.float32, device=device)
        loss_fn = CBFocalLoss(alpha=cb_alpha, gamma=cfg.GAMMA)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler(device_type, enabled=cfg.USE_AMP)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    print(f"Class counts: {class_counts.astype(int)}")
    print(f"Priors: {priors}")
    if cfg.LOSS == "ce":
        print(f"CE weights: {alpha.detach().cpu().numpy()}")
    elif cfg.LOSS == "focal":
        print(f"Focal alpha: {alpha.detach().cpu().numpy()}, gamma={cfg.GAMMA}")
    else:
        print(f"CB-Focal alpha: {cb_alpha.detach().cpu().numpy()}, gamma={cfg.GAMMA}, beta={cfg.CB_BETA}")

    wandb.log({
        "data/num_classes": int(num_classes),
        "data/class0_count": int(class_counts[0]),
        "data/class1_count": int(class_counts[1]) if len(class_counts) > 1 else 0,
        "data/class2_count": int(class_counts[2]) if len(class_counts) > 2 else 0,
    })

    best_val_f1_macro = -1.0
    epochs_no_improve = 0
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(cfg.MODEL_DIR, 'best_localization_model.pth')

    print("\n--- Starting Training ---")
    for epoch in range(cfg.NUM_EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, num_classes,
            scaler=scaler, use_amp=cfg.USE_AMP, max_norm=cfg.MAX_GRAD_NORM
        )

        if len(val_loader) > 0:
            # Evaluate with (possibly) a fixed tau each epoch
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _pc = evaluate(
                model, val_loader, loss_fn, device, num_classes,
                logit_adj_tau=cfg.LOGIT_ADJ_TAU, prior_log=prior_log
            )
            # macro metrics from per-class
            try:
                import numpy as _np
                val_f1_macro = float(_np.mean(_pc["f1"])) if isinstance(_pc.get("f1"), list) else val_f1
                val_prec_macro = float(_np.mean(_pc["precision"])) if isinstance(_pc.get("precision"), list) else val_prec
                val_rec_macro = float(_np.mean(_pc["recall"])) if isinstance(_pc.get("recall"), list) else val_rec
            except Exception:
                val_f1_macro, val_prec_macro, val_rec_macro = val_f1, val_prec, val_rec

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc":  val_acc,
                "val/precision_w": val_prec,
                "val/recall_w":    val_rec,
                "val/f1_w":        val_f1,
                "val/precision_m": val_prec_macro,
                "val/recall_m":    val_rec_macro,
                "val/f1_m":        val_f1_macro,
            })

            scheduler.step(val_f1_macro)

            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}  time={dt:.2f}s  "
                  f"train_loss={train_loss:.4f}  val_f1_w={val_f1:.4f}  val_f1_m={val_f1_macro:.4f}  val_acc={val_acc:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

            improved = val_f1_macro - best_val_f1_macro > cfg.MIN_DELTA
            if improved:
                best_val_f1_macro = val_f1_macro
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
                print(f"  -> Saved new best model to {best_model_path} (F1_m={best_val_f1_macro:.4f})")
                art = wandb.Artifact("best_localization_model", type="model")
                art.add_file(best_model_path)
                wandb.log_artifact(art)
                log_confmat_wandb(model, val_loader, device, num_classes, split_name="val")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} (no improve for {cfg.PATIENCE} epochs)")
                    break
        else:
            # No val set → step on train_loss
            scheduler.step(train_loss)
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}  train_loss={train_loss:.4f}  (model saved)  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

    print("--- Training Finished ---")

    # ----------------- Load best and τ-scan on VAL -----------------
    if os.path.exists(best_model_path):
        try:
            state = torch.load(best_model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best checkpoint from {best_model_path}")
    else:
        print("Best model not found on disk; using last-epoch weights.")

    best_tau = cfg.LOGIT_ADJ_TAU
    if cfg.TAU_SCAN and len(val_loader) > 0:
        print("\n--- τ scan on VAL (macro-F1) ---")
        best_macro = -1.0
        for tau in cfg.TAU_SCAN:
            _, _, _, _, _, _, pc = evaluate(
                model, val_loader, loss_fn, device, num_classes,
                logit_adj_tau=tau, prior_log=prior_log
            )
            macro = float(np.mean(pc["f1"])) if isinstance(pc.get("f1"), list) else -1.0
            print(f"VAL τ={tau:.2f}: macro-F1={macro:.4f}")
            if macro > best_macro:
                best_macro = macro
                best_tau = tau
        print("Best τ on VAL:", best_tau)

    # ----------------- TEST EVALUATION -----------------
    test_loss, test_acc, test_prec, test_rec, test_f1, test_report, test_pc = evaluate(
        model, test_loader, loss_fn, device, num_classes,
        logit_adj_tau=best_tau, prior_log=prior_log
    )

    print("\n--- Test Set Performance (Best Checkpoint) ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision (weighted): {test_prec:.4f}")
    print(f"Recall (weighted):    {test_rec:.4f}")
    print(f"F1-Score (weighted):  {test_f1:.4f}")
    print("\nDetailed Classification Report (per-node):")
    print(test_report)

    print("\nPer-class metrics:")
    for i in range(num_classes):
        print(
            f"  Class {i}: Precision={test_pc['precision'][i]:.4f}, "
            f"Recall={test_pc['recall'][i]:.4f}, F1={test_pc['f1'][i]:.4f}, "
            f"Support={int(test_pc['support'][i])}"
        )

    wandb.log({
        "test/loss": test_loss,
        "test/acc":  test_acc,
        "test/precision_w": test_prec,
        "test/recall_w":    test_rec,
        "test/f1_w":        test_f1,
        "test/best_tau":    best_tau,
    })
    for i in range(num_classes):
        wandb.log({
            f"test/precision_c{i}": test_pc["precision"][i],
            f"test/recall_c{i}":    test_pc["recall"][i],
            f"test/f1_c{i}":        test_pc["f1"][i],
            f"test/support_c{i}":   int(test_pc["support"][i]),
        })
    log_confmat_wandb(model, test_loader, device, num_classes, split_name="test")

    # Save scaler for inference
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    with open(os.path.join(cfg.MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_data, f)
    print(f"\nScaler saved to {os.path.join(cfg.MODEL_DIR, 'scaler.pkl')}")

    wandb.finish()
