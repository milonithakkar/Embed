# train_v3.py — TrustGate with Sensor Importance Auxiliary Loss
# Save as: C:\Users\HP\Downloads\trustgate\train_v3.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)
import time
import os
import csv

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH      = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
BASELINE_PATH  = r'D:\trustgate_pcaps\sensor_normal_baseline.npy'
SAVE_PATH      = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
LOG_PATH       = r'D:\trustgate_pcaps\training_log_full_v3.csv'

BATCH_SIZE     = 128
LR             = 1e-4
WEIGHT_DECAY   = 0.01
MAX_EPOCHS     = 150
PATIENCE       = 40
MIN_DELTA      = 0.001
GRAD_CLIP      = 1.0
LABEL_SMOOTH   = 0.05
WARMUP_EPOCHS  = 10
LR_WARMUP_EP   = 5

# Auxiliary loss weight
# This controls how much sensor_imp is pushed toward deviation targets
# Start small — if zero-shot improves but detection degrades, reduce it
W_IMP          = 0.15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── LOSSES ────────────────────────────────────────────────────────────────────
class BCELossSmooth(nn.Module):
    def __init__(self, label_smooth=0.05):
        super().__init__()
        self.ls = label_smooth

    def forward(self, logits, targets):
        t = targets * (1 - self.ls) + 0.5 * self.ls
        return F.binary_cross_entropy_with_logits(
            logits, t, reduction='mean')


def sensor_importance_loss(sensor_imp_logits, x_sensor, baseline,
                            y_b, eps=1e-8):
    """
    KL divergence loss that pushes sensor_imp toward
    the observed per-sensor deviation from normal baseline.

    Only computed on ATTACK windows — normal windows have
    no meaningful deviation signal.

    Args:
        sensor_imp_logits: (B, 71) raw logits from sensor_imp
        x_sensor:          (B, T, 71) input sensor windows
        baseline:          (71,) normal baseline tensor
        y_b:               (B,) binary labels
        eps:               numerical stability

    Returns:
        scalar KL loss (0.0 if no attack windows in batch)
    """
    attack_mask = (y_b == 1)
    if attack_mask.sum() == 0:
        return torch.tensor(0.0, device=sensor_imp_logits.device)

    # Get attack windows only
    x_att  = x_sensor[attack_mask]          # (N_att, T, 71)
    logits = sensor_imp_logits[attack_mask]  # (N_att, 71)

    # Per-sensor absolute deviation from normal baseline
    # baseline: (71,) → broadcast to (N_att, T, 71)
    deviation = torch.abs(x_att - baseline.unsqueeze(0).unsqueeze(0))
    deviation = deviation.mean(dim=1)        # (N_att, 71) — avg over time

    # Normalize to a probability distribution (target for sensor_imp)
    deviation_sum = deviation.sum(dim=1, keepdim=True).clamp(min=eps)
    target_dist   = deviation / deviation_sum   # (N_att, 71) sums to 1

    # Add small epsilon to prevent log(0)
    target_dist = target_dist.clamp(min=eps)

    # KL(target || model) = target * log(target / model_prob)
    # Use F.kl_div which expects log-probabilities as input
    log_probs = F.log_softmax(logits, dim=-1)   # (N_att, 71)

    kl = F.kl_div(log_probs, target_dist, reduction='batchmean')
    return kl


# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    pct_cands   = np.percentile(probs, np.arange(1, 100, 1))
    fixed_cands = np.arange(0.05, 0.96, 0.01)
    candidates  = np.unique(np.concatenate([pct_cands, fixed_cands]))
    for t in candidates:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


# ── LR SCHEDULE ───────────────────────────────────────────────────────────────
def make_scheduler(optimizer, warmup_ep, max_ep):
    def lr_lambda(epoch):
        if epoch < warmup_ep:
            return (epoch + 1) / max(warmup_ep, 1)
        progress = (epoch - warmup_ep) / max(max_ep - warmup_ep, 1)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TrustGate v3 — Training with Sensor Importance Auxiliary Loss")
print("=" * 65)
print(f"\nDevice: {device}")

# [1/6] Load data
print(f"\n[1/6] Loading data...")
data   = np.load(DATA_PATH, allow_pickle=True)

X_s_tr = torch.FloatTensor(data['X_s_train'])
X_n_tr = torch.FloatTensor(data['X_n_train'])
y_b_tr = torch.FloatTensor(data['y_b_train'])
y_c_tr = torch.LongTensor(data['y_c_train'])
y_p_tr = torch.FloatTensor(data['y_p_train'])

y_b_vl_np = data['y_b_val'].astype(int)

n_normal = int((y_b_tr == 0).sum())
n_attack = int((y_b_tr == 1).sum())

print(f"  Train normal/attack: {n_normal:,}/{n_attack:,}")
print(f"  Train: {len(X_s_tr):,}  Val: {len(y_b_vl_np):,}")

# Load normal baseline for KL loss
print(f"\n  Loading baseline from {BASELINE_PATH}...")
baseline_np = np.load(BASELINE_PATH)
baseline    = torch.FloatTensor(baseline_np).to(device)
print(f"  Baseline shape: {baseline.shape}")
print(f"  Baseline range: {baseline_np.min():.4f} to {baseline_np.max():.4f}")

# [2/6] DataLoaders
print(f"\n[2/6] Building DataLoaders...")
w_sample = torch.where(y_b_tr == 1,
                        torch.tensor(float(n_normal)),
                        torch.tensor(float(n_attack)))
sampler  = WeightedRandomSampler(w_sample,
                                  num_samples=len(w_sample),
                                  replacement=True)

train_ds = TensorDataset(X_s_tr, X_n_tr, y_b_tr, y_c_tr, y_p_tr)
val_ds   = TensorDataset(
    torch.FloatTensor(data['X_s_val']),
    torch.FloatTensor(data['X_n_val']),
    torch.FloatTensor(data['y_b_val']),
    torch.LongTensor(data['y_c_val']),
    torch.FloatTensor(data['y_p_val']),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=512,
                          shuffle=False,
                          num_workers=0, pin_memory=True)

print(f"  Balanced sampler ON")
print(f"  Train: {len(train_loader)} batches  Val: {len(val_loader)} batches")

# [3/6] Build model
print(f"\n[3/6] Building model...")
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

model    = TrustGateModel().to(device)
n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"  Parameters: {n_params:,}")
if n_params < 800_000:
    print("  ABORT: Model too small. Fix model.py.")
    exit(1)
print(f"  CONFIRMED")

# [4/6] Losses + optimizer
print(f"\n[4/6] Losses + optimizer...")
bin_criterion = BCELossSmooth(label_smooth=LABEL_SMOOTH)
cls_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = make_scheduler(optimizer, LR_WARMUP_EP, MAX_EPOCHS)

print(f"  Binary: BCE + label_smooth={LABEL_SMOOTH}")
print(f"  Sensor imp: KL divergence vs deviation target (w={W_IMP})")
print(f"  LR={LR}, warmup={LR_WARMUP_EP} epochs, cosine decay")
print(f"  Patience={PATIENCE}")

# [5/6] Training
print(f"\n[5/6] Training (max {MAX_EPOCHS} epochs)...")
print(f"\n| {'Ep':>4} {'Ph':>6} | {'TLoss':>6} {'TBin':>6} {'TImp':>6} | "
      f"{'AUC':>6} {'AP':>6} {'F1*':>6} "
      f"{'T*':>5} {'Prec*':>6} {'Rec*':>6} {'CAcc':>6} | "
      f"{'LR':>9} {'Time':>4} {'Status':>8} |")
print("-" * 125)

best_auc    = -1.0
patience_ct = 0
log_rows    = []

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    if epoch <= WARMUP_EPOCHS:
        phase  = "BINARY"
        w_bin, w_cls = 1.0, 0.0
    else:
        ramp   = min(1.0, (epoch - WARMUP_EPOCHS) / 10.0)
        phase  = "MULTI"
        w_bin, w_cls = 1.0, 0.5 * ramp

    # ── Train ─────────────────────────────────────────────────
    model.train()
    total_loss = bin_loss_sum = imp_loss_sum = 0.0
    n_batches  = 0

    for xs, xn, yb, yc, yp in train_loader:
        xs = xs.to(device)
        xn = xn.to(device)
        yb = yb.to(device)
        yc = yc.to(device)

        optimizer.zero_grad()
        out = model(xs, xn)
        # Unpack 7 outputs
        bin_logit         = out[0]
        cls_logits        = out[1]
        sensor_imp_logits = out[6]   # raw logits for KL loss

        # Binary loss
        loss_bin = bin_criterion(bin_logit.squeeze(-1), yb)

        # Class loss (phase 2 only)
        loss_cls = (cls_criterion(cls_logits, yc)
                    if w_cls > 0 else torch.tensor(0.0, device=device))

        # Sensor importance auxiliary loss
        # Only meaningful for attack windows — normal deviation = 0
        loss_imp = sensor_importance_loss(
            sensor_imp_logits, xs, baseline, yb)

        # Total loss
        loss = (w_bin * loss_bin
                + w_cls * loss_cls
                + W_IMP * loss_imp)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss   += loss.item()
        bin_loss_sum += loss_bin.item()
        imp_loss_sum += loss_imp.item()
        n_batches    += 1

    scheduler.step()
    avg_loss = total_loss   / n_batches
    avg_bin  = bin_loss_sum / n_batches
    avg_imp  = imp_loss_sum / n_batches

    # ── Validate ──────────────────────────────────────────────
    model.eval()
    all_probs, all_labels   = [], []
    all_cls_pred, all_cls_t = [], []

    with torch.no_grad():
        for xs, xn, yb, yc, yp in val_loader:
            xs, xn = xs.to(device), xn.to(device)
            out    = model(xs, xn)
            probs  = torch.sigmoid(out[0].squeeze(-1)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())
            all_cls_pred.append(out[1].argmax(-1).cpu().numpy())
            all_cls_t.append(yc.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    cls_p  = np.concatenate(all_cls_pred)
    cls_t  = np.concatenate(all_cls_t)

    if len(np.unique(labels)) < 2:
        continue

    auc     = roc_auc_score(labels, probs)
    ap      = average_precision_score(labels, probs)
    best_f1, best_t = find_best_threshold(probs, labels)
    preds_b = (probs >= best_t).astype(int)
    prec_b  = precision_score(labels, preds_b, zero_division=0)
    rec_b   = recall_score(labels,  preds_b, zero_division=0)
    cacc    = (cls_p == cls_t).mean()
    cur_lr  = optimizer.param_groups[0]['lr']
    elapsed = int(time.time() - t0)

    if auc > best_auc + MIN_DELTA:
        best_auc    = auc
        patience_ct = 0
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'monitor_value':   best_auc,
            'best_threshold':  best_t,
        }, SAVE_PATH)
        status = "SAVED"
    else:
        patience_ct += 1
        status = f"wait {patience_ct}/{PATIENCE}"

    print(f"| {epoch:>4} {phase:>6} | "
          f"{avg_loss:>6.4f} {avg_bin:>6.4f} {avg_imp:>6.4f} | "
          f"{auc:>6.4f} {ap:>6.4f} {best_f1:>6.4f} "
          f"{best_t:>5.2f} {prec_b:>6.4f} {rec_b:>6.4f} {cacc:>6.4f} | "
          f"{cur_lr:>9.2e} {elapsed:>3}s {status:>8} |")

    log_rows.append(dict(
        epoch=epoch, phase=phase,
        train_loss=avg_loss, bin_loss=avg_bin, imp_loss=avg_imp,
        val_auc=auc, val_ap=ap, val_f1_best=best_f1,
        best_threshold=best_t, val_prec=prec_b,
        val_rec=rec_b, class_acc=cacc,
        lr=cur_lr, status=status
    ))

    if patience_ct >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}")
        break

# ── Save log ──────────────────────────────────────────────────
if log_rows:
    with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

print(f"\n[6/6] Training complete")
print(f"  Best val_auc : {best_auc:.4f}")
print(f"  Checkpoint   : {SAVE_PATH}")
print(f"\n{'='*65}")
print(f"Next steps:")
print(f"  python 02_evaluate.py")
print(f"  python evaluate_zero_shot.py")
print(f"{'='*65}")