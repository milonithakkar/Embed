# train_v3.py — TrustGate Path B, Single Rebalancing Strategy
# Save as: C:\Users\HP\Downloads\trustgate\train_v3.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)
import time
import os
import csv

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH    = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
SAVE_PATH    = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
LOG_PATH     = r'D:\trustgate_pcaps\training_log_full_v3.csv'

BATCH_SIZE   = 128
LR           = 2e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS   = 150
PATIENCE     = 30
MIN_DELTA    = 0.001
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.05
WARMUP_EPOCHS = 15
T_0          = 25
MIXUP_ALPHA  = 0.1

# Single rebalancing strategy: pos_weight only
# No balanced sampler. No focal loss boosting.
USE_BALANCED_SAMPLER = False
FOCAL_GAMMA          = 0.0   # gamma=0 means standard BCE with pos_weight only

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── WEIGHTED BCE LOSS (Focal with gamma=0 = standard weighted BCE) ────────────
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight, label_smooth=0.05):
        super().__init__()
        self.pos_weight   = pos_weight
        self.label_smooth = label_smooth

    def forward(self, logits, targets):
        # Label smoothing
        targets_s = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_s,
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        return loss

# ── MIXUP ─────────────────────────────────────────────────────────────────────
def mixup_batch(xs, xn, yb, alpha=0.1):
    if alpha <= 0:
        return xs, xn, yb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xs.size(0), device=xs.device)
    return (lam * xs  + (1-lam) * xs[idx],
            lam * xn  + (1-lam) * xn[idx],
            lam * yb  + (1-lam) * yb[idx])

# ── FIND OPTIMAL THRESHOLD ────────────────────────────────────────────────────
def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TrustGate v3 — Fixed Training (Single Rebalancing)")
print("=" * 60)
print(f"\nDevice: {device}")

# [1/6] Load data
print(f"\n[1/6] Loading {DATA_PATH}...")
data = np.load(DATA_PATH, allow_pickle=True)

X_s_tr = torch.FloatTensor(data['X_s_train'])
X_n_tr = torch.FloatTensor(data['X_n_train'])
y_b_tr = torch.FloatTensor(data['y_b_train'])
y_c_tr = torch.LongTensor(data['y_c_train'])
y_p_tr = torch.FloatTensor(data['y_p_train'])

X_s_vl = torch.FloatTensor(data['X_s_val'])
X_n_vl = torch.FloatTensor(data['X_n_val'])
y_b_vl = data['y_b_val'].astype(int)
y_c_vl = data['y_c_val'].astype(int)
y_p_vl = torch.FloatTensor(data['y_p_val'])

n_normal = int((y_b_tr == 0).sum())
n_attack = int((y_b_tr == 1).sum())
ratio    = n_normal / max(n_attack, 1)
pos_w    = torch.tensor([ratio], device=device)

print(f"  Train normal/attack: {n_normal:,}/{n_attack:,}")
print(f"  pos_weight: {ratio:.2f}  (single rebalancing strategy)")
print(f"  Train: {len(X_s_tr):,}  Val: {len(X_s_vl):,}")

# [2/6] DataLoaders — standard shuffle, no balanced sampler
print(f"\n[2/6] Building DataLoaders...")
print(f"  Balanced sampler: OFF")
print(f"  pos_weight: {ratio:.2f} handles class imbalance")

train_ds = TensorDataset(X_s_tr, X_n_tr, y_b_tr, y_c_tr, y_p_tr)
val_ds   = TensorDataset(X_s_vl, X_n_vl,
                         torch.FloatTensor(data['y_b_val']),
                         torch.LongTensor(data['y_c_val']),
                         y_p_vl)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=512,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# [3/6] Build model
print(f"\n[3/6] Building model...")
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

model   = TrustGateModel().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")

if n_params < 800_000:
    print(f"  ABORT: Model too small ({n_params:,}). Fix model.py first.")
    exit(1)
print(f"  CONFIRMED: Full architecture")

# [4/6] Optimizer + losses
print(f"\n[4/6] Building optimizer + losses...")

# Single rebalancing: weighted BCE only
bin_criterion = WeightedBCELoss(pos_w, label_smooth=LABEL_SMOOTH)
cls_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0)

print(f"  Loss: WeightedBCE (pos_weight={ratio:.2f}) — NO focal, NO sampler")
print(f"  Label smoothing: {LABEL_SMOOTH}")
print(f"  Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0={T_0})")
print(f"  Mixup: alpha={MIXUP_ALPHA}")
print(f"  Warmup binary-only: {WARMUP_EPOCHS} epochs")
print(f"  Patience: {PATIENCE}")

# [5/6] Training
print(f"\n[5/6] Training (max {MAX_EPOCHS} epochs)...")
print(f"\n| {'Ep':>4} {'Phase':>6} | {'TLoss':>6} {'TBin':>6} | "
      f"{'AUC':>6} {'AP':>6} {'F1@.5':>6} {'F1*':>6} "
      f"{'T*':>5} {'Prec*':>6} {'Rec*':>6} {'CAcc':>6} | "
      f"{'LR':>9} {'Time':>4} {'Status':>10} |")
print("-" * 118)

best_auc    = -1.0
patience_ct = 0
log_rows    = []

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    # Phase
    if epoch <= WARMUP_EPOCHS:
        phase = "BINARY"
        w_bin, w_cls = 1.0, 0.0
    else:
        ramp = min(1.0, (epoch - WARMUP_EPOCHS) / 10.0)
        phase = "MULTI"
        w_bin, w_cls = 1.0, 0.5 * ramp

    # ── Train ─────────────────────────────────────────────────
    model.train()
    total_loss = bin_loss_sum = 0.0
    n_batches  = 0

    for xs, xn, yb, yc, yp in train_loader:
        xs  = xs.to(device)
        xn  = xn.to(device)
        yb  = yb.to(device)
        yc  = yc.to(device)

        if MIXUP_ALPHA > 0 and phase == "BINARY":
            xs, xn, yb = mixup_batch(xs, xn, yb, MIXUP_ALPHA)

        optimizer.zero_grad()

        out = model(xs, xn)
        bin_logit, cls_logits = out[0], out[1]

        loss_bin = bin_criterion(bin_logit.squeeze(-1), yb)
        loss_cls = (cls_criterion(cls_logits, yc)
                    if w_cls > 0 else torch.tensor(0.0, device=device))
        loss     = w_bin * loss_bin + w_cls * loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss   += loss.item()
        bin_loss_sum += loss_bin.item()
        n_batches    += 1

    scheduler.step()
    avg_loss = total_loss   / n_batches
    avg_bin  = bin_loss_sum / n_batches

    # ── Validate ──────────────────────────────────────────────
    model.eval()
    all_probs, all_labels = [], []
    all_cls_pred, all_cls_true = [], []

    with torch.no_grad():
        for xs, xn, yb, yc, yp in val_loader:
            xs, xn = xs.to(device), xn.to(device)
            out    = model(xs, xn)
            probs  = torch.sigmoid(out[0].squeeze(-1)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())
            all_cls_pred.append(out[1].argmax(-1).cpu().numpy())
            all_cls_true.append(yc.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    cls_p  = np.concatenate(all_cls_pred)
    cls_t  = np.concatenate(all_cls_true)

    if len(np.unique(labels)) < 2:
        print(f"  WARNING epoch {epoch}: only one class in val. Skipping.")
        continue

    auc      = roc_auc_score(labels, probs)
    ap       = average_precision_score(labels, probs)
    f1_half  = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    best_f1, best_t = find_best_threshold(probs, labels)
    preds_b  = (probs >= best_t).astype(int)
    prec_b   = precision_score(labels, preds_b, zero_division=0)
    rec_b    = recall_score(labels,  preds_b, zero_division=0)
    cacc     = (cls_p == cls_t).mean()
    cur_lr   = optimizer.param_groups[0]['lr']
    elapsed  = int(time.time() - t0)

    # ── Early stopping ─────────────────────────────────────────
    if auc > best_auc + MIN_DELTA:
        best_auc    = auc
        patience_ct = 0
        torch.save({
            'epoch':          epoch,
            'model_state':    model.state_dict(),
            'optimizer_state':optimizer.state_dict(),
            'monitor_value':  best_auc,
            'best_threshold': best_t,
        }, SAVE_PATH)
        status = "SAVED ✓"
    else:
        patience_ct += 1
        status = f"wait {patience_ct}/{PATIENCE}"

    print(f"| {epoch:>4} {phase:>6} | {avg_loss:>6.4f} {avg_bin:>6.4f} | "
          f"{auc:>6.4f} {ap:>6.4f} {f1_half:>6.4f} {best_f1:>6.4f} "
          f"{best_t:>5.2f} {prec_b:>6.4f} {rec_b:>6.4f} {cacc:>6.4f} | "
          f"{cur_lr:>9.2e} {elapsed:>3}s {status:>10} |")

    log_rows.append(dict(epoch=epoch, phase=phase,
                         train_loss=avg_loss, val_auc=auc, val_ap=ap,
                         val_f1_half=f1_half, val_f1_best=best_f1,
                         best_threshold=best_t, val_prec=prec_b,
                         val_rec=rec_b, class_acc=cacc,
                         lr=cur_lr, status=status))

    if patience_ct >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}")
        break

# ── Save log ──────────────────────────────────────────────────
if log_rows:
    with open(LOG_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

print(f"\n[6/6] Training complete")
print(f"  Best val_auc : {best_auc:.4f}")
print(f"  Checkpoint   : {SAVE_PATH}")
print(f"  Log          : {LOG_PATH}")
print(f"\n{'='*60}")
print(f"Next: python 02_evaluate.py")
print(f"{'='*60}")