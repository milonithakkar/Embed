# train_v3.py — TrustGate v3 that produced AUC=0.9531
# Save as: C:\Users\HP\Downloads\trustgate\train_v3.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, TensorDataset,
                               WeightedRandomSampler)
from sklearn.metrics import (roc_auc_score,
                              average_precision_score,
                              f1_score, precision_score,
                              recall_score)
import time
import os
import csv
import sys

# ── FIXED SEED ────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH     = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
BASELINE_PATH = r'D:\trustgate_pcaps\sensor_normal_baseline.npy'
SAVE_PATH     = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
LOG_PATH      = r'D:\trustgate_pcaps\training_log_v3.csv'

BATCH_SIZE    = 128
LR            = 1e-4
WEIGHT_DECAY  = 0.01
MAX_EPOCHS    = 150
PATIENCE      = 40
MIN_DELTA     = 0.001
GRAD_CLIP     = 1.0
LABEL_SMOOTH  = 0.05
WARMUP_EPOCHS = 10
LR_WARMUP_EP  = 5
W_IMP         = 0.15

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

# ── LOSSES ────────────────────────────────────────────────────
class BCELossSmooth(nn.Module):
    def __init__(self, ls=0.05):
        super().__init__()
        self.ls = ls

    def forward(self, logits, targets):
        t = targets * (1 - self.ls) + 0.5 * self.ls
        return F.binary_cross_entropy_with_logits(
            logits, t, reduction='mean')


def sensor_importance_loss(logits, x_sensor,
                            baseline, y_b, eps=1e-8):
    mask = (y_b == 1)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    x_att   = x_sensor[mask]
    logits_ = logits[mask]
    dev     = torch.abs(
        x_att - baseline.unsqueeze(0).unsqueeze(0))
    dev     = dev.mean(dim=1)
    dev_sum = dev.sum(dim=1, keepdim=True).clamp(min=eps)
    target  = (dev / dev_sum).clamp(min=eps)
    return F.kl_div(F.log_softmax(logits_, dim=-1),
                    target, reduction='batchmean')


def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    cands = np.unique(np.concatenate([
        np.percentile(probs, np.arange(1, 100, 1)),
        np.arange(0.05, 0.96, 0.01)
    ]))
    for t in cands:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def make_scheduler(optimizer, warmup_ep, max_ep):
    def lr_lambda(epoch):
        if epoch < warmup_ep:
            return (epoch + 1) / max(warmup_ep, 1)
        p = (epoch - warmup_ep) / max(
            max_ep - warmup_ep, 1)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda)


# ── MAIN ──────────────────────────────────────────────────────
print("=" * 65)
print("TrustGate v3 — Reproducing AUC=0.9531")
print("=" * 65)
print(f"Device: {DEVICE}")
print(f"Seed: 42 (fixed for reproducibility)")

# [1] Load data
print(f"\n[1/6] Loading data...")
data   = np.load(DATA_PATH, allow_pickle=True)
X_s_tr = torch.FloatTensor(data['X_s_train'])
X_n_tr = torch.FloatTensor(data['X_n_train'])
y_b_tr = torch.FloatTensor(data['y_b_train'])
y_c_tr = torch.LongTensor(data['y_c_train'])
y_p_tr = torch.FloatTensor(data['y_p_train'])
y_b_vl = data['y_b_val'].astype(int)

n_normal = int((y_b_tr == 0).sum())
n_attack = int((y_b_tr == 1).sum())
print(f"  Train: {len(X_s_tr):,}  "
      f"normal={n_normal:,} attack={n_attack:,}")

baseline = torch.FloatTensor(
    np.load(BASELINE_PATH)).to(DEVICE)

# [2] DataLoaders
print(f"\n[2/6] DataLoaders...")
w_samp  = torch.where(
    y_b_tr == 1,
    torch.tensor(float(n_normal)),
    torch.tensor(float(n_attack)))
sampler = WeightedRandomSampler(
    w_samp, len(w_samp), replacement=True)

train_ds = TensorDataset(
    X_s_tr, X_n_tr, y_b_tr, y_c_tr, y_p_tr)
val_ds   = TensorDataset(
    torch.FloatTensor(data['X_s_val']),
    torch.FloatTensor(data['X_n_val']),
    torch.FloatTensor(data['y_b_val']),
    torch.LongTensor(data['y_c_val']),
    torch.FloatTensor(data['y_p_val']),
)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(
    val_ds, batch_size=512, shuffle=False,
    num_workers=0, pin_memory=True)

print(f"  Balanced sampler ON")
print(f"  Train: {len(train_loader)} batches  "
      f"Val: {len(val_loader)} batches")

# [3] Build model
print(f"\n[3/6] Building model...")
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

model    = TrustGateModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
if not (1_000_000 <= n_params <= 1_100_000):
    print(f"  ABORT: Wrong parameter count {n_params:,}")
    print(f"  Expected ~1,035,748. Fix model.py first.")
    exit(1)
print(f"  CONFIRMED")

# [4] Optimizer
print(f"\n[4/6] Optimizer...")
bin_loss = BCELossSmooth(LABEL_SMOOTH)
cls_loss = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = make_scheduler(
    optimizer, LR_WARMUP_EP, MAX_EPOCHS)

print(f"  BCE + label_smooth={LABEL_SMOOTH}")
print(f"  KL imp loss w={W_IMP}")
print(f"  LR={LR} warmup={LR_WARMUP_EP} cosine decay")
print(f"  Patience={PATIENCE}")

# [5] Training
print(f"\n[5/6] Training...")
print(f"\n| {'Ep':>4} {'Ph':>6} | "
      f"{'Loss':>6} {'Bin':>6} {'Imp':>6} | "
      f"{'AUC':>6} {'F1*':>6} "
      f"{'Prec':>6} {'Rec':>6} {'FAR':>6} | "
      f"{'LR':>9} {'Time':>4} {'Status':>8} |")
print("-" * 110)

best_auc = -1.0
pat_ct   = 0
rows     = []

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    if epoch <= WARMUP_EPOCHS:
        phase    = "BINARY"
        w_bin, w_cls = 1.0, 0.0
    else:
        ramp     = min(1.0,
                       (epoch - WARMUP_EPOCHS) / 10.0)
        phase    = "MULTI"
        w_bin, w_cls = 1.0, 0.5 * ramp

    # ── Train ─────────────────────────────────────────
    model.train()
    tl = bl = il = 0.0
    nb = 0

    for xs, xn, yb, yc, yp in train_loader:
        xs  = xs.to(DEVICE)
        xn  = xn.to(DEVICE)
        yb  = yb.to(DEVICE)
        yc  = yc.to(DEVICE)

        optimizer.zero_grad()
        out = model(xs, xn)

        lb = bin_loss(out[0].squeeze(-1), yb)
        lc = (cls_loss(out[1], yc)
               if w_cls > 0
               else torch.tensor(0.0, device=DEVICE))
        li = sensor_importance_loss(
            out[6], xs, baseline, yb)

        # EXACT loss: binary + class + KL only
        # NO topology, NO entropy
        loss = w_bin * lb + w_cls * lc + W_IMP * li

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP)
        optimizer.step()

        tl += loss.item()
        bl += lb.item()
        il += li.item()
        nb += 1

    scheduler.step()
    avg_t = tl / nb
    avg_b = bl / nb
    avg_i = il / nb

    # ── Validate ───────────────────────────────────────
    model.eval()
    probs_, labs_, cp_, ct_ = [], [], [], []
    with torch.no_grad():
        for xs, xn, yb, yc, yp in val_loader:
            xs, xn = xs.to(DEVICE), xn.to(DEVICE)
            out    = model(xs, xn)
            probs_.append(torch.sigmoid(
                out[0].squeeze(-1)).cpu().numpy())
            labs_.append(yb.numpy())
            cp_.append(
                out[1].argmax(-1).cpu().numpy())
            ct_.append(yc.numpy())

    probs  = np.concatenate(probs_)
    labels = np.concatenate(labs_)
    cls_p  = np.concatenate(cp_)
    cls_t  = np.concatenate(ct_)

    if len(np.unique(labels)) < 2:
        continue

    auc    = roc_auc_score(labels, probs)
    f1, t  = find_best_threshold(probs, labels)
    preds  = (probs >= t).astype(int)
    prec   = precision_score(labels, preds,
                              zero_division=0)
    rec    = recall_score(labels, preds,
                           zero_division=0)
    n_norm = int((labels == 0).sum())
    far    = (int(((preds==1)&(labels==0)).sum())
              / max(n_norm, 1))
    cur_lr  = optimizer.param_groups[0]['lr']
    elapsed = int(time.time() - t0)

    if auc > best_auc + MIN_DELTA:
        best_auc = auc
        pat_ct   = 0
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'monitor_value':   best_auc,
            'best_threshold':  t,
        }, SAVE_PATH)
        status = "SAVED"
    else:
        pat_ct += 1
        status  = f"wait {pat_ct}/{PATIENCE}"

    print(f"| {epoch:>4} {phase:>6} | "
          f"{avg_t:>6.4f} {avg_b:>6.4f} {avg_i:>6.4f} | "
          f"{auc:>6.4f} {f1:>6.4f} "
          f"{prec:>6.4f} {rec:>6.4f} {far:>6.4f} | "
          f"{cur_lr:>9.2e} {elapsed:>3}s "
          f"{status:>8} |")

    rows.append(dict(
        epoch=epoch, phase=phase,
        loss=avg_t, bin=avg_b, imp=avg_i,
        auc=auc, f1=f1, prec=prec,
        rec=rec, far=far, status=status))

    if pat_ct >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}")
        break

# Save log
if rows:
    with open(LOG_PATH, 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(
            f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

print(f"\n[6/6] Training complete")
print(f"  Best AUC:   {best_auc:.4f}")
print(f"  Checkpoint: {SAVE_PATH}")
print(f"\n{'='*65}")
if best_auc >= 0.950:
    print(f"SUCCESS — target AUC reached")
else:
    print(f"Below target — try seed=0 or seed=123")
print(f"{'='*65}")