# train_v3.py
# Same architecture as model.py (DO NOT MODIFY model.py)
# Smart training with:
#   1. AUC-based early stopping (not F1)
#   2. Two-phase training: warmup binary, then add multi-task
#   3. Threshold scanning in eval (shows true potential)
#   4. Stronger regularization (dropout 0.5, weight_decay 1e-2)
#   5. Mixup augmentation (smooth decision boundaries)
#   6. Class-balanced batch sampling option
# Save as: C:\Users\HP\Downloads\trustgate\train_v3.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score
)
from pathlib import Path

from model import TrustGateModel    # KEEP existing architecture

# ── CONFIG ────────────────────────────────────────────────────────
DATA_PATH      = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
CHECKPOINT     = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
LOG_FILE       = r'D:\trustgate_pcaps\training_log_full_v3.csv'

BATCH_SIZE     = 64           # smaller batch = better generalization
EPOCHS         = 150          # plenty of room with early stop
LR             = 2e-4         # gentler
WEIGHT_DECAY   = 1e-2         # strong regularization
GRAD_CLIP      = 0.5          # tighter clipping

# Early stopping on AUC
PATIENCE       = 25
MIN_DELTA      = 0.002
MONITOR        = 'val_auc'    # KEY: AUC, not F1

# Multi-task — schedule activation
WARMUP_EPOCHS  = 10           # binary only for first N epochs
CLASS_WEIGHT_FINAL = 0.3      # ramps up after warmup
COMP_WEIGHT_FINAL  = 0.15

# Focal + label smoothing
FOCAL_GAMMA    = 2.5          # slightly stronger focus
LABEL_SMOOTH   = 0.05         # smooths hard labels

# Mixup augmentation
USE_MIXUP      = True
MIXUP_ALPHA    = 0.2

# Sampling: balanced batches (50% attack per batch)
USE_BALANCED_SAMPLER = True

NUM_WORKERS    = 0
PIN_MEMORY     = False


# ── Dataset ──────────────────────────────────────────────────────
class TrustGateDataset(Dataset):
    def __init__(self, X_s, X_n, y_b, y_c, y_p):
        self.X_s = torch.from_numpy(X_s.astype(np.float32))
        self.X_n = torch.from_numpy(X_n.astype(np.float32))
        self.y_b = torch.from_numpy(y_b.astype(np.float32))
        self.y_c = torch.from_numpy(y_c.astype(np.int64))
        self.y_p = torch.from_numpy(y_p.astype(np.float32))

    def __len__(self):
        return len(self.y_b)

    def __getitem__(self, idx):
        return (self.X_s[idx], self.X_n[idx],
                self.y_b[idx], self.y_c[idx], self.y_p[idx])


def build_balanced_sampler(y_binary):
    """Sample so each batch has ~50% attacks."""
    n_total  = len(y_binary)
    n_attack = int(y_binary.sum())
    n_normal = n_total - n_attack
    
    w_normal = 1.0 / n_normal
    w_attack = 1.0 / n_attack
    
    weights = np.where(y_binary == 1, w_attack, w_normal)
    weights = torch.from_numpy(weights).float()
    
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=n_total, replacement=True
    )
    return sampler


# ── Focal Loss with label smoothing ──────────────────────────────
class FocalLossLS(nn.Module):
    """Focal loss with label smoothing and pos_weight."""
    def __init__(self, gamma=FOCAL_GAMMA, pos_weight=1.0,
                 label_smooth=LABEL_SMOOTH):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smooth = label_smooth

    def forward(self, logits, targets):
        # Label smoothing
        targets_smooth = targets * (1 - self.label_smooth) + \
                         0.5 * self.label_smooth
        
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        focal_w = (1 - p_t) ** self.gamma
        
        return (focal_w * bce).mean()


# ── Mixup augmentation ──────────────────────────────────────────
def mixup_data(x_s, x_n, y_b, y_c, y_p, alpha=MIXUP_ALPHA):
    """Apply mixup to dual-stream input."""
    if alpha <= 0:
        return x_s, x_n, y_b, y_c, y_p, 1.0, None
    
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)   # ensure lam >= 0.5
    
    batch_size = x_s.size(0)
    index = torch.randperm(batch_size, device=x_s.device)
    
    mixed_x_s = lam * x_s + (1 - lam) * x_s[index]
    mixed_x_n = lam * x_n + (1 - lam) * x_n[index]
    
    # For labels, keep both (compute weighted loss later)
    return mixed_x_s, mixed_x_n, y_b, y_c, y_p, lam, index


def mixup_loss(criterion, logits, targets_a, targets_b, lam):
    """Compute mixup-weighted loss."""
    return lam * criterion(logits, targets_a) + \
           (1 - lam) * criterion(logits, targets_b)


# ── Training functions ───────────────────────────────────────────
def train_epoch(model, loader, criterions, optimizer, device,
                epoch, warmup_epochs):
    """
    Smart training:
    - Epochs 1 to warmup: binary loss only
    - After warmup: gradually add class and component losses
    """
    model.train()
    totals = {'total': 0, 'bin': 0, 'cls': 0, 'comp': 0}
    n_batches = 0
    
    focal_loss, ce_loss, bce_comp_loss = criterions
    
    # Loss weight schedule
    if epoch <= warmup_epochs:
        w_binary, w_class, w_comp = 1.0, 0.0, 0.0
    else:
        # Gradual ramp-up over next 5 epochs
        progress = min((epoch - warmup_epochs) / 5.0, 1.0)
        w_binary = 1.0
        w_class  = CLASS_WEIGHT_FINAL * progress
        w_comp   = COMP_WEIGHT_FINAL * progress
    
    for x_s, x_n, y_b, y_c, y_p in loader:
        x_s = x_s.to(device, non_blocking=True)
        x_n = x_n.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        y_c = y_c.to(device, non_blocking=True)
        y_p = y_p.to(device, non_blocking=True)
        
        # Apply mixup (only on binary task)
        if USE_MIXUP and np.random.rand() < 0.5:
            x_s_m, x_n_m, y_b_m, y_c_m, y_p_m, lam, idx = mixup_data(
                x_s, x_n, y_b, y_c, y_p
            )
            
            optimizer.zero_grad()
            bin_logit, cls_logits, comp_logits, _, _, _ = model(x_s_m, x_n_m)
            
            # Mixup applied to binary head
            loss_b = mixup_loss(
                focal_loss, bin_logit.squeeze(1),
                y_b, y_b[idx], lam
            )
            # Class/comp use original labels (mixup hurts them)
            loss_c = ce_loss(cls_logits, y_c)
            loss_p = bce_comp_loss(comp_logits, y_p)
        else:
            optimizer.zero_grad()
            bin_logit, cls_logits, comp_logits, _, _, _ = model(x_s, x_n)
            
            loss_b = focal_loss(bin_logit.squeeze(1), y_b)
            loss_c = ce_loss(cls_logits, y_c)
            loss_p = bce_comp_loss(comp_logits, y_p)
        
        loss = w_binary * loss_b + w_class * loss_c + w_comp * loss_p
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        totals['total'] += loss.item()
        totals['bin']   += loss_b.item()
        totals['cls']   += loss_c.item()
        totals['comp']  += loss_p.item()
        n_batches       += 1
    
    return {k: v/n_batches for k, v in totals.items()}, \
           (w_binary, w_class, w_comp)


def evaluate(model, loader, criterions, device):
    """
    Evaluate with threshold scanning to find best F1.
    Reports BOTH F1@0.5 and F1 at optimal threshold.
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_bin_probs  = []
    all_bin_labels = []
    all_cls_probs  = []
    all_cls_labels = []
    all_comp_probs = []
    all_comp_labels = []
    
    focal_loss, ce_loss, bce_comp_loss = criterions
    
    with torch.no_grad():
        for x_s, x_n, y_b, y_c, y_p in loader:
            x_s = x_s.to(device); x_n = x_n.to(device)
            y_b = y_b.to(device); y_c = y_c.to(device); y_p = y_p.to(device)
            
            bin_logit, cls_logits, comp_logits, _, _, _ = model(x_s, x_n)
            
            loss_b = focal_loss(bin_logit.squeeze(1), y_b)
            loss_c = ce_loss(cls_logits, y_c)
            loss_p = bce_comp_loss(comp_logits, y_p)
            loss = loss_b + 0.3 * loss_c + 0.15 * loss_p
            
            total_loss += loss.item()
            n_batches += 1
            
            all_bin_probs.append(torch.sigmoid(bin_logit.squeeze(1)).cpu().numpy())
            all_bin_labels.append(y_b.cpu().numpy())
            all_cls_probs.append(F.softmax(cls_logits, dim=-1).cpu().numpy())
            all_cls_labels.append(y_c.cpu().numpy())
            all_comp_probs.append(torch.sigmoid(comp_logits).cpu().numpy())
            all_comp_labels.append(y_p.cpu().numpy())
    
    bin_probs  = np.concatenate(all_bin_probs)
    bin_labels = np.concatenate(all_bin_labels)
    cls_probs  = np.concatenate(all_cls_probs)
    cls_labels = np.concatenate(all_cls_labels)
    comp_probs = np.concatenate(all_comp_probs)
    comp_labels = np.concatenate(all_comp_labels)
    
    # Binary metrics
    metrics = {'loss': total_loss / n_batches}
    
    if len(np.unique(bin_labels)) > 1:
        metrics['val_auc'] = roc_auc_score(bin_labels, bin_probs)
        metrics['val_ap']  = average_precision_score(bin_labels, bin_probs)
    else:
        metrics['val_auc'] = 0.5
        metrics['val_ap']  = 0.0
    
    # F1 at default threshold 0.5
    preds_50 = (bin_probs >= 0.5).astype(int)
    if preds_50.sum() > 0:
        metrics['val_f1_50'] = f1_score(bin_labels, preds_50, pos_label=1, zero_division=0)
    else:
        metrics['val_f1_50'] = 0.0
    
    # Best F1 across all thresholds
    best_f1, best_t, best_p, best_r = 0, 0.5, 0, 0
    for t in np.arange(0.05, 0.95, 0.02):
        preds = (bin_probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(bin_labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_p = precision_score(bin_labels, preds, pos_label=1, zero_division=0)
            best_r = recall_score(bin_labels, preds, pos_label=1, zero_division=0)
    
    metrics['val_f1_best']  = best_f1
    metrics['val_thresh']   = best_t
    metrics['val_prec']     = best_p
    metrics['val_recall']   = best_r
    
    # Multi-class
    cls_preds = cls_probs.argmax(axis=-1)
    metrics['val_cls_acc'] = accuracy_score(cls_labels, cls_preds)
    
    # Components
    comp_preds = (comp_probs >= 0.5).astype(int)
    if comp_labels.sum() > 0:
        metrics['val_comp_f1'] = f1_score(
            comp_labels.flatten(), comp_preds.flatten(),
            zero_division=0
        )
    else:
        metrics['val_comp_f1'] = 0.0
    
    return metrics


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA, path=CHECKPOINT):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best = -1.0
        self.counter = 0
        self.stop = False

    def __call__(self, metric_value, model, epoch, metrics):
        if metric_value > self.best + self.min_delta:
            self.best = metric_value
            self.counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'metrics': metrics,
                'monitor_value': metric_value,
            }, self.path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("TrustGate v3 — Full Architecture, Smart Training")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Load data ────────────────────────────────────────────────
    print(f"\n[1/6] Loading {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)

    n_normal = int((data['y_b_train'] == 0).sum())
    n_attack = int((data['y_b_train'] == 1).sum())
    pos_weight = n_normal / n_attack
    print(f"  Train normal/attack: {n_normal:,}/{n_attack:,}  (pos_weight: {pos_weight:.2f})")

    train_ds = TrustGateDataset(
        data['X_s_train'], data['X_n_train'],
        data['y_b_train'], data['y_c_train'], data['y_p_train'],
    )
    val_ds = TrustGateDataset(
        data['X_s_val'], data['X_n_val'],
        data['y_b_val'], data['y_c_val'], data['y_p_val'],
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── DataLoaders ──────────────────────────────────────────────
    print(f"\n[2/6] Building DataLoaders...")
    if USE_BALANCED_SAMPLER:
        sampler = build_balanced_sampler(data['y_b_train'])
        print(f"  Using balanced sampler (50% attack per batch)")
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            drop_last=True,
        )
    
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model — FULL ARCHITECTURE ────────────────────────────────
    print(f"\n[3/6] Building model (FULL architecture from model.py)...")
    model = TrustGateModel().to(device)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,} (~{total*4/(1024**2):.1f} MB)")
    print(f"  KEEPING: dual-stream BiLSTM + bidirectional cross-attention")
    print(f"  KEEPING: 3 heads (binary + class + components)")

    # ── Losses + optimizer ───────────────────────────────────────
    print(f"\n[4/6] Building optimizer + losses...")
    focal_loss    = FocalLossLS(gamma=FOCAL_GAMMA, pos_weight=pos_weight,
                                 label_smooth=LABEL_SMOOTH)
    ce_loss       = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    bce_comp_loss = nn.BCEWithLogitsLoss()
    criterions    = (focal_loss, ce_loss, bce_comp_loss)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    early_stop = EarlyStopping()

    print(f"  Loss: Focal (gamma={FOCAL_GAMMA}) + label_smooth={LABEL_SMOOTH}")
    print(f"  pos_weight: {pos_weight:.2f}")
    print(f"  Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=20)")
    print(f"  Mixup: {'ON' if USE_MIXUP else 'OFF'} (alpha={MIXUP_ALPHA})")
    print(f"  Warmup binary-only: {WARMUP_EPOCHS} epochs")
    print(f"  Monitor: {MONITOR}, patience: {PATIENCE}")

    with open(LOG_FILE, 'w') as f:
        f.write("epoch,phase,train_loss,train_bin,train_cls,train_comp,"
                "w_bin,w_cls,w_comp,val_loss,val_auc,val_ap,"
                "val_f1_50,val_f1_best,val_thresh,val_prec,val_recall,"
                "val_cls_acc,val_comp_f1,lr,time\n")

    # ── Training loop ────────────────────────────────────────────
    print(f"\n[5/6] Training (max {EPOCHS} epochs, monitor={MONITOR})...")
    print(f"\n  {'Ep':>3} {'Phase':>7} | {'TLoss':>6} {'TBin':>6} | "
          f"{'AUC':>6} {'AP':>6} {'F1@.5':>6} {'F1*':>6} {'T*':>4} "
          f"{'Prec*':>6} {'Rec*':>6} {'CAcc':>6} {'CmpF1':>6} | "
          f"{'LR':>9} {'Time':>4} Status")
    print(f"  {'-'*145}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        
        tr_loss, weights = train_epoch(
            model, train_loader, criterions, optimizer, device,
            epoch, WARMUP_EPOCHS
        )
        m = evaluate(model, val_loader, criterions, device)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        improved = early_stop(m[MONITOR], model, epoch, m)
        status = "SAVED ✓" if improved else f"wait {early_stop.counter}/{PATIENCE}"
        t = time.time() - t0
        
        phase = "BINARY" if epoch <= WARMUP_EPOCHS else "MULTI"
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"{epoch},{phase},{tr_loss['total']:.6f},{tr_loss['bin']:.6f},"
                    f"{tr_loss['cls']:.6f},{tr_loss['comp']:.6f},"
                    f"{weights[0]:.3f},{weights[1]:.3f},{weights[2]:.3f},"
                    f"{m['loss']:.6f},{m['val_auc']:.6f},{m['val_ap']:.6f},"
                    f"{m['val_f1_50']:.6f},{m['val_f1_best']:.6f},"
                    f"{m['val_thresh']:.4f},{m['val_prec']:.6f},{m['val_recall']:.6f},"
                    f"{m['val_cls_acc']:.6f},{m['val_comp_f1']:.6f},{lr:.8f},{t:.1f}\n")
        
        print(f"  {epoch:>3} {phase:>7} | {tr_loss['total']:>6.4f} {tr_loss['bin']:>6.4f} | "
              f"{m['val_auc']:>6.4f} {m['val_ap']:>6.4f} {m['val_f1_50']:>6.4f} "
              f"{m['val_f1_best']:>6.4f} {m['val_thresh']:>4.2f} "
              f"{m['val_prec']:>6.4f} {m['val_recall']:>6.4f} "
              f"{m['val_cls_acc']:>6.4f} {m['val_comp_f1']:>6.4f} | "
              f"{lr:>9.2e} {t:>3.0f}s {status}")
        
        if early_stop.stop:
            print(f"\n  Early stopping at epoch {epoch}")
            print(f"  Best {MONITOR}: {early_stop.best:.4f}")
            break

    print(f"\n[6/6] Training complete")
    print(f"  Best {MONITOR}: {early_stop.best:.4f}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Log: {LOG_FILE}")
    print(f"\n{'='*60}")
    print("Next: run evaluate.py to get test metrics + threshold calibration")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()