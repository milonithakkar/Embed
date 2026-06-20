# =================================================================
# train.py v2
# Fixed: no sampler, pos_weight focal, rebalanced multi-task
# Save as: C:\Users\HP\Downloads\trustgate\train.py (overwrite)
# =================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
from pathlib import Path

from model import TrustGateModel

# ── CONFIG ────────────────────────────────────────────────────────
DATA_PATH      = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
CHECKPOINT     = r'D:\trustgate_pcaps\trained_model_A12_v3.pth'
LOG_FILE       = r'D:\trustgate_pcaps\training_log_A12_v3.csv'

BATCH_SIZE     = 128
EPOCHS         = 100
LR             = 3e-4         # reduced from 5e-4
WEIGHT_DECAY   = 5e-3         # increased from 1e-3
GRAD_CLIP      = 1.0

PATIENCE       = 20           # longer patience
MIN_DELTA      = 0.001
MONITOR        = 'val_f1'

# Multi-task weights (rebalanced)
LOSS_WEIGHT_BINARY = 1.0
LOSS_WEIGHT_CLASS  = 0.2      # reduced
LOSS_WEIGHT_COMP   = 0.1      # reduced

FOCAL_GAMMA    = 2.0

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
        return (
            self.X_s[idx], self.X_n[idx],
            self.y_b[idx], self.y_c[idx], self.y_p[idx]
        )


# ── Focal Loss with pos_weight ───────────────────────────────────
class FocalLossWithPosWeight(nn.Module):
    """
    Focal loss with class imbalance correction.
    
    pos_weight: weight for positive class (attacks)
                Higher = harder penalty for missing attacks
                = n_normal / n_attack (auto-calculated)
    
    gamma: focusing parameter (down-weight easy examples)
           Higher = focus more on hard examples
    """
    def __init__(self, gamma=FOCAL_GAMMA, pos_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Base BCE with class weighting
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight,
                                    device=logits.device),
            reduction='none'
        )
        
        # Focal modulation
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        focal_w = (1 - p_t) ** self.gamma
        
        return (focal_w * bce).mean()


# ── Training functions ───────────────────────────────────────────
def train_one_epoch(model, loader, criterions, optimizer, device):
    model.train()
    totals = {'total': 0, 'bin': 0, 'cls': 0, 'comp': 0}
    n_batches = 0

    focal_loss, ce_loss, bce_comp_loss = criterions

    for x_s, x_n, y_b, y_c, y_p in loader:
        x_s = x_s.to(device)
        x_n = x_n.to(device)
        y_b = y_b.to(device)
        y_c = y_c.to(device)
        y_p = y_p.to(device)

        optimizer.zero_grad()

        bin_logit, cls_logits, comp_logits, _, _, _ = model(x_s, x_n)

        loss_b = focal_loss(bin_logit.squeeze(1), y_b)
        loss_c = ce_loss(cls_logits, y_c)
        loss_p = bce_comp_loss(comp_logits, y_p)

        loss = (LOSS_WEIGHT_BINARY * loss_b
                + LOSS_WEIGHT_CLASS  * loss_c
                + LOSS_WEIGHT_COMP   * loss_p)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        totals['total'] += loss.item()
        totals['bin']   += loss_b.item()
        totals['cls']   += loss_c.item()
        totals['comp']  += loss_p.item()
        n_batches       += 1

    return {k: v/n_batches for k, v in totals.items()}


def evaluate(model, loader, criterions, device):
    model.eval()
    total_loss = 0
    n_batches  = 0

    all_bin_probs   = []
    all_bin_labels  = []
    all_cls_probs   = []
    all_cls_labels  = []
    all_comp_probs  = []
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

            loss = (LOSS_WEIGHT_BINARY * loss_b
                    + LOSS_WEIGHT_CLASS  * loss_c
                    + LOSS_WEIGHT_COMP   * loss_p)
            total_loss += loss.item()
            n_batches  += 1

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

    bin_preds = (bin_probs >= 0.5).astype(int)
    metrics = {
        'loss'       : total_loss / n_batches,
        'val_auc'    : roc_auc_score(bin_labels, bin_probs) if len(np.unique(bin_labels)) > 1 else 0.5,
        'val_f1'     : f1_score(bin_labels, bin_preds, pos_label=1, zero_division=0),
        'val_prec'   : precision_score(bin_labels, bin_preds, pos_label=1, zero_division=0),
        'val_recall' : recall_score(bin_labels, bin_preds, pos_label=1, zero_division=0),
    }

    cls_preds = cls_probs.argmax(axis=-1)
    metrics['val_cls_acc'] = accuracy_score(cls_labels, cls_preds)

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

    def __call__(self, metric_value, model, epoch, metrics_dict):
        if metric_value > self.best + self.min_delta:
            self.best = metric_value
            self.counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'metrics': metrics_dict,
                'val_f1': metric_value,
            }, self.path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


def init_log(path):
    with open(path, 'w') as f:
        f.write("epoch,train_loss,train_bin,train_cls,train_comp,"
                "val_loss,val_auc,val_f1,val_prec,val_recall,"
                "val_cls_acc,val_comp_f1,lr,time_s\n")


def log_epoch(path, ep, tr, val, lr, t):
    with open(path, 'a') as f:
        f.write(f"{ep},{tr['total']:.6f},{tr['bin']:.6f},"
                f"{tr['cls']:.6f},{tr['comp']:.6f},"
                f"{val['loss']:.6f},{val['val_auc']:.6f},"
                f"{val['val_f1']:.6f},{val['val_prec']:.6f},"
                f"{val['val_recall']:.6f},{val['val_cls_acc']:.6f},"
                f"{val['val_comp_f1']:.6f},{lr:.8f},{t:.1f}\n")


def main():
    print("="*60)
    print("TrustGate v2 — Fixed Training")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print(f"\n[1/5] Loading {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)

    # Compute pos_weight from train data
    n_normal = int((data['y_b_train'] == 0).sum())
    n_attack = int((data['y_b_train'] == 1).sum())
    pos_weight_value = n_normal / n_attack
    print(f"  n_normal: {n_normal:,}  n_attack: {n_attack:,}")
    print(f"  pos_weight: {pos_weight_value:.2f}")

    train_ds = TrustGateDataset(
        data['X_s_train'], data['X_n_train'],
        data['y_b_train'], data['y_c_train'], data['y_p_train'],
    )
    val_ds = TrustGateDataset(
        data['X_s_val'], data['X_n_val'],
        data['y_b_val'], data['y_c_val'], data['y_p_val'],
    )

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    print(f"\n[2/5] Building DataLoaders (SHUFFLE, no sampler)...")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,                    # ← KEY CHANGE
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    print(f"\n[3/5] Building model...")
    model = TrustGateModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print(f"\n[4/5] Building optimizer + losses...")
    focal_loss    = FocalLossWithPosWeight(
        gamma=FOCAL_GAMMA,
        pos_weight=pos_weight_value
    )
    ce_loss       = nn.CrossEntropyLoss()
    bce_comp_loss = nn.BCEWithLogitsLoss()
    criterions = (focal_loss, ce_loss, bce_comp_loss)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=7, min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, path=CHECKPOINT)

    print(f"  pos_weight   : {pos_weight_value:.2f}")
    print(f"  Loss weights : binary={LOSS_WEIGHT_BINARY}  "
          f"class={LOSS_WEIGHT_CLASS}  comp={LOSS_WEIGHT_COMP}")
    print(f"  LR           : {LR}")
    print(f"  Weight decay : {WEIGHT_DECAY}")
    print(f"  Patience     : {PATIENCE}")

    init_log(LOG_FILE)

    print(f"\n[5/5] Training (max {EPOCHS} epochs)...")
    print(f"\n  {'Ep':>3} | {'TLoss':>7} {'TBin':>6} {'TCls':>6} {'TCom':>6} | "
          f"{'VLoss':>7} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'CAcc':>6} {'CmpF1':>6} | {'LR':>9} {'Time':>5} {'Status'}")
    print(f"  {'-'*135}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, criterions, optimizer, device)
        val = evaluate(model, val_loader, criterions, device)

        scheduler.step(val['val_f1'])
        lr = optimizer.param_groups[0]['lr']

        improved = early_stop(val[MONITOR], model, epoch, val)
        status = "SAVED ✓" if improved else f"wait {early_stop.counter}/{PATIENCE}"

        t_epoch = time.time() - t0
        log_epoch(LOG_FILE, epoch, tr_loss, val, lr, t_epoch)

        print(f"  {epoch:>3} | "
              f"{tr_loss['total']:>7.4f} {tr_loss['bin']:>6.4f} "
              f"{tr_loss['cls']:>6.4f} {tr_loss['comp']:>6.4f} | "
              f"{val['loss']:>7.4f} {val['val_auc']:>6.4f} "
              f"{val['val_f1']:>6.4f} {val['val_prec']:>6.4f} "
              f"{val['val_recall']:>6.4f} {val['val_cls_acc']:>6.4f} "
              f"{val['val_comp_f1']:>6.4f} | "
              f"{lr:>9.2e} {t_epoch:>4.0f}s {status}")

        if early_stop.stop:
            print(f"\n  Early stopping at epoch {epoch}")
            print(f"  Best val_f1: {early_stop.best:.4f}")
            break

    print(f"\n{'='*60}")
    print(f"Training complete. Best val_f1: {early_stop.best:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()