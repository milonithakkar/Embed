import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time, os
from model import TrustGateModel, ATTACK_CLASSES

# ── Config ──────────────────────────────────────────────────
DATA_PATH  = 'trustgate_data/swat_multilabel.npz'
SAVE_PATH  = 'models/trustgate_best.pt'
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.3
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


class SWaTDataset(Dataset):
    def __init__(self, X_s, X_n, y):
        self.X_s = torch.FloatTensor(X_s)
        self.X_n = torch.FloatTensor(X_n)
        self.y   = torch.LongTensor(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X_s[idx], self.X_n[idx], self.y[idx]


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total   = len(y)
    weights = torch.ones(6)
    for cls, cnt in zip(classes, counts):
        weights[cls] = total / (len(classes) * cnt)
    return weights


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_s, X_n, y in loader:
            X_s, X_n, y = X_s.to(device), X_n.to(device), y.to(device)
            with autocast():
                _, stage_logits, _, _ = model(X_s, X_n)
                loss = criterion(stage_logits, y)
            total_loss += loss.item()
            correct    += (stage_logits.argmax(-1) == y).sum().item()
            total      += len(y)
    return total_loss / len(loader), correct / total * 100


def train():
    print(f"Device : {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs('models', exist_ok=True)

    print("\nLoading data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    train_ds = SWaTDataset(data['X_s_train'], data['X_n_train'], data['y_train'])
    val_ds   = SWaTDataset(data['X_s_val'],   data['X_n_val'],   data['y_val'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train  : {len(train_ds):,} windows")
    print(f"Val    : {len(val_ds):,}   windows")

    # Class distribution
    for cls, name in ATTACK_CLASSES.items():
        cnt = (data['y_train'] == cls).sum()
        print(f"  Class {cls} ({name:>12}): {cnt:>8,}")

    model = TrustGateModel(
        sensor_dim=44, network_dim=132,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        dropout=DROPOUT, num_stages=6
    ).to(DEVICE)

    print(f"\nParams : {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(data['y_train']).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler        = GradScaler()

    best_val_loss = float('inf')
    print("\n" + "="*70)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val Acc':>8} | {'Time':>6}")
    print("="*70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for X_s, X_n, y in train_loader:
            X_s, X_n, y = X_s.to(DEVICE), X_n.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                _, stage_logits, _, _ = model(X_s, X_n)
                loss = criterion(stage_logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        marker = " ← best" if val_loss < best_val_loss else ""
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | "
              f"{val_acc:>7.2f}% | {elapsed:>5.1f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc':  val_acc,
                'config': dict(sensor_dim=44, network_dim=132,
                               hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                               dropout=DROPOUT, num_stages=6)
            }, SAVE_PATH)

    print("="*70)
    print(f"Done. Best val loss: {best_val_loss:.4f} → {SAVE_PATH}")


if __name__ == '__main__':
    train()