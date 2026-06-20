# =================================================================
# train1.py v2 — TrustGate Dual-Stream BiLSTM
# Changes from v1:
#   1. Focal loss replaces weighted BCE
#   2. SensorImportance uses attention-weighted pooling
#   3. DROPOUT increased to 0.4
#   4. weight_decay increased to 1e-3
#   5. LR reduced to 5e-4 (more stable convergence)
#   6. PATIENCE increased to 15 (more time to find optimum)
# =================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

# ================================================================
# CONSTANTS
# ================================================================
DATA_PATH     = './trustgate_data/swat_final.npz'
CHECKPOINT    = './trustgate_data/best_model_v2.pth'
LOG_FILE      = './trustgate_data/training_log_v2.csv'

BATCH_SIZE    = 256
WINDOW_SIZE   = 30
SENSOR_FEATS  = 44
NETWORK_FEATS = 132
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.4        # v1: 0.3 → increased
FUSION_DIM    = 256

EPOCHS        = 60
LR            = 5e-4       # v1: 1e-3 → halved for stability
POS_WEIGHT    = 19.5
PATIENCE      = 15         # v1: 10 → more time to converge
MIN_DELTA     = 0.001
FOCAL_GAMMA   = 2.0        # focal loss focusing parameter
NUM_WORKERS   = 0


# ================================================================
# SECTION 1 — DATASET + DATALOADER (unchanged from v1)
# ================================================================

class TrustGateDataset(Dataset):
    def __init__(self, X_s, X_n, y):
        assert X_s.ndim == 3
        assert X_n.ndim == 3
        assert X_s.shape[0] == X_n.shape[0] == y.shape[0]
        self.X_s = torch.from_numpy(X_s.astype(np.float32))
        self.X_n = torch.from_numpy(X_n.astype(np.float32))
        self.y   = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_s[idx], self.X_n[idx], self.y[idx]


def build_weighted_sampler(y):
    n_normal = int((y == 0).sum())
    n_attack = int((y == 1).sum())
    total    = len(y)
    w_normal = total / (2.0 * n_normal)
    w_attack = total / (2.0 * n_attack)
    sample_weights = np.where(y == 1, w_attack, w_normal)
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )
    print(f"  Sampler → w_normal={w_normal:.4f} | "
          f"w_attack={w_attack:.4f}")
    return sampler


def build_dataloaders(data_path=DATA_PATH):
    print("=" * 60)
    print("TrustGate v2 — Building DataLoaders")
    print("=" * 60)

    print(f"\n[1/4] Loading {data_path}...")
    data      = np.load(data_path, allow_pickle=True)

    X_s_train = data['X_s_train']
    X_n_train = data['X_n_train']
    y_train   = data['y_train']
    X_s_val   = data['X_s_val']
    X_n_val   = data['X_n_val']
    y_val     = data['y_val']
    X_s_test  = data['X_s_test']
    X_n_test  = data['X_n_test']
    y_test    = data['y_test']

    print(f"  Train: {len(y_train):,} | "
          f"Val: {len(y_val):,} | "
          f"Test: {len(y_test):,}")

    print(f"\n[2/4] Building datasets...")
    train_ds = TrustGateDataset(X_s_train, X_n_train, y_train)
    val_ds   = TrustGateDataset(X_s_val,   X_n_val,   y_val)
    test_ds  = TrustGateDataset(X_s_test,  X_n_test,  y_test)
    print(f"  [OK]")

    print(f"\n[3/4] Building sampler...")
    sampler = build_weighted_sampler(y_train)

    print(f"\n[4/4] DataLoaders (batch={BATCH_SIZE})...")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE*2,
        shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False,
    )

    xs, xn, y = next(iter(train_loader))
    assert xs.shape == (BATCH_SIZE, WINDOW_SIZE, SENSOR_FEATS)
    assert xn.shape == (BATCH_SIZE, WINDOW_SIZE, NETWORK_FEATS)
    print(f"  [OK] Shapes verified")
    print(f"  Batches → Train:{len(train_loader)} "
          f"Val:{len(val_loader)} Test:{len(test_loader)}")

    return train_loader, val_loader, test_loader, y_train


# ================================================================
# SECTION 2 — MODEL ARCHITECTURE v2
# Key change: SensorImportance now uses attention-weighted pooling
# ================================================================

class SensorStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=SENSOR_FEATS, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            bidirectional=True, dropout=DROPOUT,
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.dropout(self.layer_norm(out))


class NetworkStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=NETWORK_FEATS, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            bidirectional=True, dropout=DROPOUT,
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.dropout(self.layer_norm(out))


class BahdanauCrossAttention(nn.Module):
    def __init__(self, hidden_dim=FUSION_DIM):
        super().__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, sensor_out, network_out):
        query        = self.W_query(sensor_out)
        key          = self.W_key(network_out)
        energy       = self.dropout(torch.tanh(query + key))
        scores       = self.v(energy).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        context      = (attn_weights.unsqueeze(-1) * network_out
                        ).sum(dim=1)
        return context, attn_weights


class SensorImportance(nn.Module):
    """
    v2 FIX: Uses temporal attention weights to pool sensor stream
    before projecting to per-sensor importance scores.

    v1 problem: mean pooling ignored WHEN anomalies occurred.
    v2 fix:     attention-weighted pooling focuses on the
                timesteps where the cross-attention fired.

    This makes sensor_scores dynamic — they change based on
    WHAT the model is currently detecting, not just WHAT
    sensors are present in the window.
    """
    def __init__(self):
        super().__init__()
        self.importance_net = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, SENSOR_FEATS),
        )

    def forward(self, sensor_out, attn_weights):
        """
        sensor_out  : (batch, 30, 256)
        attn_weights: (batch, 30)      ← from cross-attention
        Returns     : (batch, 44)
        """
        # Attention-weighted pooling instead of mean pooling
        # attn_weights: (batch, 30) → (batch, 30, 1)
        w      = attn_weights.unsqueeze(-1)
        pooled = (w * sensor_out).sum(dim=1)  # (batch, 256)

        scores     = self.importance_net(pooled)
        importance = F.softmax(scores, dim=-1)
        return importance


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = HIDDEN_SIZE * 2 + FUSION_DIM
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.attack_head   = nn.Linear(64, 1)
        self.severity_head = nn.Linear(64, 1)

    def forward(self, fused):
        feat     = self.classifier(fused)
        logit    = self.attack_head(feat)
        severity = torch.sigmoid(self.severity_head(feat))
        return logit, severity


class TrustGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor_stream  = SensorStream()
        self.network_stream = NetworkStream()
        self.attention      = BahdanauCrossAttention()
        self.sensor_imp     = SensorImportance()
        self.head           = ClassificationHead()
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if 'weight_ih' in pname:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in pname:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in pname:
                        nn.init.zeros_(param.data)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

    def forward(self, x_sensor, x_network):
        sensor_out          = self.sensor_stream(x_sensor)
        network_out         = self.network_stream(x_network)
        context, attn_w     = self.attention(sensor_out, network_out)
        sensor_scores       = self.sensor_imp(sensor_out, attn_w)
        fused               = torch.cat(
            [sensor_out[:, -1, :], context], dim=-1
        )
        logit, severity     = self.head(fused)
        return logit, severity, attn_w, sensor_scores


# ================================================================
# SECTION 3 — FOCAL LOSS + TRAINING LOOP
# ================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Down-weights easy negatives, focuses on hard examples.

    FL(p) = -alpha * (1-p)^gamma * log(p)

    gamma=2.0: standard value from Lin et al. 2017
    alpha=pos_weight/(1+pos_weight): balances classes

    Why focal loss over weighted BCE here:
      Weighted BCE treats all attack windows equally.
      Many attack windows are "easy" — model already
      assigns high probability to them.
      Focal loss ignores easy windows and focuses
      training signal on hard-to-detect attacks.
      This is exactly what we need for the minority
      attack patterns the model keeps missing.

    Paper line:
      "We employ Focal Loss (Lin et al., 2017) to address
       class imbalance, dynamically down-weighting
       well-classified examples and focusing training
       on difficult attack signatures."
    """

    def __init__(self, gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT):
        super().__init__()
        self.gamma      = gamma
        self.alpha      = pos_weight / (1.0 + pos_weight)

    def forward(self, logits, targets):
        """
        logits  : (batch,) raw logits
        targets : (batch,) float 0/1
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs    = torch.sigmoid(logits)

        # p_t: probability of the TRUE class
        p_t      = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t: class-specific weight
        alpha_t  = (self.alpha * targets
                    + (1 - self.alpha) * (1 - targets))

        # Focal weight: (1-p_t)^gamma
        focal_w  = (1.0 - p_t) ** self.gamma

        loss     = alpha_t * focal_w * bce_loss
        return loss.mean()


def build_optimizer(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = 1e-3,    # v1: 1e-4 → stronger regularization
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=4, min_lr=1e-6,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for xs, xn, y in loader:
        xs = xs.to(device, non_blocking=True)
        xn = xn.to(device, non_blocking=True)
        y  = y.to(device,  non_blocking=True)

        optimizer.zero_grad()
        logit, _, _, _ = model(xs, xn)
        loss           = criterion(logit.squeeze(1), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def validate(model, loader, device):
    """
    Validation uses NO pos_weight — gives honest loss reading.
    AUC is the early stopping signal.
    """
    model.eval()
    criterion  = nn.BCEWithLogitsLoss()   # unweighted for val
    total_loss = 0.0
    n_batches  = 0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for xs, xn, y in loader:
            xs = xs.to(device)
            xn = xn.to(device)
            y  = y.to(device)

            logit, _, _, _ = model(xs, xn)
            loss           = criterion(logit.squeeze(1), y)
            total_loss    += loss.item()
            n_batches     += 1

            probs = torch.sigmoid(logit.squeeze(1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / n_batches

    if len(np.unique(all_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA,
                 checkpoint_path=CHECKPOINT):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = checkpoint_path
        self.best_auc  = -1.0
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_auc, model, epoch):
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter  = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_auc'    : val_auc,
            }, self.path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


def init_log(p):
    with open(p, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_auc,lr,time\n")


def append_log(p, ep, tl, vl, va, lr, t):
    with open(p, 'a') as f:
        f.write(f"{ep},{tl:.6f},{vl:.6f},{va:.6f},{lr:.8f},{t:.1f}\n")


def run_training(model, train_loader, val_loader, device):
    print("\n" + "="*60)
    print("Section 3 — Training v2 (Focal Loss)")
    print("="*60)

    criterion            = FocalLoss()
    optimizer, scheduler = build_optimizer(model)
    early_stopping       = EarlyStopping()

    os.makedirs(
        os.path.dirname(CHECKPOINT), exist_ok=True
    )
    init_log(LOG_FILE)

    print(f"\n  Loss       : Focal Loss (gamma={FOCAL_GAMMA})")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  LR         : {LR}")
    print(f"  Dropout    : {DROPOUT}")
    print(f"  Patience   : {PATIENCE}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"\n  {'Ep':>3} | {'TrLoss':>8} | {'VaLoss':>8} | "
          f"{'ValAUC':>7} | {'LR':>9} | {'Time':>6} | Status")
    print(f"  {'-'*68}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss           = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(
            model, val_loader, device
        )

        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]['lr']

        improved = early_stopping(val_auc, model, epoch)
        status   = "SAVED ✓" if improved else \
                   f"wait {early_stopping.counter}/{PATIENCE}"

        t = time.time() - t0
        append_log(LOG_FILE, epoch, tr_loss, val_loss, val_auc, lr, t)

        print(f"  {epoch:>3} | {tr_loss:>8.5f} | {val_loss:>8.5f} | "
              f"{val_auc:>7.4f} | {lr:>9.2e} | "
              f"{t:>5.1f}s | {status}")

        if early_stopping.stop:
            print(f"\n  Early stopping at epoch {epoch}.")
            print(f"  Best val AUC : {early_stopping.best_auc:.4f}")
            break

    print(f"\n  Done. Best checkpoint → {CHECKPOINT}")
    print("="*60)
    return early_stopping.best_auc


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == '__main__':

    # Section 1
    train_loader, val_loader, test_loader, y_train = \
        build_dataloaders()

    # Section 2 — verify
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\nDevice: {device}")
    model = TrustGateModel().to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}  (~{total*4/(1024**2):.1f} MB)")

    with torch.no_grad():
        dx = torch.randn(4, 30, 44).to(device)
        dn = torch.randn(4, 30, 132).to(device)
        lo, sv, aw, si = model(dx, dn)

    assert lo.shape == (4, 1)
    assert aw.shape == (4, 30)
    assert si.shape == (4, 44)
    dev = (aw.sum(-1) - 1).abs().max().item()
    assert dev < 1e-4, f"Attention sum deviation: {dev}"
    print(f"Forward pass OK — attn deviation: {dev:.2e}")

    # Section 3
    model = TrustGateModel().to(device)
    best  = run_training(model, train_loader, val_loader, device)
    print(f"\nBest val AUC: {best:.4f}")
    print("Run: py evaluate.py  (update CHECKPOINT to best_model_v2.pth)")