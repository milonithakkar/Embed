"""
TrustGate Phase A — Self-Supervised Pre-training on A11 Normal Baseline
========================================================================
Uses ONLY A11 normal data (zero attacks) to pre-train the BiLSTM encoders.
Two simultaneous objectives:
  1. NT-Xent contrastive loss  — encoders learn what "normal" looks like
  2. Reconstruction autoencoder — encoders capture fine-grained local patterns

Output:
  trustgate_data/pretrain_sensor_enc.pt   — SensorEncoder weights
  trustgate_data/pretrain_net_enc.pt      — NetworkEncoder weights
  trustgate_data/pretrain_autoencoder.pt  — full autoencoder (for recon_error inference)
  trustgate_data/pretrain_loss.csv        — training log

Runtime: ~20–40 min on GPU with A11 data (58k windows)

python 04_pretrain.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

# ─── ADD MODEL TO PATH ────────────────────────────────────────────
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location('_m', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '03_model.py'))
_m = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m)
SensorEncoder = _m.SensorEncoder
NetworkEncoder = _m.NetworkEncoder
SensorAutoEncoder = _m.SensorAutoEncoder
ProjectionHead = _m.ProjectionHead
NTXentLoss = _m.NTXentLoss

# ─── PATHS ────────────────────────────────────────────────────────
NPZ_PATH   = 'trustgate_data/swat_final.npz'
OUT_S_ENC  = 'trustgate_data/pretrain_sensor_enc.pt'
OUT_N_ENC  = 'trustgate_data/pretrain_net_enc.pt'
OUT_AE     = 'trustgate_data/pretrain_autoencoder.pt'
OUT_LOG    = 'trustgate_data/pretrain_loss.csv'

# ─── HYPERPARAMETERS ─────────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 256   # Large batch required for NT-Xent loss quality
LR          = 1e-3
LR_MIN      = 1e-5  # Cosine annealing floor
TEMPERATURE = 0.07  # NT-Xent temperature (optimal for time series)
RECON_WEIGHT = 0.5  # Weight of reconstruction loss vs contrastive
PROJ_DIM    = 128   # Contrastive projection head output dimension
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Force stdout to flush immediately — fixes silent output on Windows
import sys
sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}', flush=True)


# ══════════════════════════════════════════════════════════════════
# ICS-SPECIFIC AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════

def augment_sensor(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    5 augmentations that produce VALID-LOOKING ICS data.
    Teaches the encoder robustness to noise without corrupting semantics.
    x: (T, D) float32
    """
    x = x.copy().astype(np.float32)
    T, D = x.shape

    # 1. Magnitude jitter — PLC readings have ~1% Gaussian noise
    x += np.random.randn(T, D).astype(np.float32) * sigma

    # 2. Sensor dropout — simulate 3s comms failure on 10% of sensors
    n_drop     = max(1, int(D * 0.10))
    drop_feats = np.random.choice(D, n_drop, replace=False)
    drop_start = np.random.randint(0, max(1, T - 3))
    x[drop_start:drop_start + 3, drop_feats] = 0.0

    # 3. Window slicing — take a 25s sub-window, zero-pad to 30s
    if T >= 5:
        slice_len = T - np.random.randint(1, min(6, T))
        start     = np.random.randint(0, T - slice_len + 1)
        sliced    = x[start:start + slice_len]
        x         = np.zeros_like(x)
        x[:slice_len] = sliced

    # 4. Phase shift — ±2 second timestamp drift
    shift = np.random.randint(-2, 3)
    if shift > 0:
        x = np.concatenate([np.zeros((shift, D), dtype=np.float32), x[:-shift]], axis=0)
    elif shift < 0:
        x = np.concatenate([x[-shift:], np.zeros((-shift, D), dtype=np.float32)], axis=0)

    # 5. Scale jitter on continuous range — ±5% global amplitude
    scale = np.random.uniform(0.95, 1.05)
    x    *= scale

    return x


def augment_network(x: np.ndarray) -> np.ndarray:
    """
    3 augmentations for the network feature stream.
    x: (T, D) float32
    """
    x = x.copy().astype(np.float32)
    T, D = x.shape

    # 1. Feature masking — randomly zero 20% of network features
    mask    = np.random.rand(T, D) < 0.20
    x[mask] = 0.0

    # 2. Count scaling — traffic volume variation ±20%
    x *= np.random.uniform(0.80, 1.20)

    # 3. Packet reordering jitter — permute ±2 second bins
    if T >= 5:
        jitter_idx = np.arange(T)
        for t in range(T):
            swap = np.clip(t + np.random.randint(-2, 3), 0, T - 1)
            jitter_idx[t] = swap
        x = x[jitter_idx]

    return x


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class ContrastiveICSDataset(Dataset):
    """
    Returns TWO differently-augmented views of each A11 window.
    Both views come from the SAME underlying normal window.
    NT-Xent loss will push them together and apart from all other pairs.
    """
    def __init__(self, X_s: np.ndarray, X_n: np.ndarray):
        self.X_s = X_s   # (N, T, D_s)
        self.X_n = X_n   # (N, T, D_n)

    def __len__(self):
        return len(self.X_s)

    def __getitem__(self, idx):
        xs = self.X_s[idx]
        xn = self.X_n[idx]

        xs1 = torch.from_numpy(augment_sensor(xs))
        xn1 = torch.from_numpy(augment_network(xn))
        xs2 = torch.from_numpy(augment_sensor(xs))
        xn2 = torch.from_numpy(augment_network(xn))

        return xs1, xn1, xs2, xn2


# ══════════════════════════════════════════════════════════════════
# PRE-TRAINING MODEL (encoders + projection heads + autoencoder)
# ══════════════════════════════════════════════════════════════════

class PretrainModel(nn.Module):
    def __init__(self, sensor_dim: int, net_dim: int,
                 window_size: int = 30, proj_dim: int = 128):
        super().__init__()
        # Shared encoders — these weights will be transferred to TrustGateModel
        self.sensor_enc = SensorEncoder(sensor_dim)
        self.net_enc    = NetworkEncoder(net_dim)

        # Contrastive projection heads (discarded after pre-training)
        self.proj_s = ProjectionHead(SensorEncoder.OUT_DIM,  proj_dim)
        self.proj_n = ProjectionHead(NetworkEncoder.OUT_DIM, proj_dim)

        # Joint projection (fuses both streams for contrastive objective)
        self.proj_joint = ProjectionHead(SensorEncoder.OUT_DIM + NetworkEncoder.OUT_DIM,
                                          proj_dim)

        # Autoencoder decoder for sensor stream
        flat_dim = window_size * sensor_dim
        self.ae_to_lat = nn.Linear(SensorEncoder.OUT_DIM, 64)
        self.ae_dec    = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, flat_dim),
        )
        self.T = window_size
        self.D = sensor_dim

    def encode(self, xs, xn):
        """Encode both streams → context vectors."""
        _, c_s, _ = self.sensor_enc(xs)
        _, c_n, _ = self.net_enc(xn)
        return c_s, c_n

    def forward(self, xs1, xn1, xs2, xn2):
        """Two augmented view pairs → contrastive projections + reconstruction."""
        c_s1, c_n1 = self.encode(xs1, xn1)
        c_s2, c_n2 = self.encode(xs2, xn2)

        # Joint embeddings for NT-Xent (fuse sensor + network)
        z1 = self.proj_joint(torch.cat([c_s1, c_n1], dim=-1))
        z2 = self.proj_joint(torch.cat([c_s2, c_n2], dim=-1))

        # Reconstruction on view1 sensor stream
        lat      = self.ae_to_lat(c_s1)
        x_hat    = self.ae_dec(lat)           # (B, T*D)
        x_target = xs1.view(xs1.size(0), -1)  # (B, T*D)
        recon_err = F.mse_loss(x_hat, x_target)

        return z1, z2, recon_err


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def load_pretrain_data():
    """Load A11 windows from swat_final.npz, or fall back to A11 CSV."""
    npz_file = Path(NPZ_PATH)
    if npz_file.exists():
        data = np.load(NPZ_PATH, allow_pickle=True)
        if 'X_s_pretrain' in data and 'X_n_pretrain' in data:
            Xs = data['X_s_pretrain']
            Xn = data['X_n_pretrain']
            print(f'  Loaded A11 pre-train windows from {NPZ_PATH}')
            print(f'  Xs: {Xs.shape}  Xn: {Xn.shape}')
            return Xs, Xn

    # Fallback: use training split (will be mostly normal if A11 is separate)
    print('  WARNING: X_s_pretrain not found — using train split.')
    print('  Run 01_build_windows.py first for best results.')
    data = np.load('trustgate_data/swat_multilabel.npz', allow_pickle=True)
    Xs = data['X_s_train']
    Xn = data['X_n_train']
    return Xs, Xn


def train():
    Path('trustgate_data').mkdir(exist_ok=True)

    print('\n' + '='*60)
    print('TrustGate Phase A — Contrastive Pre-training')
    print('='*60)

    # ── Load data ─────────────────────────────────────────────────
    print('\n[1/4] Loading A11 normal windows...')
    Xs, Xn = load_pretrain_data()
    sensor_dim = Xs.shape[2]
    net_dim    = Xn.shape[2]
    win_size   = Xs.shape[1]
    print(f'  Windows: {len(Xs):,}  |  sensor_dim={sensor_dim}  net_dim={net_dim}')

    dataset    = ContrastiveICSDataset(Xs, Xn)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0,
                             pin_memory=True, drop_last=True)
    print(f'  Batches per epoch: {len(loader)}')

    # ── Build model ───────────────────────────────────────────────
    print('\n[2/4] Building pre-training model...')
    model      = PretrainModel(sensor_dim, net_dim, win_size, PROJ_DIM).to(DEVICE)
    ntxent     = NTXentLoss(temperature=TEMPERATURE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable parameters: {n_params:,}')

    # ── Training ──────────────────────────────────────────────────
    print('\n[3/4] Training...\n')
    log_rows = []
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_contrast = 0.0
        epoch_recon    = 0.0
        steps          = 0

        n_batches = len(loader)
        for batch_idx, (xs1, xn1, xs2, xn2) in enumerate(loader):
            xs1 = xs1.to(DEVICE); xn1 = xn1.to(DEVICE)
            xs2 = xs2.to(DEVICE); xn2 = xn2.to(DEVICE)

            optimizer.zero_grad()
            z1, z2, recon_err = model(xs1, xn1, xs2, xn2)

            L_contrast = ntxent(z1, z2)
            L_total    = L_contrast + RECON_WEIGHT * recon_err
            L_total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_contrast += L_contrast.item()
            epoch_recon    += recon_err.item()
            steps          += 1

            # Show progress every 20 batches so user sees activity
            if batch_idx % 20 == 0:
                print(f'    Epoch {epoch}/{EPOCHS}  batch {batch_idx}/{n_batches}  '
                      f'contrast={L_contrast.item():.4f}', flush=True)

        scheduler.step()

        avg_cont  = epoch_contrast / steps
        avg_recon = epoch_recon / steps
        avg_total = avg_cont + RECON_WEIGHT * avg_recon
        lr_now    = optimizer.param_groups[0]['lr']

        log_rows.append({'epoch': epoch, 'L_contrast': avg_cont,
                          'L_recon': avg_recon, 'L_total': avg_total, 'lr': lr_now})

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{EPOCHS} | '
                  f'Contrast: {avg_cont:.4f}  '
                  f'Recon: {avg_recon:.4f}  '
                  f'Total: {avg_total:.4f}  '
                  f'LR: {lr_now:.2e}', flush=True)

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.sensor_enc.state_dict(), OUT_S_ENC)
            torch.save(model.net_enc.state_dict(),    OUT_N_ENC)
            # Save full autoencoder for recon_error computation at inference
            torch.save({'model_state': model.state_dict(),
                        'sensor_dim': sensor_dim,
                        'net_dim':    net_dim,
                        'win_size':   win_size}, OUT_AE)

    # ── Save log ──────────────────────────────────────────────────
    print('\n[4/4] Saving outputs...')
    pd.DataFrame(log_rows).to_csv(OUT_LOG, index=False)
    print(f'  Best total loss: {best_loss:.4f}')
    print(f'  Encoder weights: {OUT_S_ENC}')
    print(f'                   {OUT_N_ENC}')
    print(f'  Autoencoder    : {OUT_AE}')
    print(f'  Training log   : {OUT_LOG}')
    print('\n[DONE] Run python 05_finetune.py next')


if __name__ == '__main__':
    train()