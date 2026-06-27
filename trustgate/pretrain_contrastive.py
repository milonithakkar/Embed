# pretrain_contrastive.py
# Self-supervised contrastive pre-training on A11 normal data
# NT-Xent loss: pulls augmented views of same window together
# Run: python pretrain_contrastive.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

A11_PATH  = r'D:\trustgate_pcaps\a11_pretrain_windows.npz'
CKPT_PATH = r'D:\trustgate_pcaps\pretrained_encoder.pth'

# Config
BATCH_SIZE   = 256
LR           = 3e-4
MAX_EPOCHS   = 50
TEMPERATURE  = 0.07
DEVICE       = torch.device('cuda' if torch.cuda.is_available()
                             else 'cpu')

print("=" * 60)
print("TrustGate — Contrastive Pre-training on A11")
print("=" * 60)
print(f"Device: {DEVICE}")

# ── Augmentations ─────────────────────────────────────────────
def augment(x, device):
    """
    Apply random augmentation to a batch of windows.
    x: (B, T, 71)
    Returns augmented x of same shape.
    """
    B, T, D = x.shape
    x = x.clone()

    # 1. Magnitude jitter (Gaussian noise σ=0.05)
    x = x + torch.randn_like(x) * 0.05

    # 2. Sensor dropout — zero out random 10% of sensors
    #    for a random 3-second block
    mask = torch.ones(B, T, D, device=device)
    for b in range(B):
        n_drop  = max(1, int(D * 0.10))
        drop_s  = torch.randperm(D)[:n_drop]
        t_start = torch.randint(0, max(1, T-3), (1,)).item()
        mask[b, t_start:t_start+3, drop_s] = 0.0
    x = x * mask

    # 3. Window slicing — take 25-step slice, pad to 30
    if T >= 25:
        t_start = torch.randint(0, T-24, (1,)).item()
        x_slice = x[:, t_start:t_start+25, :]
        x = F.pad(x_slice, (0, 0, 0, T-25))

    # 4. Time shift — roll by ±2 steps
    shift = torch.randint(-2, 3, (1,)).item()
    if shift != 0:
        x = torch.roll(x, shift, dims=1)

    return x


# ── NT-Xent Loss ──────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    For a batch of N windows with 2 views each (2N total),
    pulls the two views of the same window together,
    pushes all other pairs apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        # z1, z2: (B, D) — embeddings of two views
        B = z1.size(0)

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: (2B, D)
        z  = torch.cat([z1, z2], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.T

        # Mask out self-similarity
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0, B,   device=z.device)
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ── Projection Head ───────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Small MLP on top of encoder output.
    Maps pooled encoder output → contrastive embedding space.
    This is discarded after pre-training — only encoder weights kept.
    """
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Load A11 windows ──────────────────────────────────────────
print(f"\n[1/4] Loading A11 windows...")
data    = np.load(A11_PATH, allow_pickle=True)
windows = data['windows']   # (N, 30, 71)
print(f"  Windows: {windows.shape}")
print(f"  Range:   {windows.min():.3f} to {windows.max():.3f}")

# Use sensor stream only (71 sensors)
# A11 has no network stream → we pre-train sensor encoder only
X = torch.FloatTensor(windows)
ds = TensorDataset(X)
loader = DataLoader(ds, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
print(f"  Batches: {len(loader)}")

# ── Build model + projection head ────────────────────────────
print(f"\n[2/4] Building model...")
model = TrustGateModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {n_params:,}")

# Projection head
# Input dim = sensor_hidden*2 + network_hidden*2 = 256+128 = 384
# But for pre-training sensor-only, we use sensor encoder output
# sensor_enc output: (B, T, 256) → pool → (B, 256)
proj_head = ProjectionHead(
    in_dim=256,    # sensor_enc hidden*2 = 128*2
    hidden_dim=256,
    out_dim=128
).to(DEVICE)

print(f"  Projection head params: "
      f"{sum(p.numel() for p in proj_head.parameters()):,}")

# Only train sensor encoder + projection head
# Network encoder stays at random init (no A11 network data)
params_to_train = (list(model.sensor_enc.parameters()) +
                   list(proj_head.parameters()))

optimizer  = torch.optim.AdamW(params_to_train,
                                lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
criterion  = NTXentLoss(temperature=TEMPERATURE)

# ── Pre-training loop ─────────────────────────────────────────
print(f"\n[3/4] Pre-training ({MAX_EPOCHS} epochs)...")
print(f"\n  {'Ep':>4}  {'Loss':>8}  {'LR':>10}  {'Time':>6}")
print(f"  {'-'*35}")

best_loss = float('inf')

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.sensor_enc.train()
    proj_head.train()

    epoch_loss = 0.0
    n_batches  = 0

    for (xs,) in loader:
        xs = xs.to(DEVICE)   # (B, 30, 71)

        # Create two augmented views
        x1 = augment(xs, DEVICE)
        x2 = augment(xs, DEVICE)

        # Need a dummy network input for model.sensor_enc
        # since we call sensor_enc directly
        # sensor_enc: LayerNormBiLSTM
        # Input: (B, T, 71) → Output: (B, T, 256)

        h1 = model.sensor_enc(x1)   # (B, T, 256)
        h2 = model.sensor_enc(x2)

        # Pool over time
        z1 = h1.mean(dim=1)   # (B, 256)
        z2 = h2.mean(dim=1)

        # Project
        z1 = proj_head(z1)   # (B, 128)
        z2 = proj_head(z2)

        # NT-Xent loss
        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    cur_lr   = optimizer.param_groups[0]['lr']
    elapsed  = int(time.time() - t0)

    # Save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch':             epoch,
            'sensor_enc_state':  model.sensor_enc.state_dict(),
            'loss':              best_loss,
        }, CKPT_PATH)
        star = " ← SAVED"
    else:
        star = ""

    if epoch % 5 == 0 or epoch <= 3:
        print(f"  {epoch:>4}  {avg_loss:>8.4f}  "
              f"{cur_lr:>10.2e}  {elapsed:>4}s{star}")

print(f"\n[4/4] Pre-training complete.")
print(f"  Best loss:  {best_loss:.4f}")
print(f"  Checkpoint: {CKPT_PATH}")
print(f"\n{'='*60}")
print(f"Next: python finetune_v4.py")
print(f"{'='*60}")