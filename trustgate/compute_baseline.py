# compute_baseline.py
# Computes per-sensor normal baseline from training data
# Run: python compute_baseline.py

import numpy as np

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_PATH  = r'D:\trustgate_pcaps\sensor_normal_baseline.npy'

print("Computing normal baseline...")
data   = np.load(DATA_PATH, allow_pickle=True)
X_s    = data['X_s_train']   # (N, 30, 71)
y_b    = data['y_b_train']

# Use ONLY normal windows
normal_mask = (y_b == 0)
X_normal    = X_s[normal_mask]   # (N_normal, 30, 71)

# Per-sensor mean across all normal windows and all time steps
baseline = X_normal.mean(axis=(0, 1))   # (71,)

print(f"Normal windows used : {normal_mask.sum():,}")
print(f"Baseline shape      : {baseline.shape}")
print(f"Baseline range      : {baseline.min():.4f} to {baseline.max():.4f}")

np.save(OUT_PATH, baseline)
print(f"Saved: {OUT_PATH}")