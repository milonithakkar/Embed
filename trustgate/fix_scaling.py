# fix_scaling.py
# Refits RobustScaler on NORMAL training windows only
# then reapplies to all splits and saves fixed NPZ
# Run: python fix_scaling.py

import numpy as np
from sklearn.preprocessing import RobustScaler
import os

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
OUT_PATH  = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'

print("=" * 60)
print("TrustGate — Fix Scaling on NPZ Data")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────
print(f"\n[1/6] Loading {DATA_PATH}...")
data = np.load(DATA_PATH, allow_pickle=True)

X_s_train = data['X_s_train'].copy()   # (N_train, 30, 71)
X_n_train = data['X_n_train'].copy()   # (N_train, 30, 19)
X_s_val   = data['X_s_val'].copy()
X_n_val   = data['X_n_val'].copy()
X_s_test  = data['X_s_test'].copy()
X_n_test  = data['X_n_test'].copy()
y_b_train = data['y_b_train']

print(f"  X_s_train: {X_s_train.shape}")
print(f"  X_n_train: {X_n_train.shape}")
print(f"  Normal windows in train: {(y_b_train == 0).sum():,}")
print(f"  Attack windows in train: {(y_b_train == 1).sum():,}")

# ── Fit scaler on NORMAL training windows only ────────────────
print(f"\n[2/6] Fitting RobustScaler on NORMAL training windows only...")

normal_mask = (y_b_train == 0)
X_s_normal  = X_s_train[normal_mask]   # (N_normal, 30, 71)
X_n_normal  = X_n_train[normal_mask]   # (N_normal, 30, 19)

N_train, T, D_s = X_s_train.shape
D_n = X_n_train.shape[2]

# Flatten to (N*T, D) for scaler fitting
X_s_normal_flat = X_s_normal.reshape(-1, D_s)   # (N_normal*30, 71)
X_n_normal_flat = X_n_normal.reshape(-1, D_n)   # (N_normal*30, 19)

scaler_s = RobustScaler(quantile_range=(10, 90))
scaler_n = RobustScaler(quantile_range=(10, 90))

scaler_s.fit(X_s_normal_flat)
scaler_n.fit(X_n_normal_flat)

print(f"  Sensor scaler fitted on {X_s_normal_flat.shape[0]:,} timesteps")
print(f"  Network scaler fitted on {X_n_normal_flat.shape[0]:,} timesteps")

# Check scaler center and scale
print(f"\n  Sensor scaler:")
print(f"    center range: {scaler_s.center_.min():.3f} to {scaler_s.center_.max():.3f}")
print(f"    scale  range: {scaler_s.scale_.min():.4f} to {scaler_s.scale_.max():.4f}")

zero_scale_s = (scaler_s.scale_ < 1e-6).sum()
zero_scale_n = (scaler_n.scale_ < 1e-6).sum()
print(f"    zero-scale cols (constant features): {zero_scale_s}")

if zero_scale_s > 0:
    zero_idx = np.where(scaler_s.scale_ < 1e-6)[0]
    sensor_cols = list(data['sensor_cols'])
    print(f"    Constant sensor cols: {[sensor_cols[i] for i in zero_idx]}")
    print(f"    → These will be clipped to 0 after scaling (safe)")
    # Set scale to 1.0 for constant columns to avoid division by zero
    scaler_s.scale_[scaler_s.scale_ < 1e-6] = 1.0

if zero_scale_n > 0:
    zero_idx = np.where(scaler_n.scale_ < 1e-6)[0]
    network_cols = list(data['network_cols'])
    print(f"    Constant network cols: {[network_cols[i] for i in zero_idx]}")
    scaler_n.scale_[scaler_n.scale_ < 1e-6] = 1.0

# ── Apply scaler to all splits ────────────────────────────────
print(f"\n[3/6] Applying scaler to all splits...")

def scale_split(X, scaler, name):
    N, T, D = X.shape
    flat    = X.reshape(-1, D)
    scaled  = scaler.transform(flat)
    # Clip extreme outliers to [-10, 10] — attack values
    # will be large but bounded, preventing gradient explosion
    scaled  = np.clip(scaled, -10.0, 10.0)
    result  = scaled.reshape(N, T, D)
    print(f"  {name}: mean={result.mean():.4f}  std={result.std():.4f}  "
          f"min={result.min():.3f}  max={result.max():.3f}")
    return result.astype(np.float32)

X_s_train_sc = scale_split(X_s_train, scaler_s, "X_s_train")
X_n_train_sc = scale_split(X_n_train, scaler_n, "X_n_train")
X_s_val_sc   = scale_split(X_s_val,   scaler_s, "X_s_val  ")
X_n_val_sc   = scale_split(X_n_val,   scaler_n, "X_n_val  ")
X_s_test_sc  = scale_split(X_s_test,  scaler_s, "X_s_test ")
X_n_test_sc  = scale_split(X_n_test,  scaler_n, "X_n_test ")

# ── Verify separation after scaling ───────────────────────────
print(f"\n[4/6] Verifying attack vs normal separation after scaling...")

y_b = y_b_train
attack_idx = np.where(y_b == 1)[0]
normal_idx = np.where(y_b == 0)[0]

att_mean = X_s_train_sc[attack_idx].mean()
att_std  = X_s_train_sc[attack_idx].std()
nor_mean = X_s_train_sc[normal_idx].mean()
nor_std  = X_s_train_sc[normal_idx].std()

print(f"  Attack windows: mean={att_mean:.4f}  std={att_std:.4f}")
print(f"  Normal windows: mean={nor_mean:.4f}  std={nor_std:.4f}")

# Check per-column separation
att_col_means = X_s_train_sc[attack_idx].mean(axis=(0, 1))
nor_col_means = X_s_train_sc[normal_idx].mean(axis=(0, 1))
diff = np.abs(att_col_means - nor_col_means)
top5 = np.argsort(diff)[::-1][:5]
sensor_cols = list(data['sensor_cols'])

print(f"\n  Top 5 separating features after scaling:")
for rank, idx in enumerate(top5):
    print(f"    [{rank+1}] {sensor_cols[idx]:20s}: "
          f"normal={nor_col_means[idx]:.4f}  "
          f"attack={att_col_means[idx]:.4f}  "
          f"diff={diff[idx]:.4f}")

zero_sep = (diff < 0.05).sum()
print(f"\n  Columns with near-zero separation (<0.05): {zero_sep}/71")
if zero_sep > 40:
    print(f"  WARNING: Still too many non-separating columns.")
    print(f"  Consider dropping constant binary columns from training.")
else:
    print(f"  OK: Sufficient separating signal for training.")

# ── Save fixed NPZ ────────────────────────────────────────────
print(f"\n[5/6] Saving fixed NPZ to {OUT_PATH}...")

save_dict = {
    # Fixed feature arrays
    'X_s_train': X_s_train_sc,
    'X_n_train': X_n_train_sc,
    'X_s_val':   X_s_val_sc,
    'X_n_val':   X_n_val_sc,
    'X_s_test':  X_s_test_sc,
    'X_n_test':  X_n_test_sc,
    # Labels unchanged
    'y_b_train': data['y_b_train'],
    'y_c_train': data['y_c_train'],
    'y_p_train': data['y_p_train'],
    'y_b_val':   data['y_b_val'],
    'y_c_val':   data['y_c_val'],
    'y_p_val':   data['y_p_val'],
    'y_b_test':  data['y_b_test'],
    'y_c_test':  data['y_c_test'],
    'y_p_test':  data['y_p_test'],
    # Metadata unchanged
    'sensor_cols':     data['sensor_cols'],
    'network_cols':    data['network_cols'],
    'component_names': data['component_names'],
}

np.savez_compressed(OUT_PATH, **save_dict)
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"  Saved: {OUT_PATH} ({size_mb:.1f} MB)")

# ── Final verification ────────────────────────────────────────
print(f"\n[6/6] Final verification of saved file...")
verify = np.load(OUT_PATH, allow_pickle=True)
print(f"  Keys: {list(verify.keys())}")
print(f"  X_s_train: {verify['X_s_train'].shape}  "
      f"dtype={verify['X_s_train'].dtype}")
print(f"  X_n_train: {verify['X_n_train'].shape}  "
      f"dtype={verify['X_n_train'].dtype}")

# Spot check
xs = verify['X_s_train']
print(f"\n  Spot check X_s_train:")
print(f"    mean={xs.mean():.4f}  std={xs.std():.4f}")
print(f"    min={xs.min():.3f}   max={xs.max():.3f}")

if abs(xs.mean()) < 1.0 and xs.std() < 5.0:
    print(f"\n  SCALING OK — data is in reasonable range for training")
else:
    print(f"\n  WARNING — scaling may still be off, check values above")

print(f"\n{'='*60}")
print(f"Fix complete.")
print(f"Update DATA_PATH in train_v3.py to:")
print(f"  {OUT_PATH}")
print(f"Then run: python train_v3.py")
print(f"{'='*60}")