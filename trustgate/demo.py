# Save as C:\Users\HP\Downloads\trustgate\fit_network_scaler.py
# Run: python fit_network_scaler.py

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler

CSV_PATH    = r'D:\trustgate_pcaps\network_features_A12_smart.csv'
SCALER_OUT  = r'D:\trustgate_pcaps\network_scaler.pkl'
NPZ_PATH    = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'

print("=" * 60)
print("TrustGate — Fit Network Feature Scaler")
print("=" * 60)

# ── Load raw CSV ──────────────────────────────────────────────────
print(f"\n[1/4] Loading raw network features...")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── Load exact column order from training npz ─────────────────────
print(f"\n[2/4] Loading exact network_cols order from npz...")
data = np.load(NPZ_PATH, allow_pickle=True)
network_cols = [str(c) for c in data['network_cols']]
print(f"  network_cols ({len(network_cols)}):")
for i, c in enumerate(network_cols):
    print(f"    [{i:2d}] {c}")

# ── Align CSV columns to exact training order ─────────────────────
print(f"\n[3/4] Aligning CSV columns to training order...")

missing = [c for c in network_cols if c not in df.columns]
extra   = [c for c in df.columns
           if c not in network_cols and c != 'timestamp_sgt']

if missing:
    print(f"  WARNING — Missing columns (will be filled with 0):")
    for c in missing:
        print(f"    {c}")
        df[c] = 0.0

if extra:
    print(f"  Extra columns in CSV (ignored): {extra}")

# Select and order exactly as training
df_net = df[network_cols].copy()
df_net = df_net.fillna(0.0)

print(f"  Aligned shape: {df_net.shape}")
print(f"\n  Raw feature statistics (before scaling):")
print(f"  {'Feature':30s} {'Mean':>10s} {'Std':>10s} "
      f"{'Min':>10s} {'Max':>10s}")
print(f"  {'-'*65}")
for col in network_cols:
    vals = df_net[col]
    print(f"  {col:30s} {vals.mean():>10.3f} {vals.std():>10.3f} "
          f"{vals.min():>10.3f} {vals.max():>10.3f}")

# ── Fit RobustScaler ──────────────────────────────────────────────
print(f"\n[4/4] Fitting RobustScaler...")
scaler = RobustScaler()
scaler.fit(df_net.values)

print(f"  Scaler center_ (median):")
for i, (col, center) in enumerate(zip(network_cols, scaler.center_)):
    print(f"    [{i:2d}] {col:30s}  center={center:.4f}")

print(f"\n  Scaler scale_ (IQR):")
for i, (col, scale) in enumerate(zip(network_cols, scaler.scale_)):
    print(f"    [{i:2d}] {col:30s}  scale={scale:.4f}")

# ── Verify scaled output matches training data ranges ─────────────
print(f"\n  Verification — scaled output ranges vs training data:")
X_scaled = scaler.transform(df_net.values)
X_n_train = data['X_n_train'].reshape(-1, 19)

print(f"\n  {'Feature':30s} {'Train mean':>12s} {'Scaled mean':>12s} "
      f"{'Match':>8s}")
print(f"  {'-'*68}")
for i, col in enumerate(network_cols):
    train_mean  = float(X_n_train[:, i].mean())
    scaled_mean = float(X_scaled[:, i].mean())
    match = "✓" if abs(train_mean - scaled_mean) < 0.5 else "✗ CHECK"
    print(f"  {col:30s} {train_mean:>12.4f} {scaled_mean:>12.4f} "
          f"{match:>8s}")

# ── Save ──────────────────────────────────────────────────────────
with open(SCALER_OUT, 'wb') as f:
    pickle.dump({
        'scaler':       scaler,
        'feature_names':network_cols,
        'n_features':   len(network_cols),
    }, f)

print(f"\n{'='*60}")
print(f"Saved: {SCALER_OUT}")
print(f"{'='*60}")
