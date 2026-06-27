# build_a11_windows.py
# Processes A11 normal data → windowed NPZ for pre-training
# Run: python build_a11_windows.py

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import RobustScaler

A11_PATH     = r'D:\Users\HP\Downloads\SWaT.A10_OTDataset_19-Feb-2026_0930_1735.csv'
A12_NPZ      = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_PATH     = r'D:\trustgate_pcaps\a11_pretrain_windows.npz'
SCALER_PATH  = r'D:\trustgate_pcaps\a11_scaler.pkl'

WINDOW_SIZE  = 30
STRIDE       = 1

# These 71 columns match our model exactly
# Same order as A12_windowed_v3_fixed.npz sensor_cols
SENSOR_71 = [
    "P1_STATE", "LIT101.Pv", "FIT101.Pv", "MV101.Status",
    "P101.Status", "P102.Status", "P2_STATE", "FIT201.Pv",
    "AIT201.Pv", "AIT202.Pv", "AIT203.Pv", "MV201.Status",
    "P201.Status", "P202.Status", "P203.Status", "P204.Status",
    "P205.Status", "P206.Status", "P207.Status", "P208.Status",
    "P3_STATE", "AIT301.Pv", "AIT302.Pv", "AIT303.Pv",
    "LIT301.Pv", "FIT301.Pv", "DPIT301.Pv", "MV301.Status",
    "MV302.Status", "MV303.Status", "MV304.Status", "P301.Status",
    "P302.Status", "P4_STATE", "LIT401.Pv", "FIT401.Pv",
    "AIT401.Pv", "AIT402.Pv", "P401.Status", "P402.Status",
    "P403.Status", "P404.Status", "UV401.Status", "P5_STATE",
    "FIT501.Pv", "FIT502.Pv", "FIT503.Pv", "FIT504.Pv",
    "AIT501.Pv", "AIT502.Pv", "AIT503.Pv", "AIT504.Pv",
    "PIT501.Pv", "PIT502.Pv", "PIT503.Pv", "P501.Status",
    "P501.Speed", "P502.Status", "P502.Speed", "MV501.Status",
    "MV502.Status", "MV503.Status", "MV504.Status", "P6_STATE",
    "LIT601.Pv", "LIT602.Pv", "FIT601.Pv", "FIT602.Pv",
    "P601.Status", "P602.Status", "P603.Status",
]

# Categorical columns that need encoding
# Active/Inactive → 1/0,  Bad Input → 0
CATEGORICAL_MAP = {
    'Active':    1.0,
    'Inactive':  0.0,
    'Bad Input': 0.0,
    'High':      1.0,
    'Low':       0.0,
    'Normal':    0.5,
}

print("=" * 60)
print("A11 Pre-training Window Builder")
print("=" * 60)

# ── Load A11 ──────────────────────────────────────────────────
print(f"\n[1/6] Loading A11...")
df = pd.read_csv(A11_PATH, low_memory=False)
print(f"  Shape: {df.shape}")
print(f"  Rows:  {len(df):,}")

# ── Parse timestamp ───────────────────────────────────────────
print(f"\n[2/6] Parsing timestamps...")
df['t_stamp'] = pd.to_datetime(df['t_stamp'],
                                format='%d-%b-%Y %H:%M:%S',
                                errors='coerce')
n_bad = df['t_stamp'].isna().sum()
if n_bad > 0:
    print(f"  WARNING: {n_bad} unparseable timestamps — dropping")
    df = df.dropna(subset=['t_stamp'])
df = df.sort_values('t_stamp').reset_index(drop=True)
print(f"  Duration: {df['t_stamp'].min()} → {df['t_stamp'].max()}")
print(f"  Total seconds: {len(df):,}")

# ── Extract 71 sensor columns ─────────────────────────────────
print(f"\n[3/6] Extracting 71 sensor columns...")
missing = [c for c in SENSOR_71 if c not in df.columns]
if missing:
    print(f"  WARNING: Missing columns: {missing}")
    SENSOR_71_USE = [c for c in SENSOR_71 if c in df.columns]
else:
    SENSOR_71_USE = SENSOR_71
    print(f"  All 71 columns found")

df_sensors = df[SENSOR_71_USE].copy()

# ── Convert all columns to numeric ───────────────────────────
print(f"\n[4/6] Converting to numeric...")
for col in df_sensors.columns:
    if df_sensors[col].dtype == object:
        # Try direct numeric conversion first
        numeric = pd.to_numeric(df_sensors[col], errors='coerce')
        if numeric.isna().sum() > len(df_sensors) * 0.5:
            # More than 50% non-numeric → categorical
            df_sensors[col] = df_sensors[col].map(
                CATEGORICAL_MAP).fillna(0.0)
        else:
            df_sensors[col] = numeric.fillna(
                numeric.median())

# Final fillna
df_sensors = df_sensors.fillna(df_sensors.median())
df_sensors = df_sensors.astype(np.float32)

print(f"  Shape: {df_sensors.shape}")
print(f"  Range: {df_sensors.values.min():.3f} to "
      f"{df_sensors.values.max():.3f}")
print(f"  NaN remaining: {df_sensors.isna().sum().sum()}")

# ── Verify columns match A12 ──────────────────────────────────
print(f"\n[5/6] Verifying column alignment with A12...")
a12_data = np.load(A12_NPZ, allow_pickle=True)
a12_cols  = list(a12_data['sensor_cols'])

a11_cols = list(df_sensors.columns)
match    = (a11_cols == a12_cols)
print(f"  A11 cols: {len(a11_cols)}")
print(f"  A12 cols: {len(a12_cols)}")
print(f"  Column order match: {match}")

if not match:
    print("  Mismatches:")
    for i, (a, b) in enumerate(zip(a11_cols, a12_cols)):
        if a != b:
            print(f"    pos {i}: A11={a}  A12={b}")

# ── Fit RobustScaler on A11 normal data ──────────────────────
print(f"\n[6/6] Fitting RobustScaler and building windows...")

# Fit scaler on ALL A11 data (it is all normal)
scaler = RobustScaler(quantile_range=(10, 90))
X_raw  = df_sensors.values   # (29160, 71)
scaler.fit(X_raw)

# Check for zero-scale columns
zero_scale = (scaler.scale_ < 1e-6).sum()
if zero_scale > 0:
    print(f"  Fixing {zero_scale} zero-scale columns")
    scaler.scale_[scaler.scale_ < 1e-6] = 1.0

# Transform
X_scaled = scaler.transform(X_raw)
X_scaled  = np.clip(X_scaled, -10.0, 10.0).astype(np.float32)

print(f"  Scaled range: {X_scaled.min():.3f} to {X_scaled.max():.3f}")
print(f"  Scaled mean:  {X_scaled.mean():.4f}")
print(f"  Scaled std:   {X_scaled.std():.4f}")

# Build sliding windows
N = len(X_scaled)
n_windows = (N - WINDOW_SIZE) // STRIDE + 1
print(f"\n  Building {n_windows:,} windows "
      f"(size={WINDOW_SIZE}, stride={STRIDE})...")

windows = np.lib.stride_tricks.sliding_window_view(
    X_scaled, (WINDOW_SIZE, X_scaled.shape[1])
)
windows = windows[:, 0, :, :]   # (n_windows, 30, 71)
windows = windows[::STRIDE]

print(f"  Windows shape: {windows.shape}")
print(f"  Memory: {windows.nbytes / 1e6:.1f} MB")

# Save
np.savez_compressed(OUT_PATH, windows=windows,
                    sensor_cols=np.array(a11_cols))
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n  Saved: {OUT_PATH}")
print(f"  Saved: {SCALER_PATH}")

# Final check
verify = np.load(OUT_PATH, allow_pickle=True)
print(f"\n  Verified: {verify['windows'].shape}")
print(f"  Sample window mean: "
      f"{verify['windows'][0].mean():.4f}")

print(f"\n{'='*60}")
print(f"A11 windows ready.")
print(f"Next: python pretrain_contrastive.py")
print(f"{'='*60}")