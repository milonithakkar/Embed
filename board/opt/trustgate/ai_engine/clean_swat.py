"""
Day 12 Step 2 — SWaT Data Cleaning + Feature Engineering
Run on your LAPTOP in the same folder as merged.csv

What this script does:
1. Strips whitespace from column names and label values
2. Drops the 7 NaN-heavy columns (same 991,800 NaN pattern = sensor group offline)
3. Parses timestamps correctly (mixed format, dayfirst=True)
4. Encodes labels: Normal=0, Attack=1
5. Builds SENSOR features (44 physical sensor values, normalized)
6. Builds NETWORK features (synthetic Modbus behavior — register index, value deviation, rate of change)
7. Creates sliding window sequences (window=30, stride=1)
8. Saves train/val/test splits as .npz files ready for model training
9. Prints full summary so we verify before training
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# ── Config ─────────────────────────────────────────────────────────
WINDOW_SIZE   = 30      # 30 seconds of history per sample
STRIDE        = 5       # step 5 rows between windows (reduces dataset size 5x, keeps diversity)
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15
RANDOM_SEED   = 42
OUT_DIR       = './trustgate_data'

# These 7 columns have 68.8% NaNs — same sensor group offline together
# Dropping columns keeps ALL rows; dropping rows would lose 68.8% of normal data
NAN_COLUMNS_TO_DROP = ['MV101', 'AIT201', 'MV201', 'P201', 'P202', 'P204', 'MV303']

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("TrustGate — SWaT Data Cleaning")
print("=" * 60)

# ── Step 1: Load ───────────────────────────────────────────────────
print("\n[1/8] Loading merged.csv...")
df = pd.read_csv('./merged.csv', low_memory=False)
print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Step 2: Strip whitespace ───────────────────────────────────────
print("\n[2/8] Stripping whitespace from column names and labels...")
df.columns = df.columns.str.strip()
df['Normal/Attack'] = df['Normal/Attack'].astype(str).str.strip()
print(f"  Label values found: {df['Normal/Attack'].unique()}")

# ── Step 3: Drop NaN columns ──────────────────────────────────────
print(f"\n[3/8] Dropping {len(NAN_COLUMNS_TO_DROP)} NaN-heavy columns...")
# Only drop what actually exists after stripping
cols_to_drop = [c for c in NAN_COLUMNS_TO_DROP if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"  Dropped: {cols_to_drop}")
print(f"  Remaining NaNs: {df.isnull().sum().sum()}")
print(f"  Shape after drop: {df.shape}")

# ── Step 4: Parse timestamps ───────────────────────────────────────
print("\n[4/8] Parsing timestamps (mixed format, dayfirst=True)...")
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"  First: {df['Timestamp'].iloc[0]}")
    print(f"  Last:  {df['Timestamp'].iloc[-1]}")
    duration = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]
    print(f"  Duration: {duration}")
except Exception as e:
    print(f"  [!] Timestamp parse failed: {e}")
    print(f"  Continuing without timestamp sorting...")

# ── Step 5: Encode labels ──────────────────────────────────────────
print("\n[5/8] Encoding labels...")
df['label'] = (df['Normal/Attack'] == 'Attack').astype(int)
print(f"  Normal (0): {(df['label']==0).sum():,}")
print(f"  Attack (1): {(df['label']==1).sum():,}")
print(f"  Imbalance:  {(df['label']==0).sum() / (df['label']==1).sum():.1f}:1")

# ── Step 6: Sensor features ────────────────────────────────────────
print("\n[6/8] Building sensor features...")
exclude = ['Timestamp', 'Normal/Attack', 'label']
sensor_cols = [c for c in df.columns if c not in exclude]
print(f"  Sensor columns: {len(sensor_cols)}")
print(f"  Columns: {sensor_cols}")

sensor_data = df[sensor_cols].values.astype(np.float32)

# Normalize to [0,1] using MinMaxScaler — fit on normal data only to avoid attack leakage
normal_mask = df['label'].values == 0
scaler = MinMaxScaler()
scaler.fit(sensor_data[normal_mask])          # fit on normal only
sensor_data_scaled = scaler.transform(sensor_data)
joblib.dump(scaler, os.path.join(OUT_DIR, 'sensor_scaler.pkl'))
print(f"  Scaler saved → {OUT_DIR}/sensor_scaler.pkl")
print(f"  Sensor data shape: {sensor_data_scaled.shape}")

# ── Step 7: Network/Modbus features ───────────────────────────────
print("\n[7/8] Building network features (rate of change + deviation)...")
# Since SWaT has no actual Modbus packet logs, we derive network-proxy features
# from sensor behavior — this is valid because Modbus writes CAUSE sensor changes
# Features: rate of change per sensor, deviation from rolling mean
# These capture the behavioral signature of attacks even without raw network packets

ROC_WINDOW = 5   # rate of change over last 5 seconds

roc_features = np.zeros_like(sensor_data_scaled, dtype=np.float32)
dev_features  = np.zeros_like(sensor_data_scaled, dtype=np.float32)

print(f"  Computing rate-of-change and deviation (window={ROC_WINDOW})...")
for i in range(ROC_WINDOW, len(sensor_data_scaled)):
    roc_features[i] = sensor_data_scaled[i] - sensor_data_scaled[i - ROC_WINDOW]

# Rolling deviation from 60s mean (captures slow drift attacks)
DEV_WINDOW = 60
for i in range(DEV_WINDOW, len(sensor_data_scaled)):
    window_mean = sensor_data_scaled[i-DEV_WINDOW:i].mean(axis=0)
    dev_features[i] = sensor_data_scaled[i] - window_mean

# Stack: [sensor_values | rate_of_change | deviation_from_mean]
# Shape: (N, 44*3) = (N, 132)
network_data = np.concatenate([sensor_data_scaled, roc_features, dev_features], axis=1)
print(f"  Network proxy features shape: {network_data.shape}")

labels = df['label'].values

# ── Step 8: Sliding window sequences ──────────────────────────────
print(f"\n[8/8] Creating sliding windows (size={WINDOW_SIZE}, stride={STRIDE})...")

def make_windows(sensor, network, labels, window, stride):
    X_sensor, X_network, y = [], [], []
    for i in range(0, len(labels) - window, stride):
        X_sensor.append(sensor[i:i+window])
        X_network.append(network[i:i+window])
        # Label = 1 if ANY timestep in window is an attack
        y.append(1 if labels[i:i+window].sum() > 0 else 0)
    return (np.array(X_sensor, dtype=np.float32),
            np.array(X_network, dtype=np.float32),
            np.array(y, dtype=np.int32))

X_s, X_n, y = make_windows(sensor_data_scaled, network_data, labels, WINDOW_SIZE, STRIDE)
print(f"  Total windows: {len(y):,}")
print(f"  X_sensor shape:  {X_s.shape}  (windows, timesteps, sensor_features)")
print(f"  X_network shape: {X_n.shape}  (windows, timesteps, network_features)")
print(f"  Attack windows:  {y.sum():,} ({y.sum()/len(y)*100:.1f}%)")
print(f"  Normal windows:  {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")

# ── Train/Val/Test split ───────────────────────────────────────────
# Split BEFORE shuffle to avoid future data leaking into training
# (time-series data — order matters)
n = len(y)
n_test = int(n * TEST_SIZE)
n_val  = int(n * VAL_SIZE)
n_train = n - n_test - n_val

X_s_train,  X_s_val,  X_s_test  = X_s[:n_train],  X_s[n_train:n_train+n_val],  X_s[n_train+n_val:]
X_n_train,  X_n_val,  X_n_test  = X_n[:n_train],  X_n[n_train:n_train+n_val],  X_n[n_train+n_val:]
y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]

print(f"\n  Split (time-ordered, no shuffle):")
print(f"  Train: {len(y_train):,} windows  | Attacks: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"  Val:   {len(y_val):,}  windows  | Attacks: {y_val.sum():,} ({y_val.sum()/len(y_val)*100:.1f}%)")
print(f"  Test:  {len(y_test):,}  windows  | Attacks: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.1f}%)")

# ── Save ───────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'swat_windows.npz')
np.savez_compressed(out_path,
    X_s_train=X_s_train, X_n_train=X_n_train, y_train=y_train,
    X_s_val=X_s_val,     X_n_val=X_n_val,     y_val=y_val,
    X_s_test=X_s_test,   X_n_test=X_n_test,   y_test=y_test,
    sensor_cols=np.array(sensor_cols)
)
print(f"\n  Saved → {out_path}")

# Save sensor column names for dashboard mapping later
with open(os.path.join(OUT_DIR, 'sensor_cols.txt'), 'w') as f:
    f.write('\n'.join(sensor_cols))

print("\n" + "="*60)
print("Cleaning complete. Paste output before model training.")
print("="*60)
