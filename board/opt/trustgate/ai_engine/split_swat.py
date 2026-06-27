"""
Day 12 Step 3 — Episode-Aware Split
Preserves time-ordering within each split.
BiLSTM sees normal→attack→normal transitions correctly.
Run on your LAPTOP in the same folder as merged.csv + trustgate_data/
"""

import pandas as pd
import numpy as np
import joblib
import os

# ── Config ─────────────────────────────────────────────────────────
WINDOW_SIZE = 30
STRIDE      = 5
OUT_DIR     = './trustgate_data'

# Episodes assigned to each split (1-indexed, from our exploration output)
# Logic: episode 22 (35,900 rows = 65% of attacks) MUST be in train
# Test = last 5 episodes (completely held-out, most recent attacks)
# Val  = every ~4th episode from middle
VAL_EPISODES  = {4, 8, 12, 16, 20, 24, 28}       # 7 episodes
TEST_EPISODES = {31, 32, 33, 34, 35}              # 5 episodes, last in timeline
# Train = all remaining (includes episode 22 with 35,900 rows)

print("=" * 60)
print("TrustGate — Episode-Aware Train/Val/Test Split")
print("=" * 60)

# ── Load and clean (same as clean_swat.py) ────────────────────────
print("\n[1/6] Loading and cleaning merged.csv...")
df = pd.read_csv('./merged.csv', low_memory=False)
df.columns = df.columns.str.strip()
df['Normal/Attack'] = df['Normal/Attack'].astype(str).str.strip()

NAN_COLS = ['MV101', 'AIT201', 'MV201', 'P201', 'P202', 'P204', 'MV303']
df.drop(columns=[c for c in NAN_COLS if c in df.columns], inplace=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
df = df.sort_values('Timestamp').reset_index(drop=True)
df['label'] = (df['Normal/Attack'] == 'Attack').astype(int)

exclude = ['Timestamp', 'Normal/Attack', 'label']
sensor_cols = [c for c in df.columns if c not in exclude]
print(f"  Rows: {len(df):,} | Sensors: {len(sensor_cols)} | Attacks: {df['label'].sum():,}")

# ── Assign episode numbers ─────────────────────────────────────────
print("\n[2/6] Assigning episode numbers to attack rows...")
df['episode_raw'] = (df['label'] != df['label'].shift()).cumsum()

# Give each attack episode a sequential number (1-35)
attack_episodes = df[df['label']==1]['episode_raw'].unique()
episode_map = {raw: i+1 for i, raw in enumerate(sorted(attack_episodes))}
df['attack_episode'] = df['episode_raw'].map(episode_map)  # NaN for normal rows

print(f"  Attack episodes found: {len(episode_map)}")

# ── Assign split to each ROW ──────────────────────────────────────
print("\n[3/6] Assigning split labels to rows...")
# Strategy for normal rows:
# Each normal row belongs to the same split as the NEAREST attack episode
# Normal rows before episode 1: train
# Normal rows between episodes i and i+1: same split as episode i
# This ensures each split's normal data surrounds its attacks

# First mark attack rows
df['split'] = 'train'  # default

# Mark attack rows by episode
for raw_ep, ep_num in episode_map.items():
    mask = df['episode_raw'] == raw_ep
    if ep_num in TEST_EPISODES:
        df.loc[mask, 'split'] = 'test'
    elif ep_num in VAL_EPISODES:
        df.loc[mask, 'split'] = 'val'
    else:
        df.loc[mask, 'split'] = 'train'

# For normal rows: assign based on which attack episode boundary they fall between
# We forward-fill episode assignments through normal gaps
# Normal rows get the split of the NEXT attack episode (pre-attack context matters for BiLSTM)
attack_only = df[df['label']==1][['Timestamp','split']].copy()
attack_only = attack_only.set_index('Timestamp')

# Build boundary list: (timestamp, split_of_next_episode)
episode_boundaries = []
for raw_ep, ep_num in sorted(episode_map.items(), key=lambda x: x[1]):
    ep_rows = df[df['episode_raw'] == raw_ep]
    start_ts = ep_rows['Timestamp'].iloc[0]
    if ep_num in TEST_EPISODES:
        sp = 'test'
    elif ep_num in VAL_EPISODES:
        sp = 'val'
    else:
        sp = 'train'
    episode_boundaries.append((start_ts, sp))

# Assign normal rows: find which episode boundary each normal row falls before
normal_mask = df['label'] == 0
normal_timestamps = df.loc[normal_mask, 'Timestamp']

def get_split_for_normal(ts):
    # Find first episode that starts AFTER this timestamp
    for boundary_ts, sp in episode_boundaries:
        if ts < boundary_ts:
            return sp
    return 'test'  # after all episodes → assign to test

print("  Assigning splits to normal rows (this takes ~60s for 1.4M rows)...")
df.loc[normal_mask, 'split'] = normal_mask[normal_mask].index.map(
    lambda i: get_split_for_normal(df.loc[i, 'Timestamp'])
)

# ── Verify split ──────────────────────────────────────────────────
print("\n[4/6] Split verification:")
for sp in ['train', 'val', 'test']:
    subset = df[df['split'] == sp]
    attacks = subset['label'].sum()
    print(f"  {sp:>5}: {len(subset):>8,} rows | Attacks: {attacks:>6,} ({attacks/len(subset)*100:.1f}%)")

# ── Normalize ─────────────────────────────────────────────────────
print("\n[5/6] Normalizing (fit scaler on TRAIN normal only)...")
from sklearn.preprocessing import MinMaxScaler

sensor_data = df[sensor_cols].values.astype(np.float32)
train_normal_mask = (df['split'] == 'train') & (df['label'] == 0)
scaler = MinMaxScaler()
scaler.fit(sensor_data[train_normal_mask.values])
sensor_scaled = scaler.transform(sensor_data)
joblib.dump(scaler, os.path.join(OUT_DIR, 'sensor_scaler_v2.pkl'))
print(f"  Scaler fitted on {train_normal_mask.sum():,} train-normal rows")

# ── Network proxy features ─────────────────────────────────────────
print("  Computing rate-of-change and deviation features...")
roc = np.zeros_like(sensor_scaled)
dev = np.zeros_like(sensor_scaled)
for i in range(5, len(sensor_scaled)):
    roc[i] = sensor_scaled[i] - sensor_scaled[i-5]
for i in range(60, len(sensor_scaled)):
    dev[i] = sensor_scaled[i] - sensor_scaled[i-60:i].mean(axis=0)

network_data = np.concatenate([sensor_scaled, roc, dev], axis=1).astype(np.float32)
labels_arr = df['label'].values
splits_arr = df['split'].values

# ── Sliding windows per split ──────────────────────────────────────
print("\n[6/6] Creating sliding windows per split...")

def make_windows(indices, sensor, network, labels, window, stride):
    """Make windows only from consecutive rows within same split"""
    X_s, X_n, y = [], [], []
    idx_list = sorted(indices)
    i = 0
    while i < len(idx_list) - window:
        # Check if this window is fully consecutive (no gaps)
        window_indices = idx_list[i:i+window]
        if window_indices[-1] - window_indices[0] == window - 1:
            X_s.append(sensor[window_indices[0]:window_indices[0]+window])
            X_n.append(network[window_indices[0]:window_indices[0]+window])
            y.append(1 if labels[window_indices[0]:window_indices[0]+window].sum() > 0 else 0)
            i += stride
        else:
            # Gap found — skip to next valid start
            i += 1
    return np.array(X_s, dtype=np.float32), np.array(X_n, dtype=np.float32), np.array(y, dtype=np.int32)

results = {}
for sp in ['train', 'val', 'test']:
    sp_indices = np.where(splits_arr == sp)[0]
    X_s, X_n, y = make_windows(sp_indices, sensor_scaled, network_data, labels_arr, WINDOW_SIZE, STRIDE)
    results[sp] = (X_s, X_n, y)
    attacks = y.sum()
    print(f"  {sp:>5}: {len(y):>7,} windows | Attacks: {attacks:>5,} ({attacks/len(y)*100:.1f}%) | X_s: {X_s.shape}")

# ── Save ───────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'swat_windows_v2.npz')
np.savez_compressed(out_path,
    X_s_train=results['train'][0], X_n_train=results['train'][1], y_train=results['train'][2],
    X_s_val=results['val'][0],     X_n_val=results['val'][1],     y_val=results['val'][2],
    X_s_test=results['test'][0],   X_n_test=results['test'][1],   y_test=results['test'][2],
    sensor_cols=np.array(sensor_cols)
)
print(f"\n  Saved → {out_path}")
print("\n" + "="*60)
print("Split complete. Paste output before model training.")
print("="*60)
