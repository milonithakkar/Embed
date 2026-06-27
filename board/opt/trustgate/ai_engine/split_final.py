"""
TrustGate — Final Episode-Aware Split v2
Pre-attack buffer priority rule correctly implemented.
Only 1 cross-split overlap exists (SE4 val vs SE5 train) — handled.

Run on LAPTOP in same folder as merged.csv
Expected runtime: ~3-4 minutes
"""

import pandas as pd
import numpy as np
import joblib, os
from sklearn.preprocessing import MinMaxScaler

# ── Config ─────────────────────────────────────────────────────────
WINDOW_SIZE   = 30
STRIDE        = 5
MERGE_GAP_MIN = 5
BUFFER_MIN    = 7
OUT_DIR       = './trustgate_data'
NAN_COLS      = ['MV101','AIT201','MV201','P201','P202','P204','MV303']

# Split assignment — episode 22 (SE20) confirmed TRAIN
VAL_SE  = {4, 8, 12, 16}
TEST_SE = {24, 27, 31}

os.makedirs(OUT_DIR, exist_ok=True)

def get_split(se_num):
    if se_num in TEST_SE: return 'test'
    if se_num in VAL_SE:  return 'val'
    return 'train'

print("="*65)
print("TrustGate — Final Split v2 (pre-attack priority rule)")
print("="*65)

# ── Step 1: Load + clean ───────────────────────────────────────────
print("\n[1/8] Loading merged.csv...")
df = pd.read_csv('./merged.csv', low_memory=False)
df.columns = df.columns.str.strip()
df['Normal/Attack'] = df['Normal/Attack'].astype(str).str.strip()
df.drop(columns=[c for c in NAN_COLS if c in df.columns], inplace=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
df = df.sort_values('Timestamp').reset_index(drop=True)
df['label'] = (df['Normal/Attack'] == 'Attack').astype(int)
exclude = ['Timestamp', 'Normal/Attack', 'label']
sensor_cols = [c for c in df.columns if c not in exclude]
print(f"  {len(df):,} rows | {len(sensor_cols)} sensors | {df['label'].sum():,} attacks")

# ── Step 2: Build super-episodes ──────────────────────────────────
print("\n[2/8] Building super-episodes (merge gap < 5 min)...")
df['ep_raw'] = (df['label'] != df['label'].shift()).cumsum()
attack_ep_raws = sorted(df[df['label']==1]['ep_raw'].unique())
ep_map = {raw: i+1 for i, raw in enumerate(attack_ep_raws)}

ep_bounds = []
for raw_ep, ep_num in ep_map.items():
    rows = df[df['ep_raw'] == raw_ep]
    ep_bounds.append({
        'ep_num': ep_num, 'raw': raw_ep,
        'start': rows['Timestamp'].iloc[0],
        'end':   rows['Timestamp'].iloc[-1],
        'rows':  len(rows)
    })

merge_gap = pd.Timedelta(minutes=MERGE_GAP_MIN)
super_episodes = []
cur = {**ep_bounds[0], 'orig_eps': [ep_bounds[0]['ep_num']], 'orig_raws': [ep_bounds[0]['raw']]}
for i in range(1, len(ep_bounds)):
    if ep_bounds[i]['start'] - cur['end'] < merge_gap:
        cur['end'] = ep_bounds[i]['end']
        cur['rows'] += ep_bounds[i]['rows']
        cur['orig_eps'].append(ep_bounds[i]['ep_num'])
        cur['orig_raws'].append(ep_bounds[i]['raw'])
    else:
        super_episodes.append(cur)
        cur = {**ep_bounds[i], 'orig_eps': [ep_bounds[i]['ep_num']], 'orig_raws': [ep_bounds[i]['raw']]}
super_episodes.append(cur)

for i, se in enumerate(super_episodes):
    se['se_num'] = i + 1
    se['split']  = get_split(i + 1)

print(f"  35 raw episodes → {len(super_episodes)} super-episodes")

# Verify episode 22 in train
ep22 = next(se for se in super_episodes if 22 in se['orig_eps'])
assert ep22['split'] == 'train', f"Episode 22 must be TRAIN, got {ep22['split']}"
print(f"  ✓ Episode 22 in TRAIN (SE{ep22['se_num']}, {ep22['rows']:,} rows)")

for se in super_episodes:
    print(f"    SE{se['se_num']:>2} → {se['split'].upper():>5} | "
          f"rows:{se['rows']:>6} | eps:{se['orig_eps']}")

# ── Step 3: Assign splits using priority zones ─────────────────────
print("\n[3/8] Assigning splits (pre-attack priority > post-attack)...")
buf = pd.Timedelta(minutes=BUFFER_MIN)

# Initialize all rows as unassigned
df['split'] = 'unassigned'
splits = df['split'].values.copy()
ts     = df['Timestamp'].values

# Priority order (highest → lowest):
#   1. Attack rows themselves
#   2. Pre-attack buffers  ← your insight: these beat post-attack buffers
#   3. Post-attack buffers
# Process in reverse priority so higher priority overwrites lower

# Pass 3 — post-attack buffers (lowest priority, processed first)
print("  Pass 1/3: post-attack buffers...")
for se in super_episodes:
    post_start = np.datetime64(se['end'])
    post_end   = np.datetime64(se['end'] + buf)
    mask = (ts > post_start) & (ts <= post_end) & (splits == 'unassigned')
    splits[mask] = se['split']

# Pass 2 — pre-attack buffers (overwrites post-attack in overlap zones)
print("  Pass 2/3: pre-attack buffers (overwrites post-attack overlaps)...")
for se in super_episodes:
    pre_start = np.datetime64(se['start'] - buf)
    pre_end   = np.datetime64(se['start'])
    # Overwrite ANYTHING in this zone — including post-attack of previous SE
    # This is the pre-attack priority rule
    mask = (ts >= pre_start) & (ts < pre_end)
    splits[mask] = se['split']

# Pass 1 — attack rows (highest priority, always wins)
print("  Pass 3/3: attack rows (highest priority)...")
for se in super_episodes:
    for raw_ep in se['orig_raws']:
        mask = df['ep_raw'].values == raw_ep
        splits[mask] = se['split']

df['split'] = splits

# Remaining unassigned → 70/15/15 time-ordered
print("  Assigning remaining normal rows (70/15/15 time-ordered)...")
unassigned = np.where(df['split'].values == 'unassigned')[0]
n  = len(unassigned)
n_tr = int(n * 0.70)
n_v  = int(n * 0.15)
df['split'].iloc[unassigned[:n_tr]]        = 'train'
df['split'].iloc[unassigned[n_tr:n_tr+n_v]] = 'val'
df['split'].iloc[unassigned[n_tr+n_v:]]   = 'test'

assert (df['split'] == 'unassigned').sum() == 0, "Unassigned rows remain!"
print(f"  ✓ All {len(df):,} rows assigned")

# ── Step 4: Verify ────────────────────────────────────────────────
print("\n[4/8] Split verification:")
for sp in ['train','val','test']:
    s   = df[df['split'] == sp]
    atk = s['label'].sum()
    print(f"  {sp:>5}: {len(s):>9,} rows | "
          f"Attacks: {atk:>6,} ({atk/len(s)*100:.1f}%) | "
          f"Normal: {(s['label']==0).sum():>9,}")

# Verify the key overlap — SE4(val) vs SE5(train) pre-attack buffer
# Overlap zone: 11:53:55 → 12:01:08 on 2015-12-28
overlap_start = pd.Timestamp("2015-12-28 11:53:55")
overlap_end   = pd.Timestamp("2015-12-28 12:01:08")
overlap_rows  = df[(df['Timestamp'] >= overlap_start) & (df['Timestamp'] < overlap_end)]
overlap_splits = overlap_rows['split'].value_counts()
print(f"\n  Key overlap zone (SE4 val vs SE5 train pre-attack buffer):")
print(f"  {dict(overlap_splits)}")
se5_pre = (overlap_rows['split'] == 'train').sum()
se4_post = (overlap_rows['split'] == 'val').sum()
if se5_pre > se4_post:
    print(f"  ✓ Pre-attack buffer of SE5 (TRAIN) correctly won the overlap")
else:
    print(f"  ✗ Post-attack of SE4 (VAL) incorrectly won — check priority logic")

# ── Step 5: Normalize ─────────────────────────────────────────────
print("\n[5/8] Normalizing...")
sensor_data  = df[sensor_cols].values.astype(np.float32)
train_normal = (df['split'] == 'train') & (df['label'] == 0)
scaler = MinMaxScaler()
scaler.fit(sensor_data[train_normal.values])
sensor_scaled = scaler.transform(sensor_data)
joblib.dump(scaler,      f'{OUT_DIR}/scaler_final.pkl')
joblib.dump(sensor_cols, f'{OUT_DIR}/sensor_cols_final.pkl')
print(f"  Scaler fit on {train_normal.sum():,} train-normal rows")

# ── Step 6: Network proxy features ────────────────────────────────
print("\n[6/8] Computing network proxy features...")
roc = np.zeros_like(sensor_scaled, dtype=np.float32)
roc[5:] = sensor_scaled[5:] - sensor_scaled[:-5]

# Vectorized rolling mean via cumsum trick — O(n) not O(n*window)
print("  Rolling deviation (60s window, vectorized)...")
cumsum = np.cumsum(sensor_scaled, axis=0)
rolling_mean = np.zeros_like(sensor_scaled, dtype=np.float32)
rolling_mean[60:] = (cumsum[60:] - cumsum[:-60]) / 60.0
dev = np.zeros_like(sensor_scaled, dtype=np.float32)
dev[60:] = sensor_scaled[60:] - rolling_mean[60:]

network_data = np.concatenate([sensor_scaled, roc, dev], axis=1).astype(np.float32)
print(f"  Shape: {network_data.shape}  (44 sensor + 44 roc + 44 dev = 132 features)")

labels_arr = df['label'].values
splits_arr = df['split'].values

# ── Step 7: Sliding windows ────────────────────────────────────────
print("\n[7/8] Creating sliding windows...")

def make_windows(split_name, sensor, network, labels, splits, window, stride):
    idxs  = np.where(splits == split_name)[0]
    X_s, X_n, y = [], [], []
    i = 0
    total = len(idxs)
    while i <= total - window:
        chunk = idxs[i:i+window]
        if chunk[-1] - chunk[0] == window - 1:
            # Fully consecutive — valid window
            X_s.append(sensor[chunk[0]:chunk[0]+window])
            X_n.append(network[chunk[0]:chunk[0]+window])
            y.append(1 if labels[chunk[0]:chunk[0]+window].sum() > 0 else 0)
            i += stride
        else:
            # Gap — jump past it efficiently
            gaps = np.where(np.diff(chunk) > 1)[0]
            i += gaps[0] + 1 if len(gaps) > 0 else 1
    return (np.array(X_s, dtype=np.float32),
            np.array(X_n, dtype=np.float32),
            np.array(y,   dtype=np.int32))

results = {}
for sp in ['train','val','test']:
    print(f"  Building {sp}...", flush=True)
    X_s, X_n, y = make_windows(
        sp, sensor_scaled, network_data, labels_arr, splits_arr, WINDOW_SIZE, STRIDE
    )
    results[sp] = (X_s, X_n, y)
    atk = y.sum()
    print(f"  {sp:>5}: {len(y):>7,} windows | "
          f"Attacks: {atk:>5,} ({atk/max(len(y),1)*100:.1f}%) | "
          f"X_s: {X_s.shape} | X_n: {X_n.shape}")

# ── Step 8: Save ──────────────────────────────────────────────────
print("\n[8/8] Saving...")
out = f'{OUT_DIR}/swat_final.npz'
np.savez_compressed(out,
    X_s_train=results['train'][0], X_n_train=results['train'][1], y_train=results['train'][2],
    X_s_val  =results['val'][0],   X_n_val  =results['val'][1],   y_val  =results['val'][2],
    X_s_test =results['test'][0],  X_n_test =results['test'][1],  y_test =results['test'][2],
    sensor_cols=np.array(sensor_cols)
)
print(f"  Saved → {out}")

print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)
total_w = sum(len(results[s][2]) for s in ['train','val','test'])
for sp in ['train','val','test']:
    y   = results[sp][2]
    X_s = results[sp][0]
    print(f"  {sp:>5}: {len(y):>7,} windows ({len(y)/total_w*100:.0f}%) | "
          f"Attack%: {y.sum()/max(len(y),1)*100:.1f}%")
print("\nThresholds:")
print("  train: >150K windows | attack% 3-8%")
print("  val:   >15K  windows | attack% 2-10%")
print("  test:  >15K  windows | attack% 2-10%")
print("="*65)
