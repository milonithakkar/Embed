# verify_a12_features.py
# Sanity check the extracted features

import pandas as pd

df = pd.read_csv(r'D:\trustgate_pcaps\network_features_A12.csv')
df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])

print("="*60)
print("A12 Feature Verification")
print("="*60)

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print(f"\nTime range:")
print(f"  Start: {df['timestamp_sgt'].min()}")
print(f"  End  : {df['timestamp_sgt'].max()}")
print(f"  Duration: {df['timestamp_sgt'].max() - df['timestamp_sgt'].min()}")
print(f"  Expected seconds: {len(df):,}")

# Check for time gaps
df = df.sort_values('timestamp_sgt').reset_index(drop=True)
df['gap_sec'] = df['timestamp_sgt'].diff().dt.total_seconds()
big_gaps = df[df['gap_sec'] > 1.5]
print(f"\nTime gaps > 1.5 sec: {len(big_gaps)}")
if len(big_gaps) > 0:
    print("  Largest gaps:")
    print(big_gaps.nlargest(5, 'gap_sec')[['timestamp_sgt', 'gap_sec']])

# Check feature variance
print(f"\nFeature variance check:")
print(f"  (low variance = useless for ML)")
for col in df.columns:
    if col in ['timestamp_sgt', 'gap_sec']:
        continue
    std = df[col].std()
    mean = df[col].mean()
    cv = std / mean if mean > 0 else 0
    flag = " ← LOW VARIANCE" if cv < 0.05 else ""
    print(f"  {col:25s} mean={mean:>10.2f} std={std:>10.2f} cv={cv:.3f}{flag}")

# Check attack window — A12 has attacks 13:00-16:40 SGT
print(f"\nAttack period comparison (13:00-16:40 vs normal):")
attack_mask = (
    (df['timestamp_sgt'].dt.hour >= 13) &
    (df['timestamp_sgt'].dt.hour < 17)
)
normal_mask = ~attack_mask

print(f"  Normal rows: {normal_mask.sum():,}")
print(f"  Attack rows: {attack_mask.sum():,}")
print(f"\n  {'Feature':25s} {'Normal':>12s} {'Attack':>12s} {'Diff%':>8s}")
print(f"  {'-'*60}")
for col in ['packet_count', 'cip_write_count', 'critical_writes',
            'suspicious_writes', 'reset_count', 'unique_tags']:
    if col in df.columns:
        norm_mean = df.loc[normal_mask, col].mean()
        atk_mean  = df.loc[attack_mask, col].mean()
        if norm_mean > 0:
            diff_pct = (atk_mean - norm_mean) / norm_mean * 100
        else:
            diff_pct = 0
        print(f"  {col:25s} {norm_mean:>12.2f} {atk_mean:>12.2f} "
              f"{diff_pct:>+7.1f}%")

# Show top tags that triggered "suspicious"
# (only possible if we kept raw tag data — we didn't, so skip)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())