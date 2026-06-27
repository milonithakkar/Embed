"""
Day 12 — SWaT Dataset Exploration
Run this on your LAPTOP (not the board)
Place this file next to your normal.csv and attack.csv

What this tells us:
- Column names and types
- Class balance (critical — SWaT is heavily imbalanced)
- Attack vs normal sample counts
- Any NaN / bad values
- Value ranges per sensor (needed for normalization)
- Timestamp format (needed for sequence windowing)
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Auto-detect file paths ─────────────────────────────────────────
def find_file(name):
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower() == name.lower():
                return os.path.join(root, f)
    return None

print("=" * 60)
print("TrustGate — SWaT Dataset Explorer")
print("=" * 60)

# ── Load files ─────────────────────────────────────────────────────
for fname in ['attack.csv', 'normal.csv']:
    path = find_file(fname)
    if not path:
        print(f"[!] {fname} not found — place it in the same folder as this script")
        continue

    print(f"\n{'─'*60}")
    print(f"FILE: {fname}  ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"{'─'*60}")

    # Try reading — SWaT CSVs sometimes have header issues
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [ERROR] Could not read: {e}")
        continue

    print(f"Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Label column detection ─────────────────────────────────────
    label_col = None
    for candidate in ['Normal/Attack', 'label', 'Label', 'Attack', 'attack']:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col:
        counts = df[label_col].value_counts()
        print(f"\nLabel column  : '{label_col}'")
        print(f"Class counts  :")
        for val, cnt in counts.items():
            pct = cnt / len(df) * 100
            print(f"  {str(val):20s} → {cnt:8,} rows  ({pct:.1f}%)")
        imbalance = counts.max() / counts.min()
        print(f"Imbalance ratio: {imbalance:.1f}:1  ", end="")
        if imbalance > 10:
            print("⚠ HIGH — will need class weighting or SMOTE")
        else:
            print("✓ manageable")
    else:
        print("  [!] No label column found — check column names below")

    # ── Timestamp ──────────────────────────────────────────────────
    ts_col = None
    for candidate in ['Timestamp', 'timestamp', 'Time', 'time']:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col:
        print(f"\nTimestamp col : '{ts_col}'")
        print(f"  First       : {df[ts_col].iloc[0]}")
        print(f"  Last        : {df[ts_col].iloc[-1]}")

    # ── Sensor columns ─────────────────────────────────────────────
    exclude = {label_col, ts_col} if label_col and ts_col else set()
    sensor_cols = [c for c in df.columns if c not in exclude]
    print(f"\nSensor columns: {len(sensor_cols)}")

    # ── NaN check ─────────────────────────────────────────────────
    nan_counts = df[sensor_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) == 0:
        print("NaN values    : None ✓")
    else:
        print(f"NaN values    : {len(nan_cols)} columns have NaNs")
        print(nan_cols.to_string())

    # ── Data types ────────────────────────────────────────────────
    numeric_cols = df[sensor_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric  = [c for c in sensor_cols if c not in numeric_cols]
    print(f"Numeric cols  : {len(numeric_cols)}")
    if non_numeric:
        print(f"Non-numeric   : {non_numeric}  ← these need encoding")

    # ── Value ranges for key sensors ─────────────────────────────
    print(f"\nKey sensor ranges (first 10 numeric):")
    for col in numeric_cols[:10]:
        mn = df[col].min()
        mx = df[col].max()
        mean = df[col].mean()
        print(f"  {col:15s} min={mn:10.2f}  max={mx:10.2f}  mean={mean:10.2f}")

    # ── Columns list ──────────────────────────────────────────────
    print(f"\nAll columns:")
    for i, col in enumerate(df.columns):
        print(f"  [{i:3d}] {col}")

print("\n" + "="*60)
print("Exploration complete.")
print("Share this output before we design the feature extractor.")
print("="*60)
