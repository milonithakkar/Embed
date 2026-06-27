"""
Step 1 — Explore merged.csv
Run on your LAPTOP in the same folder as your CSV files
"""

import pandas as pd
import numpy as np
import os

def find_file(name):
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower() == name.lower():
                return os.path.join(root, f)
    return None

print("=" * 60)
print("TrustGate — merged.csv Explorer")
print("=" * 60)

path = find_file('merged.csv')
if not path:
    print("[!] merged.csv not found")
    exit(1)

print(f"Found: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

# Read just first 5 rows to check header issues first
df_peek = pd.read_csv(path, nrows=5, low_memory=False)
print(f"\n--- RAW COLUMN NAMES (first 10) ---")
for i, c in enumerate(df_peek.columns[:10]):
    print(f"  [{i}] repr: {repr(c)}")  # repr shows hidden spaces

print(f"\n--- FIRST 3 ROWS ---")
print(df_peek.head(3).to_string())

# Now read full file
print(f"\nReading full file...")
df = pd.read_csv(path, low_memory=False)

# Strip whitespace from ALL column names immediately
df.columns = df.columns.str.strip()
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Label distribution
label_col = None
for c in ['Normal/Attack', 'label', 'Label']:
    if c in df.columns:
        label_col = c
        break

if label_col:
    # Strip whitespace from label values too
    df[label_col] = df[label_col].astype(str).str.strip()
    counts = df[label_col].value_counts()
    print(f"\nLabel distribution ('{label_col}'):")
    for val, cnt in counts.items():
        print(f"  {str(val):20s} → {cnt:8,} ({cnt/len(df)*100:.1f}%)")
    print(f"\nIMBALANCE RATIO: {counts.max()/counts.min():.0f}:1")
else:
    print("[!] No label column found")

# NaN analysis
print(f"\nNaN analysis:")
nan_counts = df.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) == 0:
    print("  No NaNs ✓")
else:
    print(f"  {len(nan_cols)} columns with NaNs:")
    for col, cnt in nan_cols.items():
        print(f"    {col:15s} → {cnt:,} NaN rows ({cnt/len(df)*100:.1f}%)")

# Timestamp
ts_col = None
for c in ['Timestamp', 'timestamp']:
    if c in df.columns:
        ts_col = c
        break
if ts_col:
    print(f"\nTimestamp range:")
    print(f"  First: {df[ts_col].iloc[0]}")
    print(f"  Last:  {df[ts_col].iloc[-1]}")
    # Check for gaps
    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
        diffs = df[ts_col].diff().dropna()
        most_common = diffs.mode()[0]
        gaps = diffs[diffs > most_common * 5]
        print(f"  Sample interval: {most_common}")
        print(f"  Large gaps (>5x normal): {len(gaps)}")
        if len(gaps) > 0:
            print(f"  ⚠ Gaps found — these break sequence windows")
            print(f"  Gap locations (row index):")
            for idx in gaps.index[:5]:
                print(f"    Row {idx}: gap of {diffs[idx]}")
    except Exception as e:
        print(f"  Could not parse timestamps: {e}")

print("\n" + "="*60)
print("Paste this output before we proceed to cleaning.")
print("="*60)
