# explore_a11.py
# Save as: C:\Users\HP\Downloads\trustgate\explore_a11.py
# Run: python explore_a11.py

import pandas as pd
import numpy as np
import os

A11_PATH = r'D:\Users\HP\Downloads\SWaT.A10_OTDataset_19-Feb-2026_0930_1735.csv'

print("=" * 60)
print("A11 Dataset Exploration")
print("=" * 60)

# Check file exists
if not os.path.exists(A11_PATH):
    print(f"NOT FOUND: {A11_PATH}")
    print("Searching for CSV files in Downloads...")
    downloads = r'D:\Users\HP\Downloads'
    for f in os.listdir(downloads):
        if f.endswith('.csv'):
            print(f"  Found: {f}")
    exit(1)

size_mb = os.path.getsize(A11_PATH) / 1e6
print(f"\nFile: {A11_PATH}")
print(f"Size: {size_mb:.1f} MB")

# Load
print(f"\nLoading...")
df = pd.read_csv(A11_PATH, nrows=5)
print(f"Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i:3d}: {col}")

# Full load
print(f"\nLoading full file...")
df = pd.read_csv(A11_PATH)
print(f"Shape: {df.shape}")
print(f"Rows: {len(df):,}")

# Check for attack column
attack_cols = [c for c in df.columns
               if 'attack' in c.lower() or 'label' in c.lower()
               or 'normal' in c.lower()]
print(f"\nAttack/label columns: {attack_cols}")

# Check timestamp
time_cols = [c for c in df.columns
             if 'time' in c.lower() or 'date' in c.lower()
             or 'timestamp' in c.lower()]
print(f"Time columns: {time_cols}")

# Value ranges
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

print(f"\nData types:")
print(df.dtypes.value_counts())

# Check for NaN
print(f"\nNaN counts (top 10):")
nan_counts = df.isnull().sum()
print(nan_counts[nan_counts > 0].head(10))

# Duration
if time_cols:
    try:
        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
        duration = df[time_cols[0]].max() - df[time_cols[0]].min()
        print(f"\nDuration: {duration}")
        print(f"Start: {df[time_cols[0]].min()}")
        print(f"End:   {df[time_cols[0]].max()}")
    except Exception as e:
        print(f"Could not parse time: {e}")

print(f"\n{'='*60}")
print("Exploration complete.")
print("="*60)