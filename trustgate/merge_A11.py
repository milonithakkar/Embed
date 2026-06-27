# =================================================================
# merge_A11.py
# Merges A11 sensor CSV(s) + extracted network features
# A11 is 100% NORMAL — no attack labels at all.
# Output is used for:
#   1. Fitting the RobustScaler (normal operation manifold)
#   2. Pre-training BiLSTM encoders via NT-Xent contrastive loss
#   3. Computing Granger causality matrix
#
# Save as: C:\Users\HP\Downloads\trustgate\merge_A11.py
# Run    : python merge_A11.py
# =================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── CONFIG — set these paths ──────────────────────────────────────

# Sensor CSV(s)
# If you only have ONE day, leave SENSOR_CSV_DAY2 = None
# If you have BOTH days, fill in both paths
SENSOR_CSV_DAY1 = None   # e.g. r'D:\Users\HP\Downloads\SWaT.A10_OTDataset_19-Feb-2026_0930_1736.csv'
SENSOR_CSV_DAY2 = r'D:\Users\HP\Downloads\SWaT.A10_OTDataset_20-Feb-2026_0905_1710.csv'

# Network features CSV (from extract_A11_day1.py or day2 extraction)
# If you only extracted Day 1 network, only Day 1 timestamps will match
NETWORK_CSV = r'D:\trustgate_pcaps\network_features_A11_day1.csv'

# Output
OUTPUT_CSV = r'D:\trustgate_pcaps\A11_merged.csv'

# ── SENSOR COLUMN DEFINITIONS ─────────────────────────────────────
# These are the columns we expect in A11 sensor CSV
# (same physical sensors as A12 — same plant)
EXPECTED_SENSOR_COLS = [
    # Flow indicators
    'FIT101','FIT201','FIT301','FIT401','FIT501','FIT502','FIT503','FIT504',
    # Level indicators
    'LIT101','LIT201','LIT301','LIT401','LIT501','LIT601',
    # Pressure indicators
    'PIT101','PIT501','PIT502','PIT503','DPIT301',
    # Chemical analysers
    'AIT201','AIT202','AIT203','AIT401','AIT402',
    'AIT501','AIT502','AIT503','AIT504',
    # Pumps (binary state 0/1)
    'P101','P102','P201','P202','P203','P204','P205','P206',
    'P301','P302','P401','P402','P403','P404','P501','P502',
    'P601','P602','P603',
    # Motor valves (binary state 0/1)
    'MV101','MV201','MV301','MV302','MV303','MV304','MV401',
    'MV501','MV502','MV503','MV504','UV401',
]

# Possible timestamp column names in SWaT historian exports
# (NUS uses different names across datasets — we try all of them)
TIMESTAMP_CANDIDATES = [
    't_stamp', 'Timestamp', 'timestamp', 'TIME', 'Time',
    'timestamp_sgt', 'DateTime', 'datetime', 'date_time',
]


# ── Helpers ───────────────────────────────────────────────────────

def find_timestamp_col(df, filename=''):
    """Find whichever timestamp column exists in this CSV."""
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    # Last resort: check if first column looks like a timestamp
    first_col = df.columns[0]
    sample    = str(df[first_col].iloc[0])
    if any(c in sample for c in ['/', '-', ':']):
        print(f'  [WARN] Guessing timestamp column = "{first_col}"')
        return first_col
    raise ValueError(
        f'Cannot find timestamp column in {filename}.\n'
        f'Columns found: {list(df.columns[:10])}\n'
        f'Expected one of: {TIMESTAMP_CANDIDATES}'
    )


def load_sensor_csv(path, day_label=''):
    """Load one sensor CSV, parse timestamp, forward-fill short gaps."""
    print(f'  Loading sensor CSV {day_label}: {Path(path).name}')
    df = pd.read_csv(path, low_memory=False)
    print(f'    Raw shape: {df.shape}')
    print(f'    Columns  : {list(df.columns[:8])}...')

    ts_col = find_timestamp_col(df, path)
    print(f'    Timestamp column: "{ts_col}"')

    df['timestamp_sgt'] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.drop(columns=[ts_col])

    bad = df['timestamp_sgt'].isna().sum()
    if bad > 0:
        print(f'    [WARN] {bad} rows with unparseable timestamps — dropping')
        df = df.dropna(subset=['timestamp_sgt'])

    df = df.sort_values('timestamp_sgt').reset_index(drop=True)

    # Forward-fill gaps ≤ 5 seconds (network hiccups, not sensor failures)
    df = df.ffill(limit=5)

    print(f'    Time range: {df["timestamp_sgt"].min()} '
          f'→ {df["timestamp_sgt"].max()}')
    print(f'    Rows after clean: {len(df):,}')
    return df


def find_sensor_cols(df):
    """Return intersection of expected sensor columns and what's in the CSV."""
    found   = [c for c in EXPECTED_SENSOR_COLS if c in df.columns]
    missing = [c for c in EXPECTED_SENSOR_COLS if c not in df.columns]
    if missing:
        print(f'    [INFO] {len(missing)} sensor cols not in CSV (normal for subset exports):')
        print(f'           {missing[:10]}{"..." if len(missing)>10 else ""}')
    print(f'    Sensor columns found: {len(found)}')
    return found


def check_day_mismatch(sensor_df, network_df):
    """
    Warn if sensor and network data cover different date ranges.
    This happens if you only extracted Day 1 network but have Day 2 sensor CSV.
    """
    s_dates = set(sensor_df['timestamp_sgt'].dt.date.unique())
    n_dates = set(pd.to_datetime(network_df['timestamp_sgt']).dt.date.unique())
    overlap = s_dates & n_dates

    print(f'\n  Sensor dates  : {sorted(s_dates)}')
    print(f'  Network dates : {sorted(n_dates)}')
    print(f'  Overlap dates : {sorted(overlap)}')

    if not overlap:
        print('\n  ⚠️  WARNING: Sensor and network data cover DIFFERENT DATES.')
        print('     The merge will produce ZERO rows.')
        print('     FIX: Either run extract_A11_day2.py for the Day 2 network,')
        print('          OR provide the Day 1 sensor CSV (SENSOR_CSV_DAY1).')
        return False
    if len(overlap) < len(s_dates):
        print(f'\n  ⚠️  PARTIAL OVERLAP: {len(overlap)}/{len(s_dates)} sensor days '
              f'have matching network data.')
        print('     Rows from unmatched days will be dropped during merge.')
    return True


# ── Main ─────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('TrustGate — Merge A11 Sensor + Network (NORMAL BASELINE)')
    print('=' * 60)

    # ── [1/5] Load sensor CSV(s) ──────────────────────────────────
    print('\n[1/5] Loading sensor CSV(s)...')
    sensor_dfs = []

    if SENSOR_CSV_DAY1 and Path(SENSOR_CSV_DAY1).exists():
        sensor_dfs.append(load_sensor_csv(SENSOR_CSV_DAY1, 'Day1'))
    elif SENSOR_CSV_DAY1:
        print(f'  [WARN] Day1 path not found: {SENSOR_CSV_DAY1}')

    if SENSOR_CSV_DAY2 and Path(SENSOR_CSV_DAY2).exists():
        sensor_dfs.append(load_sensor_csv(SENSOR_CSV_DAY2, 'Day2'))
    elif SENSOR_CSV_DAY2:
        print(f'  [WARN] Day2 path not found: {SENSOR_CSV_DAY2}')

    if not sensor_dfs:
        raise FileNotFoundError(
            'No sensor CSVs found. Set SENSOR_CSV_DAY1 and/or SENSOR_CSV_DAY2.')

    # Combine days (if 2 days)
    if len(sensor_dfs) == 2:
        print(f'\n  Combining Day 1 + Day 2 sensor data...')
        # Research doc: treat 8-hour gap between days as stationary period
        # Concatenate chronologically — do NOT interpolate across the gap
        sensor_df = pd.concat(sensor_dfs, ignore_index=True)
        sensor_df = sensor_df.sort_values('timestamp_sgt').reset_index(drop=True)
        print(f'  Combined rows: {len(sensor_df):,}')
    else:
        sensor_df = sensor_dfs[0]
        print(f'  Single day loaded: {len(sensor_df):,} rows')

    sensor_cols = find_sensor_cols(sensor_df)

    # ── [2/5] Load network CSV ────────────────────────────────────
    print(f'\n[2/5] Loading network features...')
    print(f'  File: {NETWORK_CSV}')

    network_df = pd.read_csv(NETWORK_CSV, low_memory=False)
    network_df['timestamp_sgt'] = pd.to_datetime(
        network_df['timestamp_sgt'], errors='coerce')
    network_df = network_df.dropna(subset=['timestamp_sgt']) \
                            .reset_index(drop=True)

    print(f'  Shape: {network_df.shape}')
    print(f'  Network features: {[c for c in network_df.columns if c != "timestamp_sgt"]}')
    print(f'  Time range: {network_df["timestamp_sgt"].min()} '
          f'→ {network_df["timestamp_sgt"].max()}')

    # ── [3/5] Check date alignment ────────────────────────────────
    print('\n[3/5] Checking date alignment...')
    ok = check_day_mismatch(sensor_df, network_df)
    if not ok:
        print('\n  *** Cannot merge — no date overlap. ***')
        print('  Options:')
        print('  A) Run extract_A11_day2.py on your Day 2 PCAPs')
        print('     (copy extract_A11_day1.py, change GZ_DIR and OUTPUT_CSV)')
        print('  B) Add SENSOR_CSV_DAY1 path if you have the Day 1 sensor file')
        print('\n  Exiting. Fix the date mismatch and re-run.')
        return

    # ── [4/5] Merge on timestamp ──────────────────────────────────
    print('\n[4/5] Merging on timestamp...')

    # Round sensor timestamps to second precision (historian usually 1Hz)
    sensor_df['timestamp_sgt'] = sensor_df['timestamp_sgt'].dt.floor('S')

    # Keep only useful columns from sensor CSV
    keep_cols = ['timestamp_sgt'] + sensor_cols
    # Also keep any extra columns that aren't sensor cols (alarm statuses, etc.)
    extra_cols = [c for c in sensor_df.columns
                  if c not in keep_cols and c != 'timestamp_sgt']
    if extra_cols:
        print(f'  Extra sensor columns (keeping): {extra_cols[:5]}')
        keep_cols += extra_cols

    sensor_df = sensor_df[keep_cols]

    merged = pd.merge(
        sensor_df,
        network_df,
        on='timestamp_sgt',
        how='inner'
    )
    merged = merged.sort_values('timestamp_sgt').reset_index(drop=True)

    print(f'  Sensor rows  : {len(sensor_df):,}')
    print(f'  Network rows : {len(network_df):,}')
    print(f'  Merged rows  : {len(merged):,}  '
          f'({len(merged)/len(sensor_df)*100:.1f}% of sensor data)')

    if len(merged) == 0:
        print('\n  *** MERGE PRODUCED ZERO ROWS — timestamp mismatch ***')
        print('  Check that sensor CSV and network CSV cover the same dates.')
        print('  Sample sensor timestamps:')
        print(f'    {sensor_df["timestamp_sgt"].head(3).tolist()}')
        print('  Sample network timestamps:')
        print(f'    {network_df["timestamp_sgt"].head(3).tolist()}')
        return

    # ── [5/5] Add NORMAL labels and save ─────────────────────────
    print('\n[5/5] Adding normal labels and saving...')

    # A11 is 100% normal — all labels are 0
    merged['label_binary'] = 0
    merged['label_class']  = 0
    merged['attack_name']  = 'NORMAL'

    # Save
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)

    size_mb = Path(OUTPUT_CSV).stat().st_size / (1024 ** 2)

    # Feature breakdown
    net_feature_cols    = [c for c in network_df.columns
                           if c != 'timestamp_sgt']
    non_sensor          = {'timestamp_sgt','label_binary',
                           'label_class','attack_name'}
    non_sensor.update(net_feature_cols)
    final_sensor_cols   = [c for c in merged.columns
                           if c not in non_sensor]

    print(f'\n  ✓ Saved: {OUTPUT_CSV}')
    print(f'  ✓ Size : {size_mb:.2f} MB')
    print(f'  ✓ Shape: {merged.shape}')
    print(f'  ✓ Time : {merged["timestamp_sgt"].min()} '
          f'→ {merged["timestamp_sgt"].max()}')
    print(f'\n  Feature counts:')
    print(f'    Sensor features  : {len(final_sensor_cols)}')
    print(f'    Network features : {len(net_feature_cols)}')
    print(f'    Total model input: {len(final_sensor_cols)+len(net_feature_cols)}')
    print(f'    All labels       : NORMAL (label_binary=0)')

    print('\n' + '=' * 60)
    print('COMPLETE')
    print('Next step: update A11_CSV in 01_build_windows.py')
    print(f'  A11_CSV = r"{OUTPUT_CSV}"')
    print('Then run: python 01_build_windows.py')
    print('=' * 60)


if __name__ == '__main__':
    main()