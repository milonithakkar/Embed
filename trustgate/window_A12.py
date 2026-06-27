# window_A12.py
# Converts A12_merged.csv into training-ready windows
# Save as: C:\Users\HP\Downloads\trustgate\window_A12.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
INPUT_CSV   = r'D:\trustgate_pcaps\A12_merged.csv'
OUTPUT_NPZ  = r'D:\trustgate_pcaps\A12_windowed.npz'

WINDOW_SIZE = 30        # 30 seconds per window
STRIDE      = 1         # sliding window step

# Train/val/test split — attack-aware (each split sees diverse attacks)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15

# Component labels (must match merge_A12.py)
COMPONENTS = [
    'MV101', 'MV201', 'MV301', 'MV302', 'MV303', 'MV304',
    'MV501', 'MV502', 'MV503', 'MV504',
    'P101', 'P102', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'LIT101', 'LIT601',
    'DPIT301', 'AIT402',
]


def main():
    print("=" * 60)
    print("TrustGate — Window A12 for Training")
    print("=" * 60)

    # ── [1/6] Load merged CSV ───────────────────────────────────
    print(f"\n[1/6] Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])
    df = df.sort_values('timestamp_sgt').reset_index(drop=True)

    print(f"  Rows: {len(df):,}")
    print(f"  Time: {df['timestamp_sgt'].min()} → {df['timestamp_sgt'].max()}")

    # ── [2/6] Identify feature columns ──────────────────────────
    print(f"\n[2/6] Identifying feature columns...")

    # Network feature columns (from smart extraction)
    network_cols = [
        'packet_count', 'total_bytes',
        'valve_write_count', 'pump_write_count',
        'valve_read_count', 'pump_read_count',
        'level_access_count', 'dosing_access_count',
        'state_poll_count',
        'valve_write_deviation', 'pump_write_deviation',
        'novel_tag_count', 'tag_entropy',
        'burst_component_max', 'rare_access_score',
        'unique_tags_this_sec', 'unique_src_ips',
        'reset_count', 'permissive_count',
    ]

    # Component label columns
    component_cols = [f'comp_{c}' for c in COMPONENTS]

    # Excluded columns
    excluded = set(
        ['timestamp_sgt', 'label_binary', 'label_class', 'attack_name'] +
        network_cols + component_cols
    )

    # Sensor columns = everything else
    sensor_cols = [c for c in df.columns if c not in excluded]

    print(f"  Sensor features  : {len(sensor_cols)}")
    print(f"  Network features : {len(network_cols)}")
    print(f"  Component labels : {len(component_cols)}")

    # ── [3/6] Clean non-numeric columns ─────────────────────────
    print(f"\n[3/6] Coercing sensor columns to numeric...")

    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill remaining NaN with column median (sensor failures)
    nan_counts = df[sensor_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"  NaN cells in sensors: {int(nan_counts.sum())}")
        for col in sensor_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # ── [4/6] Attack-aware split (chronological) ────────────────
    print(f"\n[4/6] Splitting chronologically with attack distribution check...")

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

    for name, sub in [('train', train_df), ('val', val_df), ('test', test_df)]:
        atk = int(sub['label_binary'].sum())
        print(f"  {name:5s}: {len(sub):>6,} rows | "
              f"{atk:>5,} attack ({atk/len(sub)*100:.1f}%)")

    # ── [5/6] Scaling — fit on train, transform all ─────────────
    print(f"\n[5/6] Scaling (RobustScaler fit on train only)...")

    sensor_scaler  = RobustScaler()
    network_scaler = RobustScaler()

    train_df[sensor_cols]  = sensor_scaler.fit_transform(train_df[sensor_cols])
    val_df[sensor_cols]    = sensor_scaler.transform(val_df[sensor_cols])
    test_df[sensor_cols]   = sensor_scaler.transform(test_df[sensor_cols])

    train_df[network_cols] = network_scaler.fit_transform(train_df[network_cols])
    val_df[network_cols]   = network_scaler.transform(val_df[network_cols])
    test_df[network_cols]  = network_scaler.transform(test_df[network_cols])

    # ── [6/6] Build sliding windows ─────────────────────────────
    print(f"\n[6/6] Building {WINDOW_SIZE}-second windows...")

    def make_windows(sub_df):
        """Build sliding windows from a chronological dataframe."""
        sensor_arr = sub_df[sensor_cols].values.astype(np.float32)
        net_arr    = sub_df[network_cols].values.astype(np.float32)
        bin_arr    = sub_df['label_binary'].values.astype(np.int64)
        cls_arr    = sub_df['label_class'].values.astype(np.int64)
        comp_arr   = sub_df[component_cols].values.astype(np.float32)

        n_rows = len(sub_df)
        if n_rows < WINDOW_SIZE:
            return (
                np.zeros((0, WINDOW_SIZE, len(sensor_cols)), np.float32),
                np.zeros((0, WINDOW_SIZE, len(network_cols)), np.float32),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.int64),
                np.zeros((0, len(COMPONENTS)), np.float32),
            )

        n_windows = (n_rows - WINDOW_SIZE) // STRIDE + 1

        X_s = np.zeros((n_windows, WINDOW_SIZE, len(sensor_cols)),  np.float32)
        X_n = np.zeros((n_windows, WINDOW_SIZE, len(network_cols)), np.float32)
        y_b = np.zeros((n_windows,), np.int64)
        y_c = np.zeros((n_windows,), np.int64)
        y_p = np.zeros((n_windows, len(COMPONENTS)), np.float32)

        for i in range(n_windows):
            start = i * STRIDE
            end   = start + WINDOW_SIZE

            X_s[i] = sensor_arr[start:end]
            X_n[i] = net_arr[start:end]

            # Label = max in window (if ANY attack in window → window is attack)
            y_b[i] = int(bin_arr[start:end].max())

            # Class = most frequent non-zero, else NORMAL
            window_classes = cls_arr[start:end]
            attack_classes = window_classes[window_classes > 0]
            if len(attack_classes) > 0:
                values, counts = np.unique(attack_classes, return_counts=True)
                y_c[i] = int(values[np.argmax(counts)])
            else:
                y_c[i] = 0

            # Components = union (any component touched in window)
            y_p[i] = comp_arr[start:end].max(axis=0)

        return X_s, X_n, y_b, y_c, y_p

    print(f"  Building train windows...")
    X_s_tr, X_n_tr, y_b_tr, y_c_tr, y_p_tr = make_windows(train_df)

    print(f"  Building val windows...")
    X_s_va, X_n_va, y_b_va, y_c_va, y_p_va = make_windows(val_df)

    print(f"  Building test windows...")
    X_s_te, X_n_te, y_b_te, y_c_te, y_p_te = make_windows(test_df)

    # ── Report shapes ───────────────────────────────────────────
    print(f"\n  Window counts and attack ratios:")
    for name, X_s, y_b in [
        ('train', X_s_tr, y_b_tr),
        ('val',   X_s_va, y_b_va),
        ('test',  X_s_te, y_b_te),
    ]:
        atk = int(y_b.sum())
        print(f"    {name:5s}: {len(y_b):>6,} windows | "
              f"{atk:>5,} attack ({atk/len(y_b)*100:.1f}%) | "
              f"shape {X_s.shape}")

    # ── Save NPZ ────────────────────────────────────────────────
    print(f"\nSaving {OUTPUT_NPZ}...")
    np.savez(
        OUTPUT_NPZ,
        X_s_train=X_s_tr, X_n_train=X_n_tr,
        y_b_train=y_b_tr, y_c_train=y_c_tr, y_p_train=y_p_tr,
        X_s_val=X_s_va,   X_n_val=X_n_va,
        y_b_val=y_b_va,   y_c_val=y_c_va,   y_p_val=y_p_va,
        X_s_test=X_s_te,  X_n_test=X_n_te,
        y_b_test=y_b_te,  y_c_test=y_c_te,  y_p_test=y_p_te,
        sensor_cols=np.array(sensor_cols),
        network_cols=np.array(network_cols),
        component_names=np.array(COMPONENTS),
    )

    size_mb = Path(OUTPUT_NPZ).stat().st_size / (1024**2)
    print(f"  Saved: {OUTPUT_NPZ} ({size_mb:.1f} MB)")

    print(f"\n  Class distribution in train set:")
    for cls_id in range(6):
        count = int((y_c_tr == cls_id).sum())
        print(f"    class {cls_id}: {count:>6,}")

    print("\n" + "=" * 60)
    print("Windowing complete. Ready to train.")
    print("=" * 60)


if __name__ == '__main__':
    main()