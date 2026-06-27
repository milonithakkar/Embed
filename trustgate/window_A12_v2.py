# window_A12_v2.py
# Fixes: NaN handling + attack-stratified split
# Save as: C:\Users\HP\Downloads\trustgate\window_A12_v2.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
INPUT_CSV   = r'D:\trustgate_pcaps\A12_merged.csv'
OUTPUT_NPZ  = r'D:\trustgate_pcaps\A12_windowed.npz'

WINDOW_SIZE = 30
STRIDE      = 1

# Per-attack split: split EACH attack event into train/val/test
# This ensures every class appears in every split
ATTACK_TRAIN_RATIO = 0.70
ATTACK_VAL_RATIO   = 0.15

COMPONENTS = [
    'MV101', 'MV201', 'MV301', 'MV302', 'MV303', 'MV304',
    'MV501', 'MV502', 'MV503', 'MV504',
    'P101', 'P102', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'LIT101', 'LIT601',
    'DPIT301', 'AIT402',
]


def main():
    print("=" * 60)
    print("TrustGate — Window A12 v2 (NaN-fixed, attack-stratified)")
    print("=" * 60)

    # ── [1/7] Load ───────────────────────────────────────────────
    print(f"\n[1/7] Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])
    df = df.sort_values('timestamp_sgt').reset_index(drop=True)
    print(f"  Rows: {len(df):,}")

    # ── [2/7] Identify columns ──────────────────────────────────
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
    component_cols = [f'comp_{c}' for c in COMPONENTS]
    excluded = set(
        ['timestamp_sgt', 'label_binary', 'label_class', 'attack_name']
        + network_cols + component_cols
    )
    sensor_cols = [c for c in df.columns if c not in excluded]
    print(f"\n[2/7] Initial feature counts:")
    print(f"  Sensor   : {len(sensor_cols)}")
    print(f"  Network  : {len(network_cols)}")

    # ── [3/7] Drop useless sensor columns ───────────────────────
    print(f"\n[3/7] Coercing to numeric + dropping all-NaN columns...")
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    nan_pct = df[sensor_cols].isna().sum() / len(df)
    bad_cols = nan_pct[nan_pct > 0.5].index.tolist()

    if bad_cols:
        print(f"  Dropping {len(bad_cols)} useless sensor columns "
              f"(>50% NaN):")
        for c in bad_cols:
            print(f"    {c}  ({nan_pct[c]*100:.0f}% NaN)")
        df = df.drop(columns=bad_cols)
        sensor_cols = [c for c in sensor_cols if c not in bad_cols]

    # Fill remaining NaN in surviving columns with median
    nan_remaining = df[sensor_cols].isna().sum().sum()
    if nan_remaining > 0:
        print(f"  Filling {int(nan_remaining)} remaining NaN with column median")
        for col in sensor_cols:
            if df[col].isna().any():
                med = df[col].median()
                if pd.isna(med):
                    med = 0.0   # if entire col was NaN
                df[col] = df[col].fillna(med)

    # Fill network NaN (shouldn't have any, but be safe)
    for col in network_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    print(f"  Final sensor cols  : {len(sensor_cols)}")
    print(f"  Final network cols : {len(network_cols)}")

    # ── [4/7] Per-attack stratified split ───────────────────────
    print(f"\n[4/7] Attack-stratified chronological split...")
    print(f"  Each attack event split: 70% train / 15% val / 15% test")

    # Mark each row with split assignment
    df['split'] = 'unassigned'

    # Find contiguous attack blocks
    df['attack_block'] = (df['label_binary'].diff().fillna(0) != 0).cumsum()
    attack_blocks = (
        df[df['label_binary'] == 1]
        .groupby('attack_block')
        .agg(start_idx=('attack_block', lambda x: x.index[0]),
             end_idx=('attack_block', lambda x: x.index[-1]),
             class_id=('label_class', 'first'))
        .reset_index()
    )

    print(f"  Found {len(attack_blocks)} attack blocks")

    # For each attack block: split internally 70/15/15
    for _, blk in attack_blocks.iterrows():
        s, e = int(blk['start_idx']), int(blk['end_idx'])
        block_len = e - s + 1
        n_train = int(block_len * ATTACK_TRAIN_RATIO)
        n_val   = int(block_len * ATTACK_VAL_RATIO)

        # Assign indices
        df.loc[s:s + n_train - 1, 'split']                    = 'train'
        df.loc[s + n_train:s + n_train + n_val - 1, 'split']  = 'val'
        df.loc[s + n_train + n_val:e, 'split']                = 'test'

    # Normal rows: 70/15/15 chronological
    normal_idx = df[df['label_binary'] == 0].index.tolist()
    n_normal = len(normal_idx)
    n_n_train = int(n_normal * 0.70)
    n_n_val   = int(n_normal * 0.15)

    train_normal_idx = normal_idx[:n_n_train]
    val_normal_idx   = normal_idx[n_n_train:n_n_train + n_n_val]
    test_normal_idx  = normal_idx[n_n_train + n_n_val:]

    df.loc[train_normal_idx, 'split'] = 'train'
    df.loc[val_normal_idx,   'split'] = 'val'
    df.loc[test_normal_idx,  'split'] = 'test'

    # Report
    for name in ['train', 'val', 'test']:
        sub = df[df['split'] == name]
        atk = int(sub['label_binary'].sum())
        print(f"  {name:5s}: {len(sub):>6,} rows | "
              f"{atk:>5,} attack ({atk/len(sub)*100:.1f}%)")
        # Class breakdown
        for cls_id in range(6):
            count = int(((sub['label_class'] == cls_id) &
                         (sub['label_binary'] == 1)).sum())
            if count > 0:
                print(f"      class {cls_id}: {count:>5,}")

    # ── [5/7] Scaling ────────────────────────────────────────────
    print(f"\n[5/7] Scaling (RobustScaler fit on train only)...")

    train_df = df[df['split'] == 'train'].copy()
    val_df   = df[df['split'] == 'val'].copy()
    test_df  = df[df['split'] == 'test'].copy()

    sensor_scaler  = RobustScaler()
    network_scaler = RobustScaler()

    train_df[sensor_cols]  = sensor_scaler.fit_transform(train_df[sensor_cols])
    val_df[sensor_cols]    = sensor_scaler.transform(val_df[sensor_cols])
    test_df[sensor_cols]   = sensor_scaler.transform(test_df[sensor_cols])

    train_df[network_cols] = network_scaler.fit_transform(train_df[network_cols])
    val_df[network_cols]   = network_scaler.transform(val_df[network_cols])
    test_df[network_cols]  = network_scaler.transform(test_df[network_cols])

    # ── [6/7] Re-sort each split chronologically and window ──────
    print(f"\n[6/7] Building {WINDOW_SIZE}-second windows...")

    # CRITICAL: re-sort by timestamp so windows are contiguous
    train_df = train_df.sort_values('timestamp_sgt').reset_index(drop=True)
    val_df   = val_df.sort_values('timestamp_sgt').reset_index(drop=True)
    test_df  = test_df.sort_values('timestamp_sgt').reset_index(drop=True)

    def make_windows(sub_df):
        sensor_arr = sub_df[sensor_cols].values.astype(np.float32)
        net_arr    = sub_df[network_cols].values.astype(np.float32)
        bin_arr    = sub_df['label_binary'].values.astype(np.int64)
        cls_arr    = sub_df['label_class'].values.astype(np.int64)
        comp_arr   = sub_df[component_cols].values.astype(np.float32)

        n = len(sub_df)
        if n < WINDOW_SIZE:
            return (
                np.zeros((0, WINDOW_SIZE, len(sensor_cols)), np.float32),
                np.zeros((0, WINDOW_SIZE, len(network_cols)), np.float32),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.int64),
                np.zeros((0, len(COMPONENTS)), np.float32),
            )

        n_win = (n - WINDOW_SIZE) // STRIDE + 1

        X_s = np.zeros((n_win, WINDOW_SIZE, len(sensor_cols)),  np.float32)
        X_n = np.zeros((n_win, WINDOW_SIZE, len(network_cols)), np.float32)
        y_b = np.zeros((n_win,), np.int64)
        y_c = np.zeros((n_win,), np.int64)
        y_p = np.zeros((n_win, len(COMPONENTS)), np.float32)

        for i in range(n_win):
            s = i * STRIDE
            e = s + WINDOW_SIZE
            X_s[i] = sensor_arr[s:e]
            X_n[i] = net_arr[s:e]
            y_b[i] = int(bin_arr[s:e].max())

            atk_cls = cls_arr[s:e][cls_arr[s:e] > 0]
            if len(atk_cls) > 0:
                vals, cnts = np.unique(atk_cls, return_counts=True)
                y_c[i] = int(vals[np.argmax(cnts)])
            else:
                y_c[i] = 0

            y_p[i] = comp_arr[s:e].max(axis=0)

        return X_s, X_n, y_b, y_c, y_p

    print(f"  Building train windows...")
    X_s_tr, X_n_tr, y_b_tr, y_c_tr, y_p_tr = make_windows(train_df)
    print(f"  Building val windows...")
    X_s_va, X_n_va, y_b_va, y_c_va, y_p_va = make_windows(val_df)
    print(f"  Building test windows...")
    X_s_te, X_n_te, y_b_te, y_c_te, y_p_te = make_windows(test_df)

    # ── [7/7] Save + report ─────────────────────────────────────
    print(f"\n[7/7] Saving to {OUTPUT_NPZ}...")

    print(f"\n  Window shapes:")
    for name, X_s, X_n, y_b, y_c in [
        ('train', X_s_tr, X_n_tr, y_b_tr, y_c_tr),
        ('val',   X_s_va, X_n_va, y_b_va, y_c_va),
        ('test',  X_s_te, X_n_te, y_b_te, y_c_te),
    ]:
        atk = int(y_b.sum())
        print(f"    {name:5s}: X_s={X_s.shape} X_n={X_n.shape} "
              f"y={y_b.shape}")
        print(f"           attack: {atk:,} ({atk/len(y_b)*100:.1f}%)")
        for cls_id in range(6):
            count = int((y_c == cls_id).sum())
            if count > 0:
                print(f"           class {cls_id}: {count:>5,}")

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
    print(f"\n  Saved: {OUTPUT_NPZ} ({size_mb:.1f} MB)")

    print(f"\n  FINAL SENSOR COLS USED ({len(sensor_cols)}):")
    for i, c in enumerate(sensor_cols):
        if i % 5 == 0:
            print(f"   ", end='')
        print(f" {c:18s}", end='')
        if i % 5 == 4:
            print()
    print()

    print("\n" + "=" * 60)
    print("Windowing complete (v2). Ready to train.")
    print("=" * 60)


if __name__ == '__main__':
    main()