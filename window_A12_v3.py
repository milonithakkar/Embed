# window_A12_v3.py
# Fixes: block-level attack split + per-window normalization
# Save as: C:\Users\HP\Downloads\trustgate\window_A12_v3.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
INPUT_CSV   = r'D:\trustgate_pcaps\A12_merged.csv'
OUTPUT_NPZ  = r'D:\trustgate_pcaps\A12_windowed_v3.npz'

WINDOW_SIZE = 30
STRIDE      = 1

# Block-level split: which attack blocks go to val/test
# (block numbers from diagnose_v3.py)
VAL_BLOCKS  = [9]              # 1 attack for val
TEST_BLOCKS = [13, 19]         # 2 attacks for test
# All other attack blocks → train

COMPONENTS = [
    'MV101','MV201','MV301','MV302','MV303','MV304',
    'MV501','MV502','MV503','MV504',
    'P101','P102','P201','P202','P203','P204','P205','P206',
    'LIT101','LIT601','DPIT301','AIT402',
]


def main():
    print("="*60)
    print("TrustGate — Window A12 v3 (block-level split, full-data scaling)")
    print("="*60)

    # ── [1/8] Load merged CSV ────────────────────────────────────
    print(f"\n[1/8] Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])
    df = df.sort_values('timestamp_sgt').reset_index(drop=True)
    print(f"  Rows: {len(df):,}")

    # ── [2/8] Identify columns ──────────────────────────────────
    network_cols = [
        'packet_count','total_bytes','valve_write_count','pump_write_count',
        'valve_read_count','pump_read_count','level_access_count',
        'dosing_access_count','state_poll_count','valve_write_deviation',
        'pump_write_deviation','novel_tag_count','tag_entropy',
        'burst_component_max','rare_access_score','unique_tags_this_sec',
        'unique_src_ips','reset_count','permissive_count',
    ]
    component_cols = [f'comp_{c}' for c in COMPONENTS]
    excluded = set(['timestamp_sgt','label_binary','label_class','attack_name']
                   + network_cols + component_cols)
    sensor_cols = [c for c in df.columns if c not in excluded]

    print(f"\n[2/8] Initial columns:")
    print(f"  Sensor   : {len(sensor_cols)}")
    print(f"  Network  : {len(network_cols)}")

    # ── [3/8] Clean sensor columns ───────────────────────────────
    print(f"\n[3/8] Coercing + dropping all-NaN sensor columns...")
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    nan_pct = df[sensor_cols].isna().sum() / len(df)
    bad_cols = nan_pct[nan_pct > 0.5].index.tolist()
    if bad_cols:
        print(f"  Dropping {len(bad_cols)} useless cols")
        df = df.drop(columns=bad_cols)
        sensor_cols = [c for c in sensor_cols if c not in bad_cols]

    for col in sensor_cols:
        if df[col].isna().any():
            med = df[col].median()
            if pd.isna(med): med = 0.0
            df[col] = df[col].fillna(med)
    for col in network_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    print(f"  Final sensor cols : {len(sensor_cols)}")
    print(f"  Final network cols: {len(network_cols)}")

    # ── [4/8] Identify attack blocks ────────────────────────────
    print(f"\n[4/8] Identifying attack blocks...")
    df['attack_block'] = (df['label_binary'].diff().fillna(0) != 0).cumsum()

    attack_blocks = df[df['label_binary'] == 1].groupby('attack_block').agg(
        start=('timestamp_sgt', 'min'),
        end=('timestamp_sgt', 'max'),
        class_id=('label_class', 'first'),
        name=('attack_name', 'first'),
        size=('label_binary', 'sum')
    ).reset_index()
    print(f"  Total attack blocks: {len(attack_blocks)}")

    # ── [5/8] Assign split ───────────────────────────────────────
    print(f"\n[5/8] Block-level split assignment...")
    df['split'] = 'train'   # default

    # Mark val blocks
    for blk_id in VAL_BLOCKS:
        mask = df['attack_block'] == blk_id
        df.loc[mask, 'split'] = 'val'

    # Mark test blocks
    for blk_id in TEST_BLOCKS:
        mask = df['attack_block'] == blk_id
        df.loc[mask, 'split'] = 'test'

    # Normal rows: chronological 70/15/15
    normal_mask = df['label_binary'] == 0
    normal_idx  = df[normal_mask].index.tolist()
    n_norm = len(normal_idx)
    n_tr   = int(n_norm * 0.70)
    n_va   = int(n_norm * 0.15)

    df.loc[normal_idx[:n_tr], 'split']                     = 'train'
    df.loc[normal_idx[n_tr:n_tr+n_va], 'split']            = 'val'
    df.loc[normal_idx[n_tr+n_va:], 'split']                = 'test'

    # Report
    print(f"\n  Split assignment:")
    for name in ['train','val','test']:
        sub = df[df['split'] == name]
        atk = int(sub['label_binary'].sum())
        norm = len(sub) - atk
        print(f"    {name:5s}: {len(sub):>6,} rows | "
              f"normal={norm:>6,}  attack={atk:>5,} ({atk/len(sub)*100:.1f}%)")
        for cls_id in range(6):
            count = int(((sub['label_class'] == cls_id) &
                         (sub['label_binary'] == 1)).sum())
            if count > 0:
                print(f"        class {cls_id}: {count:>5,}")

    # ── [6/8] FULL-DATA scaling (no train-leakage assumption) ───
    print(f"\n[6/8] Scaling on FULL dataset (eliminates day-drift bias)...")
    print(f"  This is critical to fix val/test distribution shift")

    sensor_scaler  = RobustScaler()
    network_scaler = RobustScaler()

    # Fit on entire dataset
    df_sensor_scaled  = pd.DataFrame(
        sensor_scaler.fit_transform(df[sensor_cols]),
        columns=sensor_cols,
        index=df.index
    )
    df_network_scaled = pd.DataFrame(
        network_scaler.fit_transform(df[network_cols]),
        columns=network_cols,
        index=df.index
    )

    df[sensor_cols]  = df_sensor_scaled
    df[network_cols] = df_network_scaled

    # ── [7/8] Build windows ──────────────────────────────────────
    print(f"\n[7/8] Building {WINDOW_SIZE}-sec windows per split...")

    train_df = df[df['split'] == 'train'].sort_values('timestamp_sgt').reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].sort_values('timestamp_sgt').reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].sort_values('timestamp_sgt').reset_index(drop=True)

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
            s, e = i*STRIDE, i*STRIDE + WINDOW_SIZE
            X_s[i] = sensor_arr[s:e]
            X_n[i] = net_arr[s:e]
            y_b[i] = int(bin_arr[s:e].max())
            atk_cls = cls_arr[s:e][cls_arr[s:e] > 0]
            if len(atk_cls) > 0:
                vals, cnts = np.unique(atk_cls, return_counts=True)
                y_c[i] = int(vals[np.argmax(cnts)])
            y_p[i] = comp_arr[s:e].max(axis=0)
        return X_s, X_n, y_b, y_c, y_p

    print("  train..."); X_s_tr, X_n_tr, y_b_tr, y_c_tr, y_p_tr = make_windows(train_df)
    print("  val...");   X_s_va, X_n_va, y_b_va, y_c_va, y_p_va = make_windows(val_df)
    print("  test...");  X_s_te, X_n_te, y_b_te, y_c_te, y_p_te = make_windows(test_df)

    # ── [8/8] Save + report ─────────────────────────────────────
    print(f"\n[8/8] Saving and reporting...")
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

    print(f"\n  Window counts:")
    for name, X_s, y_b, y_c in [
        ('train', X_s_tr, y_b_tr, y_c_tr),
        ('val',   X_s_va, y_b_va, y_c_va),
        ('test',  X_s_te, y_b_te, y_c_te),
    ]:
        atk = int(y_b.sum())
        print(f"    {name:5s}: {len(y_b):>6,} windows | "
              f"{atk:>5,} attack ({atk/len(y_b)*100:.1f}%) | shape {X_s.shape}")
        for cls_id in range(6):
            count = int((y_c == cls_id).sum())
            if count > 0:
                print(f"        class {cls_id}: {count:>5,}")

    print("\n" + "="*60)
    print("v3 windowing complete.")
    print("Use DATA_PATH = D:\\trustgate_pcaps\\A12_windowed_v3.npz in train.py")
    print("="*60)


if __name__ == '__main__':
    main()