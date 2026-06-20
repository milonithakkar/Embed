# =================================================================
# merge_A12.py
# Combines A12 sensor CSV + smart network features + attack labels
# Save as: C:\Users\HP\Downloads\trustgate\merge_A12.py
# Run    : python merge_A12.py
# =================================================================

import pandas as pd
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
SENSOR_CSV  = r'D:\Users\HP\Downloads\11-Mar-2026_0900_1700(in).csv'
NETWORK_CSV = r'D:\trustgate_pcaps\network_features_A12_smart.csv'
OUTPUT_CSV  = r'D:\trustgate_pcaps\A12_merged.csv'

# ── A12 ATTACK SCHEDULE (from your table) ────────────────────────
# All times in Singapore time (SGT)
# (start_time, end_time, attack_class, attack_name, target_components)
ATTACKS = [
    ('13:00:00', '13:05:00', 'VALVE_ATTACK',  'Stage 5 Valve Manip',        'MV501-504'),
    ('13:40:00', '13:45:00', 'PUMP_ATTACK',   'Stage 1 Flow Disrupt',       'MV101,P101,P102'),
    ('14:20:00', '14:25:00', 'CHEMICAL',      'Florida Water Scenario',     'P201-206,MV201'),
    ('14:30:00', '14:35:00', 'LEVEL_SPOOF',   'Tank Overflow LIT101',       'LIT101'),
    ('14:40:00', '14:45:00', 'VALVE_ATTACK',  'Stage 5 Valve Repeat',       'MV501-504'),
    ('15:00:00', '15:02:00', 'LEVEL_SPOOF',   'Tank Overflow Repeat',       'LIT101'),
    ('15:02:00', '15:07:00', 'PUMP_ATTACK',   'Stage 2 Parallel Pump',      'MV201,P101,P102'),
    ('15:20:00', '15:25:00', 'VALVE_ATTACK',  'RO Backwash Diversion',      'MV302,MV303'),
    ('15:45:00', '15:50:00', 'SENSOR_SPOOF',  'Forced Backwash DPIT301',    'DPIT301'),
    ('16:10:00', '16:15:00', 'SENSOR_SPOOF',  'LIT601 Spoofing',            'LIT601'),
    ('16:35:00', '16:40:00', 'SENSOR_SPOOF',  'AIT402 High Spoof',          'AIT402'),
]

ATTACK_CLASSES = {
    'NORMAL'       : 0,
    'VALVE_ATTACK' : 1,
    'PUMP_ATTACK'  : 2,
    'CHEMICAL'     : 3,
    'LEVEL_SPOOF'  : 4,
    'SENSOR_SPOOF' : 5,
}

# Components for localization head (Head 3)
COMPONENTS = [
    'MV101', 'MV201', 'MV301', 'MV302', 'MV303', 'MV304',
    'MV501', 'MV502', 'MV503', 'MV504',
    'P101', 'P102', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'LIT101', 'LIT601',
    'DPIT301', 'AIT402',
]


# ── Helpers ──────────────────────────────────────────────────────
def parse_attack_time(time_str, date_str='2026-03-11'):
    """Convert HH:MM:SS to datetime on attack day."""
    return datetime.strptime(
        f"{date_str} {time_str}",
        '%Y-%m-%d %H:%M:%S'
    )


def expand_component_token(token):
    """
    Expand patterns like:
      MV501-504 -> MV501, MV502, MV503, MV504
      P201-206  -> P201, P202, ..., P206
    """
    token = token.strip()

    if '-' not in token:
        return [token]

    left, right = token.split('-')

    prefix = ''.join(ch for ch in left if ch.isalpha())
    left_num = ''.join(ch for ch in left if ch.isdigit())
    right_num = ''.join(ch for ch in right if ch.isdigit())

    if not prefix or not left_num or not right_num:
        return [token]

    start_n = int(left_num)
    end_n   = int(right_num)

    return [f"{prefix}{n}" for n in range(start_n, end_n + 1)]


def expand_targets(targets_str):
    """Expand comma-separated target list into explicit component names."""
    raw = [x.strip() for x in targets_str.split(',')]
    expanded = []
    for token in raw:
        expanded.extend(expand_component_token(token))
    return expanded


def get_attack_for_timestamp(ts):
    """
    Returns:
      binary_label, class_id, attack_name, component_list
    """
    for start_str, end_str, atk_class, atk_name, targets in ATTACKS:
        start = parse_attack_time(start_str)
        end   = parse_attack_time(end_str)

        if start <= ts < end:
            comps = expand_targets(targets)
            return 1, ATTACK_CLASSES[atk_class], atk_name, comps

    return 0, ATTACK_CLASSES['NORMAL'], 'NORMAL', []


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TrustGate — Merge A12 Sensor + Network + Labels")
    print("=" * 60)

    # ── [1/5] Load sensor CSV ────────────────────────────────────
    print(f"\n[1/5] Loading sensor CSV...")
    print(f"  File: {SENSOR_CSV}")

    sensor_df = pd.read_csv(SENSOR_CSV, low_memory=False)

    print(f"  Shape: {sensor_df.shape}")
    print(f"  Columns: {len(sensor_df.columns)} "
          f"(first 5: {list(sensor_df.columns[:5])})")

    # Flexible timestamp parsing
    sensor_df['timestamp_sgt'] = pd.to_datetime(
        sensor_df['t_stamp'],
        errors='coerce'
    )

    bad_ts = sensor_df['timestamp_sgt'].isna().sum()
    print(f"  Bad timestamps: {bad_ts}")

    if bad_ts > 0:
        print("  Sample bad timestamp values:")
        print(sensor_df[sensor_df['timestamp_sgt'].isna()]['t_stamp']
              .head(5).tolist())

    sensor_df = sensor_df.dropna(subset=['timestamp_sgt']).reset_index(drop=True)
    sensor_df = sensor_df.drop(columns=['t_stamp'])

    print(f"  Parsed time range: {sensor_df['timestamp_sgt'].min()} "
          f"→ {sensor_df['timestamp_sgt'].max()}")

    # ── [2/5] Load network CSV ───────────────────────────────────
    print(f"\n[2/5] Loading network features...")
    print(f"  File: {NETWORK_CSV}")

    network_df = pd.read_csv(NETWORK_CSV, low_memory=False)
    network_df['timestamp_sgt'] = pd.to_datetime(
        network_df['timestamp_sgt'],
        errors='coerce'
    )
    network_df = network_df.dropna(subset=['timestamp_sgt']).reset_index(drop=True)

    print(f"  Shape: {network_df.shape}")
    print(f"  Parsed time range: {network_df['timestamp_sgt'].min()} "
          f"→ {network_df['timestamp_sgt'].max()}")

    # ── [3/5] Merge on timestamp ────────────────────────────────
    print(f"\n[3/5] Merging on timestamp...")

    merged = pd.merge(
        sensor_df,
        network_df,
        on='timestamp_sgt',
        how='inner'
    )

    merged = merged.sort_values('timestamp_sgt').reset_index(drop=True)

    print(f"  Sensor rows  : {len(sensor_df):,}")
    print(f"  Network rows : {len(network_df):,}")
    print(f"  Merged rows  : {len(merged):,}")
    print(f"  Overlap      : {len(merged) / len(sensor_df) * 100:.1f}% "
          f"of sensor data")

    # ── [4/5] Apply attack labels ───────────────────────────────
    print(f"\n[4/5] Applying attack labels from schedule...")

    label_binary = []
    label_class  = []
    attack_name  = []
    component_vectors = []

    for ts in merged['timestamp_sgt']:
        y_bin, y_cls, atk_name, comps = get_attack_for_timestamp(ts)

        label_binary.append(y_bin)
        label_class.append(y_cls)
        attack_name.append(atk_name)

        comp_vec = [1 if c in comps else 0 for c in COMPONENTS]
        component_vectors.append(comp_vec)

    merged['label_binary'] = label_binary
    merged['label_class']  = label_class
    merged['attack_name']  = attack_name

    comp_df = pd.DataFrame(
        component_vectors,
        columns=[f'comp_{c}' for c in COMPONENTS]
    )
    merged = pd.concat([merged, comp_df], axis=1)

    # Label stats
    n_total  = len(merged)
    n_attack = int(merged['label_binary'].sum())
    n_normal = n_total - n_attack

    print(f"  Total seconds : {n_total:,}")
    print(f"  Normal        : {n_normal:,} ({n_normal / n_total * 100:.1f}%)")
    print(f"  Attack        : {n_attack:,} ({n_attack / n_total * 100:.1f}%)")

    print(f"\n  Attack class distribution:")
    for class_name, class_id in ATTACK_CLASSES.items():
        count = int((merged['label_class'] == class_id).sum())
        if count > 0:
            print(f"    {class_id} {class_name:15s}: {count:>5,}")

    print(f"\n  Attacks captured by name:")
    attack_rows = merged[merged['label_binary'] == 1]
    for name in attack_rows['attack_name'].unique():
        count = int((merged['attack_name'] == name).sum())
        print(f"    {name:35s}: {count:>5,} sec")

    # ── [5/5] Save merged CSV ───────────────────────────────────
    print(f"\n[5/5] Saving merged dataset...")

    merged.to_csv(OUTPUT_CSV, index=False)

    size_mb = Path(OUTPUT_CSV).stat().st_size / (1024**2)
    print(f"  Saved: {OUTPUT_CSV}")
    print(f"  Size : {size_mb:.2f} MB")
    print(f"  Shape: {merged.shape}")
    print(f"  Cols : {len(merged.columns)}")

    # Feature breakdown
    network_feature_cols = [c for c in network_df.columns if c != 'timestamp_sgt']
    component_cols = [f'comp_{c}' for c in COMPONENTS]

    excluded = set(
        ['timestamp_sgt', 'label_binary', 'label_class', 'attack_name'] +
        network_feature_cols +
        component_cols
    )
    sensor_feature_cols = [c for c in merged.columns if c not in excluded]

    print(f"\n  Feature counts:")
    print(f"    Sensor features  : {len(sensor_feature_cols)}")
    print(f"    Network features : {len(network_feature_cols)}")
    print(f"    Component labels : {len(component_cols)}")
    print(f"    Total model input features: "
          f"{len(sensor_feature_cols) + len(network_feature_cols)}")

    print("\n" + "=" * 60)
    print("Merge complete. Ready for windowing + training.")
    print("=" * 60)


if __name__ == '__main__':
    main()