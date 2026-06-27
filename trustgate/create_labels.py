"""
TrustGate — Multi-class Attack Stage Labeler (v2)
==================================================
Reads attack names saved directly by 01_build_windows.py
and maps them to 6-class labels — no sensor-deviation guessing.

Attack name → class mapping mirrors merge_A12.py ATTACK_CLASSES.

Classes:
  0: NORMAL
  1: VALVE_ATTACK   (Stage 5 Valve Manip/Repeat, RO Backwash)
  2: PUMP_ATTACK    (Stage 1 Flow Disrupt, Stage 2 Parallel Pump)
  3: CHEMICAL       (Florida Water Scenario)
  4: LEVEL_SPOOF    (Tank Overflow LIT101/Repeat)
  5: SENSOR_SPOOF   (Forced Backwash DPIT301, LIT601 Spoofing, AIT402 Spoof)

Run: python create_labels.py
"""

import numpy as np

NPZ_IN  = 'trustgate_data/swat_final.npz'
NPZ_OUT = 'trustgate_data/swat_multilabel.npz'

# ── Attack name → integer class ───────────────────────────────────
ATTACK_CLASSES = {
    'NORMAL'                  : 0,
    'Stage 5 Valve Manip'     : 1,
    'Stage 5 Valve Repeat'    : 1,
    'RO Backwash Diversion'   : 1,
    'Stage 1 Flow Disrupt'    : 2,
    'Stage 2 Parallel Pump'   : 2,
    'Florida Water Scenario'  : 3,
    'Tank Overflow LIT101'    : 4,
    'Tank Overflow Repeat'    : 4,
    'Forced Backwash DPIT301' : 5,
    'LIT601 Spoofing'         : 5,
    'AIT402 High Spoof'       : 5,
}

CLASS_NAMES = {
    0: 'NORMAL',
    1: 'VALVE_ATTACK',
    2: 'PUMP_ATTACK',
    3: 'CHEMICAL',
    4: 'LEVEL_SPOOF',
    5: 'SENSOR_SPOOF',
}


def names_to_classes(names_arr, y_binary):
    """
    Convert array of attack name strings → integer class array.
    Falls back to sensor-deviation logic only for truly unknown names.
    """
    y_multi = np.zeros(len(names_arr), dtype=np.int64)
    unknown = []

    for i, name in enumerate(names_arr):
        name = str(name).strip()
        if name in ATTACK_CLASSES:
            y_multi[i] = ATTACK_CLASSES[name]
        elif y_binary[i] == 1:
            # Unknown attack name — default to class 1 and flag it
            y_multi[i] = 1
            unknown.append(name)

    if unknown:
        unique_unknown = set(unknown)
        print(f'  [WARN] {len(unknown)} windows had unknown attack names '
              f'(defaulted to class 1): {unique_unknown}')

    return y_multi


def main():
    print(f'Loading {NPZ_IN}...')
    data = np.load(NPZ_IN, allow_pickle=True)

    print(f'Keys: {list(data.keys())}\n')

    sensor_cols = data['sensor_cols']
    print(f'Sensor columns ({len(sensor_cols)}): {list(sensor_cols)}\n')

    # Check if attack names were saved
    has_names = 'y_names_train' in data
    if not has_names:
        print('[WARN] y_names_train not found in NPZ.')
        print('       Re-run 01_build_windows.py first to save attack names.')
        print('       Falling back to binary labels only (all attacks = class 1).')

    splits = {}
    for split in ['train', 'val', 'test']:
        print(f'[{split.upper()}] Creating class labels...')

        y_binary = data[f'y_{split}']

        if has_names:
            names = data[f'y_names_{split}']
            y_multi = names_to_classes(names, y_binary)
        else:
            # Fallback: binary → 1 for all attacks
            y_multi = y_binary.copy()

        splits[split] = y_multi

        # Distribution report
        print(f'  Distribution:')
        for cls, name in CLASS_NAMES.items():
            count = int((y_multi == cls).sum())
            pct   = count / len(y_multi) * 100
            bar   = '█' * int(pct / 2)
            print(f'    Class {cls} ({name:>12}): {count:>7,}  ({pct:5.1f}%)  {bar}')
        print()

    # ── Save ──────────────────────────────────────────────────────
    np.savez_compressed(
        NPZ_OUT,
        X_s_train=data['X_s_train'], X_n_train=data['X_n_train'],
        y_train=splits['train'],
        X_s_val  =data['X_s_val'],   X_n_val  =data['X_n_val'],
        y_val  =splits['val'],
        X_s_test =data['X_s_test'],  X_n_test =data['X_n_test'],
        y_test =splits['test'],
        sensor_cols=sensor_cols,
    )
    print(f'[OK] Saved {NPZ_OUT}')


if __name__ == '__main__':
    main()