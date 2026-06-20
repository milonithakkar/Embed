"""
TrustGate — Multi-class Attack Stage Labeler
Converts binary y (0/1) → 6-class labels
Based on: Ahmed et al. (2017) SWaT attack descriptions
"""
import numpy as np

ATTACK_CLASSES = {
    0: 'NORMAL',
    1: 'CHEMICAL',     # AIT202, AIT203, AIT501-504
    2: 'PRESSURE',     # PIT501-503, DPIT301
    3: 'FLOW_TAMPER',  # FIT101,201,301,401,501-504
    4: 'PUMP_DOS',     # P-series actuators
    5: 'VALVE_ATTACK'  # MV-series + LIT-series
}

SENSOR_GROUPS = {
    1: ['AIT202','AIT203','AIT501','AIT502','AIT503','AIT504'],
    2: ['PIT501','PIT502','PIT503','DPIT301'],
    3: ['FIT101','FIT201','FIT301','FIT401','FIT501','FIT502','FIT503','FIT504'],
    4: ['P102','P202','P203','P204','P205','P206','P301','P302',
        'P401','P402','P403','P404','P501','P502','P601','P602','P603'],
    5: ['MV101','MV201','MV301','MV302','MV303','MV304','MV401',
        'LIT101','LIT201','LIT301','LIT401','LIT501']
}

def label_attack_stages(X_s, y_binary, sensor_cols):
    sensor_cols = list(sensor_cols)
    y_multi     = np.zeros(len(y_binary), dtype=np.int64)

    # Normal baseline: mean of last timestep across all normal windows
    normal_mean = X_s[y_binary == 0, -1, :].mean(axis=0)  # (44,)

    # Map sensor names → column indices
    group_indices = {}
    for cls, sensors in SENSOR_GROUPS.items():
        idxs = [sensor_cols.index(s) for s in sensors if s in sensor_cols]
        group_indices[cls] = idxs
        print(f"  Class {cls} ({ATTACK_CLASSES[cls]}): {len(idxs)} sensors mapped")

    # Assign class to each attack window
    attack_idxs = np.where(y_binary == 1)[0]
    print(f"\n  Labeling {len(attack_idxs):,} attack windows...")

    for i in attack_idxs:
        deviation  = np.abs(X_s[i, -1, :] - normal_mean)
        best_cls   = 1
        best_score = -1
        for cls, idxs in group_indices.items():
            if not idxs:
                continue
            score = deviation[idxs].mean()
            if score > best_score:
                best_score = score
                best_cls   = cls
        y_multi[i] = best_cls

    return y_multi


if __name__ == '__main__':
    print("Loading swat_final.npz...")
    data = np.load('trustgate_data/swat_final.npz', allow_pickle=True)
    sensor_cols = data['sensor_cols']
    print(f"Sensor columns ({len(sensor_cols)}): {list(sensor_cols)}\n")

    splits = {}
    for split in ['train', 'val', 'test']:
        print(f"[{split.upper()}] Creating stage labels...")
        y_multi = label_attack_stages(
            data[f'X_s_{split}'], data[f'y_{split}'], sensor_cols)
        splits[split] = y_multi

        print(f"  Distribution:")
        for cls, name in ATTACK_CLASSES.items():
            count = (y_multi == cls).sum()
            print(f"    Class {cls} ({name:>12}): {count:>8,}")
        print()

    np.savez_compressed(
        'trustgate_data/swat_multilabel.npz',
        X_s_train=data['X_s_train'], X_n_train=data['X_n_train'], y_train=splits['train'],
        X_s_val  =data['X_s_val'],   X_n_val  =data['X_n_val'],   y_val  =splits['val'],
        X_s_test =data['X_s_test'],  X_n_test =data['X_n_test'],  y_test =splits['test'],
        sensor_cols=sensor_cols
    )
    print("[OK] Saved trustgate_data/swat_multilabel.npz")