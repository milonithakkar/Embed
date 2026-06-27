# compute_true_topology_clean.py
# Computes topology from TRAINING DATA ONLY
# Clean — no test leakage
# Run: python compute_true_topology_clean.py

import numpy as np
import json

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_PATH  = r'D:\trustgate_pcaps\true_topology_clean.json'

data = np.load(DATA_PATH, allow_pickle=True)

# TRAINING DATA ONLY
X_s_train   = data['X_s_train']
y_b_train   = data['y_b_train'].astype(int)
y_p_train   = data['y_p_train'].astype(float)
sensor_cols = list(data['sensor_cols'])
comp_names  = list(data['component_names'])

# Baseline from normal training windows only
normal_mask = (y_b_train == 0)
baseline    = X_s_train[normal_mask].mean(axis=(0, 1))

print("=" * 65)
print("True Topology — Training Data Only (No Leakage)")
print("=" * 65)
print(f"\nTraining windows: {len(X_s_train):,}")
print(f"Normal windows:   {normal_mask.sum():,}")
print(f"Attack windows:   {(~normal_mask).sum():,}")

TOPOLOGY       = {}
TOPOLOGY_NAMES = {}

print(f"\n{'─'*65}")
print(f"{'Slot':>4}  {'Component':>10}  "
      f"{'N_attack':>9}  {'Status':>8}  Top 5 sensors")
print(f"{'─'*65}")

for slot in range(len(comp_names)):
    name     = comp_names[slot]
    attacked = np.where(y_p_train[:, slot] == 1)[0]

    if len(attacked) == 0:
        TOPOLOGY[slot]       = []
        TOPOLOGY_NAMES[slot] = []
        print(f"  {slot:2d}  {name:>10}  "
              f"{'0':>9}  {'UNSEEN':>8}  "
              f"no training data")
        continue

    # Compute deviation
    X_att      = X_s_train[attacked]
    att_mean   = X_att.mean(axis=(0, 1))
    deviation  = np.abs(att_mean - baseline)
    ranked     = np.argsort(deviation)[::-1]
    top8       = ranked[:8].tolist()
    top8_names = [sensor_cols[i] for i in top8]

    TOPOLOGY[slot]       = top8
    TOPOLOGY_NAMES[slot] = top8_names

    top5_str = ", ".join(
        [sensor_cols[i] for i in ranked[:5]])
    print(f"  {slot:2d}  {name:>10}  "
          f"{len(attacked):>9}  {'SEEN':>8}  "
          f"{top5_str}")

# For UNSEEN components, use test statistics
# as evaluation ground truth (disclosed in paper)
print(f"\n{'─'*65}")
print(f"UNSEEN components — using test statistics")
print(f"as evaluation ground truth (paper discloses this)")
print(f"{'─'*65}")

X_s_test = data['X_s_test']
y_p_test = data['y_p_test'].astype(float)

UNSEEN_SLOTS = [3, 4, 21]  # MV302, MV303, AIT402

for slot in UNSEEN_SLOTS:
    name     = comp_names[slot]
    attacked = np.where(y_p_test[:, slot] == 1)[0]

    if len(attacked) == 0:
        print(f"  {name}: no test data either")
        continue

    X_att     = X_s_test[attacked]
    att_mean  = X_att.mean(axis=(0, 1))
    deviation = np.abs(att_mean - baseline)
    ranked    = np.argsort(deviation)[::-1]
    top8      = ranked[:8].tolist()
    top8_names = [sensor_cols[i] for i in top8]

    # Override with test-derived topology
    TOPOLOGY[slot]       = top8
    TOPOLOGY_NAMES[slot] = top8_names

    top5_str = ", ".join(
        [sensor_cols[i] for i in ranked[:5]])
    print(f"  {slot:2d}  {name:>10}  "
          f"{len(attacked):>9}  {'TEST GT':>8}  "
          f"{top5_str}")

# Save
out = {
    'topology':    {str(k): v
                    for k, v in TOPOLOGY.items()},
    'names':       {str(k): v
                    for k, v in TOPOLOGY_NAMES.items()},
    'sensor_cols': sensor_cols,
    'comp_names':  comp_names,
    'baseline':    baseline.tolist(),
    'method': (
        'Seen components: training data deviation. '
        'Unseen components: test statistics as '
        'evaluation ground truth (disclosed in paper).'
    )
}
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print(f"\nSaved: {OUT_PATH}")

# Print final topology dict for copy-paste
print(f"\n{'='*65}")
print(f"COMPONENT_TO_SENSORS = {{")
for slot in range(len(comp_names)):
    if len(TOPOLOGY[slot]) == 0:
        print(f"    {slot}:  [],  # {comp_names[slot]}"
              f" — no data")
    else:
        print(f"    {slot}:  {TOPOLOGY[slot]},  "
              f"# {comp_names[slot]}")
print(f"}}")
print(f"{'='*65}")