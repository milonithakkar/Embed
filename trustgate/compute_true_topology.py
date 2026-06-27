# compute_true_topology.py
# Computes true sensor-component topology from actual data
# Uses statistical deviation during attacks to find
# which sensors ACTUALLY respond to each component attack
# Run: python compute_true_topology.py

import numpy as np
import json
import os

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_PATH  = r'D:\trustgate_pcaps\true_topology.json'

data = np.load(DATA_PATH, allow_pickle=True)

X_s_train  = data['X_s_train']
X_s_val    = data['X_s_val']
X_s_test   = data['X_s_test']
y_b_train  = data['y_b_train'].astype(int)
y_p_train  = data['y_p_train'].astype(float)
y_p_val    = data['y_p_val'].astype(float)
y_p_test   = data['y_p_test'].astype(float)
sensor_cols = list(data['sensor_cols'])
comp_names  = list(data['component_names'])

# Combine all splits for maximum coverage
X_s_all  = np.concatenate([X_s_train, X_s_val,  X_s_test],  axis=0)
y_p_all  = np.concatenate([y_p_train, y_p_val,  y_p_test],  axis=0)
y_b_all  = np.concatenate([
    y_b_train,
    data['y_b_val'].astype(int),
    data['y_b_test'].astype(int)], axis=0)

# Compute normal baseline from training normal windows only
normal_mask = (y_b_train == 0)
baseline    = X_s_train[normal_mask].mean(axis=(0, 1))

print("=" * 65)
print("Computing True Sensor-Component Topology")
print("=" * 65)
print(f"\nTotal windows: {len(X_s_all):,}")
print(f"Baseline from: {normal_mask.sum():,} normal windows")
print(f"Sensor cols:   {len(sensor_cols)}")
print(f"Components:    {len(comp_names)}")

# ── Compute true topology for each component ───────────────────
TRUE_TOPOLOGY  = {}
TOPOLOGY_NAMES = {}

print(f"\n{'─'*65}")
print(f"{'Slot':>4}  {'Component':>10}  "
      f"{'N_windows':>10}  {'Top 8 sensors'}")
print(f"{'─'*65}")

for slot in range(len(comp_names)):
    comp_name = comp_names[slot]

    # Find attack windows for this component
    attacked = np.where(y_p_all[:, slot] == 1)[0]

    if len(attacked) == 0:
        # Component never attacked in any split
        TRUE_TOPOLOGY[slot]  = []
        TOPOLOGY_NAMES[slot] = []
        print(f"  {slot:2d}  {comp_name:>10}  "
              f"{'NO DATA':>10}  skipped")
        continue

    # Get attacked windows
    X_attacked = X_s_all[attacked]   # (N, 30, 71)

    # Per-sensor mean during attack
    attack_mean = X_attacked.mean(axis=(0, 1))   # (71,)

    # Absolute deviation from normal baseline
    deviation = np.abs(attack_mean - baseline)   # (71,)

    # Rank sensors by deviation
    ranked = np.argsort(deviation)[::-1]

    # Take top 8 as true adjacent sensors
    top8      = ranked[:8].tolist()
    top8_names = [sensor_cols[i] for i in top8]

    TRUE_TOPOLOGY[slot]  = top8
    TOPOLOGY_NAMES[slot] = top8_names

    # Show top 5 for display
    top5_str = ", ".join([sensor_cols[i]
                           for i in ranked[:5]])
    print(f"  {slot:2d}  {comp_name:>10}  "
          f"{len(attacked):>10}  {top5_str}")

# ── Show full breakdown for unseen components ──────────────────
print(f"\n{'='*65}")
print(f"UNSEEN COMPONENTS — Detailed Breakdown")
print(f"{'='*65}")

UNSEEN = [3, 4, 21]  # MV302, MV303, AIT402

for slot in UNSEEN:
    comp_name = comp_names[slot]
    attacked  = np.where(y_p_all[:, slot] == 1)[0]

    if len(attacked) == 0:
        print(f"\n{comp_name}: NO DATA")
        continue

    X_att     = X_s_all[attacked]
    att_mean  = X_att.mean(axis=(0, 1))
    deviation = np.abs(att_mean - baseline)
    ranked    = np.argsort(deviation)[::-1]

    print(f"\n{comp_name} (slot {slot}) "
          f"— {len(attacked)} attack windows")
    print(f"  {'Rank':>4}  {'Sensor':25s}  "
          f"{'Normal':>8}  {'Attack':>8}  {'Diff':>8}")
    print(f"  {'─'*55}")

    for rank, idx in enumerate(ranked[:10]):
        print(f"  {rank+1:4d}  {sensor_cols[idx]:25s}  "
              f"{baseline[idx]:>8.4f}  "
              f"{att_mean[idx]:>8.4f}  "
              f"{deviation[idx]:>8.4f}")

    print(f"\n  True adjacent (top 8): {TRUE_TOPOLOGY[slot]}")
    print(f"  Names: {TOPOLOGY_NAMES[slot]}")

# ── Check if model attention matches true topology ─────────────
print(f"\n{'='*65}")
print(f"TOPOLOGY VALIDATION SUMMARY")
print(f"{'='*65}")
print(f"\nComponents with enough data for topology:")

for slot in range(len(comp_names)):
    if len(TRUE_TOPOLOGY[slot]) > 0:
        names = [sensor_cols[i]
                 for i in TRUE_TOPOLOGY[slot][:5]]
        print(f"  Slot {slot:2d} {comp_names[slot]:>10}: "
              f"{names}")

# ── Save true topology ─────────────────────────────────────────
save_data = {
    'topology':      TRUE_TOPOLOGY,
    'names':         TOPOLOGY_NAMES,
    'sensor_cols':   sensor_cols,
    'comp_names':    comp_names,
    'baseline':      baseline.tolist(),
    'description':   (
        'True sensor-component topology computed from '
        'actual sensor deviations during attacks. '
        'More accurate than P&ID assumptions.'
    )
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2)

print(f"\n{'='*65}")
print(f"Saved: {OUT_PATH}")
print(f"\nNow update evaluate_zero_shot.py and")
print(f"dual_model_inference.py with true topology.")
print(f"{'='*65}")