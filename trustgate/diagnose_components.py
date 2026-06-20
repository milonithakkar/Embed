# diagnose_components.py
# FIXED VERSION — Why is component head dying?
# Save as: C:\Users\HP\Downloads\trustgate\diagnose_components.py

import numpy as np

data = np.load(r'D:\trustgate_pcaps\A12_windowed_v3.npz', allow_pickle=True)

# ── Hardcoded component names (must match attention_attribution.py) ──
COMPONENT_NAMES = [
    "FIT101", "LIT101", "MV101", "P101", "P102",
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
    "P201", "P202", "P203", "P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302",   # ← TEST UNSEEN
    "MV303", "P301", "P302",                            # ← TEST UNSEEN
    "AIT401", "AIT402", "AIT501", "FIT401", "LIT401",   # AIT402 ← TEST UNSEEN
    "P401", "P402", "UV401",
    "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504",
    "P501", "P502",
    "FIT601", "P601", "P602", "P603"
]

print("="*60)
print("DIAGNOSE: Why component head is dying")
print("="*60)

y_p_train = data['y_p_train']    # (N, 22) multi-label

# Handle case where component_names might exist or not
if 'component_names' in data:
    component_names = list(data['component_names'])
    print(f"\n✓ Loaded component_names from NPZ ({len(component_names)} items)")
else:
    # Slice to match y_p_train shape (22 slots, not 46 sensors)
    # The 22 slots are a SUBSET of the 46 sensors
    # We need to know WHICH 22. Common SWaT mapping:
    component_names = [
        "MV101", "MV201", "MV301", "MV302", "MV303", "MV304",
        "MV501", "MV502", "MV503", "MV504",
        "P101", "P102", "P201", "P202", "P203", "P204", "P205", "P206",
        "LIT101", "LIT601", "DPIT301", "AIT402"
    ]
    print(f"\n⚠ Using hardcoded 22-slot mapping (NPZ missing 'component_names')")

print(f"\nShape: {y_p_train.shape}")
print(f"Total component labels: {int(y_p_train.sum()):,}")
print(f"Total samples: {len(y_p_train):,}")

print(f"\nComponent activation rates (% of windows where each is involved):")
print(f"  {'Slot':>4} {'Component':>12s} {'Count':>6s} {'Rate %':>8s} {'Status':>12s}")
print(f"  {'-'*50}")

train_active = set()
train_dead = set()

for i, name in enumerate(component_names):
    count = int(y_p_train[:, i].sum())
    rate = count / len(y_p_train) * 100
    if count == 0:
        status = "NEVER ←"
        train_dead.add(i)
    else:
        status = "ACTIVE"
        train_active.add(i)
    print(f"  {i:>4} {name:>12s} {count:>6,} {rate:>7.3f}% {status:>12s}")

# Total positive labels per sample
positives_per_sample = y_p_train.sum(axis=1)
print(f"\nLabels per sample:")
print(f"  0 components:  {int((positives_per_sample == 0).sum()):,}  ({100*(positives_per_sample == 0).sum()/len(y_p_train):.1f}%)")
print(f"  1 component :  {int((positives_per_sample == 1).sum()):,}  ({100*(positives_per_sample == 1).sum()/len(y_p_train):.1f}%)")
print(f"  2+ components: {int((positives_per_sample >= 2).sum()):,}  ({100*(positives_per_sample >= 2).sum()/len(y_p_train):.1f}%)")

# ── VAL SET ──
print(f"\n{'─'*50}")
print("VAL SET")
print(f"{'─'*50}")
y_p_val = data['y_p_val']
val_active = set()
for i, name in enumerate(component_names):
    count = int(y_p_val[:, i].sum())
    if count > 0:
        val_active.add(i)
        print(f"  Slot {i:>2} {name:>12s}: {count:>5,} windows")

# ── TEST SET ──
print(f"\n{'─'*50}")
print("TEST SET")
print(f"{'─'*50}")
y_p_test = data['y_p_test']
test_active = set()
for i, name in enumerate(component_names):
    count = int(y_p_test[:, i].sum())
    if count > 0:
        test_active.add(i)
        print(f"  Slot {i:>2} {name:>12s}: {count:>5,} windows")

# ── CROSS-SPLIT ANALYSIS ──
print(f"\n{'='*60}")
print("CROSS-SPLIT MISMATCH ANALYSIS")
print(f"{'='*60}")

print(f"\n  Active in TRAIN: {sorted(train_active)}")
print(f"  Active in VAL:   {sorted(val_active)}")
print(f"  Active in TEST:  {sorted(test_active)}")

test_unseen = test_active - train_active
val_unseen = val_active - train_active

if test_unseen:
    print(f"\n  🔴 CRITICAL: Test has {len(test_unseen)} UNSEEN components:")
    for slot in sorted(test_unseen):
        print(f"     Slot {slot}: {component_names[slot]} — NEVER seen in training!")
    print(f"\n  → Component head CANNOT predict these. Use attention attribution instead.")
else:
    print(f"\n  ✓ No unseen components in test.")

if val_unseen:
    print(f"\n  🟡 WARNING: Val has {len(val_unseen)} unseen components:")
    for slot in sorted(val_unseen):
        print(f"     Slot {slot}: {component_names[slot]}")

print(f"\n{'='*60}")