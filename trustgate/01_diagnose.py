# 01_diagnose.py
# PURPOSE: Understand exactly what is in your data before evaluating.
# Tells you: split sizes, class balance, component distribution,
#            which components appear in train vs val vs test.
# Run: python 01_diagnose.py

import numpy as np
import os

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3.npz'

print("=" * 60)
print("TrustGate — Data Diagnostic")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────
print(f"\nLoading {DATA_PATH}...")
data = np.load(DATA_PATH, allow_pickle=True)

print(f"\nAll keys in NPZ file:")
for key in sorted(data.keys()):
    arr = data[key]
    print(f"  {key:25s} shape={arr.shape}  dtype={arr.dtype}")

# ── Helper ────────────────────────────────────────────────────
def analyze_split(data, split):
    print(f"\n{'─'*50}")
    print(f"  SPLIT: {split.upper()}")
    print(f"{'─'*50}")

    # Try both naming conventions
    def get(name_a, name_b=None):
        if name_a in data:
            return data[name_a]
        if name_b and name_b in data:
            return data[name_b]
        return None

    X_s  = get(f'X_s_{split}',  f'{split}_X_sensor')
    X_n  = get(f'X_n_{split}',  f'{split}_X_network')
    y_b  = get(f'y_b_{split}',  f'{split}_y_binary')
    y_c  = get(f'y_c_{split}',  f'{split}_y_class')
    y_p  = get(f'y_p_{split}',  f'{split}_y_comp')

    if y_b is None:
        print(f"  ✗ No data found for split '{split}'")
        return

    print(f"\n  Shapes:")
    print(f"    X_sensor  : {X_s.shape if X_s is not None else 'NOT FOUND'}")
    print(f"    X_network : {X_n.shape if X_n is not None else 'NOT FOUND'}")
    print(f"    y_binary  : {y_b.shape}")
    print(f"    y_class   : {y_c.shape if y_c is not None else 'NOT FOUND'}")
    print(f"    y_comp    : {y_p.shape if y_p is not None else 'NOT FOUND'}")

    # Binary stats
    n_total  = len(y_b)
    n_attack = int(y_b.sum())
    n_normal = n_total - n_attack
    print(f"\n  Binary Labels:")
    print(f"    Total     : {n_total:,}")
    print(f"    Normal    : {n_normal:,}  ({100*n_normal/n_total:.1f}%)")
    print(f"    Attack    : {n_attack:,}  ({100*n_attack/n_total:.1f}%)")
    print(f"    Imbalance : {n_normal/max(n_attack,1):.1f}:1 (normal:attack)")

    # Class distribution
    if y_c is not None:
        unique_classes, counts = np.unique(y_c, return_counts=True)
        print(f"\n  Attack Classes:")
        for cls, cnt in zip(unique_classes, counts):
            print(f"    Class {cls}: {cnt:,} windows")

    # Component distribution — THE CRITICAL DIAGNOSTIC
    if y_p is not None:
        n_components = y_p.shape[1]
        print(f"\n  Component Slots ({n_components} total):")
        print(f"  {'Slot':>4} | {'Active Windows':>14} | {'% of Attacks':>12} | Status")
        print(f"  {'─'*55}")

        active_slots  = []
        inactive_slots = []

        for slot in range(n_components):
            slot_active = int(y_p[:, slot].sum())
            pct = 100 * slot_active / max(n_attack, 1)

            if slot_active > 0:
                status = "ACTIVE"
                active_slots.append(slot)
            else:
                status = "DEAD (always 0)"
                inactive_slots.append(slot)

            indicator = "✓" if slot_active > 0 else "✗"
            print(f"  {slot:>4} | {slot_active:>14,} | {pct:>11.1f}% | "
                  f"{indicator} {status}")

        print(f"\n  Summary:")
        print(f"    Active slots   : {len(active_slots)}  → {active_slots}")
        print(f"    Dead slots     : {len(inactive_slots)} → {inactive_slots}")

    return {
        'n_total': n_total,
        'n_attack': n_attack,
        'active_slots': active_slots if y_p is not None else [],
        'dead_slots': inactive_slots if y_p is not None else [],
    }

# ── Analyze all splits ─────────────────────────────────────────
results = {}
for split in ['train', 'val', 'test']:
    results[split] = analyze_split(data, split)

# ── Cross-split comparison ─────────────────────────────────────
print(f"\n{'='*60}")
print("  CROSS-SPLIT COMPONENT MISMATCH ANALYSIS")
print(f"{'='*60}")

if all(r is not None for r in results.values()):
    train_active = set(results['train']['active_slots'])
    val_active   = set(results['val']['active_slots'])
    test_active  = set(results['test']['active_slots'])

    print(f"\n  Train-only components  : {sorted(train_active - val_active - test_active)}")
    print(f"  Val-only components    : {sorted(val_active - train_active - test_active)}")
    print(f"  Test-only components   : {sorted(test_active - train_active - val_active)}")
    print(f"  Shared train+val       : {sorted(train_active & val_active)}")
    print(f"  Shared train+test      : {sorted(train_active & test_active)}")
    print(f"  Shared all three       : {sorted(train_active & val_active & test_active)}")

    test_unseen = test_active - train_active
    if test_unseen:
        print(f"\n  ⚠ WARNING: Test has {len(test_unseen)} components never seen in training!")
        print(f"    These slots will always output 0 (label starvation).")
        print(f"    Unseen slots: {sorted(test_unseen)}")
        print(f"    Solution: Use attention attribution instead of component head.")
    else:
        print(f"\n  ✓ No unseen components in test set.")

print(f"\n{'='*60}")
print("Diagnostic complete. Check results above before evaluating.")
print(f"{'='*60}")