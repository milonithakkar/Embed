"""
TrustGate — Data Loader Verification
Targets swat_final.npz specifically
Run from: /opt/trustgate/ai_engine/
"""

import numpy as np
import os, sys

# ── Config ─────────────────────────────────────────────────────────
DATA_PATH  = './trustgate_data/swat_final.npz'
BATCH_SIZE = 256

print("=" * 60)
print("TrustGate — Verifying swat_final.npz")
print("=" * 60)

# ── Check 1: File exists ───────────────────────────────────────────
print("\n[1/6] File check...")
if not os.path.exists(DATA_PATH):
    print(f"  [FAIL] Not found: {DATA_PATH}")
    sys.exit(1)

size_mb = os.path.getsize(DATA_PATH) / (1024**2)
print(f"  [OK] Found — {size_mb:.1f} MB")

# ── Check 2: Load + keys ───────────────────────────────────────────
print("\n[2/6] Loading and checking keys...")
data = np.load(DATA_PATH, allow_pickle=True)
found = list(data.keys())
print(f"  Keys: {found}")

expected = [
    'X_s_train','X_n_train','y_train',
    'X_s_val',  'X_n_val',  'y_val',
    'X_s_test', 'X_n_test', 'y_test',
    'sensor_cols'
]
missing = [k for k in expected if k not in found]
if missing:
    print(f"  [FAIL] Missing: {missing}")
    sys.exit(1)
print(f"  [OK] All keys present")

# ── Check 3: Shapes ────────────────────────────────────────────────
print("\n[3/6] Shape verification...")
splits = {
    'train': (data['X_s_train'], data['X_n_train'], data['y_train']),
    'val':   (data['X_s_val'],   data['X_n_val'],   data['y_val']),
    'test':  (data['X_s_test'],  data['X_n_test'],  data['y_test']),
}

all_ok = True
for name, (xs, xn, y) in splits.items():
    s_ok = (xs.ndim==3 and xs.shape[1]==30 and xs.shape[2]==44)
    n_ok = (xn.ndim==3 and xn.shape[1]==30 and xn.shape[2]==132)
    y_ok = (y.ndim==1)
    status = "[OK]" if (s_ok and n_ok and y_ok) else "[FAIL]"
    if not (s_ok and n_ok and y_ok):
        all_ok = False
    print(f"  {status} {name:>5}: "
          f"X_s={xs.shape} "
          f"X_n={xn.shape} "
          f"y={y.shape}")

if all_ok:
    print(f"  [OK] All shapes correct (N, 30, 44) and (N, 30, 132)")
else:
    print(f"  [FAIL] Shape mismatch — paste output before continuing")
    sys.exit(1)

# ── Check 4: NaN / Inf ────────────────────────────────────────────
print("\n[4/6] NaN and Inf check...")
clean = True
for name, (xs, xn, y) in splits.items():
    issues = {
        'xs_nan': int(np.isnan(xs).sum()),
        'xn_nan': int(np.isnan(xn).sum()),
        'xs_inf': int(np.isinf(xs).sum()),
        'xn_inf': int(np.isinf(xn).sum()),
    }
    total_issues = sum(issues.values())
    if total_issues > 0:
        print(f"  [FAIL] {name}: {issues}")
        clean = False
    else:
        print(f"  [OK]   {name}: clean")

if not clean:
    print("\n  [ACTION] NaNs found — tell me and we add a fix step")
else:
    print("  [OK] Zero NaN/Inf across all splits")

# ── Check 5: Label distribution ───────────────────────────────────
print("\n[5/6] Label distribution...")
print(f"  {'Split':>5} | {'Windows':>8} | {'Normal':>8} | {'Attack':>8} | {'Atk%':>6} | {'Ratio':>8}")
print(f"  {'-'*55}")
for name, (xs, xn, y) in splits.items():
    total  = len(y)
    atk    = int(y.sum())
    norm   = total - atk
    pct    = atk / total * 100
    ratio  = norm / atk if atk > 0 else float('inf')
    print(f"  {name:>5} | {total:>8,} | {norm:>8,} | "
          f"{atk:>8,} | {pct:>5.1f}% | {ratio:>6.1f}:1")

# ── Check 6: Value ranges + RAM ───────────────────────────────────
print("\n[6/6] Value ranges and memory...")
xs_tr = data['X_s_train']
xn_tr = data['X_n_train']

print(f"  Sensor  — min:{xs_tr.min():>8.4f} | max:{xs_tr.max():>8.4f} | mean:{xs_tr.mean():>8.4f}")
print(f"  Network — min:{xn_tr.min():>8.4f} | max:{xn_tr.max():>8.4f} | mean:{xn_tr.mean():>8.4f}")
print(f"  [INFO] Network going negative = EXPECTED (ROC + deviation features)")

# RAM
total_mb = sum(
    data[k].nbytes for k in expected if k != 'sensor_cols'
) / (1024**2)

# Single batch VRAM
batch_s_mb  = (BATCH_SIZE * 30 * 44  * 4) / (1024**2)
batch_n_mb  = (BATCH_SIZE * 30 * 132 * 4) / (1024**2)
batch_total = batch_s_mb + batch_n_mb

print(f"\n  RAM to hold full dataset:     {total_mb:.0f} MB ({total_mb/1024:.2f} GB)")
print(f"  VRAM per batch (size={BATCH_SIZE}):")
print(f"    Sensor  batch: {batch_s_mb:.1f} MB")
print(f"    Network batch: {batch_n_mb:.1f} MB")
print(f"    Total + grads: ~{batch_total*3 + 200:.0f} MB  ← RTX 3050 safe zone")

# Sensor columns
sensor_cols = list(data['sensor_cols'])
print(f"\n  Sensor columns ({len(sensor_cols)}): {sensor_cols}")

# ── Final verdict ─────────────────────────────────────────────────
print("\n" + "="*60)
print("VERIFICATION COMPLETE — paste full output before model step")
print("="*60)
