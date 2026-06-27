# debug_data.py
# Save as: C:\Users\HP\Downloads\trustgate\debug_data.py
# Run: python debug_data.py

import numpy as np
import torch

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3.npz'

print("=" * 60)
print("TrustGate — Data Integrity Diagnostic")
print("=" * 60)

data = np.load(DATA_PATH, allow_pickle=True)

print(f"\nKeys in NPZ: {list(data.keys())}")

X_s = data['X_s_train']
X_n = data['X_n_train']
y_b = data['y_b_train']
y_c = data['y_c_train']

print(f"\n── Shape Check ──────────────────────────────────────")
print(f"  X_s_train : {X_s.shape}   (expect: N x 30 x 71)")
print(f"  X_n_train : {X_n.shape}   (expect: N x 30 x 19)")
print(f"  y_b_train : {y_b.shape}   (expect: N,)")
print(f"  y_c_train : {y_c.shape}   (expect: N,)")

print(f"\n── Label Distribution ───────────────────────────────")
unique, counts = np.unique(y_b, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  y_b = {u}: {c:,} samples ({100*c/len(y_b):.1f}%)")

unique_c, counts_c = np.unique(y_c, return_counts=True)
print(f"\n  Class distribution (y_c):")
for u, c in zip(unique_c, counts_c):
    print(f"  class {u}: {c:,} samples ({100*c/len(y_c):.1f}%)")

print(f"\n── NaN / Inf Check ──────────────────────────────────")
print(f"  X_s NaN  : {np.isnan(X_s).sum():,}")
print(f"  X_s Inf  : {np.isinf(X_s).sum():,}")
print(f"  X_n NaN  : {np.isnan(X_n).sum():,}")
print(f"  X_n Inf  : {np.isinf(X_n).sum():,}")
print(f"  y_b NaN  : {np.isnan(y_b.astype(float)).sum():,}")

print(f"\n── Value Range Check ────────────────────────────────")
print(f"  X_s  min={X_s.min():.4f}  max={X_s.max():.4f}  "
      f"mean={X_s.mean():.4f}  std={X_s.std():.4f}")
print(f"  X_n  min={X_n.min():.4f}  max={X_n.max():.4f}  "
      f"mean={X_n.mean():.4f}  std={X_n.std():.4f}")

print(f"\n── Scaling Check (should be ~N(0,1) for good training) ──")
for stream, X, name in [(X_s, X_s, "sensor"), (X_n, X_n, "network")]:
    flat = X.reshape(-1, X.shape[-1])
    col_means = flat.mean(axis=0)
    col_stds  = flat.std(axis=0)
    bad_mean  = (np.abs(col_means) > 3).sum()
    bad_std   = ((col_stds < 0.01) | (col_stds > 100)).sum()
    print(f"\n  {name} stream:")
    print(f"    Column means  — min:{col_means.min():.3f}  "
          f"max:{col_means.max():.3f}  "
          f"abs>3: {bad_mean} cols")
    print(f"    Column stds   — min:{col_stds.min():.4f}  "
          f"max:{col_stds.max():.4f}  "
          f"bad: {bad_std} cols")
    if bad_mean > 0:
        bad_idx = np.where(np.abs(col_means) > 3)[0]
        print(f"    BAD MEAN cols: {bad_idx.tolist()}")
    if bad_std > 0:
        bad_idx = np.where((col_stds < 0.01) | (col_stds > 100))[0]
        print(f"    BAD STD cols : {bad_idx.tolist()}")

print(f"\n── Window Sanity Check ──────────────────────────────")
print(f"  Checking 5 random attack windows vs 5 normal windows...")
attack_idx  = np.where(y_b == 1)[0]
normal_idx  = np.where(y_b == 0)[0]

print(f"\n  Attack windows (y_b=1) — sensor stream mean per window:")
for idx in attack_idx[:5]:
    print(f"    window {idx:5d}: mean={X_s[idx].mean():.4f}  "
          f"std={X_s[idx].std():.4f}  "
          f"class={y_c[idx]}")

print(f"\n  Normal windows (y_b=0) — sensor stream mean per window:")
for idx in normal_idx[:5]:
    print(f"    window {idx:5d}: mean={X_s[idx].mean():.4f}  "
          f"std={X_s[idx].std():.4f}  "
          f"class={y_c[idx]}")

print(f"\n── Separability Check ───────────────────────────────")
attack_means = X_s[attack_idx].mean(axis=(0, 1))
normal_means = X_s[normal_idx].mean(axis=(0, 1))
diff = np.abs(attack_means - normal_means)
top5 = np.argsort(diff)[::-1][:5]

sensor_cols = list(data['sensor_cols']) if 'sensor_cols' in data else \
              [f"sensor_{i}" for i in range(X_s.shape[-1])]

print(f"\n  Top 5 most separating sensor features:")
for rank, idx in enumerate(top5):
    print(f"    [{rank+1}] col {idx:2d} ({sensor_cols[idx]:20s}): "
          f"normal={normal_means[idx]:.4f}  "
          f"attack={attack_means[idx]:.4f}  "
          f"diff={diff[idx]:.4f}")

zero_diff = (diff < 0.001).sum()
print(f"\n  Columns with near-zero separation: {zero_diff}/{len(diff)}")
if zero_diff > len(diff) * 0.5:
    print(f"  WARNING: More than 50% of features have no separation.")
    print(f"  This means normal and attack windows look identical to the model.")
    print(f"  → Data scaling or label alignment is likely wrong.")

print(f"\n── Val Set Check ────────────────────────────────────")
y_b_val = data['y_b_val']
unique_v, counts_v = np.unique(y_b_val, return_counts=True)
print(f"  Val set size: {len(y_b_val):,}")
for u, c in zip(unique_v, counts_v):
    print(f"  y_b_val = {u}: {c:,} ({100*c/len(y_b_val):.1f}%)")

if len(unique_v) < 2:
    print(f"\n  CRITICAL: Val set has only ONE class!")
    print(f"  AUC is undefined — this is why AUC oscillates randomly.")

print(f"\n{'='*60}")
print("Diagnostic complete. Paste full output to get fix.")
print("=" * 60)