# quick_check.py
# Run: python quick_check.py
# Takes 10 seconds, confirms training is safe to start

import numpy as np

data = np.load(r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz', allow_pickle=True)
sensor_cols = list(data['sensor_cols'])

X_s     = data['X_s_train']
y_b     = data['y_b_train']

attack_idx = np.where(y_b == 1)[0]
normal_idx = np.where(y_b == 0)[0]

att_means = X_s[attack_idx].mean(axis=(0,1))
nor_means = X_s[normal_idx].mean(axis=(0,1))
diff      = np.abs(att_means - nor_means)

att_stds  = X_s[attack_idx].std(axis=(0,1))
nor_stds  = X_s[normal_idx].std(axis=(0,1))

print("=" * 65)
print("Column-Level Separation Report")
print("=" * 65)
print(f"\n{'Idx':>4}  {'Sensor':25s}  {'NorMean':>8}  "
      f"{'AttMean':>8}  {'Diff':>7}  {'Status'}")
print("-" * 65)

dead_cols   = []
weak_cols   = []
strong_cols = []

for i, name in enumerate(sensor_cols):
    d = diff[i]
    if d < 0.05:
        status = "DEAD"
        dead_cols.append(i)
    elif d < 0.3:
        status = "weak"
        weak_cols.append(i)
    else:
        status = "STRONG"
        strong_cols.append(i)

    if d < 0.05 or d > 0.5:   # only print notable ones
        print(f"  {i:2d}  {name:25s}  {nor_means[i]:>8.4f}  "
              f"{att_means[i]:>8.4f}  {d:>7.4f}  {status}")

print(f"\nSummary:")
print(f"  STRONG (diff > 0.30) : {len(strong_cols):2d} columns → "
      f"{strong_cols}")
print(f"  weak   (0.05–0.30)   : {len(weak_cols):2d} columns")
print(f"  DEAD   (diff < 0.05) : {len(dead_cols):2d} columns → "
      f"{dead_cols}")

print(f"\nDead column names:")
for i in dead_cols:
    print(f"  col {i:2d}: {sensor_cols[i]}")

# Check val set has both classes
y_b_val = data['y_b_val']
print(f"\nVal set: {(y_b_val==0).sum()} normal, "
      f"{(y_b_val==1).sum()} attack")
print(f"  → AUC is computable: {len(np.unique(y_b_val)) == 2}")

# Check network stream too
X_n     = data['X_n_train']
net_cols = list(data['network_cols'])
att_n   = X_n[attack_idx].mean(axis=(0,1))
nor_n   = X_n[normal_idx].mean(axis=(0,1))
diff_n  = np.abs(att_n - nor_n)

print(f"\nNetwork stream separation:")
print(f"{'Idx':>4}  {'Feature':25s}  {'Diff':>7}  Status")
print("-" * 45)
for i, name in enumerate(net_cols):
    d = diff_n[i]
    status = "STRONG" if d > 0.3 else ("weak" if d > 0.05 else "DEAD")
    print(f"  {i:2d}  {name:25s}  {d:>7.4f}  {status}")

print(f"\n{'='*65}")
print("If val AUC is computable and STRONG cols > 10:")
print("→ Safe to start training now.")
print("=" * 65)