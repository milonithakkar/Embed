# check_mv302_sensors.py
# Find which sensors actually change during MV302 attacks

import numpy as np

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
data      = np.load(DATA_PATH, allow_pickle=True)

X_s_test  = data['X_s_test']
y_p_test  = data['y_p_test'].astype(float)
X_s_train = data['X_s_train']
y_b_train = data['y_b_train'].astype(int)
sensor_cols = list(data['sensor_cols'])

# Get normal baseline from training
normal_mask = (y_b_train == 0)
baseline    = X_s_train[normal_mask].mean(axis=(0, 1))

# Get MV302 attack windows
mv302_mask = y_p_test[:, 3] == 1
X_mv302    = X_s_test[mv302_mask]   # (300, 30, 71)

# Per-sensor deviation during MV302 attacks
mv302_mean = X_mv302.mean(axis=(0, 1))   # (71,)
deviation  = np.abs(mv302_mean - baseline)

# Rank sensors by deviation
ranked = np.argsort(deviation)[::-1]

print("=" * 60)
print("Sensors that ACTUALLY change during MV302 attacks")
print("=" * 60)
print(f"\n{'Rank':>4}  {'Sensor':25s}  "
      f"{'Normal':>8}  {'Attack':>8}  {'Diff':>8}")
print("-" * 60)

for rank, idx in enumerate(ranked[:20]):
    print(f"  {rank+1:2d}  {sensor_cols[idx]:25s}  "
          f"{baseline[idx]:>8.4f}  "
          f"{mv302_mean[idx]:>8.4f}  "
          f"{deviation[idx]:>8.4f}")

print(f"\nThese are the TRUE adjacent sensors for MV302")
print(f"Update COMPONENT_TO_SENSORS with top 8 indices:")
print(f"\nMV302 true adjacent: {list(ranked[:8])}")
print(f"Sensor names: {[sensor_cols[i] for i in ranked[:8]]}")

# Same for AIT402 (our working case — verify it)
ait402_mask = y_p_test[:, 21] == 1
X_ait402    = X_s_test[ait402_mask]
ait402_mean = X_ait402.mean(axis=(0,1))
dev_ait402  = np.abs(ait402_mean - baseline)
ranked_402  = np.argsort(dev_ait402)[::-1]

print(f"\n{'='*60}")
print(f"AIT402 actual deviating sensors (verify our mapping)")
print(f"{'='*60}")
for rank, idx in enumerate(ranked_402[:10]):
    print(f"  {rank+1:2d}  {sensor_cols[idx]:25s}  "
          f"diff={dev_ait402[idx]:.4f}")

print(f"\nAIT402 true adjacent: {list(ranked_402[:10])}")