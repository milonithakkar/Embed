# diagnose_v1.py
import numpy as np

data = np.load(r'D:\trustgate_pcaps\A12_windowed.npz', allow_pickle=True)

# Get attack windows only
train_attacks_s = data['X_s_train'][data['y_b_train'] == 1]
train_attacks_n = data['X_n_train'][data['y_b_train'] == 1]
val_attacks_s   = data['X_s_val'][data['y_b_val'] == 1]
val_attacks_n   = data['X_n_val'][data['y_b_val'] == 1]

# Get normal windows
train_normal_s = data['X_s_train'][data['y_b_train'] == 0]
val_normal_s   = data['X_s_val'][data['y_b_val'] == 0]

print("="*60)
print("DIAGNOSIS: Are val attacks similar to train attacks?")
print("="*60)

# Compare means
print(f"\nFeature means comparison:")
print(f"  {'Stream':10s} {'TrainAtk':>12s} {'ValAtk':>12s} {'TrainNorm':>12s} {'ValNorm':>12s}")

# Sensor
print(f"  {'Sensor':10s} "
      f"{train_attacks_s.mean():>12.4f} "
      f"{val_attacks_s.mean():>12.4f} "
      f"{train_normal_s.mean():>12.4f} "
      f"{val_normal_s.mean():>12.4f}")

# Network
val_normal_n = data['X_n_val'][data['y_b_val'] == 0]
train_normal_n = data['X_n_train'][data['y_b_train'] == 0]
print(f"  {'Network':10s} "
      f"{train_attacks_n.mean():>12.4f} "
      f"{val_attacks_n.mean():>12.4f} "
      f"{train_normal_n.mean():>12.4f} "
      f"{val_normal_n.mean():>12.4f}")

# Distance: train attacks to val attacks (should be small for good generalization)
from scipy.spatial.distance import cdist
train_atk_centroid = train_attacks_s.mean(axis=(0,1))
val_atk_centroid   = val_attacks_s.mean(axis=(0,1))
train_norm_centroid = train_normal_s.mean(axis=(0,1))
val_norm_centroid   = val_normal_s.mean(axis=(0,1))

print(f"\nSensor centroid distances (cosine):")
def cos_dist(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
print(f"  Train_atk vs Val_atk    : {cos_dist(train_atk_centroid, val_atk_centroid):.4f}")
print(f"  Train_norm vs Val_norm  : {cos_dist(train_norm_centroid, val_norm_centroid):.4f}")
print(f"  Train_atk vs Train_norm : {cos_dist(train_atk_centroid, train_norm_centroid):.4f}")
print(f"  Val_atk vs Val_norm     : {cos_dist(val_atk_centroid, val_norm_centroid):.4f}")

print(f"\nINTERPRETATION:")
print(f"  If train_atk vs val_atk distance > val_atk vs val_norm distance:")
print(f"    → Val attacks look MORE like val normal than train attacks")
print(f"    → Model trained on train attacks CANNOT generalize")
print(f"    → This is the core problem")