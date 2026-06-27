"""
TrustGate — Label Verification + Sensor Deviation Analysis
Two purposes:
1. Verify binary class distribution across train/val/test splits
2. Find which sensors deviate most during attacks per stage
   → This becomes our attention weight validation later
   → Also confirms our rule-based post-processor logic

Run on LAPTOP in trustgate_data/ folder (needs swat_final.npz + sensor_cols_final.pkl)
"""

import numpy as np
import joblib
import os

print("="*65)
print("TrustGate — Label + Sensor Deviation Verification")
print("="*65)

# ── Load data ──────────────────────────────────────────────────────
data_path = './trustgate_data/swat_final.npz'
cols_path = './trustgate_data/sensor_cols_final.pkl'

if not os.path.exists(data_path):
    print(f"[!] {data_path} not found — run split_final_v2.py first")
    exit(1)

print("\n[1/5] Loading split data...")
data = np.load(data_path, allow_pickle=True)
sensor_cols = joblib.load(cols_path)

X_s_train = data['X_s_train']  # (N, 30, 44) sensor features
X_n_train = data['X_n_train']  # (N, 30, 132) network features
y_train   = data['y_train']

X_s_val   = data['X_s_val']
X_n_val   = data['X_n_val']
y_val     = data['y_val']

X_s_test  = data['X_s_test']
X_n_test  = data['X_n_test']
y_test    = data['y_test']

print(f"  Train: {len(y_train):,} windows | Val: {len(y_val):,} | Test: {len(y_test):,}")
print(f"  Sensor features: {X_s_train.shape[2]} | Network features: {X_n_train.shape[2]}")

# ── Step 2: Binary class distribution ─────────────────────────────
print("\n[2/5] Binary class distribution:")
for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
    n_total  = len(y)
    n_attack = y.sum()
    n_normal = n_total - n_attack
    ratio    = n_normal / max(n_attack, 1)
    flag = "⚠ HIGH IMBALANCE" if ratio > 50 else ("✓" if ratio < 30 else "~ acceptable")
    print(f"  {name:>5}: Normal={n_normal:>7,} | Attack={n_attack:>5,} "
          f"({n_attack/n_total*100:.1f}%) | Ratio={ratio:.0f}:1 {flag}")

# Recommended class weight for loss function
total = len(y_train)
n_pos = y_train.sum()
n_neg = total - n_pos
pos_weight = n_neg / n_pos
print(f"\n  Recommended pos_weight for BCEWithLogitsLoss: {pos_weight:.1f}")
print(f"  (This tells the model: one attack window = {pos_weight:.0f} normal windows)")

# ── Step 3: Sensor deviation during attacks ────────────────────────
print("\n[3/5] Sensor deviation analysis (which sensors change most during attacks)...")
print("  Using TRAINING data only (no test data leakage)")

# For each sensor: compute mean absolute deviation during attack vs normal windows
# Using the middle timestep (t=15) of each window as representative
MID = 15  # middle of 30-timestep window

attack_mask = y_train == 1
normal_mask = y_train == 0

attack_sensor = X_s_train[attack_mask, MID, :]   # (n_attack, 44)
normal_sensor = X_s_train[normal_mask, MID, :]   # (n_normal, 44)

attack_mean = attack_sensor.mean(axis=0)   # (44,)
normal_mean = normal_sensor.mean(axis=0)   # (44,)
attack_std  = attack_sensor.std(axis=0)    # (44,)
normal_std  = normal_sensor.std(axis=0)    # (44,)

# Deviation score = |attack_mean - normal_mean| / (normal_std + 1e-8)
# This is basically a Cohen's d effect size — how many std deviations different
deviation = np.abs(attack_mean - normal_mean) / (normal_std + 1e-8)

# Sort by deviation
sorted_idx = np.argsort(deviation)[::-1]

print(f"\n  Top 15 most deviated sensors during attacks:")
print(f"  {'Rank':>4} | {'Sensor':>10} | {'Deviation Score':>15} | "
      f"{'Normal Mean':>12} | {'Attack Mean':>12} | Stage")
print(f"  {'-'*75}")

# Stage mapping based on sensor prefix
def get_stage(col):
    if any(x in col for x in ['FIT101','LIT101','P101','P102']): return 'S1'
    if any(x in col for x in ['AIT20','FIT201','P20','MV20']): return 'S2'
    if any(x in col for x in ['DPIT','FIT301','LIT301','MV30','P301','P302']): return 'S3'
    if any(x in col for x in ['AIT40','FIT401','LIT401','P40','UV4']): return 'S4'
    if any(x in col for x in ['AIT50','FIT50','P501','P502','PIT5']): return 'S5'
    if any(x in col for x in ['FIT601','P601','P602','P603']): return 'S6'
    return '??'

for rank, idx in enumerate(sorted_idx[:15]):
    col = sensor_cols[idx]
    print(f"  {rank+1:>4} | {col:>10} | {deviation[idx]:>15.3f} | "
          f"{normal_mean[idx]:>12.4f} | {attack_mean[idx]:>12.4f} | {get_stage(col)}")

# ── Step 4: Stage-level deviation for post-processor ──────────────
print("\n[4/5] Stage-level deviation (for attention post-processor logic):")

stage_sensors = {
    'S1_raw_water':    [i for i,c in enumerate(sensor_cols) if any(x in c for x in ['FIT101','LIT101','P101','P102'])],
    'S2_chemical':     [i for i,c in enumerate(sensor_cols) if any(x in c for x in ['AIT20','FIT201','P20'])],
    'S3_filtration':   [i for i,c in enumerate(sensor_cols) if any(x in c for x in ['DPIT','FIT301','LIT301','MV30','P301','P302'])],
    'S4_dechlorin':    [i for i,c in enumerate(sensor_cols) if any(x in c for x in ['AIT40','FIT401','LIT401','P40','UV4'])],
    'S5_pressure':     [i for i,c in enumerate(sensor_cols) if any(x in c for x in ['AIT50','FIT50','P501','P502','PIT5'])],
    'S6_distribution': [i for i,c in enumerate(sensor_cols) if any(x in c for x in ['FIT601','P601','P602','P603'])],
}

print(f"  {'Stage':>20} | {'Sensors':>7} | {'Avg Deviation':>14} | {'Max Sensor':>12}")
print(f"  {'-'*65}")
stage_deviations = {}
for stage, idxs in stage_sensors.items():
    if not idxs:
        continue
    stage_dev = deviation[idxs].mean()
    max_idx   = idxs[np.argmax(deviation[idxs])]
    stage_deviations[stage] = stage_dev
    print(f"  {stage:>20} | {len(idxs):>7} | {stage_dev:>14.3f} | {sensor_cols[max_idx]:>12}")

print(f"\n  → Most attacked stage: {max(stage_deviations, key=stage_deviations.get)}")
print(f"  → This should be Stage 5 (pressure) or Stage 2 (chemical) based on SWaT attacks")

# ── Step 5: Attention post-processor logic preview ─────────────────
print("\n[5/5] Rule-based post-processor preview:")
print("  When model outputs severity > 0.5, attention weights will be checked:")
print("  (These thresholds will be tuned after model training)")
print()

rules = [
    ("CHEMICAL_ATTACK",   "S2_chemical",     0.3, "AIT202 pH or AIT203 ORP deviation → toxic chemical"),
    ("PRESSURE_ATTACK",   "S5_pressure",     0.3, "PIT501-503 pressure anomaly → pipe rupture risk"),
    ("FLOW_TAMPER",       "S1_raw_water",    0.3, "FIT101/LIT101 anomaly → water supply manipulation"),
    ("FILTRATION_ATTACK", "S3_filtration",   0.3, "DPIT301/MV301-304 → filter bypass"),
    ("PUMP_DOS",          None,              0.0, "Multiple stage pumps → operational DoS"),
]

for attack_type, stage, threshold, consequence in rules:
    print(f"  IF attention['{stage}'] > {threshold}:")
    print(f"    → {attack_type}: {consequence}")
    print()

print("="*65)
print("Verification complete.")
print("\nKey outputs for model design:")
print("  1. pos_weight value → use in BCEWithLogitsLoss")
print("  2. Top deviated sensors → validate attention weights after training")
print("  3. Stage deviations → tune post-processor thresholds")
print("="*65)
