# evaluate_zero_shot.py
# Zero-Shot Component Localization via Attention Attribution
# Evaluates whether sensor_imp correctly localizes UNSEEN components
# Run AFTER training completes: python evaluate_zero_shot.py

import numpy as np
import torch
import sys
import os
import json
from collections import defaultdict

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel
from sensor_component_topology import (
    COMPONENT_NAMES, UNSEEN_COMPONENTS, SEEN_COMPONENTS,
    get_adjacent_sensors, get_component_name, get_sensor_name, is_unseen
)

# ── Config ──
DATA_PATH   = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
CHECKPOINT  = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
OUTPUT_DIR  = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
TOP_K = 5  # How many top sensors to consider

# ── Load Model ──
print("="*70)
print("TrustGate — Zero-Shot Component Localization Evaluation")
print("="*70)

print(f"\n[1/5] Loading model from {CHECKPOINT}...")
ckpt = torch.load(CHECKPOINT, map_location=device)
model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  ✓ Model loaded (epoch {ckpt['epoch']}, val_auc={ckpt['monitor_value']:.4f})")

# ── Load Data ──
print(f"\n[2/5] Loading test data...")
data = np.load(DATA_PATH, allow_pickle=True)

X_s_test = torch.FloatTensor(data['X_s_test'])
X_n_test = torch.FloatTensor(data['X_n_test'])
y_b_test = data['y_b_test'].astype(int)
y_p_test = data['y_p_test'].astype(float)

print(f"  Test samples: {len(y_b_test):,}")
print(f"  Attack windows: {y_b_test.sum():,}")
print(f"  Unseen components in test: {[get_component_name(s) for s in UNSEEN_COMPONENTS]}")

# ── Collect Predictions + sensor_imp ──
print(f"\n[3/5] Running inference to collect sensor_imp...")

all_sensor_imp = []
all_bin_probs = []
all_labels = []
all_comp_labels = []

with torch.no_grad():
    for i in range(0, len(X_s_test), BATCH_SIZE):
        xs = X_s_test[i:i+BATCH_SIZE].to(device)
        xn = X_n_test[i:i+BATCH_SIZE].to(device)
        
        bin_logit, cls_logits, comp_logits, attn_s2n, attn_n2s, sensor_imp = model(xs, xn)
        
        all_bin_probs.append(torch.sigmoid(bin_logit.squeeze(-1)).cpu().numpy())
        all_sensor_imp.append(sensor_imp.cpu().numpy())
        all_labels.append(y_b_test[i:i+BATCH_SIZE])
        all_comp_labels.append(y_p_test[i:i+BATCH_SIZE])

bin_probs = np.concatenate(all_bin_probs)
sensor_imp = np.concatenate(all_sensor_imp)  # (N_test, 71)
labels = np.concatenate(all_labels)
comp_labels = np.concatenate(all_comp_labels)  # (N_test, 22)

# ── Evaluate Zero-Shot Localization ──
print(f"\n[4/5] Evaluating zero-shot localization...")

def evaluate_localization(component_slot, threshold=0.5):
    """
    For a given component slot, find all test windows where that component
    is attacked (comp_labels[:, slot] == 1). Then check if sensor_imp
    ranks the adjacent sensors in the top-K.
    """
    # Find windows where this component is attacked
    attacked_mask = comp_labels[:, component_slot] == 1
    attacked_indices = np.where(attacked_mask)[0]
    
    if len(attacked_indices) == 0:
        return None  # No attacks on this component in test
    
    # Get adjacent sensors for this component
    adjacent_sensors = get_adjacent_sensors(component_slot)
    if not adjacent_sensors:
        return None
    
    # For each attacked window, check if adjacent sensors are in top-K
    hits = 0
    total = 0
    mrr_sum = 0.0  # Mean Reciprocal Rank
    
    for idx in attacked_indices:
        imp = sensor_imp[idx]  # (71,) importance scores
        top_k_sensors = np.argsort(imp)[::-1][:TOP_K]
        
        # Check if ANY adjacent sensor is in top-K
        hit = any(s in top_k_sensors for s in adjacent_sensors)
        if hit:
            hits += 1
        
        # Reciprocal Rank: 1 / rank of first adjacent sensor
        ranks = []
        for s in adjacent_sensors:
            rank = np.where(np.argsort(imp)[::-1] == s)[0]
            if len(rank) > 0:
                ranks.append(rank[0] + 1)  # 1-indexed
        if ranks:
            mrr_sum += 1.0 / min(ranks)
        
        total += 1
    
    precision_at_k = hits / total if total > 0 else 0.0
    mrr = mrr_sum / total if total > 0 else 0.0
    
    return {
        'component': get_component_name(component_slot),
        'slot': component_slot,
        'n_attacks': total,
        'precision_at_k': precision_at_k,
        'mrr': mrr,
        'is_unseen': is_unseen(component_slot),
        'adjacent_sensors': adjacent_sensors,
    }

# Evaluate all components present in test set
results = []
for slot in range(22):
    if comp_labels[:, slot].sum() > 0:  # Only evaluate if component appears in test
        r = evaluate_localization(slot)
        if r:
            results.append(r)

# ── Report Results ──
print(f"\n[5/5] Results (Top-{TOP_K} evaluation):")
print(f"\n{'='*70}")
print(f"{'Slot':>4} | {'Component':>8} | {'Seen?':>6} | {'N':>5} | {'P@K':>6} | {'MRR':>6}")
print(f"{'-'*70}")

seen_results = []
unseen_results = []

for r in results:
    seen_str = "SEEN" if not r['is_unseen'] else "UNSEEN"
    print(f"{r['slot']:>4} | {r['component']:>8} | {seen_str:>6} | "
          f"{r['n_attacks']:>5} | {r['precision_at_k']:>6.3f} | {r['mrr']:>6.3f}")
    
    if r['is_unseen']:
        unseen_results.append(r)
    else:
        seen_results.append(r)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

if seen_results:
    avg_seen_pk = np.mean([r['precision_at_k'] for r in seen_results])
    avg_seen_mrr = np.mean([r['mrr'] for r in seen_results])
    print(f"\nSEEN Components (n={len(seen_results)}):")
    print(f"  Avg Precision@{TOP_K}: {avg_seen_pk:.3f}")
    print(f"  Avg MRR:              {avg_seen_mrr:.3f}")

if unseen_results:
    avg_unseen_pk = np.mean([r['precision_at_k'] for r in unseen_results])
    avg_unseen_mrr = np.mean([r['mrr'] for r in unseen_results])
    print(f"\nUNSEEN Components (n={len(unseen_results)}) — ZERO-SHOT:")
    print(f"  Avg Precision@{TOP_K}: {avg_unseen_pk:.3f}")
    print(f"  Avg MRR:              {avg_unseen_mrr:.3f}")
    print(f"\n  🎯 Paper Claim: Model localizes {len(unseen_results)} unseen components")
    print(f"     without ever training on their labels.")

# Save results
results_path = os.path.join(OUTPUT_DIR, 'zero_shot_results.json')
with open(results_path, 'w') as f:
    json.dump({
        'top_k': TOP_K,
        'seen': seen_results,
        'unseen': unseen_results,
    }, f, indent=2)
print(f"\n  Saved: {results_path}")

print(f"\n{'='*70}")
print("Zero-shot evaluation complete.")
print(f"{'='*70}")