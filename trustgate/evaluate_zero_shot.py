# evaluate_zero_shot.py — TrustGate Zero-Shot Localization Evaluation
# Uses sensor_imp attention weights to localize UNSEEN components
# Run: python evaluate_zero_shot.py

import numpy as np
import torch
import sys
import os
import json
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
CKPT_PATH = r'D:\trustgate_pcaps\trained_model_full_v3.pth'  
OUT_DIR   = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_K  = 5   # how many top sensors to check

# ── EXACT SENSOR COLUMNS FROM YOUR NPZ (71 columns) ──────────────────────────
SENSOR_COLUMNS = [
    "P1_STATE",     "LIT101.Pv",    "FIT101.Pv",    "MV101.Status",
    "P101.Status",  "P102.Status",  "P2_STATE",     "FIT201.Pv",
    "AIT201.Pv",    "AIT202.Pv",    "AIT203.Pv",    "MV201.Status",
    "P201.Status",  "P202.Status",  "P203.Status",  "P204.Status",
    "P205.Status",  "P206.Status",  "P207.Status",  "P208.Status",
    "P3_STATE",     "AIT301.Pv",    "AIT302.Pv",    "AIT303.Pv",
    "LIT301.Pv",    "FIT301.Pv",    "DPIT301.Pv",   "MV301.Status",
    "MV302.Status", "MV303.Status", "MV304.Status", "P301.Status",
    "P302.Status",  "P4_STATE",     "LIT401.Pv",    "FIT401.Pv",
    "AIT401.Pv",    "AIT402.Pv",    "P401.Status",  "P402.Status",
    "P403.Status",  "P404.Status",  "UV401.Status", "P5_STATE",
    "FIT501.Pv",    "FIT502.Pv",    "FIT503.Pv",    "FIT504.Pv",
    "AIT501.Pv",    "AIT502.Pv",    "AIT503.Pv",    "AIT504.Pv",
    "PIT501.Pv",    "PIT502.Pv",    "PIT503.Pv",    "P501.Status",
    "P501.Speed",   "P502.Status",  "P502.Speed",   "MV501.Status",
    "MV502.Status", "MV503.Status", "MV504.Status", "P6_STATE",
    "LIT601.Pv",    "LIT602.Pv",    "FIT601.Pv",    "FIT602.Pv",
    "P601.Status",  "P602.Status",  "P603.Status",
]

# ── COMPONENT NAMES (22 slots) ────────────────────────────────────────────────
COMPONENT_NAMES = [
    "MV101",   "MV201",   "MV301",   "MV302",   "MV303",
    "MV304",   "MV501",   "MV502",   "MV503",   "MV504",
    "P101",    "P102",    "P201",    "P202",    "P203",
    "P204",    "P205",    "P206",    "LIT101",  "LIT601",
    "DPIT301", "AIT402",
]

# ── TOPOLOGY: component slot → adjacent sensor indices ────────────────────────
# Based on SWaT plant P&ID using your exact sensor column indices
COMPONENT_TO_SENSORS = {
    # Stage 1
    0:  [0, 1, 2, 3],           # MV101  → P1_STATE, LIT101, FIT101, MV101
    10: [0, 1, 2, 4],           # P101   → P1_STATE, LIT101, FIT101, P101
    11: [0, 1, 2, 5],           # P102   → P1_STATE, LIT101, FIT101, P102
    18: [0, 1, 2, 3, 4, 5],     # LIT101 → all Stage 1

    # Stage 2
    1:  [6, 7, 8, 9, 10, 11],   # MV201  → P2_STATE, FIT201, AIT201-203, MV201
    12: [6, 7, 8, 9, 10, 12],   # P201
    13: [6, 7, 8, 9, 10, 13],   # P202
    14: [6, 7, 8, 9, 10, 14],   # P203
    15: [6, 7, 8, 9, 10, 15],   # P204
    16: [6, 7, 8, 9, 10, 16],   # P205
    17: [6, 7, 8, 9, 10, 17],   # P206

    # Stage 3 UF
    2:  [20, 21, 22, 23, 24, 25, 26, 27],  # MV301 (DEAD)
    3:  [20, 21, 22, 23, 24, 25, 26, 28],  # MV302 ← UNSEEN TEST TARGET
    4:  [20, 21, 22, 23, 24, 25, 26, 29],  # MV303 ← UNSEEN TEST TARGET
    5:  [20, 21, 22, 23, 24, 25, 26, 30],  # MV304 (DEAD)
    20: [20, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # DPIT301

    # Stage 5 Backwash
    6:  [43, 44, 52, 53, 54, 55, 56, 59],  # MV501
    7:  [43, 45, 52, 53, 54, 55, 56, 60],  # MV502
    8:  [43, 46, 52, 53, 54, 57, 58, 61],  # MV503
    9:  [43, 47, 52, 53, 54, 57, 58, 62],  # MV504

    # Stage 6
    19: [63, 64, 65, 66, 67, 68, 69, 70],  # LIT601

    # Stage 4 RO Analyzer
    21: [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],  # AIT402 ← UNSEEN TEST TARGET
}

UNSEEN_SLOTS = [3, 4, 21]   # MV302, MV303, AIT402
SEEN_SLOTS   = [0, 1, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("=" * 65)
print("TrustGate — Zero-Shot Component Localization")
print("=" * 65)
print(f"\nDevice: {device}")

ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Checkpoint: epoch={ckpt['epoch']}, AUC={ckpt['monitor_value']:.4f}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print(f"\nLoading {DATA_PATH}...")
data       = np.load(DATA_PATH, allow_pickle=True)
X_s_test   = data['X_s_test']
X_n_test   = data['X_n_test']
y_b_test   = data['y_b_test'].astype(int)
y_p_test   = data['y_p_test'].astype(float)

print(f"Test samples : {len(y_b_test):,}")
print(f"Test attacks : {y_b_test.sum():,}")
print(f"Unseen targets: {[COMPONENT_NAMES[s] for s in UNSEEN_SLOTS]}")

# ── COLLECT sensor_imp FOR ALL TEST WINDOWS ───────────────────────────────────
print(f"\nRunning inference on test set...")
all_sensor_imp = []
all_bin_probs  = []

ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_s_test),
    torch.FloatTensor(X_n_test)
)
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)

with torch.no_grad():
    for xs, xn in loader:
        xs, xn = xs.to(device), xn.to(device)
        out = model(xs, xn)
        bin_probs  = torch.sigmoid(out[0].squeeze(-1)).cpu().numpy()
        sensor_imp = out[5].cpu().numpy()   # (B, 71)
        all_bin_probs.append(bin_probs)
        all_sensor_imp.append(sensor_imp)

bin_probs  = np.concatenate(all_bin_probs)   # (N_test,)
sensor_imp = np.concatenate(all_sensor_imp)  # (N_test, 71)

print(f"sensor_imp shape : {sensor_imp.shape}")
print(f"sensor_imp range : {sensor_imp.min():.4f} – {sensor_imp.max():.4f}")
print(f"sensor_imp sum   : {sensor_imp.sum(axis=1).mean():.4f} (should be 1.0)")

# ── EVALUATE LOCALIZATION ─────────────────────────────────────────────────────
def evaluate_component(slot, sensor_imp, y_p_test, top_k=5):
    """
    For windows where component[slot] is attacked,
    check if sensor_imp ranks adjacent sensors in top-K.
    """
    attacked = np.where(y_p_test[:, slot] == 1)[0]
    if len(attacked) == 0:
        return None

    adjacent = COMPONENT_TO_SENSORS.get(slot, [])
    if not adjacent:
        return None

    hits      = 0
    mrr_sum   = 0.0
    precision_at_k_sum = 0.0

    for idx in attacked:
        imp          = sensor_imp[idx]               # (71,)
        ranked       = np.argsort(imp)[::-1]         # descending
        top_k_set    = set(ranked[:top_k])
        adjacent_set = set(adjacent)

        # Precision@K: fraction of top-K that are adjacent
        overlap = len(top_k_set & adjacent_set)
        precision_at_k_sum += overlap / top_k

        # Hit@K: at least one adjacent sensor in top-K
        if overlap > 0:
            hits += 1

        # MRR: reciprocal rank of first adjacent sensor
        for rank, sensor_idx in enumerate(ranked):
            if sensor_idx in adjacent_set:
                mrr_sum += 1.0 / (rank + 1)
                break

    n          = len(attacked)
    hit_at_k   = hits / n
    mean_prec  = precision_at_k_sum / n
    mrr        = mrr_sum / n

    return {
        'slot':       slot,
        'component':  COMPONENT_NAMES[slot],
        'n_attacked': n,
        'adjacent':   adjacent,
        'hit_at_k':   hit_at_k,
        'prec_at_k':  mean_prec,
        'mrr':        mrr,
        'is_unseen':  slot in UNSEEN_SLOTS,
    }

# ── RUN FOR ALL ACTIVE TEST COMPONENTS ───────────────────────────────────────
print(f"\n{'='*65}")
print(f"Zero-Shot Localization Results  (Top-{TOP_K})")
print(f"{'='*65}")
print(f"\n{'Slot':>4}  {'Component':>8}  {'Status':>8}  "
      f"{'N':>5}  {'Hit@K':>7}  {'P@K':>7}  {'MRR':>7}")
print(f"{'─'*65}")

results       = []
seen_results  = []
unseen_results= []

for slot in range(22):
    if y_p_test[:, slot].sum() == 0:
        continue
    r = evaluate_component(slot, sensor_imp, y_p_test, TOP_K)
    if r is None:
        continue
    results.append(r)
    status = "UNSEEN" if r['is_unseen'] else "seen"
    print(f"  {slot:2d}  {r['component']:>8}  {status:>8}  "
          f"{r['n_attacked']:>5}  "
          f"{r['hit_at_k']:>7.4f}  "
          f"{r['prec_at_k']:>7.4f}  "
          f"{r['mrr']:>7.4f}")
    if r['is_unseen']:
        unseen_results.append(r)
    else:
        seen_results.append(r)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY")
print(f"{'='*65}")

if seen_results:
    avg_hit  = np.mean([r['hit_at_k']  for r in seen_results])
    avg_prec = np.mean([r['prec_at_k'] for r in seen_results])
    avg_mrr  = np.mean([r['mrr']       for r in seen_results])
    print(f"\nSEEN components (n={len(seen_results)}):")
    print(f"  Avg Hit@{TOP_K}  : {avg_hit:.4f}")
    print(f"  Avg P@{TOP_K}    : {avg_prec:.4f}")
    print(f"  Avg MRR    : {avg_mrr:.4f}")

if unseen_results:
    avg_hit  = np.mean([r['hit_at_k']  for r in unseen_results])
    avg_prec = np.mean([r['prec_at_k'] for r in unseen_results])
    avg_mrr  = np.mean([r['mrr']       for r in unseen_results])
    print(f"\nUNSEEN components — ZERO-SHOT (n={len(unseen_results)}):")
    print(f"  Avg Hit@{TOP_K}  : {avg_hit:.4f}")
    print(f"  Avg P@{TOP_K}    : {avg_prec:.4f}")
    print(f"  Avg MRR    : {avg_mrr:.4f}")
    print(f"\n  Components: {[r['component'] for r in unseen_results]}")
    print(f"\n  Paper claim:")
    print(f"  'TrustGate localizes {len(unseen_results)} unseen components")
    print(f"   (MV302, MV303, AIT402) with Avg P@5={avg_prec:.4f}")
    print(f"   and MRR={avg_mrr:.4f}, using only attention attribution")
    print(f"   without any component labels during training.'")

# ── DETAILED BREAKDOWN FOR UNSEEN ─────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Detailed breakdown — UNSEEN components:")
print(f"{'─'*65}")

for r in unseen_results:
    slot     = r['slot']
    adjacent = r['adjacent']
    print(f"\n  {r['component']} (slot {slot}):")
    print(f"  Adjacent sensors: "
          f"{[SENSOR_COLUMNS[s] for s in adjacent]}")
    print(f"  Windows attacked: {r['n_attacked']}")
    print(f"  Hit@{TOP_K}: {r['hit_at_k']:.4f}  "
          f"P@{TOP_K}: {r['prec_at_k']:.4f}  "
          f"MRR: {r['mrr']:.4f}")

    # Show top-5 sensors for a sample attack window
    attacked_windows = np.where(y_p_test[:, slot] == 1)[0]
    if len(attacked_windows) > 0:
        sample_idx = attacked_windows[0]
        imp        = sensor_imp[sample_idx]
        top5_idx   = np.argsort(imp)[::-1][:5]
        print(f"  Sample window {sample_idx} — top-5 sensors by attention:")
        for rank, sidx in enumerate(top5_idx):
            in_adj = "(ADJACENT)" if sidx in adjacent else ""
            print(f"    [{rank+1}] sensor_{sidx:2d}: "
                  f"{SENSOR_COLUMNS[sidx]:20s} "
                  f"imp={imp[sidx]:.4f}  {in_adj}")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'zero_shot_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'top_k':   TOP_K,
        'seen':    seen_results,
        'unseen':  unseen_results,
    }, f, indent=2)
print(f"\nSaved: {out_path}")

# ── RANDOM BASELINE COMPARISON ────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Random baseline comparison:")
print(f"{'─'*65}")
for r in unseen_results:
    n_adj     = len(r['adjacent'])
    random_pk = n_adj / 71         # expected P@K if picking randomly
    random_hk = 1 - ((71-n_adj)/71)**TOP_K
    print(f"  {r['component']:8s}: "
          f"model P@{TOP_K}={r['prec_at_k']:.4f}  "
          f"random={random_pk:.4f}  "
          f"lift={r['prec_at_k']/max(random_pk,1e-6):.2f}x")

print(f"\n{'='*65}")
print(f"Zero-shot evaluation complete.")
print(f"{'='*65}")