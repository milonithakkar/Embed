# evaluate_zero_shot_v2.py
# TrustGate Zero-Shot Localization — Clean Version
# Uses data-driven topology (no P&ID assumptions)
# Seen components: training data deviation
# Unseen components: test statistics as ground truth
# Run: python evaluate_zero_shot_v2.py

import numpy as np
import torch
import sys
import os
import json
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
CKPT_PATH = r'D:\trustgate_pcaps\trained_model_v3_FINAL_AUC9620.pth'
OUT_DIR   = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')
TOP_K  = 5

# ── SENSOR COLUMNS ────────────────────────────────────────────
SENSOR_COLS = [
    "P1_STATE","LIT101.Pv","FIT101.Pv","MV101.Status",
    "P101.Status","P102.Status","P2_STATE","FIT201.Pv",
    "AIT201.Pv","AIT202.Pv","AIT203.Pv","MV201.Status",
    "P201.Status","P202.Status","P203.Status","P204.Status",
    "P205.Status","P206.Status","P207.Status","P208.Status",
    "P3_STATE","AIT301.Pv","AIT302.Pv","AIT303.Pv",
    "LIT301.Pv","FIT301.Pv","DPIT301.Pv","MV301.Status",
    "MV302.Status","MV303.Status","MV304.Status",
    "P301.Status","P302.Status","P4_STATE","LIT401.Pv",
    "FIT401.Pv","AIT401.Pv","AIT402.Pv","P401.Status",
    "P402.Status","P403.Status","P404.Status",
    "UV401.Status","P5_STATE","FIT501.Pv","FIT502.Pv",
    "FIT503.Pv","FIT504.Pv","AIT501.Pv","AIT502.Pv",
    "AIT503.Pv","AIT504.Pv","PIT501.Pv","PIT502.Pv",
    "PIT503.Pv","P501.Status","P501.Speed","P502.Status",
    "P502.Speed","MV501.Status","MV502.Status",
    "MV503.Status","MV504.Status","P6_STATE","LIT601.Pv",
    "LIT602.Pv","FIT601.Pv","FIT602.Pv","P601.Status",
    "P602.Status","P603.Status",
]

# ── COMPONENT NAMES ───────────────────────────────────────────
COMPONENT_NAMES = [
    "MV101","MV201","MV301","MV302","MV303",
    "MV304","MV501","MV502","MV503","MV504",
    "P101","P102","P201","P202","P203",
    "P204","P205","P206","LIT101","LIT601",
    "DPIT301","AIT402",
]

# ── DATA-DRIVEN TOPOLOGY ──────────────────────────────────────
# Seen components: derived from training attack deviation
# Unseen (MV302, MV303, AIT402): test statistics as
# evaluation ground truth (disclosed in paper)
COMPONENT_TO_SENSORS = {
    0:  [35, 40, 44, 20, 31, 25, 26, 28],  # MV101  SEEN
    1:  [35, 44, 40, 54, 52, 16, 4,  7 ],  # MV201  SEEN
    2:  [],                                  # MV301  no data
    3:  [46, 47, 54, 35, 52, 44, 40, 34],  # MV302  TEST GT
    4:  [46, 47, 54, 35, 52, 44, 40, 34],  # MV303  TEST GT
    5:  [],                                  # MV304  no data
    6:  [64, 20, 31, 25, 9,  26, 28, 8 ],  # MV501  SEEN
    7:  [64, 20, 31, 25, 9,  26, 28, 8 ],  # MV502  SEEN
    8:  [64, 20, 31, 25, 9,  26, 28, 8 ],  # MV503  SEEN
    9:  [64, 20, 31, 25, 9,  26, 28, 8 ],  # MV504  SEEN
    10: [35, 44, 40, 54, 52, 53, 20, 31],  # P101   SEEN
    11: [35, 44, 40, 54, 52, 53, 20, 31],  # P102   SEEN
    12: [35, 44, 40, 10, 20, 16, 54, 31],  # P201   SEEN
    13: [35, 44, 40, 10, 20, 16, 54, 31],  # P202   SEEN
    14: [35, 44, 40, 10, 20, 16, 54, 31],  # P203   SEEN
    15: [35, 44, 40, 10, 20, 16, 54, 31],  # P204   SEEN
    16: [35, 44, 40, 10, 20, 16, 54, 31],  # P205   SEEN
    17: [35, 44, 40, 10, 20, 16, 54, 31],  # P206   SEEN
    18: [35, 44, 40, 54, 52, 45, 10, 49],  # LIT101 SEEN
    19: [46, 47, 26, 35, 54, 44, 52, 40],  # LIT601 SEEN
    20: [46, 47, 35, 44, 54, 52, 29, 40],  # DPIT301 SEEN
    21: [46, 47, 35, 54, 44, 52, 40, 45],  # AIT402 TEST GT
}

UNSEEN_SLOTS = [3, 4, 21]
SEEN_SLOTS   = [0, 1, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20]

# ── LOAD MODEL ────────────────────────────────────────────────
print("=" * 65)
print("TrustGate v2 — Zero-Shot Localization Evaluation")
print("Data-driven topology (training deviation + test GT)")
print("=" * 65)
print(f"\nDevice: {DEVICE}")

ckpt  = torch.load(CKPT_PATH, map_location=DEVICE,
                   weights_only=False)
model = TrustGateModel().to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Checkpoint: epoch={ckpt['epoch']}  "
      f"AUC={ckpt['monitor_value']:.4f}")

# ── LOAD DATA ─────────────────────────────────────────────────
print(f"\nLoading data...")
data      = np.load(DATA_PATH, allow_pickle=True)
X_s_test  = data['X_s_test']
X_n_test  = data['X_n_test']
y_b_test  = data['y_b_test'].astype(int)
y_p_test  = data['y_p_test'].astype(float)

print(f"Test samples : {len(y_b_test):,}")
print(f"Test attacks : {y_b_test.sum():,}")
print(f"Unseen targets: "
      f"{[COMPONENT_NAMES[s] for s in UNSEEN_SLOTS]}")

# ── COLLECT sensor_imp ────────────────────────────────────────
print(f"\nRunning inference...")
all_imp   = []
all_probs = []

ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_s_test),
    torch.FloatTensor(X_n_test))
loader = torch.utils.data.DataLoader(
    ds, batch_size=256, shuffle=False)

with torch.no_grad():
    for xs, xn in loader:
        xs, xn = xs.to(DEVICE), xn.to(DEVICE)
        out    = model(xs, xn)
        probs  = torch.sigmoid(
            out[0].squeeze(-1)).cpu().numpy()
        imp    = out[5].cpu().numpy()
        all_probs.append(probs)
        all_imp.append(imp)

bin_probs  = np.concatenate(all_probs)
sensor_imp = np.concatenate(all_imp)

print(f"sensor_imp range: "
      f"{sensor_imp.min():.4f} – {sensor_imp.max():.4f}")
print(f"sensor_imp sum:   "
      f"{sensor_imp.sum(axis=1).mean():.4f} (should be 1.0)")

# ── EVALUATE LOCALIZATION ─────────────────────────────────────
def evaluate_component(slot):
    attacked = np.where(y_p_test[:, slot] == 1)[0]
    adjacent = COMPONENT_TO_SENSORS.get(slot, [])

    if len(attacked) == 0 or len(adjacent) == 0:
        return None

    hits   = 0
    prec_k = 0.0
    mrr    = 0.0

    for idx in attacked:
        imp      = sensor_imp[idx]
        ranked   = np.argsort(imp)[::-1]
        top_k    = set(ranked[:TOP_K])
        adj_set  = set(adjacent)

        overlap  = len(top_k & adj_set)
        prec_k  += overlap / TOP_K

        if overlap > 0:
            hits += 1

        for rank, s in enumerate(ranked):
            if s in adj_set:
                mrr += 1.0 / (rank + 1)
                break

    n = len(attacked)
    return {
        'slot':      slot,
        'component': COMPONENT_NAMES[slot],
        'n':         n,
        'hit_at_k':  hits / n,
        'prec_at_k': prec_k / n,
        'mrr':       mrr / n,
        'is_unseen': slot in UNSEEN_SLOTS,
        'adjacent':  adjacent,
    }

# ── RESULTS ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Results (Top-{TOP_K})")
print(f"{'='*65}")
print(f"\n{'Slot':>4}  {'Component':>10}  "
      f"{'Status':>8}  {'N':>5}  "
      f"{'Hit@5':>7}  {'P@5':>7}  {'MRR':>7}")
print(f"{'─'*65}")

results        = []
seen_results   = []
unseen_results = []

for slot in range(22):
    if y_p_test[:, slot].sum() == 0:
        continue
    r = evaluate_component(slot)
    if r is None:
        continue
    results.append(r)

    status = "UNSEEN" if r['is_unseen'] else "seen"
    print(f"  {slot:2d}  {r['component']:>10}  "
          f"{status:>8}  {r['n']:>5}  "
          f"{r['hit_at_k']:>7.4f}  "
          f"{r['prec_at_k']:>7.4f}  "
          f"{r['mrr']:>7.4f}")

    if r['is_unseen']:
        unseen_results.append(r)
    else:
        seen_results.append(r)

# ── SUMMARY ───────────────────────────────────────────────────
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
    print(f"\nUNSEEN components — ZERO-SHOT "
          f"(n={len(unseen_results)}):")
    print(f"  Avg Hit@{TOP_K}  : {avg_hit:.4f}")
    print(f"  Avg P@{TOP_K}    : {avg_prec:.4f}")
    print(f"  Avg MRR    : {avg_mrr:.4f}")
    print(f"\n  Components: "
          f"{[r['component'] for r in unseen_results]}")
    print(f"\n  Paper claim:")
    print(f"  TrustGate localizes {len(unseen_results)} "
          f"unseen attack components")
    print(f"  with Avg Hit@5={avg_hit:.4f}, "
          f"P@5={avg_prec:.4f}, MRR={avg_mrr:.4f}")
    print(f"  using attention attribution without")
    print(f"  component labels during training.")

# ── DETAILED BREAKDOWN ────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Detailed breakdown — UNSEEN components:")
print(f"{'─'*65}")

for r in unseen_results:
    slot     = r['slot']
    adjacent = r['adjacent']
    attacked = np.where(y_p_test[:, slot] == 1)[0]

    print(f"\n  {r['component']} (slot {slot}):")
    print(f"  Adjacent sensors: "
          f"{[SENSOR_COLS[s] for s in adjacent[:5]]}")
    print(f"  Windows: {r['n']}  "
          f"Hit@5={r['hit_at_k']:.4f}  "
          f"P@5={r['prec_at_k']:.4f}  "
          f"MRR={r['mrr']:.4f}")

    if len(attacked) > 0:
        sample = attacked[0]
        imp    = sensor_imp[sample]
        top5   = np.argsort(imp)[::-1][:5]
        adj_set = set(adjacent)
        print(f"  Sample window {sample} top-5 sensors:")
        for rank, sidx in enumerate(top5):
            hit = "(ADJACENT)" if sidx in adj_set else ""
            print(f"    [{rank+1}] {SENSOR_COLS[sidx]:20s} "
                  f"imp={imp[sidx]:.4f}  {hit}")

# ── RANDOM BASELINE ───────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Random baseline comparison:")
print(f"{'─'*65}")

for r in unseen_results:
    n_adj    = len(r['adjacent'])
    rand_pk  = n_adj / 71
    lift     = (r['prec_at_k'] / max(rand_pk, 1e-6))
    print(f"  {r['component']:8s}: "
          f"model P@5={r['prec_at_k']:.4f}  "
          f"random={rand_pk:.4f}  "
          f"lift={lift:.2f}x")

# ── SAVE ──────────────────────────────────────────────────────
out_path = os.path.join(
    OUT_DIR, 'zero_shot_results_v2.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'top_k':   TOP_K,
        'seen':    seen_results,
        'unseen':  unseen_results,
        'topology_method': (
            'Seen: training deviation. '
            'Unseen: test GT (disclosed).'
        )
    }, f, indent=2)

print(f"\nSaved: {out_path}")
print(f"\n{'='*65}")
print(f"Zero-shot evaluation v2 complete.")
print(f"{'='*65}")