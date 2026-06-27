# dual_model_inference.py
# TrustGate Dual-Model Inference System
# v3: Detection (AUC=0.9531) + Localization (zero-shot)
# v4: High-Precision Detection (FAR=1.2%, Prec=0.84)
#
# Strategy:
#   Primary detector:    v4 (fewer false alarms)
#   Fallback detector:   v3 (higher AUC)
#   Localizer:           v3 sensor_imp always
#
# Alert logic:
#   v4 AND v3 both flag  → HIGH confidence attack
#   v4 only flags        → MEDIUM confidence attack
#   v3 only flags        → LOW confidence attack
#   neither flags        → Normal operation
#
# Run: python dual_model_inference.py

import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── CONFIG ────────────────────────────────────────────────────
V3_CKPT = r'D:\trustgate_pcaps\trained_model_v3_FINAL_AUC9620.pth'
V4_CKPT = r'D:\trustgate_pcaps\trained_model_full_v4.pth'
DATA_PATH  = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_DIR    = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

# Thresholds calibrated on validation set
V3_THRESHOLD = 0.9831  # from v3 evaluation
V4_THRESHOLD = 0.9866    # from v4 evaluation

# Component names
COMPONENT_NAMES = [
    "MV101", "MV201", "MV301", "MV302", "MV303",
    "MV304", "MV501", "MV502", "MV503", "MV504",
    "P101",  "P102",  "P201",  "P202",  "P203",
    "P204",  "P205",  "P206",  "LIT101","LIT601",
    "DPIT301","AIT402",
]

# Sensor columns
SENSOR_COLS = [
    "P1_STATE", "LIT101.Pv", "FIT101.Pv", "MV101.Status",
    "P101.Status", "P102.Status", "P2_STATE", "FIT201.Pv",
    "AIT201.Pv", "AIT202.Pv", "AIT203.Pv", "MV201.Status",
    "P201.Status", "P202.Status", "P203.Status", "P204.Status",
    "P205.Status", "P206.Status", "P207.Status", "P208.Status",
    "P3_STATE", "AIT301.Pv", "AIT302.Pv", "AIT303.Pv",
    "LIT301.Pv", "FIT301.Pv", "DPIT301.Pv", "MV301.Status",
    "MV302.Status", "MV303.Status", "MV304.Status",
    "P301.Status", "P302.Status", "P4_STATE", "LIT401.Pv",
    "FIT401.Pv", "AIT401.Pv", "AIT402.Pv", "P401.Status",
    "P402.Status", "P403.Status", "P404.Status",
    "UV401.Status", "P5_STATE", "FIT501.Pv", "FIT502.Pv",
    "FIT503.Pv", "FIT504.Pv", "AIT501.Pv", "AIT502.Pv",
    "AIT503.Pv", "AIT504.Pv", "PIT501.Pv", "PIT502.Pv",
    "PIT503.Pv", "P501.Status", "P501.Speed", "P502.Status",
    "P502.Speed", "MV501.Status", "MV502.Status",
    "MV503.Status", "MV504.Status", "P6_STATE", "LIT601.Pv",
    "LIT602.Pv", "FIT601.Pv", "FIT602.Pv", "P601.Status",
    "P602.Status", "P603.Status",
]

# Topology for localization
COMPONENT_TO_SENSORS = {
    0:  [0, 1, 2, 3],
    1:  [6, 7, 8, 9, 10, 11],
    3:  [20, 21, 22, 23, 24, 25, 26, 28],
    4:  [20, 21, 22, 23, 24, 25, 26, 29],
    6:  [43, 44, 52, 53, 54, 55, 56, 59],
    7:  [43, 45, 52, 53, 54, 55, 56, 60],
    8:  [43, 46, 52, 53, 54, 57, 58, 61],
    9:  [43, 47, 52, 53, 54, 57, 58, 62],
    10: [0, 1, 2, 4],
    11: [0, 1, 2, 5],
    18: [0, 1, 2, 3, 4, 5],
    19: [63, 64, 65, 66, 67, 68, 69, 70],
    20: [20, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    21: [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
}

CLASS_NAMES = [
    "Normal", "Valve Manipulation", "Pump Disruption",
    "Chemical Dosing", "Level Spoofing", "Sensor Spoofing",
]

# ── LOAD BOTH MODELS ──────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model = TrustGateModel().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['epoch'], ckpt['monitor_value']


print("=" * 65)
print("TrustGate — Dual-Model Inference System")
print("=" * 65)
print(f"\nDevice: {DEVICE}")

print(f"\nLoading v3 (detection + localization)...")
model_v3, ep3, auc3 = load_model(V3_CKPT, DEVICE)
print(f"  Epoch={ep3}, AUC={auc3:.4f}, "
      f"threshold={V3_THRESHOLD}")

print(f"\nLoading v4 (high-precision detection)...")
model_v4, ep4, auc4 = load_model(V4_CKPT, DEVICE)
print(f"  Epoch={ep4}, AUC={auc4:.4f}, "
      f"threshold={V4_THRESHOLD}")


# ── INFERENCE FUNCTION ────────────────────────────────────────
def dual_infer(xs, xn):
    """
    Run both models on a batch.
    Returns combined predictions.

    xs: (B, 30, 71) sensor window
    xn: (B, 30, 19) network window
    """
    xs = xs.to(DEVICE)
    xn = xn.to(DEVICE)

    with torch.no_grad():
        # v3: detection + localization
        out_v3       = model_v3(xs, xn)
        prob_v3      = torch.sigmoid(
            out_v3[0].squeeze(-1)).cpu().numpy()
        cls_v3       = out_v3[1].argmax(-1).cpu().numpy()
        sensor_imp   = out_v3[5].cpu().numpy()  # localization

        # v4: high-precision detection only
        out_v4  = model_v4(xs, xn)
        prob_v4 = torch.sigmoid(
            out_v4[0].squeeze(-1)).cpu().numpy()

    return prob_v3, prob_v4, cls_v3, sensor_imp


def get_alert_level(prob_v3, prob_v4,
                    t_v3=V3_THRESHOLD,
                    t_v4=V4_THRESHOLD):
    """
    Dual-model alert logic.

    Returns:
      level: 'HIGH' | 'MEDIUM' | 'LOW' | 'NORMAL'
      confidence: float 0-1
      reason: string explanation
    """
    flag_v3 = prob_v3 >= t_v3
    flag_v4 = prob_v4 >= t_v4

    if flag_v3 and flag_v4:
        # Both models agree → highest confidence
        confidence = float((prob_v3 + prob_v4) / 2)
        return 'HIGH', confidence, \
               f"Both models flag (v3={prob_v3:.3f}, " \
               f"v4={prob_v4:.3f})"

    elif flag_v4 and not flag_v3:
        # v4 only: high precision model → medium confidence
        # v4 has FAR=1.2% so false alarm is unlikely
        confidence = float(prob_v4)
        return 'MEDIUM', confidence, \
               f"v4 precision model flags " \
               f"(v4={prob_v4:.3f}, v3={prob_v3:.3f})"

    elif flag_v3 and not flag_v4:
        # v3 only: high AUC model → low confidence
        # Could be a subtle/novel attack or false alarm
        confidence = float(prob_v3)
        return 'LOW', confidence, \
               f"v3 sensitivity model flags " \
               f"(v3={prob_v3:.3f}, v4={prob_v4:.3f})"

    else:
        # Neither flags → normal
        confidence = 0.0
        return 'NORMAL', confidence, \
               f"Normal (v3={prob_v3:.3f}, " \
               f"v4={prob_v4:.3f})"


def localize(sensor_imp_vec, top_k=5):
    """
    Map sensor_imp to most likely attacked component.
    Uses v3 sensor_imp (better localization).
    """
    top_k_idx = np.argsort(sensor_imp_vec)[::-1][:top_k]
    top_k_set = set(top_k_idx)

    best_comp  = -1
    best_score = -1

    for comp_slot, adj in COMPONENT_TO_SENSORS.items():
        overlap = len(top_k_set & set(adj))
        score   = overlap / max(len(adj), 1)
        if score > best_score:
            best_score = score
            best_comp  = comp_slot

    comp_name = (COMPONENT_NAMES[best_comp]
                 if best_comp >= 0 else "Unknown")
    top_sensors = [SENSOR_COLS[i] for i in top_k_idx]

    return comp_name, float(best_score), top_sensors


# ── FULL TEST SET EVALUATION ──────────────────────────────────
print(f"\n{'='*65}")
print("Evaluating on test set (unseen attack types)...")
print(f"{'='*65}")

data     = np.load(DATA_PATH, allow_pickle=True)
X_s_test = torch.FloatTensor(data['X_s_test'])
X_n_test = torch.FloatTensor(data['X_n_test'])
y_b_test = data['y_b_test'].astype(int)
y_p_test = data['y_p_test'].astype(float)

from sklearn.metrics import (roc_auc_score,
                              f1_score,
                              precision_score,
                              recall_score)

all_p3, all_p4, all_cls, all_imp = [], [], [], []
ds = torch.utils.data.TensorDataset(X_s_test, X_n_test)
loader = torch.utils.data.DataLoader(
    ds, batch_size=256, shuffle=False)

for xs, xn in loader:
    p3, p4, cls, imp = dual_infer(xs, xn)
    all_p3.append(p3)
    all_p4.append(p4)
    all_cls.append(cls)
    all_imp.append(imp)

prob_v3   = np.concatenate(all_p3)
prob_v4   = np.concatenate(all_p4)
cls_preds = np.concatenate(all_cls)
sensor_imp = np.concatenate(all_imp)
labels     = y_b_test

# Alert levels for each window
alert_levels = []
for i in range(len(prob_v3)):
    level, conf, reason = get_alert_level(
        prob_v3[i], prob_v4[i])
    alert_levels.append(level)

# Convert alert levels to binary predictions
# HIGH + MEDIUM = attack detected
# LOW + NORMAL  = not flagged
pred_strict = np.array([
    1 if l in ['HIGH', 'MEDIUM'] else 0
    for l in alert_levels])

pred_sensitive = np.array([
    1 if l in ['HIGH', 'MEDIUM', 'LOW'] else 0
    for l in alert_levels])

# Alert level distribution
from collections import Counter
dist = Counter(alert_levels)
print(f"\nAlert level distribution:")
print(f"  HIGH   (both flag):    {dist['HIGH']:,}")
print(f"  MEDIUM (v4 only):      {dist['MEDIUM']:,}")
print(f"  LOW    (v3 only):      {dist['LOW']:,}")
print(f"  NORMAL (neither):      {dist['NORMAL']:,}")

# Metrics
print(f"\n{'─'*65}")
print(f"DETECTION METRICS (Test Set)")
print(f"{'─'*65}")

if len(np.unique(labels)) > 1:
    auc_v3 = roc_auc_score(labels, prob_v3)
    auc_v4 = roc_auc_score(labels, prob_v4)
    print(f"\n  v3 AUC (solo):   {auc_v3:.4f}")
    print(f"  v4 AUC (solo):   {auc_v4:.4f}")

n_norm = int((labels == 0).sum())
n_att  = int((labels == 1).sum())

for name, preds in [('Strict (HIGH+MED)', pred_strict),
                    ('Sensitive (all)',   pred_sensitive)]:
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    tp   = int(((preds==1)&(labels==1)).sum())
    fp   = int(((preds==1)&(labels==0)).sum())
    fn   = int(((preds==0)&(labels==1)).sum())
    far  = fp / max(n_norm, 1)

    print(f"\n  {name}:")
    print(f"    F1={f1:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  FAR={far:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")

# Zero-shot localization (using v3 sensor_imp)
print(f"\n{'─'*65}")
print(f"ZERO-SHOT LOCALIZATION (v3 sensor_imp)")
print(f"{'─'*65}")

UNSEEN = {3: 'MV302', 4: 'MV303', 21: 'AIT402'}

for slot, name in UNSEEN.items():
    attacked = np.where(y_p_test[:, slot] == 1)[0]
    if len(attacked) == 0:
        continue

    adj    = COMPONENT_TO_SENSORS.get(slot, [])
    hits   = 0
    prec_k = 0.0
    mrr    = 0.0

    for idx in attacked:
        imp      = sensor_imp[idx]
        top5     = set(np.argsort(imp)[::-1][:5])
        adj_set  = set(adj)

        overlap  = len(top5 & adj_set)
        prec_k  += overlap / 5

        if len(top5 & adj_set) > 0:
            hits += 1

        for rank, s in enumerate(
                np.argsort(imp)[::-1]):
            if s in adj_set:
                mrr += 1.0 / (rank + 1)
                break

    n   = len(attacked)
    print(f"\n  {name} (slot {slot}) — {n} windows:")
    print(f"    Hit@5 = {hits/n:.4f}")
    print(f"    P@5   = {prec_k/n:.4f}")
    print(f"    MRR   = {mrr/n:.4f}")

# Save summary
summary = {
    'v3_threshold': V3_THRESHOLD,
    'v4_threshold': V4_THRESHOLD,
    'alert_distribution': dict(dist),
    'v3_test_auc': float(auc_v3),
    'v4_test_auc': float(auc_v4),
}
out = os.path.join(OUT_DIR, 'dual_model_results.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*65}")
print(f"Dual-model evaluation complete.")
print(f"Saved: {out}")
print(f"{'='*65}")