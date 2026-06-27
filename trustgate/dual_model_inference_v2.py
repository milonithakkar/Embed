# dual_model_inference_v2.py
# TrustGate Dual-Model System — Clean Version
# v3: AUC=0.9620, localization
# v4: FAR=1.21%, precision detection
# Run: python dual_model_inference_v2.py

import numpy as np
import torch
import json
import os
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, roc_auc_score)

# ── CONFIG ────────────────────────────────────────────────────
V3_CKPT   = r'D:\trustgate_pcaps\trained_model_v3_FINAL_AUC9620.pth'
V4_CKPT   = r'D:\trustgate_pcaps\trained_model_full_v4.pth'
DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_DIR   = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

V3_THRESHOLD = 0.9831
V4_THRESHOLD = 0.9866

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

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

COMPONENT_NAMES = [
    "MV101","MV201","MV301","MV302","MV303",
    "MV304","MV501","MV502","MV503","MV504",
    "P101","P102","P201","P202","P203",
    "P204","P205","P206","LIT101","LIT601",
    "DPIT301","AIT402",
]

CLASS_NAMES = [
    "Normal","Valve Manipulation","Pump Disruption",
    "Chemical Dosing","Level Spoofing","Sensor Spoofing",
]

# Data-driven topology
COMPONENT_TO_SENSORS = {
    0:  [35, 40, 44, 20, 31, 25, 26, 28],
    1:  [35, 44, 40, 54, 52, 16, 4,  7 ],
    2:  [],
    3:  [46, 47, 54, 35, 52, 44, 40, 34],
    4:  [46, 47, 54, 35, 52, 44, 40, 34],
    5:  [],
    6:  [64, 20, 31, 25, 9,  26, 28, 8 ],
    7:  [64, 20, 31, 25, 9,  26, 28, 8 ],
    8:  [64, 20, 31, 25, 9,  26, 28, 8 ],
    9:  [64, 20, 31, 25, 9,  26, 28, 8 ],
    10: [35, 44, 40, 54, 52, 53, 20, 31],
    11: [35, 44, 40, 54, 52, 53, 20, 31],
    12: [35, 44, 40, 10, 20, 16, 54, 31],
    13: [35, 44, 40, 10, 20, 16, 54, 31],
    14: [35, 44, 40, 10, 20, 16, 54, 31],
    15: [35, 44, 40, 10, 20, 16, 54, 31],
    16: [35, 44, 40, 10, 20, 16, 54, 31],
    17: [35, 44, 40, 10, 20, 16, 54, 31],
    18: [35, 44, 40, 54, 52, 45, 10, 49],
    19: [46, 47, 26, 35, 54, 44, 52, 40],
    20: [46, 47, 35, 44, 54, 52, 29, 40],
    21: [46, 47, 35, 54, 44, 52, 40, 45],
}

UNSEEN_SLOTS = [3, 4, 21]

# ── LOAD MODELS ───────────────────────────────────────────────
def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE,
                      weights_only=False)
    m    = TrustGateModel().to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m, ckpt['epoch'], ckpt['monitor_value']

print("=" * 65)
print("TrustGate — Dual-Model System v2")
print("=" * 65)
print(f"\nDevice: {DEVICE}")

print(f"\nLoading v3 (localization + detection)...")
model_v3, ep3, auc3 = load_model(V3_CKPT)
print(f"  epoch={ep3}  AUC={auc3:.4f}  "
      f"threshold={V3_THRESHOLD}")

print(f"\nLoading v4 (high-precision detection)...")
model_v4, ep4, auc4 = load_model(V4_CKPT)
print(f"  epoch={ep4}  AUC={auc4:.4f}  "
      f"threshold={V4_THRESHOLD}")

# ── INFERENCE ─────────────────────────────────────────────────
def get_all_probs(Xs, Xn, batch=256):
    p3s, p4s, clss, imps = [], [], [], []
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(Xs),
        torch.FloatTensor(Xn))
    ld = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for xs, xn in ld:
            xs, xn = xs.to(DEVICE), xn.to(DEVICE)
            o3 = model_v3(xs, xn)
            o4 = model_v4(xs, xn)
            p3s.append(torch.sigmoid(
                o3[0].squeeze(-1)).cpu().numpy())
            p4s.append(torch.sigmoid(
                o4[0].squeeze(-1)).cpu().numpy())
            clss.append(
                o3[1].argmax(-1).cpu().numpy())
            imps.append(o3[5].cpu().numpy())
    return (np.concatenate(p3s),
            np.concatenate(p4s),
            np.concatenate(clss),
            np.concatenate(imps))

# ── ALERT LOGIC ───────────────────────────────────────────────
def get_alert(p3, p4):
    f3 = p3 >= V3_THRESHOLD
    f4 = p4 >= V4_THRESHOLD
    if f3 and f4:
        return 'HIGH',   (p3+p4)/2
    elif f4:
        return 'MEDIUM', p4
    elif f3:
        return 'LOW',    p3
    return 'NORMAL', 0.0

# ── LOCALIZATION ──────────────────────────────────────────────
def localize(imp, top_k=5):
    top_k_idx = np.argsort(imp)[::-1][:top_k]
    top_k_set = set(top_k_idx)
    best_comp  = -1
    best_score = -1
    for slot, adj in COMPONENT_TO_SENSORS.items():
        if not adj:
            continue
        score = len(top_k_set & set(adj)) / len(adj)
        if score > best_score:
            best_score, best_comp = score, slot
    name = (COMPONENT_NAMES[best_comp]
            if best_comp >= 0 else "Unknown")
    top_sensors = [SENSOR_COLS[i] for i in top_k_idx]
    return name, float(best_score), top_sensors

# ── ZERO-SHOT LOCALIZATION EVAL ───────────────────────────────
def eval_localization(sensor_imp, y_p_test, top_k=5):
    results = {}
    for slot in UNSEEN_SLOTS:
        attacked = np.where(y_p_test[:, slot] == 1)[0]
        adjacent = COMPONENT_TO_SENSORS.get(slot, [])
        if len(attacked) == 0 or not adjacent:
            continue
        hits   = mrr = prec_k = 0.0
        adj_set = set(adjacent)
        for idx in attacked:
            imp    = sensor_imp[idx]
            ranked = np.argsort(imp)[::-1]
            top_k_set = set(ranked[:top_k])
            overlap   = len(top_k_set & adj_set)
            prec_k   += overlap / top_k
            if overlap > 0:
                hits += 1
            for rank, s in enumerate(ranked):
                if s in adj_set:
                    mrr += 1.0 / (rank+1)
                    break
        n = len(attacked)
        results[COMPONENT_NAMES[slot]] = {
            'hit_at_k':  hits/n,
            'prec_at_k': prec_k/n,
            'mrr':       mrr/n,
            'n':         n,
        }
    return results

# ── LOAD DATA ─────────────────────────────────────────────────
print(f"\nLoading data...")
data      = np.load(DATA_PATH, allow_pickle=True)
X_s_test  = data['X_s_test']
X_n_test  = data['X_n_test']
y_b_test  = data['y_b_test'].astype(int)
y_p_test  = data['y_p_test'].astype(float)

print(f"Test: {len(y_b_test):,} samples  "
      f"({y_b_test.sum()} attacks)")

# ── RUN ───────────────────────────────────────────────────────
print(f"\nRunning dual-model inference...")
p3, p4, cls_preds, sensor_imp = get_all_probs(
    X_s_test, X_n_test)

alerts = [get_alert(p3[i], p4[i])
          for i in range(len(p3))]
levels = [a[0] for a in alerts]

from collections import Counter
dist = Counter(levels)

# Binary predictions
pred_strict = np.array([
    1 if l in ['HIGH','MEDIUM'] else 0
    for l in levels])
pred_sensitive = np.array([
    1 if l in ['HIGH','MEDIUM','LOW'] else 0
    for l in levels])

# ── DETECTION RESULTS ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"DETECTION RESULTS")
print(f"{'='*65}")

print(f"\nAlert distribution:")
for level in ['HIGH','MEDIUM','LOW','NORMAL']:
    print(f"  {level:8s}: {dist[level]:,}")

n_norm = int((y_b_test==0).sum())
n_att  = int((y_b_test==1).sum())

auc_v3 = roc_auc_score(y_b_test, p3)
auc_v4 = roc_auc_score(y_b_test, p4)
print(f"\n  v3 AUC: {auc_v3:.4f}")
print(f"  v4 AUC: {auc_v4:.4f}")

for name, preds in [
        ('Strict (HIGH+MED)', pred_strict),
        ('Sensitive (all)',   pred_sensitive)]:
    f1   = f1_score(y_b_test, preds, zero_division=0)
    prec = precision_score(y_b_test, preds,
                           zero_division=0)
    rec  = recall_score(y_b_test, preds,
                        zero_division=0)
    tp   = int(((preds==1)&(y_b_test==1)).sum())
    fp   = int(((preds==1)&(y_b_test==0)).sum())
    fn   = int(((preds==0)&(y_b_test==1)).sum())
    far  = fp / max(n_norm, 1)
    print(f"\n  {name}:")
    print(f"    F1={f1:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  FAR={far:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")

# ── ZERO-SHOT RESULTS ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"ZERO-SHOT LOCALIZATION (data-driven topology)")
print(f"{'='*65}")

loc_results = eval_localization(sensor_imp, y_p_test)

print(f"\n{'Component':>10}  {'N':>5}  "
      f"{'Hit@5':>7}  {'P@5':>7}  {'MRR':>7}  "
      f"{'vs Random':>10}")
print(f"{'─'*55}")

for comp, r in loc_results.items():
    slot    = COMPONENT_NAMES.index(comp)
    n_adj   = len(COMPONENT_TO_SENSORS.get(slot, []))
    rand_pk = n_adj / 71
    lift    = r['prec_at_k'] / max(rand_pk, 1e-6)
    print(f"  {comp:>10}  {r['n']:>5}  "
          f"{r['hit_at_k']:>7.4f}  "
          f"{r['prec_at_k']:>7.4f}  "
          f"{r['mrr']:>7.4f}  "
          f"{lift:>8.2f}x")

# ── SAVE ──────────────────────────────────────────────────────
out = {
    'v3_threshold':     V3_THRESHOLD,
    'v4_threshold':     V4_THRESHOLD,
    'alert_dist':       dict(dist),
    'v3_test_auc':      float(auc_v3),
    'v4_test_auc':      float(auc_v4),
    'localization':     loc_results,
}
path = os.path.join(OUT_DIR, 'dual_model_v2_results.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"Saved: {path}")
print(f"{'='*65}")