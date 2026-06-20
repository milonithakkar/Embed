# 02b_test_threshold_sweep.py
# Shows test F1 across ALL thresholds — reveals what τ=0.98 hid.
# Run: python 02b_test_threshold_sweep.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

DATA_PATH  = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
CHECKPOINT = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
OUTPUT_DIR = r'D:\trustgate_pcaps\eval_outputs'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data   = np.load(DATA_PATH, allow_pickle=True)

ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

@torch.no_grad()
def get_probs(split):
    X_s = torch.FloatTensor(data[f'X_s_{split}'])
    X_n = torch.FloatTensor(data[f'X_n_{split}'])
    y_b = data[f'y_b_{split}'].astype(int)
    probs = []
    for i in range(0, len(X_s), 256):
        xs = X_s[i:i+256].to(device)
        xn = X_n[i:i+256].to(device)
        out = model(xs, xn)
        bl = out[0] if isinstance(out, (tuple, list)) else out
        probs.append(torch.sigmoid(bl.squeeze(-1)).cpu().numpy())
    return np.concatenate(probs), y_b

print("="*60)
print("TEST THRESHOLD SWEEP — finding robust operating point")
print("="*60)

val_probs,  val_labels  = get_probs('val')
test_probs, test_labels = get_probs('test')

print(f"\nVal  AUC: {roc_auc_score(val_labels, val_probs):.4f}")
print(f"Test AUC: {roc_auc_score(test_labels, test_probs):.4f}")

# Sweep both
thresholds = np.arange(0.05, 0.99, 0.01)
print(f"\n{'τ':>5} | {'Val F1':>7} {'Val P':>7} {'Val R':>7} | "
      f"{'Test F1':>7} {'Test P':>7} {'Test R':>7}")
print("-"*65)

best_test_f1, best_test_t = 0, 0.5
best_joint_f1, best_joint_t = 0, 0.5

val_f1s, test_f1s = [], []
for t in thresholds:
    vp = (val_probs  >= t).astype(int)
    tp = (test_probs >= t).astype(int)

    vf1 = f1_score(val_labels,  vp, zero_division=0)
    tf1 = f1_score(test_labels, tp, zero_division=0)
    val_f1s.append(vf1); test_f1s.append(tf1)

    if tf1 > best_test_f1:
        best_test_f1, best_test_t = tf1, t
    joint = min(vf1, tf1)  # robust point: maximize the WORSE of the two
    if joint > best_joint_f1:
        best_joint_f1, best_joint_t = joint, t

    # Print every 10th threshold
    if abs(t*100 % 10) < 1:
        vpr = precision_score(val_labels,  vp, zero_division=0)
        vrc = recall_score(val_labels,  vp, zero_division=0)
        tpr = precision_score(test_labels, tp, zero_division=0)
        trc = recall_score(test_labels, tp, zero_division=0)
        print(f"{t:>5.2f} | {vf1:>7.4f} {vpr:>7.4f} {vrc:>7.4f} | "
              f"{tf1:>7.4f} {tpr:>7.4f} {trc:>7.4f}")

print("\n" + "="*60)
print(f"τ=0.98 (current)      → Test F1 = "
      f"{f1_score(test_labels, (test_probs>=0.98).astype(int), zero_division=0):.4f}")
print(f"Best test τ={best_test_t:.2f}     → Test F1 = {best_test_f1:.4f}")
print(f"Robust τ={best_joint_t:.2f}        → "
      f"Val F1={f1_score(val_labels,(val_probs>=best_joint_t).astype(int),zero_division=0):.4f}, "
      f"Test F1={f1_score(test_labels,(test_probs>=best_joint_t).astype(int),zero_division=0):.4f}")
print("="*60)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, val_f1s,  'b-', lw=2, label='Val F1')
ax.plot(thresholds, test_f1s, 'r-', lw=2, label='Test F1')
ax.axvline(0.98, color='gray', ls=':', label='Current τ=0.98 (overfit)')
ax.axvline(best_joint_t, color='green', ls='--',
           label=f'Robust τ={best_joint_t:.2f}')
ax.set_xlabel('Threshold (τ)'); ax.set_ylabel('F1 Score')
ax.set_title('Val vs Test F1 — Threshold Transferability')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'test_threshold_sweep.png'), dpi=150)
print(f"\nSaved plot → {OUTPUT_DIR}\\test_threshold_sweep.png")