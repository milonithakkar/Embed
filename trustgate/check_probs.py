# check_probs.py
# Run in a SEPARATE terminal while training continues
# python check_probs.py

import numpy as np
import torch
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
CKPT_PATH = r'D:\trustgate_pcaps\trained_model_full_v3.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading checkpoint...")
ckpt  = torch.load(CKPT_PATH, map_location=device)
model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Checkpoint from epoch {ckpt['epoch']}, AUC={ckpt['monitor_value']:.4f}")

data   = np.load(DATA_PATH, allow_pickle=True)
X_s_vl = torch.FloatTensor(data['X_s_val'])
X_n_vl = torch.FloatTensor(data['X_n_val'])
y_b_vl = data['y_b_val'].astype(int)

all_probs = []
with torch.no_grad():
    for i in range(0, len(X_s_vl), 256):
        xs = X_s_vl[i:i+256].to(device)
        xn = X_n_vl[i:i+256].to(device)
        logits = model(xs, xn)[0].squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

probs  = np.concatenate(all_probs)
labels = y_b_vl

print(f"\nProbability Distribution:")
print(f"  All samples  — min:{probs.min():.4f}  "
      f"max:{probs.max():.4f}  "
      f"mean:{probs.mean():.4f}  "
      f"std:{probs.std():.4f}")

attack_probs = probs[labels == 1]
normal_probs = probs[labels == 0]

print(f"  Attack probs — min:{attack_probs.min():.4f}  "
      f"max:{attack_probs.max():.4f}  "
      f"mean:{attack_probs.mean():.4f}")
print(f"  Normal probs — min:{normal_probs.min():.4f}  "
      f"max:{normal_probs.max():.4f}  "
      f"mean:{normal_probs.mean():.4f}")

print(f"\nPercentile breakdown:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:2d}: all={np.percentile(probs,p):.4f}  "
          f"attack={np.percentile(attack_probs,p):.4f}  "
          f"normal={np.percentile(normal_probs,p):.4f}")

print(f"\nManual threshold sweep (fine-grained):")
print(f"  {'Thresh':>8}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'TP':>5}  {'FP':>5}")
from sklearn.metrics import f1_score, precision_score, recall_score
best_f1, best_t = 0, 0.5
for t in np.percentile(probs, np.arange(1, 100, 1)):
    preds = (probs >= t).astype(int)
    if preds.sum() == 0 or preds.sum() == len(preds):
        continue
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    tp   = int(((preds==1) & (labels==1)).sum())
    fp   = int(((preds==1) & (labels==0)).sum())
    if f1 > best_f1:
        best_f1, best_t = f1, t
    if prec > 0.05:
        print(f"  {t:>8.4f}  {f1:>6.4f}  {prec:>6.4f}  "
              f"{rec:>6.4f}  {tp:>5}  {fp:>5}")

print(f"\nBest F1: {best_f1:.4f} at threshold {best_t:.4f}")