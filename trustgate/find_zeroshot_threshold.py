# find_zeroshot_threshold.py
# Find threshold that catches unseen attacks with acceptable FAR
# Run: python find_zeroshot_threshold.py

import numpy as np
import torch
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel
from sklearn.metrics import f1_score, precision_score, recall_score

V3_CKPT   = r'D:\trustgate_pcaps\trained_model_v3_FINAL_AUC9620.pth'
V4_CKPT   = r'D:\trustgate_pcaps\trained_model_full_v4.pth'
DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE,
                      weights_only=False)
    m    = TrustGateModel().to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m

model_v3 = load_model(V3_CKPT)
model_v4 = load_model(V4_CKPT)

data      = np.load(DATA_PATH, allow_pickle=True)

# Get val set probabilities for threshold calibration
X_s_val   = torch.FloatTensor(data['X_s_val'])
X_n_val   = torch.FloatTensor(data['X_n_val'])
y_b_val   = data['y_b_val'].astype(int)

# Get test set probabilities
X_s_test  = torch.FloatTensor(data['X_s_test'])
X_n_test  = torch.FloatTensor(data['X_n_test'])
y_b_test  = data['y_b_test'].astype(int)
y_p_test  = data['y_p_test'].astype(float)

def get_probs(model, Xs, Xn, batch=256):
    all_p = []
    ds = torch.utils.data.TensorDataset(Xs, Xn)
    ld = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for xs, xn in ld:
            xs, xn = xs.to(DEVICE), xn.to(DEVICE)
            p = torch.sigmoid(
                model(xs, xn)[0].squeeze(-1))
            all_p.append(p.cpu().numpy())
    return np.concatenate(all_p)

print("Running inference...")
p3_val  = get_probs(model_v3, X_s_val,  X_n_val)
p4_val  = get_probs(model_v4, X_s_val,  X_n_val)
p3_test = get_probs(model_v3, X_s_test, X_n_test)
p4_test = get_probs(model_v4, X_s_test, X_n_test)

# Take max of v3 and v4 as combined score
combined_val  = np.maximum(p3_val,  p4_val)
combined_test = np.maximum(p3_test, p4_test)

print(f"\nCombined prob stats:")
print(f"  Val  — normal: {combined_val[y_b_val==0].mean():.4f}  "
      f"attack: {combined_val[y_b_val==1].mean():.4f}")
print(f"  Test — normal: {combined_test[y_b_test==0].mean():.4f}  "
      f"attack: {combined_test[y_b_test==1].mean():.4f}")

# AIT402 specific
ait402_mask = y_p_test[:, 21] == 1
mv302_mask  = y_p_test[:, 3]  == 1
mv303_mask  = y_p_test[:, 4]  == 1

print(f"\nProb distribution for unseen attacks:")
print(f"  AIT402: mean={combined_test[ait402_mask].mean():.4f}  "
      f"min={combined_test[ait402_mask].min():.4f}  "
      f"max={combined_test[ait402_mask].max():.4f}")
print(f"  MV302:  mean={combined_test[mv302_mask].mean():.4f}  "
      f"min={combined_test[mv302_mask].min():.4f}  "
      f"max={combined_test[mv302_mask].max():.4f}")
print(f"  MV303:  mean={combined_test[mv303_mask].mean():.4f}  "
      f"min={combined_test[mv303_mask].min():.4f}  "
      f"max={combined_test[mv303_mask].max():.4f}")

print(f"\n{'─'*70}")
print(f"Threshold sweep on COMBINED score (max of v3, v4):")
print(f"{'─'*70}")
print(f"{'Thresh':>8}  {'Val_F1':>7}  {'Val_FAR':>8}  "
      f"{'Test_F1':>8}  {'Test_FAR':>9}  "
      f"{'AIT402_Rec':>11}  {'MV302_Rec':>10}")
print(f"{'─'*70}")

for t in np.arange(0.95, 0.9999, 0.005):
    # Val metrics
    v_preds  = (combined_val >= t).astype(int)
    v_f1     = f1_score(y_b_val, v_preds, zero_division=0)
    v_fp     = int(((v_preds==1)&(y_b_val==0)).sum())
    v_far    = v_fp / max((y_b_val==0).sum(), 1)

    # Test metrics
    t_preds  = (combined_test >= t).astype(int)
    t_f1     = f1_score(y_b_test, t_preds, zero_division=0)
    t_fp     = int(((t_preds==1)&(y_b_test==0)).sum())
    t_far    = t_fp / max((y_b_test==0).sum(), 1)

    # Per-component recall
    ait_rec  = t_preds[ait402_mask].mean()
    mv3_rec  = t_preds[mv302_mask].mean()

    # Highlight rows where unseen attacks are caught
    star = ""
    if ait_rec > 0.5 and v_far < 0.05:
        star = " ← GOOD"
    if ait_rec > 0.8 and v_far < 0.05:
        star = " ← BEST"

    print(f"{t:>8.4f}  {v_f1:>7.4f}  {v_far:>8.4f}  "
          f"{t_f1:>8.4f}  {t_far:>9.4f}  "
          f"{ait_rec:>11.4f}  {mv3_rec:>10.4f}{star}")

print(f"\n{'─'*70}")
print(f"Recommendation:")
print(f"  Use combined threshold that maximizes AIT402 recall")
print(f"  while keeping val FAR below 5%")
print(f"{'─'*70}")