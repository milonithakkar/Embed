# baselines.py
# Runs 4 standard baselines on your exact data
# for direct comparison in paper Table 1

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'

print("=" * 60)
print("Baseline Comparisons")
print("=" * 60)

data = np.load(DATA_PATH, allow_pickle=True)

# Flatten windows to 2D for sklearn
X_tr = data['X_s_train'].reshape(
    len(data['X_s_train']), -1)   # (N, 30*71)
X_vl = data['X_s_val'].reshape(
    len(data['X_s_val']), -1)
X_te = data['X_s_test'].reshape(
    len(data['X_s_test']), -1)

y_vl = data['y_b_val'].astype(int)
y_te = data['y_b_test'].astype(int)

# Normal training data only
normal_mask = (data['y_b_train'] == 0)
X_normal    = X_tr[normal_mask]

print(f"\nTrain normal: {X_normal.shape[0]:,}")
print(f"Val samples:  {X_vl.shape[0]:,}")
print(f"Test samples: {X_te.shape[0]:,}")

results = {}

# ── Baseline 1: Isolation Forest ──────────────────────────────
print(f"\n[1/3] Isolation Forest...")
iso = IsolationForest(
    n_estimators=100,
    contamination=0.113,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_normal)

# Scores: more negative = more anomalous
scores_vl = -iso.score_samples(X_vl)
scores_te = -iso.score_samples(X_te)

# Find best threshold on val
best_f1, best_t = 0.0, 0.5
for t in np.percentile(scores_vl, np.arange(80, 100, 0.5)):
    preds = (scores_vl >= t).astype(int)
    if preds.sum() == 0:
        continue
    f1 = f1_score(y_vl, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

auc_vl = roc_auc_score(y_vl, scores_vl)
auc_te = roc_auc_score(y_te, scores_te)
preds_te = (scores_te >= best_t).astype(int)
f1_te    = f1_score(y_te, preds_te, zero_division=0)
far_vl   = ((scores_vl >= best_t).astype(int) *
             (1 - y_vl)).sum() / max((y_vl==0).sum(), 1)

results['Isolation Forest'] = {
    'val_auc': auc_vl, 'test_auc': auc_te,
    'val_f1': best_f1, 'test_f1': f1_te,
    'val_far': float(far_vl)
}
print(f"  Val  AUC={auc_vl:.4f}  F1={best_f1:.4f}  "
      f"FAR={far_vl:.4f}")
print(f"  Test AUC={auc_te:.4f}  F1={f1_te:.4f}")

# ── Baseline 2: One-Class SVM ─────────────────────────────────
print(f"\n[2/3] One-Class SVM...")
# Use subset for speed (SVM is slow on large data)
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
X_normal_pca = pca.fit_transform(X_normal[:5000])
X_vl_pca     = pca.transform(X_vl)
X_te_pca     = pca.transform(X_te)

ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
ocsvm.fit(X_normal_pca[:5000])

scores_vl_svm = -ocsvm.score_samples(X_vl_pca)
scores_te_svm = -ocsvm.score_samples(X_te_pca)

best_f1_svm, best_t_svm = 0.0, 0.0
for t in np.percentile(
        scores_vl_svm, np.arange(80, 100, 0.5)):
    preds = (scores_vl_svm >= t).astype(int)
    if preds.sum() == 0:
        continue
    f1 = f1_score(y_vl, preds, zero_division=0)
    if f1 > best_f1_svm:
        best_f1_svm, best_t_svm = f1, t

auc_vl_svm = roc_auc_score(y_vl, scores_vl_svm)
auc_te_svm = roc_auc_score(y_te, scores_te_svm)
preds_te_svm = (scores_te_svm >= best_t_svm).astype(int)
f1_te_svm    = f1_score(y_te, preds_te_svm, zero_division=0)
far_vl_svm   = ((scores_vl_svm >= best_t_svm).astype(int) *
                 (1-y_vl)).sum() / max((y_vl==0).sum(), 1)

results['One-Class SVM'] = {
    'val_auc': auc_vl_svm, 'test_auc': auc_te_svm,
    'val_f1': best_f1_svm, 'test_f1': f1_te_svm,
    'val_far': float(far_vl_svm)
}
print(f"  Val  AUC={auc_vl_svm:.4f}  "
      f"F1={best_f1_svm:.4f}  FAR={far_vl_svm:.4f}")
print(f"  Test AUC={auc_te_svm:.4f}  F1={f1_te_svm:.4f}")

# ── Baseline 3: Simple LSTM (single stream) ───────────────────
print(f"\n[3/3] Simple LSTM (single stream, no attention)...")
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=71, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=0.3)
        self.fc = nn.Linear(hidden*2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out.mean(dim=1))

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
lstm   = SimpleLSTM().to(device)
opt    = torch.optim.Adam(lstm.parameters(), lr=1e-3)
crit   = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([7.87]).to(device))

# Quick training (10 epochs)
from torch.utils.data import DataLoader, TensorDataset
Xs_tr = torch.FloatTensor(data['X_s_train'])
yb_tr = torch.FloatTensor(data['y_b_train'])
dl    = DataLoader(
    TensorDataset(Xs_tr, yb_tr),
    batch_size=256, shuffle=True)

print(f"  Training Simple LSTM (10 epochs)...")
for ep in range(10):
    lstm.train()
    for xs, yb in dl:
        xs, yb = xs.to(device), yb.to(device)
        loss = crit(lstm(xs).squeeze(-1), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if ep % 5 == 4:
        print(f"    Epoch {ep+1}/10")

# Evaluate
lstm.eval()
def get_lstm_probs(X):
    all_p = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.FloatTensor(X[i:i+256]).to(device)
            p  = torch.sigmoid(
                lstm(xb).squeeze(-1)).cpu().numpy()
            all_p.append(p)
    return np.concatenate(all_p)

p_vl_lstm = get_lstm_probs(data['X_s_val'])
p_te_lstm = get_lstm_probs(data['X_s_test'])

best_f1_lstm, best_t_lstm = 0.0, 0.5
for t in np.arange(0.1, 0.95, 0.05):
    preds = (p_vl_lstm >= t).astype(int)
    if preds.sum() == 0:
        continue
    f1 = f1_score(y_vl, preds, zero_division=0)
    if f1 > best_f1_lstm:
        best_f1_lstm, best_t_lstm = f1, t

auc_vl_lstm = roc_auc_score(y_vl, p_vl_lstm)
auc_te_lstm = roc_auc_score(y_te, p_te_lstm)
preds_te_lstm = (p_te_lstm >= best_t_lstm).astype(int)
f1_te_lstm    = f1_score(
    y_te, preds_te_lstm, zero_division=0)
far_vl_lstm   = ((p_vl_lstm >= best_t_lstm).astype(int) *
                  (1-y_vl)).sum() / max((y_vl==0).sum(), 1)

results['Simple BiLSTM'] = {
    'val_auc': auc_vl_lstm, 'test_auc': auc_te_lstm,
    'val_f1': best_f1_lstm, 'test_f1': f1_te_lstm,
    'val_far': float(far_vl_lstm)
}
print(f"  Val  AUC={auc_vl_lstm:.4f}  "
      f"F1={best_f1_lstm:.4f}  FAR={far_vl_lstm:.4f}")
print(f"  Test AUC={auc_te_lstm:.4f}  F1={f1_te_lstm:.4f}")

# ── Final comparison table ────────────────────────────────────
print(f"\n{'='*65}")
print(f"COMPARISON TABLE (for paper)")
print(f"{'='*65}")
print(f"\n{'Method':25s}  "
      f"{'Val AUC':>8}  {'Val F1':>7}  {'Val FAR':>8}  "
      f"{'Test AUC':>9}  {'Test F1':>8}")
print(f"{'─'*65}")

for name, r in results.items():
    print(f"{name:25s}  "
          f"{r['val_auc']:>8.4f}  {r['val_f1']:>7.4f}  "
          f"{r['val_far']:>8.4f}  "
          f"{r['test_auc']:>9.4f}  {r['test_f1']:>8.4f}")

# Add TrustGate results
print(f"{'TrustGate v3':25s}  "
      f"{'0.9620':>8}  {'0.7389':>7}  {'0.0297':>8}  "
      f"{'0.6563':>9}  {'0.0000':>8}")
print(f"{'TrustGate Dual':25s}  "
      f"{'0.9620':>8}  {'0.7389':>7}  {'0.0297':>8}  "
      f"{'—':>9}  {'0.4582':>8}")

print(f"\n{'='*65}")
print(f"Saved results for paper Table 1")
print(f"{'='*65}")

import json
with open(
    r'D:\trustgate_pcaps\eval_outputs\baseline_results.json',
    'w') as f:
    json.dump(results, f, indent=2)