# clean_baseline.py
# TrustGate — Baseline Comparison (Clean Version)
# Compares TrustGate against standard baselines
# Outputs to baseline_results_clean.json
# Run: python clean_baseline.py

import numpy as np
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score,
                              average_precision_score)

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')

DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'
OUT_PATH  = r'D:\trustgate_pcaps\eval_outputs\baseline_results_clean.json'
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

print("=" * 65)
print("TrustGate — Baseline Comparison (Clean)")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────
print(f"\nLoading data...")
data = np.load(DATA_PATH, allow_pickle=True)

X_s_tr = data['X_s_train']
X_n_tr = data['X_n_train']
X_s_vl = data['X_s_val']
X_n_vl = data['X_n_val']
X_s_te = data['X_s_test']
X_n_te = data['X_n_test']
y_b_tr = data['y_b_train'].astype(int)
y_b_vl = data['y_b_val'].astype(int)
y_b_te = data['y_b_test'].astype(int)

# Normal training windows only
normal_mask = (y_b_tr == 0)
X_s_normal  = X_s_tr[normal_mask]

# Flatten for sklearn
X_tr_flat = X_s_tr.reshape(len(X_s_tr), -1)
X_vl_flat = X_s_vl.reshape(len(X_s_vl), -1)
X_te_flat = X_s_te.reshape(len(X_s_te), -1)
X_no_flat = X_s_normal.reshape(len(X_s_normal), -1)

print(f"  Train normal: {X_s_normal.shape[0]:,}")
print(f"  Val samples:  {X_s_vl.shape[0]:,}")
print(f"  Test samples: {X_s_te.shape[0]:,}")

results = {}

def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.percentile(probs, np.arange(1, 100, 0.5)):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t

def compute_metrics(probs_vl, probs_te,
                    y_vl, y_te, name):
    best_f1, best_t = find_best_threshold(probs_vl, y_vl)
    preds_vl  = (probs_vl >= best_t).astype(int)
    preds_te  = (probs_te >= best_t).astype(int)

    auc_vl = roc_auc_score(y_vl, probs_vl)
    ap_vl  = average_precision_score(y_vl, probs_vl)
    f1_vl  = f1_score(y_vl, preds_vl, zero_division=0)
    pr_vl  = precision_score(y_vl, preds_vl,
                              zero_division=0)
    rc_vl  = recall_score(y_vl, preds_vl,
                           zero_division=0)
    far_vl = (((preds_vl==1)&(y_vl==0)).sum()
               / max((y_vl==0).sum(), 1))

    auc_te = roc_auc_score(y_te, probs_te)
    ap_te  = average_precision_score(y_te, probs_te)
    f1_te  = f1_score(y_te, preds_te, zero_division=0)
    pr_te  = precision_score(y_te, preds_te,
                              zero_division=0)
    rc_te  = recall_score(y_te, preds_te,
                           zero_division=0)
    far_te = (((preds_te==1)&(y_te==0)).sum()
               / max((y_te==0).sum(), 1))

    r = dict(
        val_auc=float(auc_vl),  val_ap=float(ap_vl),
        val_f1=float(f1_vl),    val_prec=float(pr_vl),
        val_rec=float(rc_vl),   val_far=float(far_vl),
        test_auc=float(auc_te), test_ap=float(ap_te),
        test_f1=float(f1_te),   test_prec=float(pr_te),
        test_rec=float(rc_te),  test_far=float(far_te),
    )
    print(f"\n  {name}:")
    print(f"    Val  AUC={auc_vl:.4f}  AP={ap_vl:.4f}  "
          f"F1={f1_vl:.4f}  FAR={far_vl:.4f}")
    print(f"    Test AUC={auc_te:.4f}  AP={ap_te:.4f}  "
          f"F1={f1_te:.4f}  FAR={far_te:.4f}")
    return r

# ── Baseline 1: Isolation Forest ──────────────────────────────
print(f"\n[1/4] Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=float(normal_mask.sum()==0 or
                        y_b_tr.mean()),
    random_state=42,
    n_jobs=-1)
iso.fit(X_no_flat)

s_vl = -iso.score_samples(X_vl_flat)
s_te = -iso.score_samples(X_te_flat)
# Normalize to [0,1]
s_min, s_max = s_vl.min(), s_vl.max()
s_vl_n = (s_vl - s_min) / max(s_max - s_min, 1e-8)
s_te_n = (s_te - s_min) / max(s_max - s_min, 1e-8)

results['Isolation Forest'] = compute_metrics(
    s_vl_n, s_te_n, y_b_vl, y_b_te,
    'Isolation Forest')

# ── Baseline 2: One-Class SVM ─────────────────────────────────
print(f"\n[2/4] One-Class SVM (with PCA)...")
pca = PCA(n_components=50, random_state=42)
X_no_pca = pca.fit_transform(X_no_flat[:5000])
X_vl_pca = pca.transform(X_vl_flat)
X_te_pca = pca.transform(X_te_flat)

svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
svm.fit(X_no_pca)

s_vl_svm = -svm.score_samples(X_vl_pca)
s_te_svm = -svm.score_samples(X_te_pca)
s_min = s_vl_svm.min()
s_max = s_vl_svm.max()
s_vl_svm = (s_vl_svm - s_min) / max(s_max - s_min, 1e-8)
s_te_svm = (s_te_svm - s_min) / max(s_max - s_min, 1e-8)

results['One-Class SVM'] = compute_metrics(
    s_vl_svm, s_te_svm, y_b_vl, y_b_te,
    'One-Class SVM')

# ── Baseline 3: Simple BiLSTM (no attention) ──────────────────
print(f"\n[3/4] Simple BiLSTM (no cross-attention)...")

class SimpleBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            71, 128, 2, batch_first=True,
            bidirectional=True, dropout=0.3)
        self.norm = nn.LayerNorm(256)
        self.fc   = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.norm(out).mean(dim=1))

simple = SimpleBiLSTM().to(DEVICE)
opt    = torch.optim.AdamW(
    simple.parameters(), lr=1e-3,
    weight_decay=0.01)

# Weighted BCE
pos_w = torch.tensor(
    [float((y_b_tr==0).sum()) /
     max((y_b_tr==1).sum(), 1)]).to(DEVICE)
crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)

Xs_tr_t = torch.FloatTensor(X_s_tr)
yb_tr_t = torch.FloatTensor(y_b_tr)
ds      = TensorDataset(Xs_tr_t, yb_tr_t)
loader  = DataLoader(ds, batch_size=256, shuffle=True)

print(f"  Training Simple BiLSTM (20 epochs)...")
for ep in range(20):
    simple.train()
    ep_loss = 0.0
    for xs, yb in loader:
        xs, yb = xs.to(DEVICE), yb.to(DEVICE)
        loss   = crit(simple(xs).squeeze(-1), yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            simple.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    if (ep+1) % 5 == 0:
        print(f"    Epoch {ep+1}/20  "
              f"loss={ep_loss/len(loader):.4f}")

simple.eval()
def lstm_probs(X):
    p = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.FloatTensor(
                X[i:i+256]).to(DEVICE)
            pb = torch.sigmoid(
                simple(xb).squeeze(-1))
            p.append(pb.cpu().numpy())
    return np.concatenate(p)

p_vl_lstm = lstm_probs(X_s_vl)
p_te_lstm = lstm_probs(X_s_te)

results['Simple BiLSTM'] = compute_metrics(
    p_vl_lstm, p_te_lstm, y_b_vl, y_b_te,
    'Simple BiLSTM')

# ── Baseline 4: Autoencoder ───────────────────────────────────
print(f"\n[4/4] Autoencoder (reconstruction error)...")

class Autoencoder(nn.Module):
    def __init__(self, input_dim=71*30):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128),       nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128),        nn.ReLU(),
            nn.Linear(128, 512),       nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

ae      = Autoencoder().to(DEVICE)
opt_ae  = torch.optim.Adam(ae.parameters(), lr=1e-3)
crit_ae = nn.MSELoss()

# Train on normal windows only
X_no_flat_t = torch.FloatTensor(X_no_flat)
ds_ae = TensorDataset(X_no_flat_t)
ld_ae = DataLoader(ds_ae, batch_size=256, shuffle=True)

print(f"  Training Autoencoder (20 epochs)...")
for ep in range(20):
    ae.train()
    ep_loss = 0.0
    for (xb,) in ld_ae:
        xb   = xb.to(DEVICE)
        loss = crit_ae(ae(xb), xb)
        opt_ae.zero_grad()
        loss.backward()
        opt_ae.step()
        ep_loss += loss.item()
    if (ep+1) % 5 == 0:
        print(f"    Epoch {ep+1}/20  "
              f"loss={ep_loss/len(ld_ae):.4f}")

ae.eval()
def ae_scores(X_flat):
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_flat), 256):
            xb   = torch.FloatTensor(
                X_flat[i:i+256]).to(DEVICE)
            recon = ae(xb)
            err  = ((recon - xb)**2).mean(dim=1)
            scores.append(err.cpu().numpy())
    s = np.concatenate(scores)
    return s

s_vl_ae = ae_scores(X_vl_flat)
s_te_ae = ae_scores(X_te_flat)
s_min   = s_vl_ae.min()
s_max   = s_vl_ae.max()
s_vl_ae = (s_vl_ae - s_min) / max(s_max-s_min, 1e-8)
s_te_ae = (s_te_ae - s_min) / max(s_max-s_min, 1e-8)

results['Autoencoder'] = compute_metrics(
    s_vl_ae, s_te_ae, y_b_vl, y_b_te,
    'Autoencoder')

# ── TrustGate results (from evaluation) ───────────────────────
results['TrustGate v3'] = dict(
    val_auc=0.9620, val_ap=0.7440,
    val_f1=0.7389,  val_prec=0.6962,
    val_rec=0.7872, val_far=0.0297,
    test_auc=0.6563, test_ap=0.3495,
    test_f1=0.0000,  test_prec=0.0000,
    test_rec=0.0000, test_far=0.0000,
)
results['TrustGate Dual'] = dict(
    val_auc=0.9620, val_ap=0.7440,
    val_f1=0.7389,  val_prec=0.6962,
    val_rec=0.7872, val_far=0.0297,
    test_auc=0.7381, test_ap=0.0,
    test_f1=0.4582,  test_prec=0.8197,
    test_rec=0.3180, test_far=0.0116,
)

# ── Final table ───────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"COMPARISON TABLE")
print(f"{'='*75}")
print(f"\n{'Method':22s}  "
      f"{'ValAUC':>7}  {'ValF1':>6}  {'ValFAR':>7}  "
      f"{'TestAUC':>8}  {'TestF1':>7}  {'TestFAR':>8}")
print(f"{'─'*75}")

for name, r in results.items():
    print(f"{name:22s}  "
          f"{r['val_auc']:>7.4f}  "
          f"{r['val_f1']:>6.4f}  "
          f"{r['val_far']:>7.4f}  "
          f"{r['test_auc']:>8.4f}  "
          f"{r['test_f1']:>7.4f}  "
          f"{r['test_far']:>8.4f}")

# ── Save ──────────────────────────────────────────────────────
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*75}")
print(f"Saved: {OUT_PATH}")
print(f"{'='*75}")