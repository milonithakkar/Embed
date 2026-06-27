# 02_evaluate.py — TrustGate Full Evaluation
# Save as: C:\Users\HP\Downloads\trustgate\02_evaluate.py
# Run: python 02_evaluate.py

import numpy as np
import torch
import torch.nn as nn
import sys
import os
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── CONFIG — must match train_v3.py exactly ───────────────────────────────────
DATA_PATH  = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'   # FIXED NPZ
CKPT_PATH = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
OUT_DIR    = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    pct_cands   = np.percentile(probs, np.arange(1, 100, 1))
    fixed_cands = np.arange(0.05, 0.96, 0.01)
    candidates  = np.unique(np.concatenate([pct_cands, fixed_cands]))
    for t in candidates:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t

# ── PRINT RESULTS BLOCK ───────────────────────────────────────────────────────
def print_results(split_name, probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    auc   = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)

    tn = int(cm[0,0]) if cm.shape == (2,2) else 0
    fp = int(cm[0,1]) if cm.shape == (2,2) else 0
    fn = int(cm[1,0]) if cm.shape == (2,2) else 0
    tp = int(cm[1,1]) if cm.shape == (2,2) else 0

    n_normal = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    far  = fp / max(n_normal, 1)
    miss = fn / max(n_attack, 1)

    print(f"\n  -- {split_name} Results (threshold={threshold:.4f}) --")
    print(f"  AUC              : {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  False Alarm Rate : {far:.4f}  ({far*100:.1f}%)")
    print(f"  Miss Rate        : {miss:.4f}  ({miss*100:.1f}%)")

    return dict(split=split_name, auc=auc, ap=ap, f1=f1,
                prec=prec, rec=rec, tp=tp, fp=fp, tn=tn, fn=fn,
                far=far, miss=miss, threshold=threshold)

# ── GET PREDICTIONS ───────────────────────────────────────────────────────────
def get_predictions(model, X_s, X_n, batch_size=512):
    model.eval()
    all_probs      = []
    all_cls_pred   = []
    all_comp_logits= []
    all_sensor_imp = []

    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_s),
        torch.FloatTensor(X_n)
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for xs, xn in loader:
            xs, xn = xs.to(device), xn.to(device)
            out = model(xs, xn)
            # out: bin_logit, cls_logits, comp_logits, attn_s2n, attn_n2s, sensor_imp
            bin_logit   = out[0].squeeze(-1)
            cls_logits  = out[1]
            comp_logits = out[2]
            sensor_imp  = out[5]

            probs = torch.sigmoid(bin_logit).cpu().numpy()
            all_probs.append(probs)
            all_cls_pred.append(cls_logits.argmax(-1).cpu().numpy())
            all_comp_logits.append(torch.sigmoid(comp_logits).cpu().numpy())
            all_sensor_imp.append(sensor_imp.cpu().numpy())

    return (np.concatenate(all_probs),
            np.concatenate(all_cls_pred),
            np.concatenate(all_comp_logits),
            np.concatenate(all_sensor_imp))

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TrustGate — Full Evaluation")
print("=" * 60)
print(f"\nDevice : {device}")
print(f"Data   : {DATA_PATH}")
print(f"Checkpoint: {CKPT_PATH}")

# Verify data path
if not os.path.exists(DATA_PATH):
    print(f"\nABORT: Data file not found: {DATA_PATH}")
    print(f"Make sure you are pointing to the FIXED npz file.")
    exit(1)

if not os.path.exists(CKPT_PATH):
    print(f"\nABORT: Checkpoint not found: {CKPT_PATH}")
    print(f"Run train_v3.py first.")
    exit(1)

# [1] Load model
print(f"\n[1/5] Loading model...")
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Checkpoint epoch : {ckpt['epoch']}")
print(f"  Checkpoint AUC   : {ckpt['monitor_value']:.4f}")
print(f"  Best threshold   : {ckpt.get('best_threshold', 'N/A')}")

# [2] Load data
print(f"\n[2/5] Loading data from {DATA_PATH}...")
data = np.load(DATA_PATH, allow_pickle=True)

X_s_val  = data['X_s_val']
X_n_val  = data['X_n_val']
y_b_val  = data['y_b_val'].astype(int)
y_c_val  = data['y_c_val'].astype(int)
y_p_val  = data['y_p_val'].astype(float)

X_s_test = data['X_s_test']
X_n_test = data['X_n_test']
y_b_test = data['y_b_test'].astype(int)
y_c_test = data['y_c_test'].astype(int)
y_p_test = data['y_p_test'].astype(float)

comp_names   = list(data['component_names'])
sensor_cols  = list(data['sensor_cols'])

print(f"  Val  : {len(y_b_val):,} samples "
      f"({(y_b_val==1).sum()} attacks, {(y_b_val==0).sum()} normal)")
print(f"  Test : {len(y_b_test):,} samples "
      f"({(y_b_test==1).sum()} attacks, {(y_b_test==0).sum()} normal)")

# Sanity check — confirm we have both classes
if len(np.unique(y_b_val)) < 2:
    print(f"\n  ABORT: Val set has only one class. Cannot compute AUC.")
    exit(1)

# [3] Get predictions
print(f"\n[3/5] Running inference...")
val_probs, val_cls, val_comp, val_simp = get_predictions(
    model, X_s_val, X_n_val)
test_probs, test_cls, test_comp, test_simp = get_predictions(
    model, X_s_test, X_n_test)

print(f"  Val  probs — min:{val_probs.min():.4f}  "
      f"max:{val_probs.max():.4f}  "
      f"mean:{val_probs.mean():.4f}")
print(f"  Test probs — min:{test_probs.min():.4f}  "
      f"max:{test_probs.max():.4f}  "
      f"mean:{test_probs.mean():.4f}")

# Quick AUC check before threshold calibration
quick_auc = roc_auc_score(y_b_val, val_probs)
print(f"\n  Quick AUC check (val): {quick_auc:.4f}")
if quick_auc < 0.70:
    print(f"  WARNING: AUC is low ({quick_auc:.4f}).")
    print(f"  Expected ~0.94 from training. Check data path is correct.")
elif quick_auc > 0.85:
    print(f"  AUC confirmed good ({quick_auc:.4f}) — matches training.")

# [4] Threshold calibration on val set
print(f"\n[4/5] Threshold calibration on validation set...")
best_f1, best_t = find_best_threshold(val_probs, y_b_val)
print(f"  Optimal threshold : {best_t:.4f}")
print(f"  Best F1 at tau*   : {best_f1:.4f}")

# Save threshold
thresh_path = os.path.join(OUT_DIR, 'optimal_threshold.json')
with open(thresh_path, 'w') as f:
    json.dump({'threshold': best_t, 'val_f1': best_f1,
               'val_auc': float(quick_auc)}, f, indent=2)
print(f"  Saved: {thresh_path}")

# [5] Full evaluation
print(f"\n[5/5] Full evaluation...")

# ── Validation ────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  VALIDATION SET")
print(f"{'─'*60}")
val_results = print_results("VAL", val_probs, y_b_val, best_t)

# Class accuracy on val
val_cls_acc = (val_cls == y_c_val).mean()
print(f"  Class accuracy   : {val_cls_acc:.4f}")

# Component F1 on val
print(f"\n  Component head F1 (val):")
val_comp_f1s = []
for i, name in enumerate(comp_names):
    true_i = y_p_val[:, i].astype(int)
    if true_i.sum() == 0:
        print(f"    Slot {i:2d} ({name:8s}): DEAD in val")
        continue
    pred_i = (val_comp[:, i] >= 0.5).astype(int)
    f1_i   = f1_score(true_i, pred_i, zero_division=0)
    val_comp_f1s.append(f1_i)
    print(f"    Slot {i:2d} ({name:8s}): F1={f1_i:.4f}  "
          f"(active={true_i.sum()} windows)")

# ── Test ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TEST SET")
print(f"{'─'*60}")
test_results = print_results("TEST", test_probs, y_b_test, best_t)

# Class accuracy on test
test_cls_acc = (test_cls == y_c_test).mean()
print(f"  Class accuracy   : {test_cls_acc:.4f}")

# Component F1 on test (zero-shot targets)
print(f"\n  Component head F1 (test) — zero-shot targets:")
for i, name in enumerate(comp_names):
    true_i = y_p_test[:, i].astype(int)
    if true_i.sum() == 0:
        continue
    pred_i = (test_comp[:, i] >= 0.5).astype(int)
    f1_i   = f1_score(true_i, pred_i, zero_division=0)
    seen   = "UNSEEN" if i in [3, 4, 21] else "seen"
    print(f"    Slot {i:2d} ({name:8s}): F1={f1_i:.4f}  "
          f"active={true_i.sum()}  [{seen}]")

# ── Threshold sweep table ──────────────────────────────────────
print(f"\n{'─'*60}")
print("  THRESHOLD SWEEP (val set)")
print(f"{'─'*60}")
print(f"  {'Thresh':>8}  {'F1':>6}  {'Prec':>6}  "
      f"{'Rec':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")

sweep_results = []
pct_cands   = np.percentile(val_probs, np.arange(5, 100, 5))
fixed_cands = np.arange(0.05, 0.96, 0.05)
candidates  = np.unique(np.concatenate([pct_cands, fixed_cands]))

for t in sorted(candidates):
    preds = (val_probs >= t).astype(int)
    if preds.sum() == 0 or preds.sum() == len(preds):
        continue
    f1   = f1_score(y_b_val, preds, zero_division=0)
    prec = precision_score(y_b_val, preds, zero_division=0)
    rec  = recall_score(y_b_val, preds, zero_division=0)
    tp   = int(((preds==1)&(y_b_val==1)).sum())
    fp   = int(((preds==1)&(y_b_val==0)).sum())
    fn   = int(((preds==0)&(y_b_val==1)).sum())
    sweep_results.append(dict(t=t, f1=f1, prec=prec, rec=rec,
                               tp=tp, fp=fp, fn=fn))
    marker = " <-- best" if abs(t - best_t) < 0.005 else ""
    print(f"  {t:>8.4f}  {f1:>6.4f}  {prec:>6.4f}  "
          f"{rec:>6.4f}  {tp:>5}  {fp:>5}  {fn:>5}{marker}")

# ── Save plots ─────────────────────────────────────────────────
print(f"\n  Saving plots...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ROC curve
    fpr_v, tpr_v, _ = roc_curve(y_b_val, val_probs)
    fpr_t, tpr_t, _ = roc_curve(y_b_test, test_probs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(fpr_v, tpr_v,
                 label=f"Val  AUC={val_results['auc']:.3f}", lw=2)
    axes[0].plot(fpr_t, tpr_t,
                 label=f"Test AUC={test_results['auc']:.3f}",
                 lw=2, linestyle='--')
    axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR curve
    pr_v, rc_v, _ = precision_recall_curve(y_b_val, val_probs)
    pr_t, rc_t, _ = precision_recall_curve(y_b_test, test_probs)

    axes[1].plot(rc_v, pr_v,
                 label=f"Val  AP={val_results['ap']:.3f}", lw=2)
    axes[1].plot(rc_t, pr_t,
                 label=f"Test AP={test_results['ap']:.3f}",
                 lw=2, linestyle='--')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 vs threshold
    if sweep_results:
        ts  = [r['t']  for r in sweep_results]
        f1s = [r['f1'] for r in sweep_results]
        prs = [r['prec'] for r in sweep_results]
        rcs = [r['rec'] for r in sweep_results]

        axes[2].plot(ts, f1s,   label='F1',       lw=2)
        axes[2].plot(ts, prs,   label='Precision', lw=2, linestyle='--')
        axes[2].plot(ts, rcs,   label='Recall',    lw=2, linestyle=':')
        axes[2].axvline(best_t, color='red', linestyle='--',
                        label=f'Best tau={best_t:.3f}')
        axes[2].set_xlabel("Threshold")
        axes[2].set_ylabel("Score")
        axes[2].set_title("F1 / Precision / Recall vs Threshold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'evaluation_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

except ImportError:
    print("  matplotlib not available — skipping plots")
except Exception as e:
    print(f"  Plot error: {e}")

# ── Save results JSON ──────────────────────────────────────────
results_path = os.path.join(OUT_DIR, 'evaluation_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump({
        'checkpoint_epoch': ckpt['epoch'],
        'checkpoint_auc':   ckpt['monitor_value'],
        'optimal_threshold': best_t,
        'val':  val_results,
        'test': test_results,
    }, f, indent=2)
print(f"  Saved: {results_path}")

# ── Final summary ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"\n  Val  AUC : {val_results['auc']:.4f}")
print(f"  Val  AP  : {val_results['ap']:.4f}")
print(f"  Val  F1* : {best_f1:.4f}  (tau={best_t:.4f})")
print(f"  Val  Prec: {val_results['prec']:.4f}")
print(f"  Val  Rec : {val_results['rec']:.4f}")
print(f"  Val  FAR : {val_results['far']:.4f}")
print(f"\n  Test AUC : {test_results['auc']:.4f}")
print(f"  Test AP  : {test_results['ap']:.4f}")
print(f"  Test F1* : {test_results['f1']:.4f}")
print(f"  Test Prec: {test_results['prec']:.4f}")
print(f"  Test Rec : {test_results['rec']:.4f}")
print(f"  Test FAR : {test_results['far']:.4f}")
print(f"\n  Outputs  : {OUT_DIR}")
print(f"{'='*60}")