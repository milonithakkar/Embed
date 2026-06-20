# 02_evaluate.py
# PURPOSE: Full evaluation of the trained model.
#          - Loads best checkpoint (Epoch 5, AUC=0.9245)
#          - Calibrates optimal threshold on validation set
#          - Reports all metrics on val AND test
#          - Saves plots and JSON results
# Run: python 02_evaluate.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys

# Add trustgate directory to path so model.py is found
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── PATHS ─────────────────────────────────────────────────────
DATA_PATH   = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
CHECKPOINT  = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
OUTPUT_DIR  = r'D:\trustgate_pcaps\eval_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE  = 256

print("=" * 60)
print("TrustGate — Evaluation + Threshold Calibration")
print("=" * 60)

# ── Dataset ───────────────────────────────────────────────────
class TrustGateDataset(Dataset):
    def __init__(self, data, split):
        def get(a, b=None):
            if a in data: return data[a]
            if b and b in data: return data[b]
            raise KeyError(f"Cannot find key {a} or {b}")

        self.X_s = torch.FloatTensor(get(f'X_s_{split}', f'{split}_X_sensor'))
        self.X_n = torch.FloatTensor(get(f'X_n_{split}', f'{split}_X_network'))
        self.y_b = torch.FloatTensor(get(f'y_b_{split}', f'{split}_y_binary'))
        self.y_c = torch.LongTensor(get(f'y_c_{split}',  f'{split}_y_class'))
        self.y_p = torch.FloatTensor(get(f'y_p_{split}',  f'{split}_y_comp'))

    def __len__(self): return len(self.y_b)

    def __getitem__(self, idx):
        return self.X_s[idx], self.X_n[idx], self.y_b[idx], self.y_c[idx], self.y_p[idx]


# ── Load model ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=device)
print(f"  Saved at epoch : {ckpt['epoch']}")
print(f"  Saved AUC      : {ckpt['monitor_value']:.4f}")

model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Model loaded ✓")


# ── Collect predictions ───────────────────────────────────────
@torch.no_grad()
def collect_predictions(split):
    data = np.load(DATA_PATH, allow_pickle=True)
    ds   = TrustGateDataset(data, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, drop_last=False)

    all_bin_probs  = []
    all_bin_labels = []
    all_cls_preds  = []
    all_cls_labels = []
    all_comp_probs = []
    all_comp_labels = []

    for x_s, x_n, y_b, y_c, y_p in loader:
        x_s = x_s.to(device)
        x_n = x_n.to(device)

        # YOUR model returns 6 values — adjust if different
        outputs = model(x_s, x_n)

        # Handle both (logit,) and (logit, cls, comp, ...) outputs
        if isinstance(outputs, (tuple, list)):
            bin_logit  = outputs[0]
            cls_logits = outputs[1] if len(outputs) > 1 else None
            comp_logits= outputs[2] if len(outputs) > 2 else None
        else:
            bin_logit  = outputs
            cls_logits = None
            comp_logits= None

        bin_prob = torch.sigmoid(bin_logit.squeeze(-1))
        all_bin_probs.append(bin_prob.cpu().numpy())
        all_bin_labels.append(y_b.numpy())

        if cls_logits is not None:
            cls_pred = cls_logits.argmax(dim=-1)
            all_cls_preds.append(cls_pred.cpu().numpy())
            all_cls_labels.append(y_c.numpy())

        if comp_logits is not None:
            comp_prob = torch.sigmoid(comp_logits)
            all_comp_probs.append(comp_prob.cpu().numpy())
            all_comp_labels.append(y_p.numpy())

    result = {
        'bin_probs':   np.concatenate(all_bin_probs),
        'bin_labels':  np.concatenate(all_bin_labels).astype(int),
    }
    if all_cls_preds:
        result['cls_preds']  = np.concatenate(all_cls_preds)
        result['cls_labels'] = np.concatenate(all_cls_labels)
    if all_comp_probs:
        result['comp_probs']  = np.concatenate(all_comp_probs)
        result['comp_labels'] = np.concatenate(all_comp_labels)

    return result


# ── Threshold calibration ─────────────────────────────────────
def calibrate_threshold(probs, labels, metric='f1'):
    """
    Sweep every threshold from 0.01 to 0.99.
    Return the one that maximizes the chosen metric.
    This is the core fix — never assume 0.5 is optimal.
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    scores     = []
    best_score = 0.0
    best_t     = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            scores.append(0.0)
            continue
        if metric == 'f1':
            s = f1_score(labels, preds, zero_division=0)
        elif metric == 'precision':
            s = precision_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            s = recall_score(labels, preds, zero_division=0)
        scores.append(s)
        if s > best_score:
            best_score = s
            best_t     = t

    return best_t, best_score, thresholds, np.array(scores)


# ── Compute all metrics ────────────────────────────────────────
def compute_metrics(probs, labels, threshold, split_name):
    preds = (probs >= threshold).astype(int)

    auc  = roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0
    ap   = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    f1   = f1_score(labels, preds,         zero_division=0)
    prec = precision_score(labels, preds,  zero_division=0)
    rec  = recall_score(labels, preds,     zero_division=0)

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    far  = fp / max(fp + tn, 1)   # False Alarm Rate
    miss = fn / max(fn + tp, 1)   # Miss Rate

    print(f"\n  ── {split_name.upper()} Results (threshold={threshold:.2f}) ──")
    print(f"  AUC              : {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  False Alarm Rate : {far:.4f}  ({100*far:.1f}%)")
    print(f"  Miss Rate        : {miss:.4f}  ({100*miss:.1f}%)")

    return {
        'split': split_name, 'threshold': float(threshold),
        'auc': float(auc), 'ap': float(ap),
        'f1': float(f1), 'precision': float(prec), 'recall': float(rec),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'false_alarm_rate': float(far), 'miss_rate': float(miss),
    }


# ── Plotting ──────────────────────────────────────────────────
def plot_threshold_curve(thresholds, f1_scores, best_t, best_f1, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    ax.axvline(best_t, color='red', linestyle='--', linewidth=1.5,
               label=f'Optimal τ={best_t:.2f}  (F1={best_f1:.4f})')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1,
               label='Default τ=0.50')
    ax.set_xlabel('Decision Threshold (τ)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('TrustGate — Threshold Calibration on Validation Set', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(labels, probs, auc_val, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Binary Attack Detection', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_pr_curve(labels, probs, ap_val, path):
    prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
    baseline = labels.sum() / len(labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec_curve, prec_curve, 'g-', linewidth=2,
            label=f'PR Curve (AP={ap_val:.4f})')
    ax.axhline(baseline, color='gray', linestyle='--',
               label=f'Baseline (random) = {baseline:.3f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve — Binary Attack Detection', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(tn, fp, fn, tp, split, threshold, path):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    labels_text = ['Normal', 'Attack']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_text); ax.set_yticklabels(labels_text)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — {split} (τ={threshold:.2f})', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── MAIN EVALUATION ───────────────────────────────────────────
all_results = {}

# ── Validation Set ────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  VALIDATION SET")
print(f"{'─'*60}")
val_preds = collect_predictions('val')

# Calibrate threshold on validation set
best_t, best_f1, thr_arr, f1_arr = calibrate_threshold(
    val_preds['bin_probs'],
    val_preds['bin_labels'],
    metric='f1'
)
print(f"\n  Threshold calibration complete:")
print(f"    Optimal threshold : {best_t:.2f}")
print(f"    Best F1 at τ*     : {best_f1:.4f}")
print(f"    F1 at τ=0.50      : {f1_arr[49]:.4f}  (index 49 ≈ 0.50)")

# Save calibrated threshold
thresh_path = os.path.join(OUTPUT_DIR, 'optimal_threshold.json')
with open(thresh_path, 'w') as f:
    json.dump({'threshold': float(best_t), 'val_f1': float(best_f1)}, f)
print(f"  Saved optimal threshold → {thresh_path}")

# Val metrics
val_metrics = compute_metrics(
    val_preds['bin_probs'],
    val_preds['bin_labels'],
    best_t, 'validation'
)
all_results['validation'] = val_metrics

# Val plots
print(f"\n  Generating plots...")
plot_threshold_curve(
    thr_arr, f1_arr, best_t, best_f1,
    os.path.join(OUTPUT_DIR, 'threshold_calibration.png')
)
plot_roc_curve(
    val_preds['bin_labels'],
    val_preds['bin_probs'],
    val_metrics['auc'],
    os.path.join(OUTPUT_DIR, 'roc_curve_val.png')
)
plot_pr_curve(
    val_preds['bin_labels'],
    val_preds['bin_probs'],
    val_metrics['ap'],
    os.path.join(OUTPUT_DIR, 'pr_curve_val.png')
)
plot_confusion_matrix(
    val_metrics['tn'], val_metrics['fp'],
    val_metrics['fn'], val_metrics['tp'],
    'Validation', best_t,
    os.path.join(OUTPUT_DIR, 'confusion_matrix_val.png')
)

# Component metrics on val
if 'comp_probs' in val_preds:
    comp_preds_bin = (val_preds['comp_probs'] >= 0.5).astype(int)
    cmp_f1 = f1_score(val_preds['comp_labels'].flatten(),
                      comp_preds_bin.flatten(), zero_division=0)
    print(f"\n  Component Head F1 (val): {cmp_f1:.4f}")
    all_results['validation']['comp_f1'] = float(cmp_f1)

    # Per-slot breakdown
    n_slots = val_preds['comp_labels'].shape[1]
    print(f"\n  Per-slot Component F1 (val):")
    for slot in range(n_slots):
        sl = val_preds['comp_labels'][:, slot]
        sp = comp_preds_bin[:, slot]
        if sl.sum() > 0:
            sf1 = f1_score(sl, sp, zero_division=0)
            print(f"    Slot {slot:2d}: F1={sf1:.4f}  "
                  f"(active={int(sl.sum())} windows)")
        else:
            print(f"    Slot {slot:2d}: DEAD (always 0 in val)")

# ── Test Set ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TEST SET")
print(f"{'─'*60}")

try:
    test_preds = collect_predictions('test')

    # Use calibrated threshold from val set
    print(f"\n  Using calibrated threshold={best_t:.2f} from validation")
    test_metrics = compute_metrics(
        test_preds['bin_probs'],
        test_preds['bin_labels'],
        best_t, 'test'
    )
    all_results['test'] = test_metrics

    # Test plots
    plot_roc_curve(
        test_preds['bin_labels'],
        test_preds['bin_probs'],
        test_metrics['auc'],
        os.path.join(OUTPUT_DIR, 'roc_curve_test.png')
    )
    plot_confusion_matrix(
        test_metrics['tn'], test_metrics['fp'],
        test_metrics['fn'], test_metrics['tp'],
        'Test', best_t,
        os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png')
    )

    if 'comp_probs' in test_preds:
        comp_preds_bin = (test_preds['comp_probs'] >= 0.5).astype(int)
        n_slots = test_preds['comp_labels'].shape[1]
        print(f"\n  Per-slot Component F1 (test):")
        for slot in range(n_slots):
            sl = test_preds['comp_labels'][:, slot]
            sp = comp_preds_bin[:, slot]
            if sl.sum() > 0:
                sf1 = f1_score(sl, sp, zero_division=0)
                print(f"    Slot {slot:2d}: F1={sf1:.4f}  "
                      f"(active={int(sl.sum())} windows)")
            else:
                print(f"    Slot {slot:2d}: DEAD — label starvation confirmed")

except Exception as e:
    print(f"  ⚠ Test set evaluation failed: {e}")
    print(f"    This is expected if no test split exists in NPZ.")

# ── Save all results ──────────────────────────────────────────
results_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"\nKey Results:")
print(f"  Val AUC  : {all_results['validation']['auc']:.4f}")
print(f"  Val F1*  : {all_results['validation']['f1']:.4f}  (at τ={best_t:.2f})")
print(f"  Val Prec : {all_results['validation']['precision']:.4f}")
print(f"  Val Rec  : {all_results['validation']['recall']:.4f}")
if 'test' in all_results:
    print(f"  Test AUC : {all_results['test']['auc']:.4f}")
    print(f"  Test F1* : {all_results['test']['f1']:.4f}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"{'='*60}")