# =================================================================
# evaluate.py — TrustGate Model Evaluation
# Loads best_model.pth → threshold calibration on val →
# final metrics on test set
# Run from: C:\Users\HP\Downloads\trustgate\
# =================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix,
    precision_recall_curve
)

# ================================================================
# CONSTANTS — must match train1.py exactly
# ================================================================
DATA_PATH     = './trustgate_data/swat_final.npz'
CHECKPOINT    = './trustgate_data/best_model.pth'
THRESHOLD_OUT = './trustgate_data/threshold.json'
RESULTS_OUT   = './trustgate_data/test_results.txt'

BATCH_SIZE    = 512       # larger — no gradients needed
WINDOW_SIZE   = 30
SENSOR_FEATS  = 44
NETWORK_FEATS = 132
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
FUSION_DIM    = 256
NUM_WORKERS   = 0


# ================================================================
# MODEL DEFINITION — exact copy from train1.py
# (needed to load checkpoint)
# ================================================================

class SensorStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size    = SENSOR_FEATS,
            hidden_size   = HIDDEN_SIZE,
            num_layers    = NUM_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = DROPOUT,
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class NetworkStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size    = NETWORK_FEATS,
            hidden_size   = HIDDEN_SIZE,
            num_layers    = NUM_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = DROPOUT,
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class BahdanauCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int = FUSION_DIM):
        super().__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, sensor_out, network_out):
        query        = self.W_query(sensor_out)
        key          = self.W_key(network_out)
        energy       = torch.tanh(query + key)
        energy       = self.dropout(energy)
        scores       = self.v(energy).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        context = (attn_weights.unsqueeze(-1) * network_out).sum(dim=1)
        return context, attn_weights


class SensorImportance(nn.Module):
    def __init__(self):
        super().__init__()
        self.importance_net = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, SENSOR_FEATS),
        )

    def forward(self, sensor_out):
        pooled     = sensor_out.mean(dim=1)
        scores     = self.importance_net(pooled)
        importance = F.softmax(scores, dim=-1)
        return importance


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = HIDDEN_SIZE * 2 + FUSION_DIM
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.attack_head   = nn.Linear(64, 1)
        self.severity_head = nn.Linear(64, 1)

    def forward(self, fused):
        features     = self.classifier(fused)
        attack_logit = self.attack_head(features)
        severity     = torch.sigmoid(self.severity_head(features))
        return attack_logit, severity


class TrustGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor_stream  = SensorStream()
        self.network_stream = NetworkStream()
        self.attention      = BahdanauCrossAttention()
        self.sensor_imp     = SensorImportance()
        self.head           = ClassificationHead()
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if 'weight_ih' in pname:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in pname:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in pname:
                        nn.init.zeros_(param.data)
                        n = param.size(0)
                        param.data[n//4 : n//2].fill_(1.0)

    def forward(self, x_sensor, x_network):
        sensor_out  = self.sensor_stream(x_sensor)
        network_out = self.network_stream(x_network)
        context, attn_weights = self.attention(sensor_out, network_out)
        sensor_scores = self.sensor_imp(sensor_out)
        sensor_last   = sensor_out[:, -1, :]
        fused         = torch.cat([sensor_last, context], dim=-1)
        attack_logit, severity = self.head(fused)
        return attack_logit, severity, attn_weights, sensor_scores


# ================================================================
# DATASET — minimal version for evaluation
# ================================================================

class TrustGateDataset(Dataset):
    def __init__(self, X_s, X_n, y):
        self.X_s = torch.from_numpy(X_s.astype(np.float32))
        self.X_n = torch.from_numpy(X_n.astype(np.float32))
        self.y   = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_s[idx], self.X_n[idx], self.y[idx]


# ================================================================
# INFERENCE — collect all probs + labels from a loader
# ================================================================

def run_inference(model, loader, device):
    """
    Runs model in eval mode over full loader.
    Returns:
      all_probs        : (N,) float — attack probabilities
      all_labels       : (N,) int   — ground truth
      all_severity     : (N,) float — severity scores
      all_attn         : (N, 30)    — attention weights
      all_sensor_scores: (N, 44)    — sensor importance
    """
    model.eval()

    all_probs         = []
    all_labels        = []
    all_severity      = []
    all_attn          = []
    all_sensor_scores = []

    with torch.no_grad():
        for xs, xn, y in loader:
            xs = xs.to(device)
            xn = xn.to(device)

            logit, severity, attn, sensor_imp = model(xs, xn)

            probs = torch.sigmoid(logit.squeeze(1))

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
            all_severity.append(severity.squeeze(1).cpu().numpy())
            all_attn.append(attn.cpu().numpy())
            all_sensor_scores.append(sensor_imp.cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels).astype(int),
        np.concatenate(all_severity),
        np.concatenate(all_attn),
        np.concatenate(all_sensor_scores),
    )


# ================================================================
# THRESHOLD CALIBRATION — find best F1 threshold on val set
# ================================================================

def calibrate_threshold(val_probs, val_labels):
    """
    Searches threshold space [0.01, 0.99] in steps of 0.01.
    Finds threshold maximizing F1 on val set.
    Applied to test set — never fit on test data.

    Paper line:
      "The decision threshold was calibrated on the validation
       set by maximizing F1-score, then applied to the held-out
       test set to produce final reported metrics."
    """
    print("\n[3/5] Threshold calibration on val set...")
    print(f"  Searching threshold space [0.01 → 0.99]...")

    best_thresh = 0.5
    best_f1     = 0.0
    results     = []

    thresholds = np.arange(0.01, 1.00, 0.01)

    for t in thresholds:
        preds = (val_probs >= t).astype(int)

        # Guard: skip if all predictions same class
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue

        f1 = f1_score(val_labels, preds,
                      pos_label=1, zero_division=0)
        results.append((t, f1))

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t

    # Show top 5 thresholds by F1
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 thresholds by val F1:")
    print(f"  {'Threshold':>10} | {'F1':>7}")
    print(f"  {'-'*22}")
    for t, f1 in results[:5]:
        marker = " ← SELECTED" if t == best_thresh else ""
        print(f"  {t:>10.2f} | {f1:>7.4f}{marker}")

    print(f"\n  Best threshold : {best_thresh:.2f}")
    print(f"  Best val F1    : {best_f1:.4f}")

    return best_thresh, best_f1


# ================================================================
# METRICS — full report on test set
# ================================================================

def compute_metrics(probs, labels, threshold, split_name='TEST'):
    """
    Computes all 6 paper metrics at given threshold.
    Returns dict of metric name → value.
    """
    preds = (probs >= threshold).astype(int)

    # Guard: if no attacks predicted at all
    if preds.sum() == 0:
        print(f"  WARNING: No attacks predicted at threshold={threshold:.2f}")
        print(f"  Try lower threshold — returning zeros")
        return {
            'auc': roc_auc_score(labels, probs),
            'f1': 0.0, 'precision': 0.0,
            'recall': 0.0, 'fpr': 0.0, 'accuracy': 0.0
        }

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    auc       = roc_auc_score(labels, probs)
    f1        = f1_score(labels, preds, pos_label=1, zero_division=0)
    precision = precision_score(labels, preds,
                                pos_label=1, zero_division=0)
    recall    = recall_score(labels, preds,
                             pos_label=1, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy  = (tp + tn) / len(labels)

    print(f"\n  ── {split_name} SET METRICS "
          f"(threshold={threshold:.2f}) ──")
    print(f"\n  {'Metric':<25} {'Value':>8}  {'Target':>8}")
    print(f"  {'-'*45}")
    print(f"  {'ROC-AUC':<25} {auc:>8.4f}  {'> 0.90':>8}")
    print(f"  {'F1 (Attack)':<25} {f1:>8.4f}  {'> 0.80':>8}")
    print(f"  {'Recall (Attack)':<25} {recall:>8.4f}  {'> 0.85':>8}")
    print(f"  {'Precision (Attack)':<25} {precision:>8.4f}  {'> 0.75':>8}")
    print(f"  {'False Positive Rate':<25} {fpr:>8.4f}  {'< 0.05':>8}")
    print(f"  {'Accuracy':<25} {accuracy:>8.4f}  {'misleading':>8}")

    print(f"\n  ── Confusion Matrix ──")
    print(f"                    Predicted")
    print(f"                  Normal  Attack")
    print(f"  Actual Normal   {tn:>6}  {fp:>6}")
    print(f"  Actual Attack   {fn:>6}  {tp:>6}")

    print(f"\n  ── Operational Summary ──")
    total_attacks = tp + fn
    caught        = tp
    missed        = fn
    false_alarms  = fp

    print(f"  Total attack windows  : {total_attacks:,}")
    print(f"  Caught (TP)           : {caught:,}  "
          f"({caught/total_attacks*100:.1f}%)")
    print(f"  Missed (FN)           : {missed:,}  "
          f"({missed/total_attacks*100:.1f}%) ← physical damage risk")
    print(f"  False alarms (FP)     : {false_alarms:,}  "
          f"({false_alarms/(tn+fp)*100:.2f}% of normal windows)")

    return {
        'auc': auc, 'f1': f1, 'precision': precision,
        'recall': recall, 'fpr': fpr, 'accuracy': accuracy,
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
        'threshold': threshold
    }


# ================================================================
# ATTENTION VALIDATION — confirm model attends to right sensors
# ================================================================

def validate_attention(sensor_scores, labels, sensor_cols):
    """
    Compares model sensor importance during attack windows
    vs ground truth deviation table from verify_label.py.

    This is Table 2 in your paper:
      "Attention mechanism targets physically meaningful
       sensors consistent with Ahmed et al. 2017."
    """
    print(f"\n  ── Attention Validation ──")
    print(f"  Top 10 sensors model attends to during ATTACK windows:")

    attack_idx  = np.where(labels == 1)[0]
    normal_idx  = np.where(labels == 0)[0]

    if len(attack_idx) == 0:
        print("  No attack windows in this split — skipping")
        return

    # Mean sensor importance during attacks vs normal
    atk_importance  = sensor_scores[attack_idx].mean(axis=0)
    norm_importance = sensor_scores[normal_idx].mean(axis=0)
    delta           = atk_importance - norm_importance

    # Rank by delta
    ranked = np.argsort(delta)[::-1]

    print(f"\n  {'Rank':>4} | {'Sensor':>10} | "
          f"{'Atk Imp':>8} | {'Nrm Imp':>8} | {'Delta':>8}")
    print(f"  {'-'*50}")

    for rank, idx in enumerate(ranked[:10], 1):
        name = sensor_cols[idx] if idx < len(sensor_cols) else f"S{idx}"
        print(f"  {rank:>4} | {name:>10} | "
              f"{atk_importance[idx]:>8.4f} | "
              f"{norm_importance[idx]:>8.4f} | "
              f"{delta[idx]:>8.4f}")

    # Check if known attack sensors are in top 10
    known_attack_sensors = [
        'FIT503', 'P501', 'FIT504', 'P402',
        'PIT501', 'FIT501', 'PIT503', 'FIT502',
        'UV401', 'FIT401', 'AIT402', 'AIT501', 'LIT401'
    ]

    top10_names = [
        sensor_cols[ranked[i]] if ranked[i] < len(sensor_cols)
        else f"S{ranked[i]}"
        for i in range(10)
    ]

    matches = [s for s in known_attack_sensors if s in top10_names]
    print(f"\n  Known attack sensors in model top-10: "
          f"{len(matches)}/10")
    print(f"  Matched: {matches}")

    if len(matches) >= 5:
        print(f"  [OK] Attention mechanism validated against "
              f"Ahmed et al. 2017")
    else:
        print(f"  [NOTE] Fewer matches than expected — "
              f"model may use different sensor combinations")


# ================================================================
# SAVE RESULTS
# ================================================================

def save_results(metrics, threshold, results_path, threshold_path):
    """Saves threshold.json for infer.py and full results txt."""
    import json

    # Save threshold for deployment
    threshold_data = {
        'threshold'  : float(threshold),
        'val_f1'     : float(metrics.get('f1', 0)),
        'description': 'Optimal F1 threshold calibrated on val set',
        'apply_to'   : 'sigmoid output of attack_logit'
    }
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f"\n  Threshold saved → {threshold_path}")

    # Save full results
    with open(results_path, 'w') as f:
        f.write("TrustGate — Test Set Evaluation Results\n")
        f.write("="*50 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"  Results saved  → {results_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("TrustGate — Model Evaluation")
    print("=" * 60)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\n  Device: {device}")

    # ── [1/5] Load data ───────────────────────────────────────────
    print(f"\n[1/5] Loading data...")
    data = np.load(DATA_PATH, allow_pickle=True)

    val_ds  = TrustGateDataset(
        data['X_s_val'], data['X_n_val'], data['y_val']
    )
    test_ds = TrustGateDataset(
        data['X_s_test'], data['X_n_test'], data['y_test']
    )

    sensor_cols = list(data['sensor_cols'])

    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = False,
    )

    print(f"  Val  : {len(val_ds):,} windows  "
          f"({int(data['y_val'].sum())} attacks)")
    print(f"  Test : {len(test_ds):,} windows  "
          f"({int(data['y_test'].sum())} attacks)")

    # ── [2/5] Load model ──────────────────────────────────────────
    print(f"\n[2/5] Loading checkpoint...")
    model = TrustGateModel().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    saved_epoch = checkpoint['epoch']
    saved_auc   = checkpoint['val_auc']
    print(f"  Loaded checkpoint from epoch {saved_epoch}")
    print(f"  Saved val AUC : {saved_auc:.4f}")

    # ── [3/5] Val inference + threshold calibration ───────────────
    print(f"\n[3/5] Running inference on val set...")
    (val_probs, val_labels,
     val_severity, val_attn,
     val_sensor_scores) = run_inference(model, val_loader, device)

    print(f"  Val probs  — min:{val_probs.min():.4f} "
          f"max:{val_probs.max():.4f} "
          f"mean:{val_probs.mean():.4f}")
    print(f"  Val AUC    : "
          f"{roc_auc_score(val_labels, val_probs):.4f}")

    best_thresh, best_val_f1 = calibrate_threshold(
        val_probs, val_labels
    )

    # ── [4/5] Test inference + final metrics ──────────────────────
    print(f"\n[4/5] Running inference on test set...")
    (test_probs, test_labels,
     test_severity, test_attn,
     test_sensor_scores) = run_inference(model, test_loader, device)

    print(f"  Test probs — min:{test_probs.min():.4f} "
          f"max:{test_probs.max():.4f} "
          f"mean:{test_probs.mean():.4f}")
    print(f"  Test AUC   : "
          f"{roc_auc_score(test_labels, test_probs):.4f}")

    metrics = compute_metrics(
        test_probs, test_labels,
        best_thresh, split_name='TEST'
    )

    # ── [5/5] Attention validation + save ─────────────────────────
    print(f"\n[5/5] Attention validation...")
    validate_attention(
        test_sensor_scores, test_labels, sensor_cols
    )

    save_results(
        metrics, best_thresh, RESULTS_OUT, THRESHOLD_OUT
    )

    # ── Final verdict ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")

    auc  = metrics['auc']
    f1   = metrics['f1']
    rec  = metrics['recall']
    prec = metrics['precision']
    fpr  = metrics['fpr']

    print(f"\n  METRIC TARGETS vs ACHIEVED:")
    print(f"  {'Metric':<22} {'Target':>8} {'Got':>8} {'Pass':>6}")
    print(f"  {'-'*48}")

    checks = [
        ('ROC-AUC',   0.90, auc,  auc  >= 0.90),
        ('F1',        0.80, f1,   f1   >= 0.80),
        ('Recall',    0.85, rec,  rec  >= 0.85),
        ('Precision', 0.75, prec, prec >= 0.75),
        ('FPR',       0.05, fpr,  fpr  <= 0.05),
    ]

    passed = 0
    for name, target, got, ok in checks:
        symbol = '✓' if ok else '✗'
        cmp    = '<' if name == 'FPR' else '>'
        print(f"  {name:<22} {cmp}{target:>6.2f} "
              f"{got:>8.4f} {symbol:>6}")
        if ok:
            passed += 1

    print(f"\n  Passed {passed}/5 targets")

    if passed >= 4:
        print(f"\n  VERDICT: STRONG — proceed to OpenVINO export")
    elif passed >= 3:
        print(f"\n  VERDICT: ACCEPTABLE — consider retraining v2")
        print(f"  Current model is usable for demo and paper")
    else:
        print(f"\n  VERDICT: RETRAIN — apply fixes in train1.py")
        print(f"  Increase DROPOUT to 0.4, weight_decay to 1e-3")

    print(f"\n  Next step based on verdict:")
    print(f"    STRONG/ACCEPTABLE → python export_openvino.py")
    print(f"    RETRAIN           → apply fixes, rerun train1.py")
    print(f"{'='*60}")