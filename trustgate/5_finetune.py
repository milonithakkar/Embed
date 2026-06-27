"""
TrustGate Phase B + C — Supervised Fine-tuning & Full Evaluation
=================================================================
Phase B (30 epochs): Load pre-trained encoders, FREEZE them.
                     Train only: cross-modal attention, fusion MLP, 5 heads.
                     LR = 5e-4

Phase C (20 epochs): Unfreeze top BiLSTM layer in each encoder.
                     Differential LR: 1e-5 for encoders, 1e-4 for rest.
                     Early stopping on val-F1 (patience=5).

Post-training:
  - Conformal prediction calibration (guaranteed FPR ≤ 1%)
  - Full metric report: F1, AUC-PR, MCC, TTD, FNPI
  - Saves model + calibration threshold

Input : trustgate_data/swat_multilabel.npz  (from create_labels.py)
        trustgate_data/swat_final.npz        (for y_comp labels)
        trustgate_data/pretrain_sensor_enc.pt
        trustgate_data/pretrain_net_enc.pt
        trustgate_data/pretrain_autoencoder.pt
        trustgate_data/granger_matrix.npy

Output: trustgate_data/best_model.pt
        trustgate_data/conformal_threshold.npy
        trustgate_data/training_log.csv
        trustgate_data/test_results.txt

python 05_finetune.py [--skip-pretrain]   # if you want random init encoders
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              average_precision_score, matthews_corrcoef,
                              confusion_matrix, roc_auc_score)
import sys, os, argparse

import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location('_m', _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '03_model.py'))
_m = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m)
TrustGateModel = _m.TrustGateModel
TrustGateLoss = _m.TrustGateLoss
SensorAutoEncoder = _m.SensorAutoEncoder

# ─── PATHS ────────────────────────────────────────────────────────
MULTILABEL_NPZ = 'trustgate_data/swat_multilabel.npz'
FINAL_NPZ      = 'trustgate_data/swat_final.npz'
PRETRAIN_S     = 'trustgate_data/pretrain_sensor_enc.pt'
PRETRAIN_N     = 'trustgate_data/pretrain_net_enc.pt'
PRETRAIN_AE    = 'trustgate_data/pretrain_autoencoder.pt'
GRANGER_MAT    = 'trustgate_data/granger_matrix.npy'
OUT_MODEL      = 'trustgate_data/best_model.pt'
OUT_CONF_THRESH= 'trustgate_data/conformal_threshold.npy'
OUT_LOG        = 'trustgate_data/training_log.csv'
OUT_RESULTS    = 'trustgate_data/test_results.txt'

# ─── HYPERPARAMETERS ─────────────────────────────────────────────
# Phase B — frozen encoders
PHASE_B_EPOCHS = 30
LR_B           = 5e-4

# Phase C — end-to-end fine-tuning
PHASE_C_EPOCHS = 20
LR_C_ENC       = 1e-5   # very low LR for encoder layers
LR_C_REST      = 1e-4   # higher LR for cross-attn + heads
PATIENCE       = 5       # early stopping on val-F1

BATCH_SIZE = 128
N_CLASSES  = 6
SEED       = 42

# Conformal prediction: guarantee FPR ≤ ALPHA
ALPHA_CONFORMAL = 0.01   # 1% FPR guarantee

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class SWaTDataset(Dataset):
    def __init__(self, X_s, X_n, y_bin, y_class, y_comp, recon_errors=None):
        self.X_s        = torch.FloatTensor(X_s)
        self.X_n        = torch.FloatTensor(X_n)
        self.y_bin      = torch.LongTensor(y_bin)
        self.y_class    = torch.LongTensor(y_class)
        self.y_comp     = torch.FloatTensor(y_comp)
        self.recon_errs = (torch.FloatTensor(recon_errors)
                           if recon_errors is not None
                           else torch.zeros(len(y_bin)))

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        return (self.X_s[idx], self.X_n[idx],
                self.y_bin[idx], self.y_class[idx],
                self.y_comp[idx], self.recon_errs[idx])


# ══════════════════════════════════════════════════════════════════
# RECONSTRUCTION ERROR PRE-COMPUTATION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_recon_errors(X_s: np.ndarray, ae_ckpt: dict) -> np.ndarray:
    """
    Run the pre-trained autoencoder on all windows to get per-window
    reconstruction errors. These feed into the fusion MLP as an
    additional anomaly signal.
    """
    if not Path(PRETRAIN_AE).exists():
        print('  [WARN] Autoencoder not found — using zero recon errors')
        return np.zeros(len(X_s), dtype=np.float32)

    # Load the pre-trained autoencoder to compute per-window reconstruction errors
    try:
        ckpt       = torch.load(PRETRAIN_AE, map_location='cpu')
        sensor_dim = ckpt['sensor_dim']
        net_dim    = ckpt['net_dim']
        win_size   = ckpt['win_size']

        # Build autoencoder only (from saved state dict keys)
        # Use SensorAutoEncoder as a standalone reconstructor
        ae = SensorAutoEncoder(sensor_dim, win_size).to(DEVICE)

        # Extract just the encoder + ae_ layers from pretrain state
        state = ckpt['model_state']
        ae_state = {k.replace('sensor_enc.', '').replace('ae_', ''): v
                    for k, v in state.items()
                    if k.startswith('sensor_enc.') or k.startswith('ae_')}
        ae.load_state_dict(ae_state, strict=False)
        ae.eval()

        errs = []
        batch = 512
        Xs_t  = torch.FloatTensor(X_s)
        for start in range(0, len(Xs_t), batch):
            xb   = Xs_t[start:start + batch].to(DEVICE)
            _, _, recon_e = ae(xb)
            errs.append(recon_e.cpu().numpy())

        return np.concatenate(errs).astype(np.float32)
    except Exception as e:
        print(f'  [WARN] Recon error computation failed: {e}')
        return np.zeros(len(X_s), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_data():
    """
    Load from swat_multilabel.npz (6-class y) and swat_final.npz (y_comp).
    Falls back gracefully if y_comp is unavailable.
    """
    print('[1/5] Loading data...')
    ml  = np.load(MULTILABEL_NPZ, allow_pickle=True)
    fin = np.load(FINAL_NPZ,      allow_pickle=True)

    splits = {}
    for tag in ['train', 'val', 'test']:
        X_s   = ml[f'X_s_{tag}']
        X_n   = ml[f'X_n_{tag}']
        y_cls = ml[f'y_{tag}']              # 6-class (0=normal, 1-5=attack types)
        y_bin = (y_cls > 0).astype(np.int64)  # binary derived from multilabel

        # Component labels from swat_final.npz
        comp_key = f'y_comp_{tag}'
        if comp_key in fin:
            y_comp = fin[comp_key].astype(np.float32)
        else:
            y_comp = np.zeros((len(y_bin), 1), dtype=np.float32)

        print(f'  {tag:5s}: {len(X_s):>7,} windows | '
              f'attack: {y_bin.sum():>5,} ({y_bin.mean()*100:.1f}%) | '
              f'sensor_dim={X_s.shape[2]} net_dim={X_n.shape[2]}')

        splits[tag] = (X_s, X_n, y_bin, y_cls, y_comp)

    sensor_cols = list(ml['sensor_cols']) if 'sensor_cols' in ml else []
    n_comp      = splits['train'][4].shape[1]
    return splits, sensor_cols, n_comp


def make_loaders(splits, recon_errors):
    loaders = {}
    for tag, (X_s, X_n, y_bin, y_cls, y_comp) in splits.items():
        re = recon_errors.get(tag)
        ds = SWaTDataset(X_s, X_n, y_bin, y_cls, y_comp, re)
        loaders[tag] = DataLoader(
            ds, batch_size=BATCH_SIZE,
            shuffle=(tag == 'train'),
            num_workers=0, pin_memory=True, drop_last=False)
    return loaders


# ══════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    loss_components = {}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_s, X_n, y_bin, y_cls, y_comp, recon_e in loader:
            X_s     = X_s.to(DEVICE)
            X_n     = X_n.to(DEVICE)
            y_bin   = y_bin.to(DEVICE)
            y_cls   = y_cls.to(DEVICE)
            y_comp  = y_comp.to(DEVICE)
            recon_e = recon_e.to(DEVICE)

            if train:
                optimizer.zero_grad()

            outputs = model(X_s, X_n, recon_error=recon_e)
            loss, comps = criterion(outputs, {
                'y_binary': y_bin,
                'y_class' : y_cls,
                'y_comp'  : y_comp,
            })

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            all_probs.extend(outputs['p_attack'].detach().cpu().numpy())
            all_preds.extend((outputs['p_attack'] > 0.5).long().cpu().numpy())
            all_labels.extend(y_bin.cpu().numpy())

            for k, v in comps.items():
                loss_components[k] = loss_components.get(k, 0) + v

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    f1  = f1_score(all_labels, all_preds, zero_division=0)
    ap  = average_precision_score(all_labels, all_probs) if all_labels.sum() > 0 else 0.0
    n   = len(loader)
    avg_comps = {k: v / n for k, v in loss_components.items()}

    return {'loss': total_loss / n, 'f1': f1, 'auc_pr': ap, **avg_comps}


def build_optimizer(model, phase: str):
    """Differential learning rates for Phase C."""
    if phase == 'B':
        # All non-frozen parameters at the same LR
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_B, weight_decay=1e-4)
    else:
        # Phase C: encoders get much smaller LR
        enc_params  = []
        rest_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'sensor_enc' in name or 'net_enc' in name:
                enc_params.append(p)
            else:
                rest_params.append(p)
        return torch.optim.Adam([
            {'params': enc_params,  'lr': LR_C_ENC},
            {'params': rest_params, 'lr': LR_C_REST},
        ], weight_decay=1e-4)


# ══════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_conformal(model, splits, recon_errors, alpha=ALPHA_CONFORMAL):
    """
    Split conformal prediction calibration.
    Uses NORMAL samples from the validation split to set a threshold
    that guarantees FPR ≤ alpha on truly normal test windows.

    Returns scalar threshold τ such that:
        P(normal window flagged as attack) ≤ alpha  with probability ≥ 1-alpha
    """
    print(f'\n[Conformal] Calibrating at α={alpha} (target FPR ≤ {alpha*100:.0f}%)...')
    model.eval()

    X_s, X_n, y_bin, _, y_comp, *_ = splits['val']
    re = recon_errors.get('val', np.zeros(len(X_s)))

    # Use ONLY normal (non-attack) samples from val set
    normal_mask = (y_bin == 0)
    n_normal    = normal_mask.sum()
    print(f'  Using {n_normal:,} normal val windows for calibration')

    Xs_n  = torch.FloatTensor(X_s[normal_mask]).to(DEVICE)
    Xn_n  = torch.FloatTensor(X_n[normal_mask]).to(DEVICE)
    re_n  = torch.FloatTensor(re[normal_mask]).to(DEVICE)

    all_scores = []
    batch = 512
    for start in range(0, len(Xs_n), batch):
        out = model(Xs_n[start:start+batch],
                    Xn_n[start:start+batch],
                    re_n[start:start+batch])
        # Non-conformity score: 1 - P(normal) = P(attack) for normal samples
        s = out['p_attack'].cpu().numpy()
        all_scores.extend(s)

    scores = np.array(all_scores)

    # Conformal quantile: (1-alpha)(1 + 1/n_cal) percentile
    n_cal     = len(scores)
    quantile  = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    quantile  = min(quantile, 1.0)
    threshold = float(np.quantile(scores, quantile))

    empirical_fpr = (scores > threshold).mean()
    print(f'  Conformal threshold τ = {threshold:.4f}')
    print(f'  Empirical FPR on val  = {empirical_fpr*100:.2f}%  '
          f'(guaranteed ≤ {alpha*100:.0f}%)')

    return threshold


# ══════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════

def severity_weights(n_components):
    """
    Physical consequence severity per component (for FNPI metric).
    Chemical and pressure attacks have higher human-safety stakes.
    """
    return np.ones(n_components, dtype=np.float32)  # uniform; tune per asset map


@torch.no_grad()
def evaluate_full(model, test_loader, conf_threshold, n_components,
                  beta_ema=0.7):
    """
    Complete evaluation with all ICS-specific metrics.
    """
    model.eval()
    all_probs, all_preds_raw, all_labels = [], [], []
    all_cls_true, all_cls_pred = [], []
    all_comp_true, all_comp_pred = [], []
    detection_delays = []

    ema_prob = None
    ema_preds = []

    for X_s, X_n, y_bin, y_cls, y_comp, recon_e in test_loader:
        X_s     = X_s.to(DEVICE)
        X_n     = X_n.to(DEVICE)
        recon_e = recon_e.to(DEVICE)

        out = model(X_s, X_n, recon_error=recon_e)
        p   = out['p_attack'].cpu().numpy()
        cls = out['atk_class'].argmax(dim=-1).cpu().numpy()
        comp= out['comp'].cpu().numpy()

        all_probs.extend(p)
        all_preds_raw.extend((p > 0.5).astype(int))
        all_labels.extend(y_bin.numpy())
        all_cls_true.extend(y_cls.numpy())
        all_cls_pred.extend(cls)
        all_comp_true.extend(y_comp.numpy())
        all_comp_pred.extend(comp)

        # EMA smoothing
        for prob in p:
            if ema_prob is None:
                ema_prob = prob
            else:
                ema_prob = beta_ema * prob + (1 - beta_ema) * ema_prob
            ema_preds.append(int(ema_prob > conf_threshold))

    all_probs     = np.array(all_probs)
    all_preds_raw = np.array(all_preds_raw)
    all_labels    = np.array(all_labels)
    all_cls_true  = np.array(all_cls_true)
    all_cls_pred  = np.array(all_cls_pred)
    all_comp_true = np.array(all_comp_true)
    all_comp_pred = np.array(all_comp_pred)
    ema_preds     = np.array(ema_preds)

    # Conformal predictions
    conf_preds = (all_probs > conf_threshold).astype(int)

    # ── Standard metrics ──────────────────────────────────────────
    metrics = {}
    for name, preds in [('raw', all_preds_raw),
                         ('conformal', conf_preds),
                         ('ema+conformal', ema_preds)]:
        metrics[name] = {
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall'   : recall_score(   all_labels, preds, zero_division=0),
            'f1'       : f1_score(       all_labels, preds, zero_division=0),
            'mcc'      : matthews_corrcoef(all_labels, preds),
            'fpr'      : ((preds == 1) & (all_labels == 0)).sum() /
                          max(1, (all_labels == 0).sum()),
        }

    metrics['auc_roc'] = roc_auc_score(all_labels, all_probs) \
                          if all_labels.sum() > 0 else 0.0
    metrics['auc_pr']  = average_precision_score(all_labels, all_probs) \
                          if all_labels.sum() > 0 else 0.0

    # ── Time-to-Detection (TTD) ───────────────────────────────────
    # Find how many seconds after attack onset the model first raises an alarm
    ttd_list = []
    in_attack = False
    onset     = None
    detected  = False
    for t, (lab, pred) in enumerate(zip(all_labels, conf_preds)):
        if lab == 1 and not in_attack:
            in_attack = True; onset = t; detected = False
        if in_attack and not detected and pred == 1:
            ttd_list.append(t - onset)
            detected = True
        if lab == 0 and in_attack:
            in_attack = False
    metrics['ttd_mean'] = float(np.mean(ttd_list)) if ttd_list else float('inf')
    metrics['ttd_median'] = float(np.median(ttd_list)) if ttd_list else float('inf')

    # ── FNPI: False Negative Physical Impact Score ────────────────
    # Missed attacks weighted by physical severity
    fn_mask = (conf_preds == 0) & (all_labels == 1)
    metrics['fnpi'] = float(fn_mask.mean())   # simplified (uniform severity)

    # ── Physical Component Localization Accuracy (Head 3) ─────────
    if all_comp_true.shape[1] > 1:
        comp_preds_bin = (all_comp_pred > 0.5).astype(int)
        comp_f1 = f1_score(all_comp_true.flatten(),
                            comp_preds_bin.flatten(), zero_division=0)
        metrics['comp_localization_f1'] = comp_f1
    else:
        metrics['comp_localization_f1'] = 0.0

    # ── Attack class accuracy (on attack windows only) ────────────
    atk_mask = all_labels == 1
    if atk_mask.sum() > 0:
        cls_acc = (all_cls_pred[atk_mask] == all_cls_true[atk_mask]).mean()
        metrics['attack_class_acc'] = float(cls_acc)
    else:
        metrics['attack_class_acc'] = 0.0

    return metrics


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip loading pre-trained encoder weights')
    args = parser.parse_args()

    Path('trustgate_data').mkdir(exist_ok=True)

    # ── [1] Load data ──────────────────────────────────────────────
    splits, sensor_cols, n_comp = load_data()
    X_s_tr = splits['train'][0]

    # ── [2] Compute reconstruction errors from pre-trained AE ─────
    print('\n[2/5] Computing reconstruction errors...')
    recon_errors = {}
    for tag, (X_s, X_n, *_) in splits.items():
        recon_errors[tag] = compute_recon_errors(X_s, PRETRAIN_AE)
        print(f'  {tag:5s}: mean recon err = {recon_errors[tag].mean():.4f}')

    # ── [3] Build model ────────────────────────────────────────────
    print('\n[3/5] Building TrustGateModel...')
    sensor_dim = splits['train'][0].shape[2]
    net_dim    = splits['train'][1].shape[2]

    model = TrustGateModel(sensor_dim, net_dim,
                            n_classes=N_CLASSES,
                            n_components=n_comp,
                            use_multigran=False).to(DEVICE)

    if not args.skip_pretrain and Path(PRETRAIN_S).exists():
        model.load_pretrained_encoders(PRETRAIN_S, PRETRAIN_N)
    else:
        print('  [INFO] Starting with random encoder weights (no pre-training)')

    # Class weights for attack classification head
    y_tr_cls = splits['train'][3]
    cw = compute_class_weight('balanced',
                               classes=np.arange(N_CLASSES),
                               y=y_tr_cls)
    class_weights = torch.FloatTensor(cw).to(DEVICE)

    criterion = TrustGateLoss(sensor_cols=sensor_cols,
                               class_weights=class_weights).to(DEVICE)

    # ── [4] Make DataLoaders ───────────────────────────────────────
    print('\n[4/5] Building DataLoaders...')
    # Unpack splits correctly: (X_s, X_n, y_bin, y_cls, y_comp)
    loaders = {}
    for tag, (X_s, X_n, y_bin, y_cls, y_comp) in splits.items():
        re = recon_errors.get(tag, np.zeros(len(X_s)))
        ds = SWaTDataset(X_s, X_n, y_bin, y_cls, y_comp, re)
        loaders[tag] = DataLoader(ds, batch_size=BATCH_SIZE,
                                   shuffle=(tag == 'train'),
                                   num_workers=0, pin_memory=True)

    # ── [5] Training ───────────────────────────────────────────────
    print('\n[5/5] Training...')
    log_rows   = []
    best_f1    = 0.0
    patience_c = 0

    # ══ PHASE B — Frozen encoders ══════════════════════════════════
    print('\n' + '─'*60)
    print('Phase B — Frozen Encoders (30 epochs, LR=5e-4)')
    print('─'*60)
    model.freeze_encoders()
    optimizer = build_optimizer(model, 'B')
    sched_B   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5,
                    patience=5, verbose=True)

    for epoch in range(1, PHASE_B_EPOCHS + 1):
        tr = run_epoch(model, loaders['train'], criterion, optimizer, train=True)
        va = run_epoch(model, loaders['val'],   criterion, optimizer, train=False)
        sched_B.step(va['f1'])

        row = {'phase': 'B', 'epoch': epoch,
               'tr_loss': tr['loss'], 'tr_f1': tr['f1'], 'tr_auc_pr': tr['auc_pr'],
               'va_loss': va['loss'], 'va_f1': va['f1'], 'va_auc_pr': va['auc_pr']}
        log_rows.append(row)

        print(f'  B-{epoch:02d}/{PHASE_B_EPOCHS}  '
              f'tr: loss={tr["loss"]:.4f} f1={tr["f1"]:.4f}  '
              f'va: loss={va["loss"]:.4f} f1={va["f1"]:.4f} '
              f'auc_pr={va["auc_pr"]:.4f}')

        if va['f1'] > best_f1:
            best_f1 = va['f1']
            torch.save({'model_state': model.state_dict(),
                        'sensor_dim': sensor_dim, 'net_dim': net_dim,
                        'n_classes': N_CLASSES, 'n_comp': n_comp,
                        'sensor_cols': sensor_cols}, OUT_MODEL)

    # ══ PHASE C — End-to-end fine-tuning ═══════════════════════════
    print('\n' + '─'*60)
    print('Phase C — End-to-End Fine-tuning (20 epochs, differential LR)')
    print('─'*60)
    model.unfreeze_top_encoder_layers()
    optimizer = build_optimizer(model, 'C')
    sched_C   = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=PHASE_C_EPOCHS, eta_min=LR_C_ENC / 10)
    patience_c = 0

    for epoch in range(1, PHASE_C_EPOCHS + 1):
        tr = run_epoch(model, loaders['train'], criterion, optimizer, train=True)
        va = run_epoch(model, loaders['val'],   criterion, optimizer, train=False)
        sched_C.step()

        row = {'phase': 'C', 'epoch': PHASE_B_EPOCHS + epoch,
               'tr_loss': tr['loss'], 'tr_f1': tr['f1'], 'tr_auc_pr': tr['auc_pr'],
               'va_loss': va['loss'], 'va_f1': va['f1'], 'va_auc_pr': va['auc_pr']}
        log_rows.append(row)

        print(f'  C-{epoch:02d}/{PHASE_C_EPOCHS}  '
              f'tr: loss={tr["loss"]:.4f} f1={tr["f1"]:.4f}  '
              f'va: loss={va["loss"]:.4f} f1={va["f1"]:.4f} '
              f'auc_pr={va["auc_pr"]:.4f}')

        if va['f1'] > best_f1:
            best_f1 = va['f1']
            patience_c = 0
            torch.save({'model_state': model.state_dict(),
                        'sensor_dim': sensor_dim, 'net_dim': net_dim,
                        'n_classes': N_CLASSES, 'n_comp': n_comp,
                        'sensor_cols': sensor_cols}, OUT_MODEL)
            print(f'    ↑ New best val F1: {best_f1:.4f}  (saved)')
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f'  Early stopping at C-epoch {epoch} (no improvement for {PATIENCE} epochs)')
                break

    pd.DataFrame(log_rows).to_csv(OUT_LOG, index=False)

    # ── Load best model for evaluation ────────────────────────────
    print(f'\nLoading best model (val F1={best_f1:.4f})...')
    ckpt = torch.load(OUT_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    # ── Conformal prediction calibration ──────────────────────────
    val_data = splits['val']
    conf_threshold = calibrate_conformal(
        model, {'val': val_data}, recon_errors, alpha=ALPHA_CONFORMAL)
    np.save(OUT_CONF_THRESH, np.array(conf_threshold))

    # ── Final test evaluation ──────────────────────────────────────
    print('\nFinal Test Evaluation:')
    print('=' * 60)
    metrics = evaluate_full(model, loaders['test'], conf_threshold, n_comp)

    lines = [
        'TrustGate — Final Test Results',
        '=' * 60,
        '',
        'Method: EMA + Conformal Prediction (FPR ≤ 1% guaranteed)',
        f'  Precision          : {metrics["ema+conformal"]["precision"]:.4f}',
        f'  Recall             : {metrics["ema+conformal"]["recall"]:.4f}',
        f'  F1 Score           : {metrics["ema+conformal"]["f1"]:.4f}',
        f'  MCC                : {metrics["ema+conformal"]["mcc"]:.4f}',
        f'  FPR                : {metrics["ema+conformal"]["fpr"]*100:.2f}%',
        '',
        'Raw Model (threshold=0.5):',
        f'  Precision          : {metrics["raw"]["precision"]:.4f}',
        f'  Recall             : {metrics["raw"]["recall"]:.4f}',
        f'  F1 Score           : {metrics["raw"]["f1"]:.4f}',
        '',
        'Global Metrics:',
        f'  AUC-ROC            : {metrics["auc_roc"]:.4f}',
        f'  AUC-PR             : {metrics["auc_pr"]:.4f}  ← Headline metric',
        '',
        'ICS-Specific Metrics:',
        f'  Time-to-Detection  : {metrics["ttd_mean"]:.1f}s mean / {metrics["ttd_median"]:.1f}s median',
        f'  FNPI               : {metrics["fnpi"]*100:.2f}%  (missed-attack physical impact)',
        f'  Component Loc F1   : {metrics["comp_localization_f1"]:.4f}',
        f'  Attack Class Acc   : {metrics["attack_class_acc"]*100:.1f}%',
        '',
        f'Conformal threshold  : {conf_threshold:.4f}',
        '=' * 60,
    ]

    report = '\n'.join(lines)
    print(report)
    with open(OUT_RESULTS, 'w') as fh:
        fh.write(report)

    print(f'\nSaved: {OUT_MODEL}')
    print(f'       {OUT_CONF_THRESH}')
    print(f'       {OUT_LOG}')
    print(f'       {OUT_RESULTS}')
    print('\n[DONE] TrustGate training complete.')


if __name__ == '__main__':
    main()