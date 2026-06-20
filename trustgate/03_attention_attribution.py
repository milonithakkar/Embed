# 03_attention_attribution.py
# PURPOSE: For every attack window in the test set, identify
#          WHICH physical component is being attacked using
#          attention weights + integrated gradients.
#          Works on components never seen during training.
# Run: python 03_attention_attribution.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── PATHS ─────────────────────────────────────────────────────
DATA_PATH   = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
CHECKPOINT  = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
THRESH_FILE = r'D:\trustgate_pcaps\eval_outputs\optimal_threshold.json'
OUTPUT_DIR  = r'D:\trustgate_pcaps\attribution_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Sensor column → component name ───────────────────────────
# ADJUST THIS TO MATCH YOUR ACTUAL DATA COLUMNS
# Order must match the columns in X_s_train
SENSOR_COL_MAP = {
    0:  "FIT101",  1:  "LIT101",  2:  "MV101",
    3:  "P101",    4:  "P102",
    5:  "AIT201",  6:  "AIT202",  7:  "AIT203",
    8:  "FIT201",  9:  "MV201",
    10: "P201",    11: "P202",    12: "P203",
    13: "P204",    14: "P205",    15: "P206",
    16: "DPIT301", 17: "FIT301",  18: "LIT301",
    19: "MV301",   20: "MV302",   21: "MV303",
    22: "P301",    23: "P302",
    24: "AIT401",  25: "AIT402",  26: "FIT401",
    27: "LIT401",  28: "P401",    29: "P402",
    30: "UV401",
    31: "AIT501",  32: "AIT502",  33: "AIT503",
    34: "AIT504",  35: "FIT501",  36: "FIT502",
    37: "FIT503",  38: "FIT504",
    39: "P501",    40: "P502",
    41: "FIT601",  42: "P601",    43: "P602",
    44: "P603",
}

print("=" * 60)
print("TrustGate — Attention + IG Attribution")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

data = np.load(DATA_PATH, allow_pickle=True)

ckpt = torch.load(CHECKPOINT, map_location=device)
model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded (epoch {ckpt['epoch']}, AUC={ckpt['monitor_value']:.4f})")

# Load calibrated threshold
with open(THRESH_FILE) as f:
    thresh_data = json.load(f)
THRESHOLD = thresh_data['threshold']
print(f"Using threshold: {THRESHOLD:.2f}")


# ── Attention Hook Manager ────────────────────────────────────
class AttentionHookManager:
    """
    Registers forward hooks on all MultiheadAttention layers.
    Captures attention weight tensors during forward pass.
    """
    def __init__(self, model):
        self.hooks   = []
        self.weights = {}

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                handle = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(handle)
                self.weights[name] = None
                print(f"  Hook registered: {name}")

    def _make_hook(self, name):
        def fn(module, inp, output):
            if isinstance(output, tuple) and len(output) >= 2:
                w = output[1]
                if w is not None:
                    self.weights[name] = w.detach().cpu()
        return fn

    def clear(self):
        for name in self.weights:
            self.weights[name] = None

    def remove(self):
        for h in self.hooks:
            h.remove()

    def get_combined_importance(self):
        """
        Combine all attention layers into one importance vector
        over the TIME dimension.
        Returns: np.array of shape (T,) — importance per timestep
        """
        combined = None
        for name, w in self.weights.items():
            if w is None:
                continue
            # w shape: (B, H, T_q, T_k)
            # Average over batch and heads → (T_q, T_k)
            w_mean = w[0].mean(dim=0).numpy()
            # Average over query dim → (T_k,) = per-timestep importance
            temporal = w_mean.mean(axis=0)
            if combined is None:
                combined = temporal
            else:
                combined = combined + temporal

        if combined is not None:
            combined = combined / combined.sum()  # Normalize
        return combined


print("\nRegistering attention hooks:")
hook_mgr = AttentionHookManager(model)


# ── Integrated Gradients ──────────────────────────────────────
def integrated_gradients(model, x_s, x_n, n_steps=30, device='cpu'):
    """
    Compute IG attribution for sensor and network inputs.

    FIX: cuDNN LSTM cannot backprop in eval() mode.
    We disable cuDNN for the backward pass — uses native LSTM
    which supports gradients in eval mode. Model stays in eval()
    so dropout/norm behavior is unchanged (correct for attribution).

    Returns:
        ig_s: (T, sensor_features) — per-feature importance
        ig_n: (T, network_features)
    """
    model.eval()

    x_s = x_s.to(device)
    x_n = x_n.to(device)

    # Baseline: all zeros (represents "no signal")
    b_s = torch.zeros_like(x_s)
    b_n = torch.zeros_like(x_n)

    grad_s_acc = torch.zeros_like(x_s)
    grad_n_acc = torch.zeros_like(x_n)

    # ── KEY FIX: disable cuDNN so LSTM backward works in eval mode ──
    with torch.backends.cudnn.flags(enabled=False):
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps

            interp_s = (b_s + alpha * (x_s - b_s)).detach().requires_grad_(True)
            interp_n = (b_n + alpha * (x_n - b_n)).detach().requires_grad_(True)

            outputs = model(interp_s, interp_n)
            bin_logit = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            score = torch.sigmoid(bin_logit).sum()
            model.zero_grad()
            score.backward()

            if interp_s.grad is not None:
                grad_s_acc += interp_s.grad.detach()
            if interp_n.grad is not None:
                grad_n_acc += interp_n.grad.detach()

    # IG = (input - baseline) * average_gradient
    ig_s = ((x_s - b_s) * grad_s_acc / n_steps).squeeze(0).cpu().numpy()
    ig_n = ((x_n - b_n) * grad_n_acc / n_steps).squeeze(0).cpu().numpy()

    return ig_s, ig_n  # (T, F) each

def get_top_components_from_ig(ig_s, top_k=5, col_map=None):
    """
    Given (T, S) IG map, return top-K sensor features by importance.
    """
    col_map = col_map or SENSOR_COL_MAP
    # L1 importance: absolute IG values averaged over time
    feature_importance = np.abs(ig_s).mean(axis=0)  # (S,)
    top_k = min(top_k, len(feature_importance))
    top_idx = np.argsort(feature_importance)[::-1][:top_k]

    results = []
    for rank, col in enumerate(top_idx):
        results.append({
            'rank':       rank + 1,
            'col_idx':    int(col),
            'component':  col_map.get(int(col), f'sensor_{col}'),
            'importance': float(feature_importance[col]),
        })
    return results


# ── Attribution on test attack windows ───────────────────────
def get_array(key_a, key_b=None):
    if key_a in data: return data[key_a]
    if key_b and key_b in data: return data[key_b]
    return None

X_s_test  = get_array('X_s_test',  'test_X_sensor')
X_n_test  = get_array('X_n_test',  'test_X_network')
y_b_test  = get_array('y_b_test',  'test_y_binary')
y_p_test  = get_array('y_p_test',  'test_y_comp')

if X_s_test is None:
    print("\n⚠ No test split found. Running on validation split instead.")
    X_s_test = get_array('X_s_val', 'val_X_sensor')
    X_n_test = get_array('X_n_val', 'val_X_network')
    y_b_test = get_array('y_b_val', 'val_y_binary')
    y_p_test = get_array('y_p_val', 'val_y_comp')

# Find attack windows
attack_idx = np.where(y_b_test == 1)[0]
print(f"\nFound {len(attack_idx)} attack windows in test/val split")

# Analyze up to 20 attack windows for speed
MAX_WINDOWS = min(20, len(attack_idx))
print(f"Analyzing first {MAX_WINDOWS} attack windows with IG + Attention...")
print(f"(IG uses 30 interpolation steps per window — ~5-10s each)\n")

attribution_results = []

for i, win_idx in enumerate(attack_idx[:MAX_WINDOWS]):
    print(f"  Window {i+1:2d}/{MAX_WINDOWS}  (dataset idx={win_idx})")

    x_s = torch.FloatTensor(X_s_test[win_idx]).unsqueeze(0)  # (1, T, S)
    x_n = torch.FloatTensor(X_n_test[win_idx]).unsqueeze(0)  # (1, T, N)

    # ── Binary prediction ──────────────────────────────────
    with torch.no_grad():
        hook_mgr.clear()
        outputs  = model(x_s.to(device), x_n.to(device))
        bin_logit = outputs[0] if isinstance(outputs, (tuple,list)) else outputs
        attack_prob = torch.sigmoid(bin_logit).item()

    detected = attack_prob >= THRESHOLD
    temporal_importance = hook_mgr.get_combined_importance()

    # ── Integrated Gradients ────────────────────────────────
    ig_s, ig_n = integrated_gradients(model, x_s, x_n,
                                       n_steps=30, device=device)
    top_components = get_top_components_from_ig(ig_s, top_k=5)

    # ── True component labels ───────────────────────────────
    if y_p_test is not None:
        true_slots = np.where(y_p_test[win_idx] == 1)[0].tolist()
    else:
        true_slots = []

    result = {
        'window_idx':         int(win_idx),
        'attack_prob':        round(float(attack_prob), 4),
        'detected':           bool(detected),
        'true_component_slots': true_slots,
        'ig_top_components':  top_components,
    }
    attribution_results.append(result)

    print(f"    Attack prob    : {attack_prob:.4f}  "
          f"({'DETECTED' if detected else 'MISSED'})")
    print(f"    True slots     : {true_slots}")
    print(f"    IG Top-3       : "
          + ", ".join(f"{c['component']}({c['importance']:.3f})"
                      for c in top_components[:3]))


# ── Summary statistics ─────────────────────────────────────
print(f"\n{'─'*50}")
print("  ATTRIBUTION SUMMARY")
print(f"{'─'*50}")

detected_count = sum(1 for r in attribution_results if r['detected'])
print(f"\n  Windows analyzed    : {len(attribution_results)}")
print(f"  Detected by model   : {detected_count} "
      f"({100*detected_count/max(len(attribution_results),1):.1f}%)")

# Most frequently implicated components
from collections import Counter
component_counts = Counter()
for r in attribution_results:
    if r['detected']:
        for c in r['ig_top_components'][:3]:
            component_counts[c['component']] += 1

print(f"\n  Most frequently implicated components (IG Top-3):")
for comp, cnt in component_counts.most_common(10):
    print(f"    {comp:12s} : {cnt} windows")


# ── Plot: IG heatmap for first detected window ─────────────
detected_results = [r for r in attribution_results if r['detected']]
if detected_results:
    first_detected_idx = detected_results[0]['window_idx']
    x_s_plot = torch.FloatTensor(X_s_test[first_detected_idx]).unsqueeze(0)
    x_n_plot = torch.FloatTensor(X_n_test[first_detected_idx]).unsqueeze(0)

    ig_s_plot, _ = integrated_gradients(model, x_s_plot, x_n_plot,
                                         n_steps=30, device=device)
    ig_importance = np.abs(ig_s_plot).mean(axis=0)  # (S,)

    # Only show top 20 features for readability
    top20_idx   = np.argsort(ig_importance)[::-1][:20]
    top20_names = [SENSOR_COL_MAP.get(int(i), f'col_{i}') for i in top20_idx]
    top20_vals  = ig_importance[top20_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728' if v > np.percentile(top20_vals, 75) else '#1f77b4'
              for v in top20_vals]
    bars = ax.barh(range(len(top20_names)), top20_vals[::-1],
                   color=colors[::-1])
    ax.set_yticks(range(len(top20_names)))
    ax.set_yticklabels(top20_names[::-1], fontsize=10)
    ax.set_xlabel('IG Importance (|IG| averaged over time)', fontsize=11)
    ax.set_title(f'TrustGate — Sensor Feature Attribution\n'
                 f'Window idx={first_detected_idx}, '
                 f'Attack prob={detected_results[0]["attack_prob"]:.3f}',
                 fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, 'ig_feature_importance.png')
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"\n  Saved IG plot → {heatmap_path}")


# ── Save attribution results ───────────────────────────────
attr_path = os.path.join(OUTPUT_DIR, 'attribution_results.json')
with open(attr_path, 'w') as f:
    json.dump(attribution_results, f, indent=2)
print(f"  Saved attribution JSON → {attr_path}")

hook_mgr.remove()

print(f"\n{'='*60}")
print("Attribution complete.")
print(f"{'='*60}")