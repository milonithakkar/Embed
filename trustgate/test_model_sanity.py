# test_model_sanity.py
# Confirms model works correctly on known data
# Run: python test_model_sanity.py

import numpy as np
import torch
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

V3_CKPT   = r'D:\trustgate_pcaps\trained_model_v3_FINAL_AUC9620.pth'
V4_CKPT   = r'D:\trustgate_pcaps\trained_model_full_v4.pth'
DATA_PATH = r'D:\trustgate_pcaps\A12_windowed_v3_fixed.npz'

V3_THRESHOLD = 0.9831
V4_THRESHOLD = 0.9866

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')

COMPONENT_NAMES = [
    "MV101","MV201","MV301","MV302","MV303",
    "MV304","MV501","MV502","MV503","MV504",
    "P101","P102","P201","P202","P203",
    "P204","P205","P206","LIT101","LIT601",
    "DPIT301","AIT402",
]

SENSOR_COLS = [
    "P1_STATE","LIT101.Pv","FIT101.Pv","MV101.Status",
    "P101.Status","P102.Status","P2_STATE","FIT201.Pv",
    "AIT201.Pv","AIT202.Pv","AIT203.Pv","MV201.Status",
    "P201.Status","P202.Status","P203.Status","P204.Status",
    "P205.Status","P206.Status","P207.Status","P208.Status",
    "P3_STATE","AIT301.Pv","AIT302.Pv","AIT303.Pv",
    "LIT301.Pv","FIT301.Pv","DPIT301.Pv","MV301.Status",
    "MV302.Status","MV303.Status","MV304.Status","P301.Status",
    "P302.Status","P4_STATE","LIT401.Pv","FIT401.Pv",
    "AIT401.Pv","AIT402.Pv","P401.Status","P402.Status",
    "P403.Status","P404.Status","UV401.Status","P5_STATE",
    "FIT501.Pv","FIT502.Pv","FIT503.Pv","FIT504.Pv",
    "AIT501.Pv","AIT502.Pv","AIT503.Pv","AIT504.Pv",
    "PIT501.Pv","PIT502.Pv","PIT503.Pv","P501.Status",
    "P501.Speed","P502.Status","P502.Speed","MV501.Status",
    "MV502.Status","MV503.Status","MV504.Status","P6_STATE",
    "LIT601.Pv","LIT602.Pv","FIT601.Pv","FIT602.Pv",
    "P601.Status","P602.Status","P603.Status",
]

COMPONENT_TO_SENSORS = {
    0:[0,1,2,3], 1:[6,7,8,9,10,11],
    3:[20,21,22,23,24,25,26,28],
    4:[20,21,22,23,24,25,26,29],
    6:[43,44,52,53,54,55,56,59],
    7:[43,45,52,53,54,55,56,60],
    8:[43,46,52,53,54,57,58,61],
    9:[43,47,52,53,54,57,58,62],
    10:[0,1,2,4], 11:[0,1,2,5],
    18:[0,1,2,3,4,5],
    19:[63,64,65,66,67,68,69,70],
    20:[20,24,25,26,27,28,29,30,31,32],
    21:[33,34,35,36,37,38,39,40,41,42],
}

CLASS_NAMES = [
    "Normal","Valve Manipulation","Pump Disruption",
    "Chemical Dosing","Level Spoofing","Sensor Spoofing",
]

# ── Load models ───────────────────────────────────────────────
print("=" * 65)
print("TrustGate — Model Sanity Test")
print("=" * 65)

def load_model(path):
    ckpt  = torch.load(path, map_location=DEVICE,
                       weights_only=False)
    m     = TrustGateModel().to(DEVICE)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m, ckpt['monitor_value']

print(f"\nLoading v3...")
model_v3, auc3 = load_model(V3_CKPT)
print(f"  AUC={auc3:.4f}")

print(f"Loading v4...")
model_v4, auc4 = load_model(V4_CKPT)
print(f"  AUC={auc4:.4f}")

# ── Load data ─────────────────────────────────────────────────
data      = np.load(DATA_PATH, allow_pickle=True)
X_s_val   = data['X_s_val']
X_n_val   = data['X_n_val']
y_b_val   = data['y_b_val'].astype(int)
y_c_val   = data['y_c_val'].astype(int)
y_p_val   = data['y_p_val'].astype(float)
X_s_train = data['X_s_train']
X_n_train = data['X_n_train']
y_b_train = data['y_b_train'].astype(int)

def run_dual(xs_np, xn_np):
    xs = torch.FloatTensor(xs_np).unsqueeze(0).to(DEVICE)
    xn = torch.FloatTensor(xn_np).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o3 = model_v3(xs, xn)
        o4 = model_v4(xs, xn)
    p3   = torch.sigmoid(o3[0]).item()
    p4   = torch.sigmoid(o4[0]).item()
    cls3 = o3[1].argmax(-1).item()
    imp  = o3[5][0].cpu().numpy()
    return p3, p4, cls3, imp

def get_alert(p3, p4):
    f3 = p3 >= V3_THRESHOLD
    f4 = p4 >= V4_THRESHOLD
    if f3 and f4:
        return "HIGH"
    elif f4:
        return "MEDIUM"
    elif f3:
        return "LOW"
    return "NORMAL"

def localize(imp):
    top5    = set(np.argsort(imp)[::-1][:5])
    best_c  = -1
    best_sc = -1
    for slot, adj in COMPONENT_TO_SENSORS.items():
        sc = len(top5 & set(adj)) / max(len(adj), 1)
        if sc > best_sc:
            best_sc, best_c = sc, slot
    name = COMPONENT_NAMES[best_c] if best_c >= 0 else "Unknown"
    top5_names = [SENSOR_COLS[i]
                  for i in np.argsort(imp)[::-1][:5]]
    return name, best_sc, top5_names

# ── TEST 1: Known attack windows from val set ─────────────────
print(f"\n{'='*65}")
print("TEST 1 — Known Attack Windows (Val Set)")
print(f"{'='*65}")

attack_idx = np.where(y_b_val == 1)[0]
print(f"Testing 10 attack windows from val set...")
print(f"\n{'Win':>5}  {'Class':20s}  {'p_v3':>6}  "
      f"{'p_v4':>6}  {'Alert':>8}  {'Location':>10}")
print("-" * 65)

correct_alerts = 0
for i, idx in enumerate(attack_idx[:10]):
    p3, p4, cls3, imp = run_dual(
        X_s_val[idx], X_n_val[idx])
    alert = get_alert(p3, p4)
    loc, sc, top5 = localize(imp)
    true_class = CLASS_NAMES[y_c_val[idx]]

    if alert != "NORMAL":
        correct_alerts += 1

    print(f"{idx:>5}  {true_class:20s}  {p3:>6.4f}  "
          f"{p4:>6.4f}  {alert:>8}  {loc:>10}")

print(f"\nDetected: {correct_alerts}/10 attack windows")

# ── TEST 2: Known normal windows ──────────────────────────────
print(f"\n{'='*65}")
print("TEST 2 — Normal Windows (False Alarm Check)")
print(f"{'='*65}")

normal_idx = np.where(y_b_val == 0)[0]
print(f"Testing 10 normal windows from val set...")
print(f"\n{'Win':>5}  {'p_v3':>6}  {'p_v4':>6}  "
      f"{'Alert':>8}  {'FalseAlarm':>12}")
print("-" * 50)

false_alarms = 0
for idx in normal_idx[:10]:
    p3, p4, cls3, imp = run_dual(
        X_s_val[idx], X_n_val[idx])
    alert = get_alert(p3, p4)
    is_fa = alert != "NORMAL"
    if is_fa:
        false_alarms += 1
    print(f"{idx:>5}  {p3:>6.4f}  {p4:>6.4f}  "
          f"{alert:>8}  "
          f"{'FALSE ALARM' if is_fa else 'OK':>12}")

print(f"\nFalse alarms: {false_alarms}/10 normal windows")

# ── TEST 3: AIT402 attack windows (zero-shot) ─────────────────
print(f"\n{'='*65}")
print("TEST 3 — AIT402 Attack Windows (Zero-Shot Test)")
print(f"{'='*65}")

X_s_test = data['X_s_test']
X_n_test = data['X_n_test']
y_b_test = data['y_b_test'].astype(int)
y_p_test = data['y_p_test'].astype(float)

ait402_idx = np.where(y_p_test[:, 21] == 1)[0]
print(f"AIT402 attack windows in test: {len(ait402_idx)}")
print(f"Testing first 10...")
print(f"\n{'Win':>5}  {'p_v3':>6}  {'p_v4':>6}  "
      f"{'Alert':>8}  {'Location':>10}  {'Correct':>8}")
print("-" * 60)

correct_loc = 0
detected    = 0
for idx in ait402_idx[:10]:
    p3, p4, cls3, imp = run_dual(
        X_s_test[idx], X_n_test[idx])
    alert     = get_alert(p3, p4)
    loc, sc, top5 = localize(imp)

    # Check if FIT401 (sensor 35, adjacent to AIT402) is in top5
    top5_idx  = np.argsort(imp)[::-1][:5]
    adj_402   = set(COMPONENT_TO_SENSORS[21])
    hit       = len(set(top5_idx) & adj_402) > 0

    if alert != "NORMAL":
        detected += 1
    if hit:
        correct_loc += 1

    print(f"{idx:>5}  {p3:>6.4f}  {p4:>6.4f}  "
          f"{alert:>8}  {loc:>10}  "
          f"{'HIT' if hit else 'miss':>8}")

print(f"\nDetected:  {detected}/10 AIT402 attack windows")
print(f"Localized: {correct_loc}/10 AIT402 windows (Hit@5)")

# ── TEST 4: MV302 attack windows (hardest zero-shot) ──────────
print(f"\n{'='*65}")
print("TEST 4 — MV302 Attack Windows (Hardest Zero-Shot)")
print(f"{'='*65}")

mv302_idx = np.where(y_p_test[:, 3] == 1)[0]
print(f"MV302 attack windows in test: {len(mv302_idx)}")
print(f"Testing first 10...")
print(f"\n{'Win':>5}  {'p_v3':>6}  {'p_v4':>6}  "
      f"{'Alert':>8}  {'TopSensor':>20}  {'Rank_DPIT':>10}")
print("-" * 70)

for idx in mv302_idx[:10]:
    p3, p4, cls3, imp = run_dual(
        X_s_test[idx], X_n_test[idx])
    alert    = get_alert(p3, p4)
    top5_idx = np.argsort(imp)[::-1]

    # Find rank of DPIT301 (sensor 26, adjacent to MV302)
    dpit_rank = int(np.where(top5_idx == 26)[0][0]) + 1
    top1_name = SENSOR_COLS[top5_idx[0]]

    print(f"{idx:>5}  {p3:>6.4f}  {p4:>6.4f}  "
          f"{alert:>8}  {top1_name:>20}  "
          f"rank={dpit_rank:>3}")

# ── SUMMARY ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SANITY TEST SUMMARY")
print(f"{'='*65}")
print(f"\n  Test 1 (known attacks):     {correct_alerts}/10 detected")
print(f"  Test 2 (normal windows):    {false_alarms}/10 false alarms")
print(f"  Test 3 (AIT402 zero-shot):  {detected}/10 detected, "
      f"{correct_loc}/10 localized")
print(f"\n  Expected results:")
print(f"    Test 1: 7-10/10 detected")
print(f"    Test 2: 0-1/10 false alarms")
print(f"    Test 3: 3-6/10 detected (v4 catches some)")
print(f"\n{'='*65}")