# inference_engine.py
# Runs on DK-2500 board
# Reads live sensor data → runs AI model → writes predictions
# to /tmp/ai_predictions.json for dashboard to read
# Run: python3 /opt/trustgate/inference_engine.py

import numpy as np
import json
import time
import os
import sys
from collections import deque
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
MODEL_PATH      = '/opt/trustgate/model/trustgate_model.xml'
TWIN_REPORT     = '/tmp/twin_report.json'
AI_OUTPUT       = '/tmp/ai_predictions.json'
BASELINE_PATH   = '/opt/trustgate/model/sensor_baseline.npy'
THRESHOLD       = 0.9915   # from validation calibration

WINDOW_SIZE     = 30       # seconds
N_SENSORS       = 71
POLL_INTERVAL   = 1.0      # seconds

# Attack class names
CLASS_NAMES = [
    "Normal",
    "Valve Manipulation",
    "Pump Disruption",
    "Chemical Dosing Attack",
    "Level Spoofing",
    "Sensor Spoofing",
]

# Component names for localization
COMPONENT_NAMES = [
    "MV101", "MV201", "MV301", "MV302", "MV303",
    "MV304", "MV501", "MV502", "MV503", "MV504",
    "P101",  "P102",  "P201",  "P202",  "P203",
    "P204",  "P205",  "P206",  "LIT101","LIT601",
    "DPIT301","AIT402",
]

# Sensor columns in exact order (must match training)
SENSOR_COLS = [
    "P1_STATE", "LIT101.Pv", "FIT101.Pv", "MV101.Status",
    "P101.Status", "P102.Status", "P2_STATE", "FIT201.Pv",
    "AIT201.Pv", "AIT202.Pv", "AIT203.Pv", "MV201.Status",
    "P201.Status", "P202.Status", "P203.Status", "P204.Status",
    "P205.Status", "P206.Status", "P207.Status", "P208.Status",
    "P3_STATE", "AIT301.Pv", "AIT302.Pv", "AIT303.Pv",
    "LIT301.Pv", "FIT301.Pv", "DPIT301.Pv", "MV301.Status",
    "MV302.Status", "MV303.Status", "MV304.Status", "P301.Status",
    "P302.Status", "P4_STATE", "LIT401.Pv", "FIT401.Pv",
    "AIT401.Pv", "AIT402.Pv", "P401.Status", "P402.Status",
    "P403.Status", "P404.Status", "UV401.Status", "P5_STATE",
    "FIT501.Pv", "FIT502.Pv", "FIT503.Pv", "FIT504.Pv",
    "AIT501.Pv", "AIT502.Pv", "AIT503.Pv", "AIT504.Pv",
    "PIT501.Pv", "PIT502.Pv", "PIT503.Pv", "P501.Status",
    "P501.Speed", "P502.Status", "P502.Speed", "MV501.Status",
    "MV502.Status", "MV503.Status", "MV504.Status", "P6_STATE",
    "LIT601.Pv", "LIT602.Pv", "FIT601.Pv", "FIT602.Pv",
    "P601.Status", "P602.Status", "P603.Status",
]

# Topology: component → adjacent sensor indices
# Used for zero-shot localization
COMPONENT_TO_SENSORS = {
    0:  [0, 1, 2, 3],
    1:  [6, 7, 8, 9, 10, 11],
    3:  [20, 21, 22, 23, 24, 25, 26, 28],
    4:  [20, 21, 22, 23, 24, 25, 26, 29],
    6:  [43, 44, 52, 53, 54, 55, 56, 59],
    7:  [43, 45, 52, 53, 54, 55, 56, 60],
    8:  [43, 46, 52, 53, 54, 57, 58, 61],
    9:  [43, 47, 52, 53, 54, 57, 58, 62],
    10: [0, 1, 2, 4],
    11: [0, 1, 2, 5],
    18: [0, 1, 2, 3, 4, 5],
    19: [63, 64, 65, 66, 67, 68, 69, 70],
    20: [20, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    21: [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
}

print("=" * 55)
print("TrustGate AI Inference Engine")
print("=" * 55)

# ── Load OpenVINO model ───────────────────────────────────────
print(f"\nLoading OpenVINO model from {MODEL_PATH}...")
try:
    from openvino.runtime import Core
    ie      = Core()
    devices = ie.available_devices
    print(f"Available devices: {devices}")

    # Use NPU if available, else CPU
    target = 'NPU' if 'NPU' in devices else 'CPU'
    print(f"Using: {target}")

    model    = ie.read_model(MODEL_PATH)
    compiled = ie.compile_model(model, target)
    infer_req = compiled.create_infer_request()
    print(f"Model loaded successfully on {target}")
    USE_OPENVINO = True

except Exception as e:
    print(f"OpenVINO not available: {e}")
    print("Falling back to PyTorch CPU inference")
    USE_OPENVINO = False

    # PyTorch fallback
    import torch
    sys.path.insert(0, '/opt/trustgate')
    from model import TrustGateModel
    TORCH_CKPT = '/opt/trustgate/model/trustgate_model.pth'
    torch_model = TrustGateModel()
    ckpt = torch.load(TORCH_CKPT, map_location='cpu',
                      weights_only=False)
    torch_model.load_state_dict(ckpt['model_state'])
    torch_model.eval()
    print("PyTorch model loaded on CPU")

# ── Load baseline for deviation scoring ───────────────────────
print(f"\nLoading sensor baseline...")
baseline = np.load(BASELINE_PATH).astype(np.float32)
print(f"Baseline shape: {baseline.shape}")

# ── Sensor window buffer ──────────────────────────────────────
# Rolling 30-second window of sensor readings
sensor_buffer = deque(maxlen=WINDOW_SIZE)

# Fill buffer with baseline initially
for _ in range(WINDOW_SIZE):
    sensor_buffer.append(baseline.copy())

# ── Robust scaler parameters ──────────────────────────────────
# Load from saved scaler (center and scale arrays)
# These were fitted on A11 normal data
try:
    import pickle
    SCALER_PATH = '/opt/trustgate/model/a11_scaler.pkl'
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    CENTER = scaler.center_.astype(np.float32)
    SCALE  = scaler.scale_.astype(np.float32)
    print(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"Scaler not found ({e}) — using identity scaling")
    CENTER = np.zeros(N_SENSORS, dtype=np.float32)
    SCALE  = np.ones(N_SENSORS, dtype=np.float32)

# ── Helper functions ──────────────────────────────────────────
def scale_sensor_reading(raw_values):
    """Apply same RobustScaler as training."""
    scaled = (raw_values - CENTER) / SCALE
    return np.clip(scaled, -10.0, 10.0).astype(np.float32)


def read_sensor_values():
    """
    Read current sensor values from twin_report.json.
    Returns numpy array of shape (71,) in SENSOR_COLS order.
    Falls back to baseline if file unavailable.
    """
    try:
        with open(TWIN_REPORT, 'r') as f:
            report = json.load(f)

        # Extract sensor values in correct order
        state   = report.get('current_state', {})
        values  = np.zeros(N_SENSORS, dtype=np.float32)

        for i, col in enumerate(SENSOR_COLS):
            # Try exact match first
            if col in state:
                v = state[col]
            else:
                # Try without .Pv or .Status suffix
                short = col.replace('.Pv', '').replace(
                    '.Status', '')
                v = state.get(short, baseline[i])

            # Convert to float
            if isinstance(v, str):
                v = {'Active': 1.0, 'Inactive': 0.0,
                     'Bad Input': 0.0,
                     'Open': 1.0, 'Close': 0.0}.get(v, 0.0)
            values[i] = float(v)

        return values

    except Exception:
        return baseline.copy()


def run_inference(window):
    """
    Run model inference on a 30-second window.
    window: numpy (30, 71) scaled sensor values
    Returns: (attack_prob, class_idx, sensor_imp)
    """
    # Add batch + network stream dimensions
    xs = window[np.newaxis, :, :]   # (1, 30, 71)
    xn = np.zeros((1, 30, 19),
                   dtype=np.float32)  # dummy network stream

    if USE_OPENVINO:
        inputs = {
            compiled.input(0): xs,
            compiled.input(1): xn,
        }
        results    = infer_req.infer(inputs)
        bin_logit  = results[compiled.output(0)][0, 0]
        cls_logits = results[compiled.output(1)][0]
        sensor_imp = results[compiled.output(5)][0]

        attack_prob = 1.0 / (1.0 + np.exp(-bin_logit))
        class_idx   = int(np.argmax(cls_logits))
    else:
        import torch
        with torch.no_grad():
            out = torch_model(
                torch.FloatTensor(xs),
                torch.FloatTensor(xn)
            )
            attack_prob = float(
                torch.sigmoid(out[0][0, 0]).item())
            class_idx   = int(out[1][0].argmax().item())
            sensor_imp  = out[5][0].numpy()

    return attack_prob, class_idx, sensor_imp


def localize_attack(sensor_imp):
    """
    Use sensor importance weights to identify
    which component is most likely attacked.
    Returns (component_name, confidence, top_sensors)
    """
    top5_idx = np.argsort(sensor_imp)[::-1][:5]

    # Score each component by overlap with top-5 sensors
    best_comp  = -1
    best_score = -1

    for comp_slot, adj_sensors in COMPONENT_TO_SENSORS.items():
        overlap = len(set(top5_idx) & set(adj_sensors))
        score   = overlap / max(len(adj_sensors), 1)
        if score > best_score:
            best_score = score
            best_comp  = comp_slot

    comp_name  = (COMPONENT_NAMES[best_comp]
                  if best_comp >= 0 else "Unknown")
    top_sensor_names = [SENSOR_COLS[i] for i in top5_idx]

    return comp_name, float(best_score), top_sensor_names


def write_ai_output(attack_prob, class_idx, sensor_imp,
                    is_attack, comp_name, comp_conf,
                    top_sensors, inference_ms):
    """Write AI predictions to /tmp/ai_predictions.json"""
    output = {
        "timestamp":        datetime.now().isoformat(),
        "attack_probability": float(attack_prob),
        "is_attack":         bool(is_attack),
        "attack_class":      CLASS_NAMES[class_idx],
        "class_confidence":  float(
            np.exp(attack_prob) / (1 + np.exp(attack_prob))),
        "localization": {
            "component":    comp_name,
            "confidence":   comp_conf,
            "top_sensors":  top_sensors,
        },
        "inference_ms":      inference_ms,
        "model_version":     "v4",
        "device":            "NPU" if USE_OPENVINO else "CPU",
    }
    with open(AI_OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)


# ── Main inference loop ───────────────────────────────────────
print(f"\nStarting inference loop...")
print(f"Threshold: {THRESHOLD}")
print(f"Poll interval: {POLL_INTERVAL}s")
print(f"Writing to: {AI_OUTPUT}")
print(f"\n{'─'*55}")

attack_start_time = None
n_inferences = 0

while True:
    loop_start = time.time()

    try:
        # Read current sensor values
        raw_values    = read_sensor_values()
        scaled_values = scale_sensor_reading(raw_values)

        # Add to rolling buffer
        sensor_buffer.append(scaled_values)

        # Only run inference when buffer is full
        if len(sensor_buffer) < WINDOW_SIZE:
            time.sleep(POLL_INTERVAL)
            continue

        # Build window tensor
        window = np.array(sensor_buffer,
                          dtype=np.float32)  # (30, 71)

        # Run model
        t_infer = time.time()
        attack_prob, class_idx, sensor_imp = run_inference(
            window)
        inference_ms = (time.time() - t_infer) * 1000

        # Threshold decision
        is_attack = attack_prob >= THRESHOLD

        # Localize if attack detected
        if is_attack:
            comp_name, comp_conf, top_sensors = \
                localize_attack(sensor_imp)
            if attack_start_time is None:
                attack_start_time = time.time()
                print(f"\n🔴 ATTACK DETECTED")
                print(f"   Class:     {CLASS_NAMES[class_idx]}")
                print(f"   Prob:      {attack_prob:.4f}")
                print(f"   Component: {comp_name} "
                      f"({comp_conf:.2f} conf)")
                print(f"   Infer:     {inference_ms:.1f}ms")
        else:
            comp_name     = "None"
            comp_conf     = 0.0
            top_sensors   = []
            if attack_start_time is not None:
                duration = time.time() - attack_start_time
                print(f"\n✅ Attack cleared "
                      f"(lasted {duration:.1f}s)")
                attack_start_time = None

        # Write output
        write_ai_output(attack_prob, class_idx, sensor_imp,
                        is_attack, comp_name, comp_conf,
                        top_sensors, inference_ms)

        n_inferences += 1
        if n_inferences % 60 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Running... "
                  f"prob={attack_prob:.3f} "
                  f"infer={inference_ms:.1f}ms")

    except KeyboardInterrupt:
        print(f"\nStopped after {n_inferences} inferences.")
        break
    except Exception as e:
        print(f"Error: {e}")

    # Sleep remainder of interval
    elapsed = time.time() - loop_start
    sleep_t = max(0, POLL_INTERVAL - elapsed)
    time.sleep(sleep_t)