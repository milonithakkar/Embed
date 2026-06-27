# /opt/trustgate/ai_engine/diagnose.py
"""
TrustGate Diagnostic Oracle
Run this STANDALONE (no other scripts needed) to pinpoint the exact failure layer.
"""

import json
import pickle
import numpy as np
import os
import sys
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NETWORK_FEATURES_PATH = "/tmp/network_features.json"
NETWORK_SCALER_PATH   = "/opt/trustgate/model/network_scaler.pkl"
MODEL_DIR             = "/opt/trustgate/model/"

EXPECTED_FEATURE_DIM  = 19
BUFFER_SIZE           = 30

SEP = "─" * 60

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─────────────────────────────────────────────
# TEST 1 — Raw network_features.json inspection
# ─────────────────────────────────────────────
def test_raw_features():
    header("TEST 1: Raw /tmp/network_features.json")
    
    if not os.path.exists(NETWORK_FEATURES_PATH):
        print("  [FAIL] File does not exist — network_extractor.py not running?")
        return None

    with open(NETWORK_FEATURES_PATH, "r") as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"  Feature count : {len(features)}  (expected {EXPECTED_FEATURE_DIM})")
    print(f"  Min value     : {min(features):.6f}")
    print(f"  Max value     : {max(features):.6f}")
    print(f"  Mean value    : {np.mean(features):.6f}")
    print(f"  Std dev       : {np.std(features):.6f}")
    print(f"\n  Raw values:")
    for i, v in enumerate(features):
        flag = "  <<<< EXTREME" if abs(v) > 10 else ""
        print(f"    [{i:02d}] {v:>12.4f}{flag}")

    # Verdict
    if max(features) > 5.0 or min(features) < -5.0:
        print("\n  [VERDICT] ❌ FEATURES ARE NOT PROPERLY SCALED")
        print("            Values outside [-3, 3] will saturate LSTM gates.")
        return features, False
    else:
        print("\n  [VERDICT] ✅ Feature range looks reasonable")
        return features, True

# ─────────────────────────────────────────────
# TEST 2 — Scaler integrity check
# ─────────────────────────────────────────────
def test_scaler(raw_features):
    header("TEST 2: Scaler Integrity & Transform Verification")
    
    if not os.path.exists(NETWORK_SCALER_PATH):
        print(f"  [FAIL] Scaler not found at {NETWORK_SCALER_PATH}")
        return None

    with open(NETWORK_SCALER_PATH, "rb") as f:
        obj = pickle.load(f)

    scaler = obj['scaler'] if isinstance(obj, dict) else obj
    print(f"  Scaler type   : {type(scaler).__name__}")
    print(f"  Expects feats : {scaler.n_features_in_}")

    # Show scaler internals (RobustScaler has center_ and scale_)
    if hasattr(scaler, 'center_'):
        print(f"  Center (median): {np.round(scaler.center_, 3)}")
    if hasattr(scaler, 'scale_'):
        print(f"  Scale (IQR)    : {np.round(scaler.scale_, 3)}")

    if raw_features is not None:
        arr = np.array(raw_features).reshape(1, -1)
        scaled = scaler.transform(arr)[0]
        print(f"\n  After scaler transform:")
        print(f"    Min  : {scaled.min():.6f}")
        print(f"    Max  : {scaled.max():.6f}")
        print(f"    Mean : {scaled.mean():.6f}")
        
        extreme_count = np.sum(np.abs(scaled) > 3.0)
        print(f"    Values with |x| > 3.0 : {extreme_count}/{len(scaled)}")
        
        if extreme_count > 0:
            print("\n  [VERDICT] ❌ SCALER IS WRONG OR WAS FIT ON DIFFERENT DATA")
            print("            network_extractor may already be scaling, causing double-scale.")
        else:
            print("\n  [VERDICT] ✅ Scaler output is well-bounded")
        
        return scaled
    return None

# ─────────────────────────────────────────────
# TEST 3 — Double-scaling detection
# ─────────────────────────────────────────────
def test_double_scaling(raw_features, scaled_features):
    header("TEST 3: Double-Scaling Detection")
    
    if raw_features is None or scaled_features is None:
        print("  [SKIP] Insufficient data from prior tests")
        return

    # If raw features are ALREADY in [-2, 2] range but scaler shifts them further
    raw_range  = max(raw_features) - min(raw_features)
    scale_range = max(scaled_features) - min(scaled_features)

    print(f"  Raw feature range    : {raw_range:.4f}")
    print(f"  Scaled feature range : {scale_range:.4f}")

    if raw_range < 4.0 and scale_range < 4.0:
        print("\n  [INFO] Both ranges look small — network_extractor may ALREADY be scaling.")
        print("         If inference.py also scales → double-scaled → garbage input to model.")
    
    # Simulate double scaling
    arr = np.array(raw_features).reshape(1, -1)
    with open(NETWORK_SCALER_PATH, "rb") as f:
        obj = pickle.load(f)
    scaler = obj['scaler'] if isinstance(obj, dict) else obj
    
    double_scaled = scaler.transform(
        scaler.transform(arr)
    )[0]
    
    print(f"\n  Simulated double-scale result:")
    print(f"    Min  : {double_scaled.min():.6f}")
    print(f"    Max  : {double_scaled.max():.6f}")
    
    if np.any(np.abs(double_scaled) > 5):
        print("  [VERDICT] ❌ Double-scaling confirmed as catastrophic")
    else:
        print("  [VERDICT] Double-scaling impact is minor in this case")

# ─────────────────────────────────────────────
# TEST 4 — Synthetic NORMAL input test
# ─────────────────────────────────────────────
def test_synthetic_normal():
    header("TEST 4: Synthetic NORMAL Input — Direct Model Probe")
    
    # Find the OpenVINO model
    xml_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".xml")]
    if not xml_files:
        print(f"  [FAIL] No .xml model found in {MODEL_DIR}")
        return

    model_xml = os.path.join(MODEL_DIR, xml_files[0])
    print(f"  Model found: {model_xml}")

    try:
        from openvino.runtime import Core
        ie = Core()
        model = ie.read_model(model=model_xml)
        compiled = ie.compile_model(model=model, device_name="CPU")
        infer_req = compiled.create_infer_request()
        
        input_layer  = compiled.input(0)
        output_layer = compiled.output(0)
        
        print(f"  Input shape  : {input_layer.shape}")
        print(f"  Output shape : {output_layer.shape}")

        # Build a perfect NORMAL synthetic sequence (all zeros = centered, scaled)
        batch, seq_len, feat_dim = input_layer.shape
        
        scenarios = {
            "All Zeros (centered)":    np.zeros((batch, seq_len, feat_dim), dtype=np.float32),
            "Small Gaussian noise":    np.random.normal(0, 0.1, (batch, seq_len, feat_dim)).astype(np.float32),
            "All Ones (unit)":         np.ones((batch, seq_len, feat_dim), dtype=np.float32),
            "Large values (attack sim)": np.ones((batch, seq_len, feat_dim), dtype=np.float32) * 5.0,
        }

        for name, synthetic_input in scenarios.items():
            infer_req.infer({input_layer: synthetic_input})
            output = infer_req.get_output_tensor(output_layer.index).data.copy()
            
            # Assuming output is [batch, 2] with softmax — [NORMAL, CRITICAL]
            if output.shape[-1] == 2:
                normal_conf   = float(output[0][0])
                critical_conf = float(output[0][1])
                verdict = "NORMAL" if normal_conf > critical_conf else "CRITICAL"
            else:
                normal_conf   = 1.0 - float(output[0][0])
                critical_conf = float(output[0][0])
                verdict = "NORMAL" if critical_conf < 0.5 else "CRITICAL"

            print(f"\n  Scenario: {name}")
            print(f"    Raw output  : {output}")
            print(f"    NORMAL conf : {normal_conf:.4f}")
            print(f"    CRITICAL conf: {critical_conf:.4f}")
            print(f"    Verdict     : {'✅' if verdict == 'NORMAL' else '❌'} {verdict}")

    except ImportError:
        print("  [FAIL] OpenVINO not importable in this environment")
    except Exception as e:
        print(f"  [FAIL] Model probe error: {e}")
        import traceback
        traceback.print_exc()

# ─────────────────────────────────────────────
# TEST 5 — Live 10-second feature drift watch
# ─────────────────────────────────────────────
def test_live_drift():
    header("TEST 5: Live Feature Drift Watch (10 samples)")
    print("  Watching /tmp/network_features.json for 10 seconds...\n")
    
    samples = []
    for i in range(10):
        try:
            with open(NETWORK_FEATURES_PATH, "r") as f:
                data = json.load(f)
            feat = data.get("features", [])
            samples.append(feat)
            rng = max(feat) - min(feat)
            print(f"  t={i+1:02d}s | min={min(feat):8.4f} | max={max(feat):8.4f} "
                  f"| mean={np.mean(feat):8.4f} | range={rng:.4f}")
        except Exception as e:
            print(f"  t={i+1:02d}s | READ ERROR: {e}")
        time.sleep(1)

    if samples:
        all_vals = np.array(samples).flatten()
        print(f"\n  Overall 10s stats:")
        print(f"    Global min  : {all_vals.min():.4f}")
        print(f"    Global max  : {all_vals.max():.4f}")
        print(f"    Global mean : {all_vals.mean():.4f}")
        print(f"    Global std  : {all_vals.std():.4f}")

        if all_vals.max() > 5.0:
            print("\n  [VERDICT] ❌ Features are consistently extreme — scaling broken")
        elif all_vals.std() < 0.001:
            print("\n  [VERDICT] ⚠️  Features are completely static — extractor frozen?")
        else:
            print("\n  [VERDICT] ✅ Features show normal drift")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  TRUSTGATE DIAGNOSTIC ORACLE v1.0")
    print("═"*60)
    
    result = test_raw_features()
    raw_features = result[0] if result else None
    
    scaled = test_scaler(raw_features)
    
    if os.path.exists(NETWORK_SCALER_PATH) and raw_features:
        test_double_scaling(raw_features, scaled)
    
    test_synthetic_normal()
    test_live_drift()
    
    print(f"\n{SEP}")
    print("  DIAGNOSTIC COMPLETE — Review verdicts above")
    print(SEP + "\n")
