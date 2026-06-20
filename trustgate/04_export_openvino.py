# 04_export_openvino.py
# PURPOSE: Export trained model to ONNX, then to OpenVINO IR (FP16).
#          Optional INT8 quantization if openvino-dev is installed.
# Run: python 04_export_openvino.py

import torch
import numpy as np
import os
import sys
import json
import subprocess

sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

# ── PATHS ─────────────────────────────────────────────────────
CHECKPOINT  = r'D:\trustgate_pcaps\trained_model_full_v3.pth'
DATA_PATH   = r'D:\trustgate_pcaps\A12_windowed_v3.npz'
EXPORT_DIR  = r'D:\trustgate_pcaps\openvino_export'
MODEL_NAME  = 'trustgate_enhanced'
os.makedirs(EXPORT_DIR, exist_ok=True)

print("=" * 60)
print("TrustGate — OpenVINO Export Pipeline")
print("=" * 60)

# ── Load model ────────────────────────────────────────────────
device = torch.device('cpu')  # Always export from CPU
print(f"\nLoading checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location=device)

model = TrustGateModel().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Loaded epoch {ckpt['epoch']} | AUC={ckpt['monitor_value']:.4f}")

# ── Detect input dimensions from data ─────────────────────────
print(f"\nDetecting input dimensions from data...")
data       = np.load(DATA_PATH, allow_pickle=True)

def get_arr(a, b=None):
    if a in data: return data[a]
    if b and b in data: return data[b]
    return None

X_s_sample = get_arr('X_s_train', 'train_X_sensor')
X_n_sample = get_arr('X_n_train', 'train_X_network')

T        = X_s_sample.shape[1]   # Time steps
S_DIM    = X_s_sample.shape[2]   # Sensor features
N_DIM    = X_n_sample.shape[2]   # Network features

print(f"  Time steps      : {T}")
print(f"  Sensor features : {S_DIM}")
print(f"  Network features: {N_DIM}")


# ── STEP 1: Export to ONNX ────────────────────────────────────
print(f"\n[Step 1] Exporting to ONNX...")

onnx_path = os.path.join(EXPORT_DIR, f'{MODEL_NAME}.onnx')

# Dummy inputs for tracing
dummy_s = torch.randn(1, T, S_DIM)
dummy_n = torch.randn(1, T, N_DIM)

# Test forward pass first
try:
    with torch.no_grad():
        test_out = model(dummy_s, dummy_n)
    if isinstance(test_out, (tuple, list)):
        print(f"  Forward pass OK — {len(test_out)} outputs")
        output_names = [f'output_{i}' for i in range(len(test_out))]
        # Name the important ones
        if len(output_names) >= 3:
            output_names[0] = 'binary_logit'
            output_names[1] = 'class_logits'
            output_names[2] = 'comp_logits'
    else:
        print(f"  Forward pass OK — single output")
        output_names = ['binary_logit']
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    raise

torch.onnx.export(
    model,
    (dummy_s, dummy_n),
    onnx_path,
    input_names    = ['sensor_input', 'network_input'],
    output_names   = output_names,
    dynamic_axes   = {
        'sensor_input':  {0: 'batch_size'},
        'network_input': {0: 'batch_size'},
        'binary_logit':  {0: 'batch_size'},
    },
    opset_version      = 17,
    do_constant_folding= True,
    export_params      = True,
    verbose            = False,
)
print(f"  ✓ ONNX saved → {onnx_path}")


# ── STEP 2: Verify ONNX ───────────────────────────────────────
print(f"\n[Step 2] Verifying ONNX model...")
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX graph is valid")

    # Print model size
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  ONNX file size: {size_mb:.1f} MB")
except ImportError:
    print(f"  ⚠ onnx package not installed — skipping graph check")
    print(f"    Install: pip install onnx")

try:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path,
                                providers=['CPUExecutionProvider'])

    # Test with batch size 4
    test_s = np.random.randn(4, T, S_DIM).astype(np.float32)
    test_n = np.random.randn(4, T, N_DIM).astype(np.float32)
    outs   = sess.run(None, {
        'sensor_input':  test_s,
        'network_input': test_n,
    })
    print(f"  ✓ ONNX Runtime inference OK")
    for i, o in enumerate(outs):
        print(f"    Output {i}: shape={o.shape}")
except ImportError:
    print(f"  ⚠ onnxruntime not installed")
    print(f"    Install: pip install onnxruntime")
except Exception as e:
    print(f"  ✗ ONNX Runtime test failed: {e}")


# ── STEP 3: Convert to OpenVINO IR ───────────────────────────
print(f"\n[Step 3] Converting to OpenVINO IR (FP16)...")

ir_dir = os.path.join(EXPORT_DIR, 'ir_fp16')
os.makedirs(ir_dir, exist_ok=True)

# Try openvino Python API first (newer versions)
try:
    from openvino.tools.mo import convert_model
    from openvino.runtime import serialize

    print(f"  Using OpenVINO Python API (mo.convert_model)...")
    ov_model = convert_model(
        onnx_path,
        compress_to_fp16=True,
    )
    xml_path = os.path.join(ir_dir, f'{MODEL_NAME}.xml')
    serialize(ov_model, xml_path)
    print(f"  ✓ IR saved → {xml_path}")

except ImportError:
    # Fall back to CLI
    print(f"  Python API not found — trying CLI (mo command)...")

    cmd = [
        'mo',
        '--input_model',     onnx_path,
        '--output_dir',      ir_dir,
        '--model_name',      MODEL_NAME,
        '--compress_to_fp16',
        '--log_level',       'ERROR',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        xml_path = os.path.join(ir_dir, f'{MODEL_NAME}.xml')
        print(f"  ✓ IR saved → {xml_path}")
    else:
        print(f"  ✗ CLI conversion failed:")
        print(f"    {result.stderr[:500]}")
        print(f"\n  Install OpenVINO dev tools:")
        print(f"    pip install openvino-dev")
        xml_path = None

except Exception as e:
    print(f"  ✗ Conversion error: {e}")
    xml_path = None


# ── STEP 4: Benchmark (if OpenVINO installed) ─────────────────
if xml_path and os.path.exists(xml_path):
    print(f"\n[Step 4] Benchmarking on CPU (DK-2500 simulation)...")

    cmd = [
        'benchmark_app',
        '-m',   xml_path,
        '-d',   'CPU',
        '-t',   '10',
        '-api', 'sync',
        '-b',   '1',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Benchmark results:")
        for line in result.stdout.splitlines():
            if any(kw in line for kw in
                   ['Throughput', 'Latency', 'Count', 'Duration']):
                print(f"    {line.strip()}")
    else:
        print(f"  ⚠ benchmark_app not available (install openvino-dev)")
else:
    print(f"\n[Step 4] Skipping benchmark — IR not generated.")


# ── STEP 5: Save export manifest ─────────────────────────────
print(f"\n[Step 5] Saving export manifest...")

# Load calibrated threshold if available
thresh_path = r'D:\trustgate_pcaps\eval_outputs\optimal_threshold.json'
if os.path.exists(thresh_path):
    with open(thresh_path) as f:
        thresh_info = json.load(f)
    threshold = thresh_info['threshold']
else:
    threshold = 0.5
    print(f"  ⚠ optimal_threshold.json not found — using default 0.5")

manifest = {
    'model_name':       MODEL_NAME,
    'source_checkpoint': CHECKPOINT,
    'training_epoch':   int(ckpt['epoch']),
    'val_auc':          float(ckpt['monitor_value']),
    'onnx_path':        onnx_path,
    'fp16_ir_xml':      xml_path if xml_path else 'NOT_GENERATED',
    'sensor_features':  int(S_DIM),
    'network_features': int(N_DIM),
    'time_steps':       int(T),
    'decision_threshold': float(threshold),
    'target_device':    'Intel DK-2500 / OpenVINO CPU Plugin',
    'output_heads': {
        'binary_logit': 'shape=(B,1) — sigmoid for attack probability',
        'class_logits': 'shape=(B,6) — softmax for attack type',
        'comp_logits':  'shape=(B,22) — sigmoid for component attribution',
    },
    'notes': [
        'Component head slots 19,20,21 have label starvation.',
        'Use attention attribution (03_attention_attribution.py)',
        'for test-set component localization instead.',
    ]
}

manifest_path = os.path.join(EXPORT_DIR, 'export_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"  ✓ Manifest saved → {manifest_path}")


# ── Final summary ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("EXPORT COMPLETE")
print(f"{'='*60}")
print(f"\nFiles generated:")
print(f"  ONNX    : {onnx_path}")
if xml_path and os.path.exists(xml_path):
    bin_path = xml_path.replace('.xml', '.bin')
    xml_size = os.path.getsize(xml_path) / 1e6
    bin_size = os.path.getsize(bin_path) / 1e6 if os.path.exists(bin_path) else 0
    print(f"  IR XML  : {xml_path} ({xml_size:.1f} MB)")
    print(f"  IR BIN  : {bin_path} ({bin_size:.1f} MB)")
print(f"  Manifest: {manifest_path}")
print(f"\nDecision threshold: {threshold:.2f}")
print(f"{'='*60}")