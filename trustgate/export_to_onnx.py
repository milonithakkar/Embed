# export_to_onnx.py
# Exports trained PyTorch model to ONNX + OpenVINO IR
# Run on laptop: python export_to_onnx.py
# Then copy output files to board

import torch
import numpy as np
import os
import sys
sys.path.insert(0, r'C:\Users\HP\Downloads\trustgate')
from model import TrustGateModel

CKPT_PATH = r'D:\trustgate_pcaps\trained_model_v3_FINAL_AUC9620.pth'
ONNX_PATH = r'D:\trustgate_pcaps\trustgate_model_v3.onnx'
OUT_DIR   = r'D:\trustgate_pcaps\openvino_model_v3'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 55)
print("TrustGate — ONNX + OpenVINO Export")
print("=" * 55)

# Load model
print(f"\n[1/4] Loading model...")
ckpt  = torch.load(CKPT_PATH, map_location='cpu',
                   weights_only=False)
model = TrustGateModel()
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Epoch: {ckpt['epoch']}")
print(f"  AUC:   {ckpt['monitor_value']:.4f}")

# Create dummy inputs
print(f"\n[2/4] Exporting to ONNX...")
xs = torch.randn(1, 30, 71)   # sensor stream
xn = torch.zeros(1, 30, 19)   # network stream (zeros for board)

# Export
torch.onnx.export(
    model,
    (xs, xn),
    ONNX_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['sensor_input', 'network_input'],
    output_names=['binary_logit', 'class_logits',
                  'comp_logits', 'attn_s2n',
                  'attn_n2s', 'sensor_imp',
                  'sensor_imp_logits'],
    dynamic_axes={
        'sensor_input':  {0: 'batch_size'},
        'network_input': {0: 'batch_size'},
        'binary_logit':  {0: 'batch_size'},
        'class_logits':  {0: 'batch_size'},
        'sensor_imp':    {0: 'batch_size'},
    }
)
print(f"  Saved: {ONNX_PATH}")

# Verify ONNX
print(f"\n[3/4] Verifying ONNX...")
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    out  = sess.run(None, {
        'sensor_input':  xs.numpy(),
        'network_input': xn.numpy(),
    })
    print(f"  binary_logit:  {out[0].shape} ✓")
    print(f"  class_logits:  {out[1].shape} ✓")
    print(f"  sensor_imp:    {out[5].shape} ✓")
    prob = 1 / (1 + np.exp(-out[0][0, 0]))
    print(f"  Test prob:     {prob:.4f}")
    print(f"  ONNX verified ✓")
except Exception as e:
    print(f"  ONNX verify failed: {e}")
    print(f"  Install: pip install onnxruntime")

# Convert to OpenVINO IR
print(f"\n[4/4] Converting to OpenVINO IR...")
try:
    import subprocess
    cmd = [
        'mo',
        f'--input_model={ONNX_PATH}',
        f'--output_dir={OUT_DIR}',
        '--model_name=trustgate_model',
        '--compress_to_fp16=True',
    ]
    result = subprocess.run(cmd, capture_output=True,
                            text=True)
    if result.returncode == 0:
        print(f"  Saved IR to: {OUT_DIR}")
        print(f"  Files:")
        for f in os.listdir(OUT_DIR):
            size = os.path.getsize(
                os.path.join(OUT_DIR, f)) / 1e6
            print(f"    {f} ({size:.1f} MB)")
    else:
        print(f"  MO failed: {result.stderr[-500:]}")
        print(f"  Try: pip install openvino-dev")
        print(f"  Or use ONNX directly on board")
except FileNotFoundError:
    print(f"  'mo' command not found")
    print(f"  Install: pip install openvino-dev")
    print(f"  For now, use ONNX file directly on board")

print(f"\n{'='*55}")
print(f"Files to copy to board:")
print(f"  {ONNX_PATH}")
print(f"  OR {OUT_DIR}/trustgate_model.xml")
print(f"  OR {OUT_DIR}/trustgate_model.bin")
print(f"  + D:\\trustgate_pcaps\\a11_scaler.pkl")
print(f"  + D:\\trustgate_pcaps\\sensor_normal_baseline.npy")
print(f"\nBoard destination: /opt/trustgate/model/")
print(f"{'='*55}")