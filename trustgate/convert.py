"""
TrustGate — PyTorch → ONNX → OpenVINO IR
Run on laptop after training completes
"""
import torch
from model import TrustGateModel

CHECKPOINT = 'models/trustgate_best.pt'
ONNX_PATH  = 'models/trustgate.onnx'
IR_PATH    = 'models/trustgate'

def convert():
    print("Loading checkpoint...")
    ckpt  = torch.load(CHECKPOINT, map_location='cpu')
    model = TrustGateModel(**ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Val loss: {ckpt['val_loss']:.4f} | Val acc: {ckpt['val_acc']:.2f}%")

    # Dummy inputs
    x_s = torch.randn(1, 30, 44)
    x_n = torch.randn(1, 30, 132)

    print("\nExporting to ONNX...")
    torch.onnx.export(
        model, (x_s, x_n), ONNX_PATH,
        opset_version=11,
        input_names =['sensor_input', 'network_input'],
        output_names=['binary_logit', 'stage_logits', 'sensor_weights', 'network_weights'],
        dynamic_axes={
            'sensor_input':    {0: 'batch'},
            'network_input':   {0: 'batch'},
            'binary_logit':    {0: 'batch'},
            'stage_logits':    {0: 'batch'},
            'sensor_weights':  {0: 'batch'},
            'network_weights': {0: 'batch'},
        }
    )
    print(f"  [OK] {ONNX_PATH}")

    print("\nConverting to OpenVINO IR...")
    import subprocess
    r = subprocess.run(['ovc', ONNX_PATH, '--output_model', IR_PATH],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  [OK] {IR_PATH}.xml + {IR_PATH}.bin")
    else:
        print(f"  [FAIL]\n{r.stderr}")
        print("  Run: pip install openvino")

if __name__ == '__main__':
    convert()