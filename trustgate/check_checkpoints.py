# check_checkpoints.py
# Save as: C:\Users\HP\Downloads\trustgate\check_checkpoints.py

import torch

paths = [
    r'D:\trustgate_pcaps\trained_model_full_v3.pth',
    r'D:\trustgate_pcaps\trained_model_full_v4.pth',
    r'D:\trustgate_pcaps\trained_model_A12.pth',
    r'D:\trustgate_pcaps\trained_model_A12_v3.pth',
]

print("=" * 60)
print("Checkpoint Inspector")
print("=" * 60)

for path in paths:
    try:
        c = torch.load(path, map_location='cpu',
                       weights_only=False)
        epoch = c.get('epoch', 'unknown')
        auc   = c.get('monitor_value', 0.0)
        thr   = c.get('best_threshold', 'unknown')
        keys  = list(c.keys())
        print(f"\n{path}")
        print(f"  epoch          : {epoch}")
        print(f"  monitor_value  : {auc:.4f}")
        print(f"  best_threshold : {thr}")
        print(f"  keys           : {keys}")
    except Exception as e:
        print(f"\n{path}")
        print(f"  ERROR: {e}")

print("\n" + "=" * 60)