# 00_check_environment.py
# ENHANCED VERSION — Run this before anything else.
# Save as: C:\Users\HP\Downloads\trustgate\00_check_environment.py

import sys
import os
import importlib.util

print("=" * 60)
print("TrustGate — Environment Check")
print("=" * 60)

# ── Python version ──
print(f"\nPython: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("  ⚠ WARNING: Python < 3.8 — some features may not work")
else:
    print("  ✓ Python version OK")

# ── Check every required package ──
packages = {
    "torch":        "PyTorch",
    "numpy":        "NumPy",
    "sklearn":      "scikit-learn",
    "matplotlib":   "Matplotlib",
    "seaborn":      "Seaborn",
    "onnx":         "ONNX",
    "onnxruntime":  "ONNX Runtime",
    "pandas":       "Pandas",        # ← ADDED (needed for CSV processing)
}

missing = []
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {name:20s} {ver}")
    except ImportError:
        print(f"  ✗ {name:20s} NOT FOUND")
        missing.append(pkg)

# ── Check CUDA ──
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  ✓ CUDA available: {device_name}")
        print(f"    VRAM: {vram:.1f} GB")
        # Check compute capability (needed for some ops)
        cc = torch.cuda.get_device_capability(0)
        print(f"    Compute Capability: {cc[0]}.{cc[1]}")
        if cc[0] < 6:
            print(f"    ⚠ WARNING: Compute capability < 6.0 — some optimizations disabled")
    else:
        print(f"\n  ⚠ CUDA not available — will use CPU (training will be slow)")
except Exception as e:
    print(f"\n  ⚠ CUDA check failed: {e}")

# ── Check data paths ──
# Try to auto-detect paths
possible_data_paths = [
    r'D:\trustgate_pcaps\A12_windowed_v3.npz',
    r'C:\Users\HP\Downloads\trustgate_pcaps\A12_windowed_v3.npz',
    r'.\A12_windowed_v3.npz',
    r'..\A12_windowed_v3.npz',
]

possible_ckpt_paths = [
    r'D:\trustgate_pcaps\trained_model_full_v3.pth',
    r'C:\Users\HP\Downloads\trustgate_pcaps\trained_model_full_v3.pth',
    r'.\trained_model_full_v3.pth',
    r'..\trained_model_full_v3.pth',
]

DATA_PATH = None
for p in possible_data_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break

CHECKPOINT = None
for p in possible_ckpt_paths:
    if os.path.exists(p):
        CHECKPOINT = p
        break

print(f"\n{'─'*60}")
print("Data & Checkpoint Paths")
print(f"{'─'*60}")

if DATA_PATH:
    size = os.path.getsize(DATA_PATH) / 1e6
    print(f"  ✓ Data file:   {DATA_PATH} ({size:.1f} MB)")
else:
    print(f"  ✗ Data file NOT FOUND")
    print(f"    Searched: {', '.join(possible_data_paths)}")
    missing.append("DATA_FILE")

if CHECKPOINT:
    size = os.path.getsize(CHECKPOINT) / 1e6
    print(f"  ✓ Checkpoint:  {CHECKPOINT} ({size:.1f} MB)")
else:
    print(f"  ⚠ Checkpoint NOT FOUND — train_v3.py has not completed yet")
    print(f"    Searched: {', '.join(possible_ckpt_paths)}")

# ── Check model.py exists ──
model_paths = [
    r'C:\Users\HP\Downloads\trustgate\model.py',
    r'.\model.py',
    r'..\model.py',
    r'.\trustgate\model.py',
]

model_found = False
for p in model_paths:
    if os.path.exists(p):
        print(f"  ✓ model.py:    {p}")
        model_found = True
        break

if not model_found:
    print(f"  ✗ model.py NOT FOUND — evaluation scripts will fail")
    missing.append("model.py")

# ── Disk space check ──
try:
    if DATA_PATH:
        drive = os.path.splitdrive(DATA_PATH)[0] or 'C:'
    else:
        drive = 'C:'
    import shutil
    total, used, free = shutil.disk_usage(drive)
    free_gb = free / (1024**3)
    print(f"\n  Disk space ({drive}): {free_gb:.1f} GB free")
    if free_gb < 5:
        print(f"  ⚠ WARNING: Less than 5 GB free — may run out of space during training")
except Exception as e:
    pass

# ── Final verdict ──
print("\n" + "=" * 60)
if missing:
    print(f"🔴 ISSUES FOUND: {missing}")
    print("\nFix missing packages:")
    print("  pip install torch numpy scikit-learn matplotlib seaborn onnx onnxruntime pandas")
    if "DATA_FILE" in missing:
        print("\nFix data path:")
        print("  1. Check that A12_windowed_v3.npz exists")
        print("  2. Update DATA_PATH in train_v3.py and evaluate scripts")
    if "model.py" in missing:
        print("\nFix model.py path:")
        print("  Ensure model.py is in the same directory or update sys.path in scripts")
else:
    print("✅ ALL CHECKS PASSED — ready to run pipeline")
print("=" * 60)