# unzip_A11_day1.py
import zipfile
import os
import shutil
from pathlib import Path

ZIP_PATH    = r'D:\Users\HP\Downloads\OneDrive_2026-06-18.zip'       # Day 1 (Feb 19)
EXTRACT_DIR = r'D:\trustgate_pcaps\A11_day1_raw'

def check_disk(path, needed_gb):
    drive = os.path.splitdrive(os.path.abspath(path))[0] + '\\'
    _, _, free = shutil.disk_usage(drive)
    free_gb = free / (1024**3)
    print(f"  Free space: {free_gb:.2f} GB")
    return free_gb >= needed_gb

print("="*60)
print("A11 Day 1 — Unzip")
print("="*60)

if not os.path.exists(ZIP_PATH):
    print(f"ERROR: {ZIP_PATH} not found")
    exit(1)

zip_size_gb = os.path.getsize(ZIP_PATH) / (1024**3)
print(f"Zip: {ZIP_PATH}")
print(f"Size: {zip_size_gb:.2f} GB")

if not check_disk(EXTRACT_DIR, zip_size_gb * 2.5):
    print("WARNING: low disk space. Continue? (y/n): ", end='')
    if input().strip().lower() != 'y':
        exit(0)

os.makedirs(EXTRACT_DIR, exist_ok=True)

print("\nInspecting zip...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    names = zf.namelist()
    total = sum(zi.file_size for zi in zf.infolist())
    print(f"  Files: {len(names)}")
    print(f"  Uncompressed: {total/(1024**3):.2f} GB")

print(f"\nExtracting...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    total = len(zf.namelist())
    for i, member in enumerate(zf.namelist(), 1):
        zf.extract(member, EXTRACT_DIR)
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}]")

print(f"\nDone. Files in {EXTRACT_DIR}")
print("\nNext: decompress_A11_day1.py")