# tshark_speed_test_v2.py
# Uses only fields guaranteed to exist in all tshark versions

import subprocess
import time
import os
from pathlib import Path

TSHARK   = r'D:\Program Files\Wireshark\tshark.exe'
PCAP_DIR = r'D:\trustgate_pcaps\A12_decompressed'

test_pcap = sorted(Path(PCAP_DIR).glob('*.pcap'))[0]
print(f"Test file: {test_pcap.name}")
print(f"Size     : {test_pcap.stat().st_size/(1024**2):.1f} MB")

temp_csv = r'D:\trustgate_pcaps\tshark_test.tsv'

# Only universally-supported fields
fields = [
    'frame.time_epoch',     # absolute timestamp (Unix epoch)
    'frame.len',            # packet length in bytes
    'ip.src',               # source IP
    'ip.dst',               # destination IP
    '_ws.col.Protocol',     # protocol column from Wireshark
    '_ws.col.Info',         # info column (contains CIP details)
]

cmd = [
    TSHARK,
    '-r', str(test_pcap),
    '-T', 'fields',
    '-E', 'separator=\\t',
    '-E', 'header=y',
    '-E', 'occurrence=f',
    '-E', 'quote=d',
]
for f in fields:
    cmd.extend(['-e', f])

print(f"\nRunning tshark with {len(fields)} fields...")
t_start = time.time()

with open(temp_csv, 'w', encoding='utf-8') as out:
    result = subprocess.run(
        cmd, stdout=out, stderr=subprocess.PIPE, text=True
    )

elapsed = time.time() - t_start
print(f"Done in {elapsed:.1f} seconds")

if result.returncode != 0:
    print(f"ERROR: {result.stderr[:500]}")
else:
    with open(temp_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"  Packets extracted: {len(lines)-1:,}")
    print(f"  Rate             : {(len(lines)-1)/elapsed:,.0f} pkt/s")
    print(f"  Output size      : {os.path.getsize(temp_csv)/(1024**2):.1f} MB")
    
    print(f"\nHeader:")
    print(f"  {lines[0].rstrip()}")
    
    print(f"\nFirst 3 data lines:")
    for line in lines[1:4]:
        print(f"  {line.rstrip()[:150]}")
    
    est_min = (elapsed * 178) / 60
    print(f"\nEstimate for all 178 files:")
    print(f"  {est_min:.1f} minutes ({est_min/60:.1f} hours)")

if os.path.exists(temp_csv):
    os.remove(temp_csv)