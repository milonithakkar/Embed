# test_tshark_separator.py
import subprocess
from pathlib import Path

TSHARK = r'D:\Program Files\Wireshark\tshark.exe'
PCAP_DIR = r'D:\trustgate_pcaps\A12_decompressed'

pcap = sorted(Path(PCAP_DIR).glob('*.pcap'))[0]
out_tsv = r'D:\trustgate_pcaps\test_sep.tsv'

cmd = [
    TSHARK, '-r', str(pcap),
    '-Y', 'cip',
    '-T', 'fields',
    '-E', 'separator=' + chr(9),    # actual tab
    '-E', 'header=y',
    '-E', 'occurrence=f',
    '-e', 'frame.time_epoch',
    '-e', 'frame.len',
    '-e', '_ws.col.Info',
]

print("Running tshark with TAB separator (chr(9))...")
import time
t0 = time.time()
with open(out_tsv, 'w', encoding='utf-8') as f:
    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
print(f"Done in {time.time()-t0:.1f}s")

# Check the output
print("\nFirst 3 lines of output:")
with open(out_tsv, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        # Show line with explicit tab markers
        marked = line.rstrip().replace('\t', '<TAB>')
        print(f"  {marked[:150]}")

# Try pandas
import pandas as pd
df = pd.read_csv(out_tsv, sep='\t')
print(f"\nColumns: {list(df.columns)}")
print(f"Shape  : {df.shape}")
print(f"First row: {df.iloc[0].tolist()}")

import os
os.remove(out_tsv)
print("\n[OK] if columns show as 3 separate items")