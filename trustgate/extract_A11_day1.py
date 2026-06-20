# extract_A11_day1.py
# Combined decompress + extract for A11 Day 1
# Disk-safe: processes one .gz at a time, cleans as it goes
# Save as: C:\Users\HP\Downloads\trustgate\extract_A11_day1.py

import subprocess
import time
import os
import sys
import re
import gc
import pickle
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# ── CONFIG ────────────────────────────────────────────────────────
TSHARK      = r'D:\Program Files\Wireshark\tshark.exe'
GZ_DIR      = r'D:\trustgate_pcaps\A11_day1_raw'   # where .gz live
TEMP_DIR    = r'D:\trustgate_pcaps\A11_day1_tmp'   # ephemeral
OUTPUT_CSV  = r'D:\trustgate_pcaps\network_features_A11_day1.csv'
LOG_FILE    = r'D:\trustgate_pcaps\extract_A11_day1_log.txt'
CHECKPOINT  = r'D:\trustgate_pcaps\A11_day1_buckets.pkl'

DISPLAY_FILTER = 'cip or (tcp.len > 0)'

CRITICAL_PATTERNS = [
    'MV101','MV201','MV301','MV302','MV303','MV304',
    'MV501','MV502','MV503','MV504',
    'P101','P102','P201','P202','P203','P204','P205','P206',
    'P207','P208','P301','P302','P401','P402','P403','P404',
    'P501','P502','P601','P602','P603',
    'LIT101','LIT301','LIT401','LIT601','LIT602',
    'FIT101','FIT201','FIT301','FIT401','FIT501','FIT502',
    'FIT503','FIT504','FIT601','FIT602',
    'AIT201','AIT202','AIT203','AIT301','AIT302','AIT303',
    'AIT401','AIT402','AIT501','AIT502','AIT503','AIT504',
    'PIT501','PIT502','PIT503','DPIT301',
]
SUSPICIOUS_PATTERNS = ['RESET','PERMISSIVE','MANUAL',
                       'OVERRIDE','FORCE','BYPASS']


def log(msg):
    stamp = datetime.now().strftime('%H:%M:%S')
    line  = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def parse_info(info):
    if not info or not isinstance(info, str):
        return None, None
    tag_match = re.search(r"'([^']+)'", info)
    tag       = tag_match.group(1) if tag_match else None
    sc_match  = re.search(r'\(0x([0-9a-fA-F]+)\)', info)
    code      = int(sc_match.group(1), 16) if sc_match else None
    return tag, code


def classify(info):
    if not info:
        return 'other', None, None
    tag, code = parse_info(info)
    if 'Connection:' in info or 'SEQ=' in info:
        return 'io', tag, code
    if code in (0x4c, 0x52):
        return 'read', tag, code
    if code in (0x4d, 0x53):
        return 'write', tag, code
    if code in (0x4b, 0x4e):
        return 'connection', tag, code
    return 'other', tag, code


def empty_bucket():
    return {
        'packet_count'     : 0,
        'total_bytes'      : 0,
        'packet_sizes'     : [],
        'src_ips'          : set(),
        'dst_ips'          : set(),
        'cip_read_count'   : 0,
        'cip_write_count'  : 0,
        'cip_io_count'     : 0,
        'cip_conn_count'   : 0,
        'cip_other_count'  : 0,
        'tags'             : [],
        'critical_writes'  : 0,
        'suspicious_writes': 0,
        'reset_count'      : 0,
        'permissive_count' : 0,
        'manual_count'     : 0,
    }


def process_one_gz(gz_path, buckets):
    """Decompress .gz → tshark → aggregate → delete intermediates."""
    pcap_path = Path(TEMP_DIR) / (gz_path.stem + '.pcap')
    tsv_path  = Path(TEMP_DIR) / (gz_path.stem + '.tsv')

    # Step 1: decompress
    try:
        log(f"  Step 1: decompress {gz_path.name}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(pcap_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024*1024)
    except Exception as e:
        log(f"  DECOMPRESS FAIL: {e}")
        return False

    # Step 2: tshark extract
    fields = ['frame.time_epoch', 'frame.len',
              'ip.src', 'ip.dst', '_ws.col.Info']
    cmd = [
        TSHARK, '-r', str(pcap_path),
        '-Y', DISPLAY_FILTER,
        '-T', 'fields',
        '-E', 'separator=' + chr(9),
        '-E', 'header=y',
        '-E', 'occurrence=f',
    ]
    for f in fields:
        cmd.extend(['-e', f])

    try:
        log(f"  Step 2: tshark")
        with open(tsv_path, 'w', encoding='utf-8') as out:
            result = subprocess.run(cmd, stdout=out,
                                    stderr=subprocess.PIPE,
                                    text=True, timeout=900)
        if result.returncode != 0:
            log(f"  TSHARK FAIL: {result.stderr[:200]}")
            pcap_path.unlink(missing_ok=True)
            tsv_path.unlink(missing_ok=True)
            return False
    except Exception as e:
        log(f"  TSHARK EXCEPTION: {e}")
        pcap_path.unlink(missing_ok=True)
        tsv_path.unlink(missing_ok=True)
        return False

    # Free pcap immediately
    pcap_path.unlink(missing_ok=True)

    # Step 3: read TSV and aggregate
    try:
        log(f"  Step 3: read TSV")
        df = pd.read_csv(tsv_path, sep='\t',
                         low_memory=False, on_bad_lines='skip')
    except Exception as e:
        log(f"  READ FAIL: {e}")
        tsv_path.unlink(missing_ok=True)
        return False

    if df.empty or 'frame.time_epoch' not in df.columns:
        log(f"  Empty or missing column")
        tsv_path.unlink(missing_ok=True)
        return False

    df['frame.time_epoch'] = pd.to_numeric(
        df['frame.time_epoch'], errors='coerce')
    df = df.dropna(subset=['frame.time_epoch'])

    log(f"  Step 4: aggregate {len(df):,} packets")
    n_packets = 0
    for row in df.itertuples(index=False):
        try:
            epoch   = row[0]
            pkt_len = int(row[1]) if not pd.isna(row[1]) else 0
            src_ip  = str(row[2]) if not pd.isna(row[2]) else ''
            dst_ip  = str(row[3]) if not pd.isna(row[3]) else ''
            info    = str(row[4]) if not pd.isna(row[4]) else ''

            dt_sgt = datetime.utcfromtimestamp(epoch) + timedelta(hours=8)
            second = dt_sgt.replace(microsecond=0)

            b = buckets[second]
            b['packet_count'] += 1
            b['total_bytes']  += pkt_len
            b['packet_sizes'].append(pkt_len)
            if src_ip and src_ip != 'nan':
                b['src_ips'].add(src_ip)
            if dst_ip and dst_ip != 'nan':
                b['dst_ips'].add(dst_ip)

            cls, tag, code = classify(info)
            if cls == 'read':
                b['cip_read_count'] += 1
            elif cls == 'write':
                b['cip_write_count'] += 1
                if tag:
                    b['tags'].append(tag)
                    tu = tag.upper()
                    for p in CRITICAL_PATTERNS:
                        if p in tu:
                            b['critical_writes'] += 1
                            break
                    for p in SUSPICIOUS_PATTERNS:
                        if p in tu:
                            b['suspicious_writes'] += 1
                            if 'RESET' in p:
                                b['reset_count'] += 1
                            elif 'PERMISSIVE' in p:
                                b['permissive_count'] += 1
                            elif 'MANUAL' in p:
                                b['manual_count'] += 1
                            break
            elif cls == 'io':
                b['cip_io_count'] += 1
            elif cls == 'connection':
                b['cip_conn_count'] += 1
            else:
                b['cip_other_count'] += 1
            n_packets += 1
        except Exception:
            continue

    del df
    tsv_path.unlink(missing_ok=True)
    log(f"  Done: {n_packets:,} aggregated")
    return True


def save_checkpoint(buckets, completed):
    serializable = {}
    for sec, b in buckets.items():
        sb = dict(b)
        sb['src_ips'] = list(sb['src_ips'])
        sb['dst_ips'] = list(sb['dst_ips'])
        serializable[sec] = sb
    with open(CHECKPOINT, 'wb') as f:
        pickle.dump({'buckets': serializable,
                     'completed': list(completed)}, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return defaultdict(empty_bucket), set()
    try:
        with open(CHECKPOINT, 'rb') as f:
            data = pickle.load(f)
        buckets = defaultdict(empty_bucket)
        for sec, b in data['buckets'].items():
            nb = empty_bucket()
            for k, v in b.items():
                if k in ('src_ips', 'dst_ips'):
                    nb[k] = set(v)
                else:
                    nb[k] = v
            buckets[sec] = nb
        log(f"Resumed: {len(buckets)} secs, "
            f"{len(data['completed'])} files done")
        return buckets, set(data['completed'])
    except Exception as e:
        log(f"Checkpoint load failed: {e}")
        return defaultdict(empty_bucket), set()


def main():
    log("="*60)
    log("TrustGate — A11 Day 1 Extraction (combined gz+extract)")
    log("="*60)
    log(f"Start: {datetime.now()}")

    os.makedirs(TEMP_DIR, exist_ok=True)

    # Find all .gz files (may be in subfolder)
    gz_files = sorted(Path(GZ_DIR).rglob('*.gz'))
    log(f"Found {len(gz_files)} .gz files")

    if not gz_files:
        log("ERROR: no .gz files found")
        sys.exit(1)

    buckets, completed = load_checkpoint()
    log(f"Already done: {len(completed)}")

    t_start = time.time()
    failed  = []

    for i, gz_path in enumerate(gz_files, 1):
        if gz_path.name in completed:
            continue

        t_file = time.time()
        ok     = process_one_gz(gz_path, buckets)

        if not ok:
            failed.append(gz_path.name)
            log(f"  [{i}/{len(gz_files)}] FAILED")
            continue

        completed.add(gz_path.name)
        t_elapsed = time.time() - t_file
        total_e   = time.time() - t_start
        rate      = len(completed) / total_e if total_e > 0 else 0
        eta_h     = (len(gz_files) - i) / rate / 3600 if rate > 0 else 0

        log(f"  [{i}/{len(gz_files)}] "
            f"{gz_path.name[-30:]:30s} "
            f"{t_elapsed:.0f}s  buckets={len(buckets):,}  "
            f"ETA: {eta_h:.1f}h")

        if i % 10 == 0:
            save_checkpoint(buckets, completed)
            log(f"  [checkpoint]")
        if i % 20 == 0:
            gc.collect()

    save_checkpoint(buckets, completed)

    log(f"\nExtraction done in {(time.time()-t_start)/3600:.2f}h")
    log(f"Failed: {len(failed)}")

    # Build CSV
    log("\nBuilding final CSV...")
    rows = []
    for sec in sorted(buckets.keys()):
        b = buckets[sec]
        sizes = b['packet_sizes']
        rows.append({
            'timestamp_sgt'    : sec,
            'packet_count'     : b['packet_count'],
            'total_bytes'      : b['total_bytes'],
            'avg_packet_size'  : float(np.mean(sizes)) if sizes else 0,
            'max_packet_size'  : int(np.max(sizes))   if sizes else 0,
            'unique_src_ips'   : len(b['src_ips']),
            'unique_dst_ips'   : len(b['dst_ips']),
            'cip_read_count'   : b['cip_read_count'],
            'cip_write_count'  : b['cip_write_count'],
            'cip_io_count'     : b['cip_io_count'],
            'cip_conn_count'   : b['cip_conn_count'],
            'cip_other_count'  : b['cip_other_count'],
            'unique_tags'      : len(set(b['tags'])),
            'critical_writes'  : b['critical_writes'],
            'suspicious_writes': b['suspicious_writes'],
            'reset_count'      : b['reset_count'],
            'permissive_count' : b['permissive_count'],
            'manual_count'     : b['manual_count'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = Path(OUTPUT_CSV).stat().st_size / (1024**2)
    log(f"Saved: {OUTPUT_CSV} ({size_mb:.2f} MB)")
    log(f"Shape: {df.shape}")
    log(f"Time range: {df['timestamp_sgt'].min()} → {df['timestamp_sgt'].max()}")

    log("="*60)
    log("COMPLETE")
    log("="*60)


if __name__ == '__main__':
    if not os.path.exists(CHECKPOINT) and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    main()