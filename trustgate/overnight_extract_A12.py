# =================================================================
# overnight_extract_A12.py  (v3 — production)
# 
# Extracts CIP network features from A12 PCAP files per second.
# Uses tshark for speed, processes interleaved for disk safety.
# Checkpoints every 10 files for crash recovery.
# 
# Save as: C:\Users\HP\Downloads\trustgate\overnight_extract_A12.py
# Run    : python overnight_extract_A12.py
# Time   : ~6-9 hours for 178 files
# =================================================================

import subprocess
import time
import os
import sys
import re
import gc
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────
TSHARK      = r'D:\Program Files\Wireshark\tshark.exe'
PCAP_DIR    = r'D:\trustgate_pcaps\A12_decompressed'
TSV_DIR     = r'D:\trustgate_pcaps\A12_tsv'
OUTPUT_CSV  = r'D:\trustgate_pcaps\network_features_A12.csv'
LOG_FILE    = r'D:\trustgate_pcaps\extract_A12_log.txt'
CHECKPOINT  = r'D:\trustgate_pcaps\A12_buckets.pkl'

# Display filter — drop noise packets
DISPLAY_FILTER = 'cip or (tcp.len > 0)'

# Tag patterns for attack-relevant writes
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


# ── Logging ──────────────────────────────────────────────────────
def log(msg):
    """Print to console AND log file with timestamp + flush."""
    stamp = datetime.now().strftime('%H:%M:%S')
    line  = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ── Parse helpers ────────────────────────────────────────────────
def parse_info(info):
    """
    Extract tag name and service code from Info column.
    
    Examples:
      "'HMI_PLANT' - Service (0x4d)"
        → tag='HMI_PLANT', code=0x4d
      "Connection Manager - Unconnected Send: 'HMI_MV201' - Service (0x4c)"
        → tag='HMI_MV201', code=0x4c
      "Success: Service (0x4d)"
        → tag=None, code=0x4d
    """
    if not info or not isinstance(info, str):
        return None, None
    
    tag_match = re.search(r"'([^']+)'", info)
    tag       = tag_match.group(1) if tag_match else None
    
    sc_match  = re.search(r'\(0x([0-9a-fA-F]+)\)', info)
    code      = int(sc_match.group(1), 16) if sc_match else None
    
    return tag, code


def classify(info):
    """
    Classify packet into category based on Info column.
    Returns: (category, tag_name, service_code)
    
    Categories:
      'read'       - CIP Read Tag (0x4c, 0x52)
      'write'      - CIP Write Tag (0x4d, 0x53)
      'io'         - CIP cyclic I/O exchange
      'connection' - CIP Forward Open/Close (0x4b, 0x4e)
      'other'      - Everything else (success responses, etc.)
    """
    if not info:
        return 'other', None, None
    
    tag, code = parse_info(info)
    
    # CIP I/O cyclic data
    if 'Connection:' in info or 'SEQ=' in info:
        return 'io', tag, code
    
    # Classify by service code
    if code in (0x4c, 0x52):
        return 'read', tag, code
    if code in (0x4d, 0x53):
        return 'write', tag, code
    if code in (0x4b, 0x4e):
        return 'connection', tag, code
    
    return 'other', tag, code


# ── Empty bucket factory ─────────────────────────────────────────
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


# ── Process one PCAP end-to-end ──────────────────────────────────
def process_one_pcap(pcap_path, buckets):
    """
    Full pipeline for ONE PCAP file:
      A. tshark extract → TSV
      B. read TSV with pandas
      C. aggregate per-second into buckets
      D. delete TSV to free disk
    Returns: True if success, False on any failure.
    """
    tsv_path = Path(TSV_DIR) / (pcap_path.stem + '.tsv')
    
    # Clean stale TSV if exists
    if tsv_path.exists():
        try:
            tsv_path.unlink()
        except Exception as e:
            log(f"  WARN: stale TSV not removed: {e}")
    
    # ── Step A: tshark with TAB separator (chr(9)) ──────────────
    fields = [
        'frame.time_epoch',
        'frame.len',
        'ip.src',
        'ip.dst',
        '_ws.col.Info',
    ]
    cmd = [
        TSHARK,
        '-r', str(pcap_path),
        '-Y', DISPLAY_FILTER,
        '-T', 'fields',
        '-E', 'separator=' + chr(9),    # actual TAB char
        '-E', 'header=y',
        '-E', 'occurrence=f',
    ]
    for f in fields:
        cmd.extend(['-e', f])
    
    try:
        log(f"  Step A: tshark on {pcap_path.name}")
        with open(tsv_path, 'w', encoding='utf-8') as out:
            result = subprocess.run(
                cmd,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=900,    # 15 min max per file
            )
        if result.returncode != 0:
            log(f"  TSHARK FAIL (rc={result.returncode})")
            log(f"  STDERR: {result.stderr[:500]}")
            return False
        
        tsv_size_mb = tsv_path.stat().st_size / (1024**2) \
            if tsv_path.exists() else 0
        log(f"  Step A done: TSV {tsv_size_mb:.1f} MB")
        
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after 15 min on {pcap_path.name}")
        return False
    except Exception as e:
        log(f"  EXCEPTION in tshark: {type(e).__name__}: {e}")
        return False
    
    # ── Step B: read TSV ────────────────────────────────────────
    try:
        log(f"  Step B: reading TSV")
        df = pd.read_csv(
            tsv_path,
            sep='\t',
            low_memory=False,
            on_bad_lines='skip',
        )
        log(f"  Step B done: {len(df):,} rows, "
            f"cols: {len(df.columns)}")
    except Exception as e:
        log(f"  TSV READ FAIL: {type(e).__name__}: {e}")
        tsv_path.unlink(missing_ok=True)
        return False
    
    if df.empty:
        log(f"  Empty TSV — skipping")
        tsv_path.unlink(missing_ok=True)
        return False
    
    # Debug print actual columns on first file only
    if not hasattr(process_one_pcap, '_logged_cols'):
        log(f"  DEBUG cols: {list(df.columns)}")
        if len(df) > 0:
            log(f"  DEBUG first row: {df.iloc[0].tolist()}")
        process_one_pcap._logged_cols = True
    
    # Verify time column exists
    if 'frame.time_epoch' not in df.columns:
        log(f"  Missing frame.time_epoch. Cols: {list(df.columns)}")
        tsv_path.unlink(missing_ok=True)
        return False
    
    # Convert epoch to numeric
    df['frame.time_epoch'] = pd.to_numeric(
        df['frame.time_epoch'], errors='coerce'
    )
    df = df.dropna(subset=['frame.time_epoch'])
    
    if df.empty:
        log(f"  All rows had bad epoch values")
        tsv_path.unlink(missing_ok=True)
        return False
    
    # ── Step C: aggregate per second ────────────────────────────
    log(f"  Step C: aggregating")
    n_packets = 0
    
    for row in df.itertuples(index=False):
        try:
            epoch   = row[0]
            pkt_len = int(row[1]) if not pd.isna(row[1]) else 0
            src_ip  = str(row[2]) if not pd.isna(row[2]) else ''
            dst_ip  = str(row[3]) if not pd.isna(row[3]) else ''
            info    = str(row[4]) if not pd.isna(row[4]) else ''
            
            # UTC epoch → SGT (+8 hours)
            dt_sgt = datetime.utcfromtimestamp(epoch) + \
                     timedelta(hours=8)
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
    
    log(f"  Step C done: {n_packets:,} packets")
    
    # ── Step D: cleanup TSV ─────────────────────────────────────
    del df
    try:
        tsv_path.unlink()
        log(f"  Step D: TSV deleted")
    except Exception as e:
        log(f"  WARN: TSV not deleted: {e}")
    
    return True


# ── Checkpoint save/load ─────────────────────────────────────────
def save_checkpoint(buckets, completed_files):
    """Pickle current state for resume capability."""
    serializable = {}
    for sec, b in buckets.items():
        sb = dict(b)
        sb['src_ips'] = list(sb['src_ips'])
        sb['dst_ips'] = list(sb['dst_ips'])
        serializable[sec] = sb
    
    with open(CHECKPOINT, 'wb') as f:
        pickle.dump({
            'buckets'  : serializable,
            'completed': list(completed_files),
        }, f)


def load_checkpoint():
    """Resume from prior checkpoint if it exists."""
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
        
        log(f"  Resumed: {len(buckets)} seconds, "
            f"{len(data['completed'])} files done")
        return buckets, set(data['completed'])
    except Exception as e:
        log(f"  Checkpoint load failed: {e}")
        return defaultdict(empty_bucket), set()


# ── Main ─────────────────────────────────────────────────────────
def main():
    log("="*60)
    log("TrustGate — A12 Extraction (v3 production)")
    log("="*60)
    log(f"Start time: {datetime.now()}")
    log(f"PCAP dir  : {PCAP_DIR}")
    log(f"TSV dir   : {TSV_DIR}")
    log(f"Output    : {OUTPUT_CSV}")
    log(f"Checkpoint: {CHECKPOINT}")
    log(f"Filter    : {DISPLAY_FILTER}")
    
    os.makedirs(TSV_DIR, exist_ok=True)
    
    pcaps = sorted(Path(PCAP_DIR).glob('*.pcap'))
    log(f"Found {len(pcaps)} PCAP files")
    
    if not pcaps:
        log("ERROR: no PCAP files found")
        sys.exit(1)
    
    # Resume from checkpoint
    buckets, completed = load_checkpoint()
    log(f"Already completed: {len(completed)} files")
    
    t_start  = time.time()
    n_done   = len(completed)
    n_failed = []
    
    for i, pcap_path in enumerate(pcaps, 1):
        if pcap_path.name in completed:
            continue
        
        t_file = time.time()
        ok     = process_one_pcap(pcap_path, buckets)
        
        if not ok:
            n_failed.append(pcap_path.name)
            log(f"  [{i:>3}/{len(pcaps)}] FAILED: {pcap_path.name}")
            continue
        
        completed.add(pcap_path.name)
        n_done += 1
        
        t_elapsed = time.time() - t_file
        total_e   = time.time() - t_start
        rate      = n_done / total_e if total_e > 0 else 0
        eta_h     = (len(pcaps) - i) / rate / 3600 if rate > 0 else 0
        
        log(f"  [{i:>3}/{len(pcaps)}] "
            f"{pcap_path.name[-32:]:32s} "
            f"{t_elapsed:>5.1f}s  "
            f"buckets={len(buckets):>5}  "
            f"ETA: {eta_h:.1f}h")
        
        # Checkpoint every 10 files
        if i % 10 == 0:
            save_checkpoint(buckets, completed)
            log(f"  [checkpoint saved at file {i}]")
        
        # GC every 20 files
        if i % 20 == 0:
            gc.collect()
    
    # Final checkpoint
    save_checkpoint(buckets, completed)
    
    log(f"\n" + "="*60)
    log("Extraction complete.")
    log(f"  Completed: {len(completed)}")
    log(f"  Failed   : {len(n_failed)}")
    log(f"  Total    : {(time.time()-t_start)/3600:.2f}h")
    log("="*60)
    
    if n_failed:
        log("Failed files:")
        for name in n_failed:
            log(f"  {name}")
    
    # ── Build final CSV ─────────────────────────────────────────
    log("\nBuilding final CSV...")
    
    rows = []
    for sec in sorted(buckets.keys()):
        b     = buckets[sec]
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
    
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024**2)
    log(f"Saved: {OUTPUT_CSV} ({size_mb:.2f} MB)")
    log(f"Shape: {df.shape}")
    log(f"Time range: {df['timestamp_sgt'].min()} → "
        f"{df['timestamp_sgt'].max()}")
    
    # Summary statistics
    log(f"\nFeature summary (per-second stats):")
    for col in ['packet_count', 'cip_read_count',
                'cip_write_count', 'cip_io_count',
                'critical_writes', 'suspicious_writes',
                'reset_count', 'unique_tags']:
        if col in df.columns:
            log(f"  {col:25s} "
                f"mean={df[col].mean():>8.2f}  "
                f"max={df[col].max():>6}  "
                f"nonzero={int((df[col]>0).sum()):>5}")
    
    log(f"\n" + "="*60)
    log(f"COMPLETE")
    log(f"End time: {datetime.now()}")
    log("="*60)


if __name__ == '__main__':
    # Reset log on fresh start (not on resume)
    if not os.path.exists(CHECKPOINT) and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    main()