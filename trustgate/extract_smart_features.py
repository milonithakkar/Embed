# extract_smart_features.py
# Re-aggregates A12_buckets.pkl with attack-aware features
# No re-extraction needed — uses existing pickle
# Save as: C:\Users\HP\Downloads\trustgate\extract_smart_features.py
# Run     : python extract_smart_features.py
# Time    : ~5-10 minutes

import pickle
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
CHECKPOINT  = r'D:\trustgate_pcaps\A12_buckets.pkl'
OUTPUT_CSV  = r'D:\trustgate_pcaps\network_features_A12_smart.csv'
BASELINE_HOURS = 4   # First 4 hours are normal — use for baseline

# Tag categorization
SAFETY_CRITICAL_VALVES = {
    'MV101','MV201','MV301','MV302','MV303','MV304',
    'MV501','MV502','MV503','MV504',
}
SAFETY_CRITICAL_PUMPS = {
    'P101','P102','P201','P202','P203','P204','P205','P206',
    'P207','P208','P301','P302','P401','P402','P403','P404',
    'P501','P502','P601','P602','P603',
}
LEVEL_SENSORS = {
    'LIT101','LIT301','LIT401','LIT601','LIT602',
}
DOSING_SENSORS = {
    'AIT201','AIT202','AIT203','AIT301','AIT302','AIT303',
    'AIT401','AIT402','AIT501','AIT502','AIT503','AIT504',
}


def categorize_tag(tag):
    """Bucket each tag into a behavioral category."""
    if not tag:
        return 'unknown'
    
    tag_upper = tag.upper()
    
    # Look for specific component in tag name
    for valve in SAFETY_CRITICAL_VALVES:
        if valve in tag_upper:
            # Is it a write or a status read?
            if 'STATUS' in tag_upper or '.PV' in tag_upper:
                return 'valve_read'
            else:
                return 'valve_write'   # actual control command
    
    for pump in SAFETY_CRITICAL_PUMPS:
        if pump in tag_upper:
            if 'STATUS' in tag_upper or '.PV' in tag_upper:
                return 'pump_read'
            else:
                return 'pump_write'
    
    for sensor in LEVEL_SENSORS:
        if sensor in tag_upper:
            return 'level_access'
    
    for sensor in DOSING_SENSORS:
        if sensor in tag_upper:
            return 'dosing_access'
    
    if 'RESET' in tag_upper:
        return 'reset_signal'
    if 'PERMISSIVE' in tag_upper:
        return 'permissive_signal'
    if 'PLANT' in tag_upper or 'STATE' in tag_upper:
        return 'state_poll'
    
    return 'other'


def compute_baseline(buckets, baseline_hours=4):
    """
    Compute baseline statistics from the first N hours.
    A12 starts at 08:55:04, normal period is 4 hours.
    Returns: dict of expected tag frequencies per category
    """
    sorted_secs = sorted(buckets.keys())
    start_time  = sorted_secs[0]
    cutoff      = start_time.replace(hour=12, minute=55)  # ~4h later
    
    baseline_tags = []
    baseline_secs = 0
    
    for sec in sorted_secs:
        if sec >= cutoff:
            break
        baseline_tags.extend(buckets[sec]['tags'])
        baseline_secs += 1
    
    print(f"  Baseline window: {baseline_secs:,} seconds")
    print(f"  Baseline tag accesses: {len(baseline_tags):,}")
    
    # Per-category baseline rate
    cat_counts = Counter()
    for tag in baseline_tags:
        cat = categorize_tag(tag)
        cat_counts[cat] += 1
    
    baseline_rates = {}
    for cat, count in cat_counts.items():
        baseline_rates[cat] = count / baseline_secs
    
    # Per-tag baseline rate (for novelty detection)
    tag_counter = Counter(baseline_tags)
    baseline_tag_rates = {
        tag: count / baseline_secs
        for tag, count in tag_counter.items()
    }
    
    print(f"  Categories seen in baseline:")
    for cat, rate in sorted(baseline_rates.items(),
                            key=lambda x: -x[1]):
        print(f"    {cat:20s}: {rate:>7.1f} per second")
    
    return baseline_rates, baseline_tag_rates, set(tag_counter.keys())


def extract_smart_features(sec, bucket, baseline_rates,
                            baseline_tag_rates, known_tags):
    """
    Compute attack-aware features for one second.
    """
    tags = bucket['tags']
    n_tags = len(tags)
    
    # Categorize all tags this second
    cat_counts = Counter()
    for tag in tags:
        cat_counts[categorize_tag(tag)] += 1
    
    # ── Feature 1: Category deviations from baseline ─────────────
    # Z-score for each category vs expected rate
    valve_writes      = cat_counts.get('valve_write', 0)
    pump_writes       = cat_counts.get('pump_write', 0)
    valve_baseline    = baseline_rates.get('valve_write', 0.1)
    pump_baseline     = baseline_rates.get('pump_write', 0.1)
    
    valve_write_dev = (valve_writes - valve_baseline) / max(valve_baseline, 0.1)
    pump_write_dev  = (pump_writes - pump_baseline)   / max(pump_baseline,  0.1)
    
    # ── Feature 2: New/unusual tag access ───────────────────────
    # Tags never seen in baseline = novel
    unique_tags_this_sec = set(tags)
    novel_tags = unique_tags_this_sec - known_tags
    novel_tag_count = len(novel_tags)
    
    # ── Feature 3: Tag entropy ──────────────────────────────────
    # Normal operation: predictable polling = low entropy
    # Attack: unusual mix = high entropy
    if n_tags > 0:
        tag_counter = Counter(tags)
        probs = np.array(list(tag_counter.values())) / n_tags
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        entropy = 0.0
    
    # ── Feature 4: Burst detection ──────────────────────────────
    # Are we seeing concentrated writes to same component?
    same_component_max = 0
    if n_tags > 0:
        # Group tags by base component (e.g., MV101 from HMI_MV101)
        components = Counter()
        for tag in tags:
            for comp in (SAFETY_CRITICAL_VALVES |
                         SAFETY_CRITICAL_PUMPS):
                if comp in tag.upper():
                    components[comp] += 1
                    break
        if components:
            same_component_max = components.most_common(1)[0][1]
    
    # ── Feature 5: Rare tag frequency ────────────────────────────
    # Sum of: count_this_sec / baseline_count, for tags that
    # appear more than expected
    rare_access_score = 0.0
    for tag, count_now in Counter(tags).items():
        baseline_rate = baseline_tag_rates.get(tag, 0.01)
        if count_now > baseline_rate * 3:   # 3x normal rate
            rare_access_score += (count_now - baseline_rate)
    
    # ── Feature 6: Reset/permissive frequency ────────────────────
    # Even if baseline has many, deviations matter
    reset_count       = sum(1 for t in tags if 'RESET' in t.upper())
    permissive_count  = sum(1 for t in tags
                            if 'PERMISSIVE' in t.upper())
    
    # ── Compile features ────────────────────────────────────────
    return {
        'timestamp_sgt'         : sec,
        
        # Volume features (kept from original)
        'packet_count'          : bucket['packet_count'],
        'total_bytes'           : bucket['total_bytes'],
        
        # Category features (new — meaningful semantics)
        'valve_write_count'     : valve_writes,
        'pump_write_count'      : pump_writes,
        'valve_read_count'      : cat_counts.get('valve_read', 0),
        'pump_read_count'       : cat_counts.get('pump_read', 0),
        'level_access_count'    : cat_counts.get('level_access', 0),
        'dosing_access_count'   : cat_counts.get('dosing_access', 0),
        'state_poll_count'      : cat_counts.get('state_poll', 0),
        
        # Deviation features (NEW — attack-sensitive)
        'valve_write_deviation' : valve_write_dev,
        'pump_write_deviation'  : pump_write_dev,
        
        # Anomaly features (NEW — attack-sensitive)
        'novel_tag_count'       : novel_tag_count,
        'tag_entropy'           : entropy,
        'burst_component_max'   : same_component_max,
        'rare_access_score'     : rare_access_score,
        
        # Signal features
        'unique_tags_this_sec'  : len(unique_tags_this_sec),
        'unique_src_ips'        : len(bucket['src_ips']),
        
        # Suspicious patterns (kept but more meaningful)
        'reset_count'           : reset_count,
        'permissive_count'      : permissive_count,
    }


def main():
    print("="*60)
    print("TrustGate — Smart Network Feature Extraction")
    print("Re-aggregating from A12_buckets.pkl")
    print("="*60)
    
    print(f"\n[1/4] Loading checkpoint pickle...")
    print(f"  File: {CHECKPOINT}")
    print(f"  Size: {Path(CHECKPOINT).stat().st_size/(1024**2):.1f} MB")
    
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct buckets
    buckets = {}
    for sec, b in data['buckets'].items():
        # b is dict with sets converted to lists for pickling
        b['src_ips'] = set(b['src_ips'])
        b['dst_ips'] = set(b['dst_ips'])
        buckets[sec] = b
    
    print(f"  Loaded: {len(buckets):,} unique seconds")
    
    # ── Compute baseline ─────────────────────────────────────────
    print(f"\n[2/4] Computing baseline from first 4 hours...")
    baseline_rates, baseline_tag_rates, known_tags = \
        compute_baseline(buckets, BASELINE_HOURS)
    print(f"  Unique tags in baseline: {len(known_tags):,}")
    
    # ── Extract smart features ───────────────────────────────────
    print(f"\n[3/4] Extracting smart features per second...")
    rows = []
    for i, sec in enumerate(sorted(buckets.keys())):
        if i % 5000 == 0:
            print(f"  Progress: {i:,} / {len(buckets):,}")
        
        features = extract_smart_features(
            sec, buckets[sec], baseline_rates,
            baseline_tag_rates, known_tags
        )
        rows.append(features)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp_sgt').reset_index(drop=True)
    
    # ── Save ────────────────────────────────────────────────────
    print(f"\n[4/4] Saving smart features...")
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = Path(OUTPUT_CSV).stat().st_size / (1024**2)
    print(f"  Saved: {OUTPUT_CSV} ({size_mb:.2f} MB)")
    print(f"  Shape: {df.shape}")
    
    # ── Verify attack discrimination ────────────────────────────
    print(f"\n" + "="*60)
    print("ATTACK vs NORMAL FEATURE COMPARISON")
    print("="*60)
    
    df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])
    
    # A12 attacks: 13:00 - 16:40 SGT (approximately)
    attack_mask = (
        (df['timestamp_sgt'].dt.hour >= 13) &
        (df['timestamp_sgt'].dt.hour < 17)
    )
    normal_mask = ~attack_mask
    
    print(f"\n  Normal rows: {normal_mask.sum():,}")
    print(f"  Attack rows: {attack_mask.sum():,}")
    print(f"\n  {'Feature':25s} {'Normal':>10s} {'Attack':>10s} "
          f"{'Diff%':>8s} {'Useful':>7s}")
    print(f"  {'-'*68}")
    
    useful_count = 0
    for col in df.columns:
        if col == 'timestamp_sgt':
            continue
        norm_mean = df.loc[normal_mask, col].mean()
        atk_mean  = df.loc[attack_mask, col].mean()
        if abs(norm_mean) > 0.001:
            diff_pct = (atk_mean - norm_mean) / abs(norm_mean) * 100
        else:
            diff_pct = (atk_mean - norm_mean) * 1000
        
        useful = "✓" if abs(diff_pct) > 10 else "·"
        if abs(diff_pct) > 10:
            useful_count += 1
        
        print(f"  {col:25s} {norm_mean:>10.3f} {atk_mean:>10.3f} "
              f"{diff_pct:>+7.1f}% {useful:>7s}")
    
    print(f"\n  Useful features (>10% diff): {useful_count}")
    
    if useful_count >= 3:
        print(f"  STATUS: GOOD — features discriminate attacks")
    elif useful_count >= 1:
        print(f"  STATUS: WEAK — some signal, marginal value")
    else:
        print(f"  STATUS: POOR — features still don't discriminate")
    
    print(f"\n" + "="*60)
    print("Smart extraction complete.")
    print(f"Output: {OUTPUT_CSV}")
    print("="*60)


if __name__ == '__main__':
    main()