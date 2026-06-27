"""
Verify episode merging logic before implementing full split.
Run on LAPTOP in same folder as merged.csv.
No file writing — just analysis.
"""
import pandas as pd
import numpy as np

MERGE_GAP_MINUTES  = 5    # merge episodes closer than this
BUFFER_MINUTES     = 7    # normal context claimed around each super-episode

print("="*65)
print("Episode Merge + Buffer Verification")
print("="*65)

# Raw episode data from our earlier exploration
episodes_raw = [
    ( 1, "2015-12-28 10:29:14", "2015-12-28 10:44:53",  940),
    ( 2, "2015-12-28 10:51:08", "2015-12-28 10:58:30",  443),
    ( 3, "2015-12-28 11:22:00", "2015-12-28 11:28:22",  383),
    ( 4, "2015-12-28 11:47:39", "2015-12-28 11:54:08",  390),
    ( 5, "2015-12-28 12:00:55", "2015-12-28 12:04:10",  196),
    ( 6, "2015-12-28 12:08:25", "2015-12-28 12:15:33",  429),
    ( 7, "2015-12-28 13:10:10", "2015-12-28 13:26:13",  964),
    ( 8, "2015-12-28 14:16:20", "2015-12-28 14:28:20",  721),
    ( 9, "2015-12-29 06:30:00", "2015-12-29 06:42:00",  721),
    (10, "2015-12-29 11:11:25", "2015-12-29 11:15:17",  233),
    (11, "2015-12-29 11:35:40", "2015-12-29 11:42:50",  431),
    (12, "2015-12-29 11:57:25", "2015-12-29 12:02:00",  276),
    (13, "2015-12-29 14:38:12", "2015-12-29 14:50:08",  717),
    (14, "2015-12-29 18:10:43", "2015-12-29 18:15:01",  259),
    (15, "2015-12-29 18:15:43", "2015-12-29 18:22:17",  395),
    (16, "2015-12-29 18:30:00", "2015-12-29 18:42:00",  721),
    (17, "2015-12-29 22:55:18", "2015-12-29 23:03:00",  463),
    (18, "2015-12-30 01:42:34", "2015-12-30 01:54:10",  697),
    (19, "2015-12-30 09:51:08", "2015-12-30 09:56:28",  321),
    (20, "2015-12-30 10:01:50", "2015-12-30 10:12:01",  612),
    (21, "2015-12-30 17:04:56", "2015-12-30 17:29:00", 1445),
    (22, "2015-12-31 01:17:08", "2015-12-31 11:15:27",35900),
    (23, "2015-12-31 15:32:00", "2015-12-31 15:34:00",  121),
    (24, "2015-12-31 15:47:40", "2015-12-31 16:07:10", 1171),
    (25, "2015-12-31 22:05:34", "2015-12-31 22:11:40",  367),
    (26, "2016-01-01 10:36:00", "2016-01-01 10:46:00",  601),
    (27, "2016-01-01 14:21:12", "2016-01-01 14:28:35",  444),
    (28, "2016-01-01 17:12:40", "2016-01-01 17:14:20",  101),
    (29, "2016-01-01 17:18:56", "2016-01-01 17:26:56",  481),
    (30, "2016-01-01 22:16:01", "2016-01-01 22:25:00",  540),
    (31, "2016-01-02 11:17:02", "2016-01-02 11:24:50",  469),
    (32, "2016-01-02 11:31:38", "2016-01-02 11:36:18",  281),
    (33, "2016-01-02 11:43:48", "2016-01-02 11:50:28",  401),
    (34, "2016-01-02 11:51:42", "2016-01-02 11:56:38",  297),
    (35, "2016-01-02 13:13:02", "2016-01-02 13:41:11", 1690),
]

df_ep = pd.DataFrame(episodes_raw, columns=['ep','start','end','rows'])
df_ep['start'] = pd.to_datetime(df_ep['start'])
df_ep['end']   = pd.to_datetime(df_ep['end'])

# ── Step 1: Show gaps between consecutive episodes ─────────────────
print("\n--- Gaps between consecutive episodes ---")
print(f"{'Ep i':>5} → {'Ep i+1':>6} | {'Gap':>12} | {'Merge?':>8}")
print("-" * 45)
merge_threshold = pd.Timedelta(minutes=MERGE_GAP_MINUTES)
for i in range(len(df_ep)-1):
    gap = df_ep.iloc[i+1]['start'] - df_ep.iloc[i]['end']
    should_merge = gap < merge_threshold
    flag = "← MERGE" if should_merge else ""
    print(f"  {df_ep.iloc[i]['ep']:>3} → {df_ep.iloc[i+1]['ep']:>3}   | "
          f"{str(gap):>12} | {flag}")

# ── Step 2: Build super-episodes ──────────────────────────────────
print(f"\n--- Super-episodes (merge gap < {MERGE_GAP_MINUTES} min) ---")
super_episodes = []
current = {
    'start': df_ep.iloc[0]['start'],
    'end':   df_ep.iloc[0]['end'],
    'original_eps': [df_ep.iloc[0]['ep']],
    'rows': df_ep.iloc[0]['rows']
}
for i in range(1, len(df_ep)):
    gap = df_ep.iloc[i]['start'] - current['end']
    if gap < merge_threshold:
        current['end'] = df_ep.iloc[i]['end']
        current['original_eps'].append(df_ep.iloc[i]['ep'])
        current['rows'] += df_ep.iloc[i]['rows']
    else:
        super_episodes.append(current)
        current = {
            'start': df_ep.iloc[i]['start'],
            'end':   df_ep.iloc[i]['end'],
            'original_eps': [df_ep.iloc[i]['ep']],
            'rows': df_ep.iloc[i]['rows']
        }
super_episodes.append(current)

print(f"Original episodes: 35 → Super-episodes: {len(super_episodes)}")
print(f"\n{'SE#':>4} | {'Original eps':>20} | {'Start':>20} | {'End':>20} | {'Rows':>6}")
print("-"*80)
for i, se in enumerate(super_episodes):
    eps_str = str(se['original_eps'])
    print(f"  {i+1:>2} | {eps_str:>20} | {str(se['start']):>20} | {str(se['end']):>20} | {se['rows']:>6}")

# ── Step 3: Show buffer zones ─────────────────────────────────────
print(f"\n--- Buffer zones ({BUFFER_MINUTES} min before + after each super-episode) ---")
buf = pd.Timedelta(minutes=BUFFER_MINUTES)
print(f"{'SE#':>4} | {'Buffer Start':>22} | {'Attack Start':>22} | {'Attack End':>22} | {'Buffer End':>22} | Overlap?")
print("-"*115)

for i, se in enumerate(super_episodes):
    buf_start = se['start'] - buf
    buf_end   = se['end']   + buf

    # Check overlap with previous
    overlap_prev = ""
    if i > 0:
        prev_buf_end = super_episodes[i-1]['end'] + buf
        if buf_start < prev_buf_end:
            overlap_min = (prev_buf_end - buf_start).seconds // 60
            overlap_prev = f"← overlaps SE{i} by {overlap_min}min"

    print(f"  {i+1:>2} | {str(buf_start):>22} | {str(se['start']):>22} | "
          f"{str(se['end']):>22} | {str(buf_end):>22} | {overlap_prev}")

# ── Step 4: Proposed split assignment ────────────────────────────
print(f"\n--- Proposed split assignment for super-episodes ---")
print("Strategy: test=last 3 SE, val=every 4th SE, train=rest")
n_se = len(super_episodes)
print(f"Total super-episodes: {n_se}")
print(f"  Test  → SE {n_se-2} to {n_se} (last 3)")
print(f"  Val   → SE 4, 8, 12, ... (every 4th)")
print(f"  Train → all others (including SE with episode 22)")

for i, se in enumerate(super_episodes):
    se_num = i + 1
    if se_num > n_se - 3:
        sp = 'TEST'
    elif se_num % 4 == 0:
        sp = 'VAL'
    else:
        sp = 'TRAIN'
    print(f"  SE{se_num:>2} (eps {se['original_eps']}) → {sp} | rows: {se['rows']}")

print("\n" + "="*65)
print("Paste this output — we verify before writing the full split.")
print("="*65)
