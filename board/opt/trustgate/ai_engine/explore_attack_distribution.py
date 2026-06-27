"""
Find WHERE attacks are distributed in the timeline
This tells us the right split strategy
"""
import pandas as pd
import numpy as np

print("Loading merged.csv (this takes ~30s)...")
df = pd.read_csv('./merged.csv', low_memory=False)
df.columns = df.columns.str.strip()
df['Normal/Attack'] = df['Normal/Attack'].astype(str).str.strip()
df['label'] = (df['Normal/Attack'] == 'Attack').astype(int)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
df = df.sort_values('Timestamp').reset_index(drop=True)

total = len(df)
print(f"\nTotal rows: {total:,}")
print(f"Total attacks: {df['label'].sum():,}")

# Find attack episodes (contiguous blocks of attack rows)
# An episode = run of consecutive attack rows
df['episode'] = (df['label'] != df['label'].shift()).cumsum()
attack_episodes = df[df['label']==1].groupby('episode').agg(
    start=('Timestamp','first'),
    end=('Timestamp','last'),
    rows=('label','count'),
    start_idx=('label', lambda x: x.index[0]),
    end_idx=('label', lambda x: x.index[-1])
).reset_index(drop=True)

print(f"\nAttack episodes found: {len(attack_episodes)}")
print(f"\n{'Episode':>7} | {'Start':>22} | {'End':>22} | {'Duration':>10} | {'Rows':>6} | {'% through dataset':>17}")
print("-" * 95)
for i, row in attack_episodes.iterrows():
    duration = row['end'] - row['start']
    pct_through = row['start_idx'] / total * 100
    print(f"{i+1:>7} | {str(row['start']):>22} | {str(row['end']):>22} | {str(duration):>10} | {row['rows']:>6} | {pct_through:>16.1f}%")

# Show what % of the dataset each 10% chunk contains
print(f"\n--- Dataset broken into 10% chunks ---")
print(f"{'Chunk':>10} | {'Rows':>8} | {'Attacks':>8} | {'Attack%':>8}")
print("-" * 45)
chunk_size = total // 10
for i in range(10):
    start = i * chunk_size
    end = (i+1) * chunk_size if i < 9 else total
    chunk = df.iloc[start:end]
    attacks = chunk['label'].sum()
    print(f"{i*10:>3}%-{(i+1)*10:>3}% | {len(chunk):>8,} | {attacks:>8,} | {attacks/len(chunk)*100:>7.1f}%")

print("\nPaste this output — it tells us exactly how to split.")
