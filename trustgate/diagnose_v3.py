# diagnose_v3.py
# Are val attacks from the SAME or DIFFERENT incidents as train?
import numpy as np
import pandas as pd

df = pd.read_csv(r'D:\trustgate_pcaps\A12_merged.csv', low_memory=False)
df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])

# Get split info from windowed data
data = np.load(r'D:\trustgate_pcaps\A12_windowed.npz', allow_pickle=True)

print("="*60)
print("DIAGNOSIS: Attack distribution across train/val/test")
print("="*60)

# We need to figure out which rows went where
# Recompute the split logic
df['attack_block'] = (df['label_binary'].diff().fillna(0) != 0).cumsum()
attack_blocks = df[df['label_binary'] == 1].groupby('attack_block').agg(
    start=('timestamp_sgt', 'min'),
    end=('timestamp_sgt', 'max'),
    class_id=('label_class', 'first'),
    name=('attack_name', 'first'),
    size=('label_binary', 'sum')
).reset_index()

print(f"\nAll attack blocks found: {len(attack_blocks)}")
print(f"\n{'Block':>5} {'Start':>12} {'End':>12} {'Class':>5} {'Size':>5} Name")
for _, b in attack_blocks.iterrows():
    print(f"  {int(b['attack_block']):>3} "
          f"{b['start'].strftime('%H:%M:%S'):>12} "
          f"{b['end'].strftime('%H:%M:%S'):>12} "
          f"{int(b['class_id']):>5} "
          f"{int(b['size']):>5} "
          f"{b['name']}")

print(f"\nINTERPRETATION:")
print(f"  Each attack block is split 70/15/15 internally")
print(f"  This means train sees SECONDS 1-3 of attack,")
print(f"  val sees SECONDS 4, test sees SECOND 5")
print(f"  Adjacent seconds in 30-sec windows OVERLAP")
print(f"  This may explain why model can't learn properly")