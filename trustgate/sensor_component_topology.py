# sensor_component_topology.py
# Add this to the existing file — stage membership for locality penalty
# Save as: C:\Users\HP\Downloads\trustgate\sensor_component_topology.py

# ── STAGE MEMBERSHIP ──────────────────────────────────────────────────────────
# Maps each sensor index to its plant stage (1-6)
# Used for topology-constrained attention penalty

SENSOR_STAGE = {
    # Stage 1 — Raw Water Intake
    0:  1,  # P1_STATE
    1:  1,  # LIT101.Pv
    2:  1,  # FIT101.Pv
    3:  1,  # MV101.Status
    4:  1,  # P101.Status
    5:  1,  # P102.Status

    # Stage 2 — Chemical Dosing
    6:  2,  # P2_STATE
    7:  2,  # FIT201.Pv
    8:  2,  # AIT201.Pv
    9:  2,  # AIT202.Pv
    10: 2,  # AIT203.Pv
    11: 2,  # MV201.Status
    12: 2,  # P201.Status
    13: 2,  # P202.Status
    14: 2,  # P203.Status
    15: 2,  # P204.Status
    16: 2,  # P205.Status
    17: 2,  # P206.Status
    18: 2,  # P207.Status
    19: 2,  # P208.Status

    # Stage 3 — Ultrafiltration
    20: 3,  # P3_STATE
    21: 3,  # AIT301.Pv
    22: 3,  # AIT302.Pv
    23: 3,  # AIT303.Pv
    24: 3,  # LIT301.Pv
    25: 3,  # FIT301.Pv
    26: 3,  # DPIT301.Pv
    27: 3,  # MV301.Status
    28: 3,  # MV302.Status
    29: 3,  # MV303.Status
    30: 3,  # MV304.Status
    31: 3,  # P301.Status
    32: 3,  # P302.Status

    # Stage 4 — De-chlorination / RO
    33: 4,  # P4_STATE
    34: 4,  # LIT401.Pv
    35: 4,  # FIT401.Pv
    36: 4,  # AIT401.Pv
    37: 4,  # AIT402.Pv
    38: 4,  # P401.Status
    39: 4,  # P402.Status
    40: 4,  # P403.Status
    41: 4,  # P404.Status
    42: 4,  # UV401.Status

    # Stage 5 — Backwash / RO membranes
    43: 5,  # P5_STATE
    44: 5,  # FIT501.Pv
    45: 5,  # FIT502.Pv
    46: 5,  # FIT503.Pv
    47: 5,  # FIT504.Pv
    48: 5,  # AIT501.Pv
    49: 5,  # AIT502.Pv
    50: 5,  # AIT503.Pv
    51: 5,  # AIT504.Pv
    52: 5,  # PIT501.Pv
    53: 5,  # PIT502.Pv
    54: 5,  # PIT503.Pv
    55: 5,  # P501.Status
    56: 5,  # P501.Speed
    57: 5,  # P502.Status
    58: 5,  # P502.Speed
    59: 5,  # MV501.Status
    60: 5,  # MV502.Status
    61: 5,  # MV503.Status
    62: 5,  # MV504.Status

    # Stage 6 — Return / Storage
    63: 6,  # P6_STATE
    64: 6,  # LIT601.Pv
    65: 6,  # LIT602.Pv
    66: 6,  # FIT601.Pv
    67: 6,  # FIT602.Pv
    68: 6,  # P601.Status
    69: 6,  # P602.Status
    70: 6,  # P603.Status
}

# ── COMPONENT STAGE MEMBERSHIP ────────────────────────────────────────────────
# Maps each component slot to its plant stage
COMPONENT_STAGE = {
    0:  1,   # MV101
    10: 1,   # P101
    11: 1,   # P102
    18: 1,   # LIT101
    1:  2,   # MV201
    12: 2,   # P201
    13: 2,   # P202
    14: 2,   # P203
    15: 2,   # P204
    16: 2,   # P205
    17: 2,   # P206
    2:  3,   # MV301
    3:  3,   # MV302  ← UNSEEN
    4:  3,   # MV303  ← UNSEEN
    5:  3,   # MV304
    20: 3,   # DPIT301
    6:  5,   # MV501
    7:  5,   # MV502
    8:  5,   # MV503
    9:  5,   # MV504
    19: 6,   # LIT601
    21: 4,   # AIT402 ← UNSEEN
}

# ── BUILD STAGE SENSOR MASK ───────────────────────────────────────────────────
# For each stage, which sensor indices belong to it
import numpy as np

N_SENSORS = 71
N_STAGES  = 7   # stages 1-6, index 0 unused

STAGE_SENSOR_MASK = np.zeros((N_STAGES, N_SENSORS), dtype=np.float32)
for sensor_idx, stage in SENSOR_STAGE.items():
    STAGE_SENSOR_MASK[stage, sensor_idx] = 1.0

# ── COMPONENT TO STAGE LOOKUP ─────────────────────────────────────────────────
def get_component_stage(comp_slot):
    return COMPONENT_STAGE.get(comp_slot, -1)

def get_stage_sensor_mask(stage):
    """Returns binary mask (71,) — 1 for sensors in this stage, 0 otherwise."""
    if stage < 0 or stage >= N_STAGES:
        return np.ones(N_SENSORS, dtype=np.float32)
    return STAGE_SENSOR_MASK[stage].copy()

if __name__ == '__main__':
    print("Stage sensor counts:")
    for s in range(1, 7):
        count = int(STAGE_SENSOR_MASK[s].sum())
        sensors = [i for i in range(N_SENSORS)
                   if SENSOR_STAGE.get(i) == s]
        print(f"  Stage {s}: {count} sensors → {sensors}")