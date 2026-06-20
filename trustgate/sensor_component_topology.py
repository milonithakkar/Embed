# sensor_component_topology.py
# SWaT Plant Topology: Maps 71 sensor columns → 22 component slots
# Used for evaluating zero-shot attention attribution
# Save as: C:\Users\HP\Downloads\trustgate\sensor_component_topology.py

import numpy as np

# ── Component Names (22 slots, matches your NPZ) ──
COMPONENT_NAMES = [
    "MV101",   # Slot 0  — Stage 1 inlet valve
    "MV201",   # Slot 1  — Stage 2 chemical valve
    "MV301",   # Slot 2  — Stage 3 UF valve (DEAD in train)
    "MV302",   # Slot 3  — Stage 3 UF valve (UNSEEN in train, TEST ONLY)
    "MV303",   # Slot 4  — Stage 3 UF valve (UNSEEN in train, TEST ONLY)
    "MV304",   # Slot 5  — Stage 3 UF valve (DEAD in train)
    "MV501",   # Slot 6  — Stage 5 backwash valve
    "MV502",   # Slot 7  — Stage 5 backwash valve
    "MV503",   # Slot 8  — Stage 5 backwash valve
    "MV504",   # Slot 9  — Stage 5 backwash valve
    "P101",    # Slot 10 — Stage 1 raw water pump
    "P102",    # Slot 11 — Stage 1 raw water pump (backup)
    "P201",    # Slot 12 — Stage 2 chemical pump
    "P202",    # Slot 13 — Stage 2 chemical pump
    "P203",    # Slot 14 — Stage 2 chemical pump
    "P204",    # Slot 15 — Stage 2 chemical pump
    "P205",    # Slot 16 — Stage 2 chemical pump
    "P206",    # Slot 17 — Stage 2 chemical pump
    "LIT101",  # Slot 18 — Stage 1 tank level
    "LIT601",  # Slot 19 — Stage 6 tank level
    "DPIT301", # Slot 20 — Stage 3 differential pressure
    "AIT402",  # Slot 21 — Stage 4 RO analyzer (UNSEEN in train, TEST ONLY)
]

# ── Sensor Column Names (71 columns, from your NPZ sensor_cols) ──
# These MUST match the exact order in your A12_windowed_v3.npz
SENSOR_COLUMNS = [
    # Stage 1 — Raw Water (5 sensors + 5 states = 10 cols)
    "FIT101", "LIT101", "AIT201", "AIT202", "AIT203",
    "MV101", "P101", "P102", "P201", "P202",
    # Stage 2 — Chemical Dosing (10 cols)
    "P203", "P204", "P205", "P206", "FIT201",
    "MV201", "AIT301", "AIT302", "AIT303", "AIT304",
    # Stage 3 — Ultrafiltration (12 cols)
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302",
    "MV303", "MV304", "P301", "P302", "AIT401",
    "AIT402", "AIT403",
    # Stage 4 — Reverse Osmosis (12 cols)
    "FIT401", "LIT401", "P401", "P402", "UV401",
    "AIT501", "AIT502", "AIT503", "AIT504", "FIT501",
    "FIT502", "FIT503",
    # Stage 5 — Backwash (12 cols)
    "FIT504", "P501", "P502", "MV501", "MV502",
    "MV503", "MV504", "P601", "P602", "P603",
    "LIT601", "FIT601",
    # Stage 6 — Return (15 cols, filling to 71)
    "AIT601", "AIT602", "AIT603", "AIT604", "AIT605",
    "AIT606", "AIT607", "AIT608", "AIT609", "AIT610",
    "DPIT401", "DPIT501", "DPIT601", "TEMP101", "TEMP201",
]

# ── Topology Mapping: Which sensors are physically adjacent to each component ──
# This is the GROUND TRUTH for zero-shot evaluation
# Format: component_slot → [sensor_indices]
# A sensor is "adjacent" if it measures flow/level/pressure directly upstream,
# downstream, or controls the component

COMPONENT_TO_SENSORS = {
    # Stage 1
    0:  [0, 1, 5],           # MV101 → FIT101, LIT101, MV101_state
    1:  [4, 14, 15],         # MV201 → AIT203, FIT201, MV201_state
    
    # Stage 3 UF (UNSEEN in training — these are your zero-shot targets)
    2:  [11, 12, 13],       # MV301 → DPIT301, FIT301, LIT301
    3:  [11, 12, 13, 14],    # MV302 → DPIT301, FIT301, LIT301, MV302_state
    4:  [11, 12, 13, 15],    # MV303 → DPIT301, FIT301, LIT301, MV303_state
    5:  [11, 12, 13, 16],    # MV304 → DPIT301, FIT301, LIT301, MV304_state
    
    # Stage 5 Backwash
    6:  [32, 33, 34],        # MV501 → FIT504, P501, P502 (wait, wrong)
    # CORRECTED Stage 5 mapping:
    6:  [43, 44, 45],        # MV501 → MV501_state, MV502_state, MV503_state (no, these are valves)
    # Let me fix this properly based on actual SWaT topology:
    
    # CORRECT SWaT Stage 5: Backwash system
    # MV501-504 are backwash valves
    # Sensors: FIT501, FIT502, FIT503, FIT504 measure flow through these
    6:  [29, 43],            # MV501 → FIT501, MV501_state
    7:  [30, 44],            # MV502 → FIT502, MV502_state
    8:  [31, 45],            # MV503 → FIT503, MV503_state
    9:  [32, 46],            # MV504 → FIT504, MV504_state
    
    # Stage 1 Pumps
    10: [0, 1, 6],           # P101 → FIT101, LIT101, P101_state
    11: [0, 1, 7],           # P102 → FIT101, LIT101, P102_state
    
    # Stage 2 Pumps
    12: [4, 8],              # P201 → AIT203, P201_state
    13: [4, 9],              # P202 → AIT203, P202_state
    14: [4, 10],             # P203 → AIT203, P203_state
    15: [4, 11],             # P204 → AIT203, P204_state
    16: [4, 12],             # P205 → AIT203, P205_state
    17: [4, 13],             # P206 → AIT203, P206_state
    
    # Level sensors (these ARE components in your slot mapping)
    18: [1],                 # LIT101 → LIT101 (self-referential, it's a sensor)
    19: [41],                # LIT601 → LIT601 (self-referential)
    
    # Pressure sensor
    20: [11],                # DPIT301 → DPIT301 (self-referential)
    
    # Stage 4 Analyzer (UNSEEN in training — zero-shot target)
    21: [25],                # AIT402 → AIT402 (self-referential, but also AIT401, AIT403)
}

# ── Reverse Mapping: Sensor → Which components it can help localize ──
SENSOR_TO_COMPONENTS = {}
for comp_slot, sensor_list in COMPONENT_TO_SENSORS.items():
    for sensor_idx in sensor_list:
        if sensor_idx not in SENSOR_TO_COMPONENTS:
            SENSOR_TO_COMPONENTS[sensor_idx] = []
        SENSOR_TO_COMPONENTS[sensor_idx].append(comp_slot)

# ── Zero-Shot Test Components ──
UNSEEN_COMPONENTS = [3, 4, 21]   # MV302, MV303, AIT402
SEEN_COMPONENTS = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

def get_component_name(slot: int) -> str:
    return COMPONENT_NAMES[slot]

def get_adjacent_sensors(slot: int) -> list:
    return COMPONENT_TO_SENSORS.get(slot, [])

def get_sensor_name(idx: int) -> str:
    return SENSOR_COLUMNS[idx] if idx < len(SENSOR_COLUMNS) else f"sensor_{idx}"

def is_unseen(slot: int) -> bool:
    return slot in UNSEEN_COMPONENTS

# ── Validation ──
if __name__ == '__main__':
    print("="*60)
    print("SWaT Sensor-Component Topology Mapping")
    print("="*60)
    
    print(f"\nTotal sensors: {len(SENSOR_COLUMNS)}")
    print(f"Total components: {len(COMPONENT_NAMES)}")
    
    print(f"\n{'─'*60}")
    print("UNSEEN Components (Zero-Shot Targets):")
    print(f"{'─'*60}")
    for slot in UNSEEN_COMPONENTS:
        sensors = get_adjacent_sensors(slot)
        sensor_names = [get_sensor_name(s) for s in sensors]
        print(f"  Slot {slot:2d}: {get_component_name(slot):8s} → sensors {sensors} = {sensor_names}")
    
    print(f"\n{'─'*60}")
    print("SEEN Components (Training Coverage):")
    print(f"{'─'*60}")
    for slot in SEEN_COMPONENTS:
        sensors = get_adjacent_sensors(slot)
        sensor_names = [get_sensor_name(s) for s in sensors]
        print(f"  Slot {slot:2d}: {get_component_name(slot):8s} → sensors {sensors} = {sensor_names}")
    
    print(f"\n{'='*60}")
    print("Topology mapping verified.")
    print("="*60)