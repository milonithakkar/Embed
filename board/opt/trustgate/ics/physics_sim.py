#!/usr/bin/env python3
"""
TrustGate Enhanced — Cyber-Physical Plant Dynamics Simulator
Intel Cup ESDC 2025 | IIT Gandhinagar

Simulates a 6-stage water treatment plant governed by first-principles
physics equations.  Reads actuator commands from the Modbus register file,
evolves the continuous state variables, writes updated sensor readings back,
and exports a unified twin report consumed by the AI inference engine.

Physics models:
  - Tank mass balance:  dV/dt = Q_in - Q_out
  - Pressure drop:      ΔP = f(flow², filter_age)
  - Chemical dosing:    dpH/dt = k·(pH_target - pH) · pump_active
  - Conductivity:       f(TDS, temperature, membrane_state)

Tick rate:  2 Hz  (Δt = 0.5 s simulation time per tick)
"""

import time
import json
import os
import math
import random
import struct
import socket
import logging
import threading

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODBUS_HOST = "127.0.0.1"
MODBUS_PORT = 5020
TWIN_REPORT_PATH = "/tmp/twin_report.json"
TICK_INTERVAL = 0.5        # seconds between physics ticks
NOISE_AMPLITUDE = 0.002    # Gaussian noise scale for sensor realism

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] PHYSICS | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("physics_sim")

# ---------------------------------------------------------------------------
# Modbus TCP Client Helper
# ---------------------------------------------------------------------------
class ModbusClient:
    """Minimal Modbus TCP master for reading/writing holding registers."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock = None
        self._tid = 0

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect((self.host, self.port))

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _next_tid(self):
        self._tid = (self._tid + 1) & 0xFFFF
        return self._tid

    def read_registers(self, start: int, count: int) -> list:
        tid = self._next_tid()
        pdu = struct.pack(">BHH", 0x03, start, count)
        mbap = struct.pack(">HHHB", tid, 0, len(pdu) + 1, 1)
        self._sock.sendall(mbap + pdu)

        resp = self._sock.recv(512)
        if len(resp) < 9:
            return [0] * count

        byte_count = resp[8]
        values = []
        for i in range(byte_count // 2):
            v = struct.unpack(">H", resp[9 + i * 2: 11 + i * 2])[0]
            values.append(v)
        return values

    def write_register(self, addr: int, value: int):
        tid = self._next_tid()
        pdu = struct.pack(">BHH", 0x06, addr, value & 0xFFFF)
        mbap = struct.pack(">HHHB", tid, 0, len(pdu) + 1, 1)
        self._sock.sendall(mbap + pdu)
        self._sock.recv(512)  # consume echo response

    def write_registers(self, start: int, values: list):
        tid = self._next_tid()
        count = len(values)
        byte_count = count * 2
        pdu = struct.pack(">BHHB", 0x10, start, count, byte_count)
        for v in values:
            pdu += struct.pack(">H", v & 0xFFFF)
        mbap = struct.pack(">HHHB", tid, 0, len(pdu) + 1, 1)
        self._sock.sendall(mbap + pdu)
        self._sock.recv(512)


# ---------------------------------------------------------------------------
# Plant State (Continuous Variables)
# ---------------------------------------------------------------------------
# FIND the PlantState.__init__ and REPLACE with this:

class PlantState:
    def __init__(self):
        # Stage 1 — Raw Water Intake
        self.lit101 = 500.0
        self.fit101 = 250.0

        # Stage 2 — Pre-Treatment
        self.ait201 = 7.05
        self.ait202 = 300.0

        # Stage 3 — UF
        self.dpit301      = 40.0
        self.fit301       = 80.0
        self.uf_filter_age = 0.0

        # Stage 4 — RO Feed Tank  (start at mid-range)
        self.lit401      = 400.0
        self.ait402_cond = 0.5
        self.fit401      = 45.0

        # Stage 5 — RO
        self.ait501         = 6.80
        self.ait502         = 50.0
        self.fit501         = 30.0
        self.membrane_hours = 0.0

        # Stage 6 — Distribution  (start at mid-range)
        self.lit601 = 500.0
        self.fit601 = 25.0

        self.last_active_component = "None"
        self.attack_active         = False
        self.tick_count            = 0

    def noise(self, scale=1.0):
        return random.gauss(0, NOISE_AMPLITUDE * scale)

    def noise(self, scale=1.0):
        return random.gauss(0, NOISE_AMPLITUDE * scale)

# ---------------------------------------------------------------------------
# Physics Evolution Engine
# ---------------------------------------------------------------------------

def evolve(state: PlantState, regs: list, dt: float) -> PlantState:
    """
    PATCHED v2 — correct flow balance throughout all stages.
    Water conservation: inflow stage N = outflow stage N-1
    """
    state.tick_count += 1

    # ── Stage 1: Raw Water Intake ─────────────────────────────────
    mv101 = regs[1]
    p101  = regs[2]

    # External source inflow — constant supply pressure
    inflow_1  = 250.0 + state.noise(3) if mv101 == 1 else 0.0

    # Outflow to Stage 3 via Stage 2 — pump must run, tank must have water
    outflow_1 = 0.0
    if p101 == 2 and state.lit101 > 50:
        outflow_1 = 248.0 + state.noise(3)   # slightly less than inflow

    # dLevel_mm = (Q_in - Q_out)[L/h] / 3600 * dt[s] / area[m²] * 1e-3[m³/L] * 1e3[mm/m]
    # Simplifies to: dh = (Q_in - Q_out) * dt / (area * 3600)
    # With area=0.5m²: dh_mm = delta_Q * dt / 1800
    state.lit101 = max(0.0, min(1000.0,
        state.lit101 + (inflow_1 - outflow_1) * dt / 1800.0))
    state.fit101 = outflow_1

    # ── Stage 2: Pre-Treatment ────────────────────────────────────
    p201 = regs[12]
    p202 = regs[13]

    ph_target    = 7.05
    state.ait201 += 0.3 * (ph_target - state.ait201) * dt + state.noise(0.003)
    state.ait201  = max(4.0, min(10.0, state.ait201))
    state.ait202  = 300.0 - 30.0 * abs(state.ait201 - 7.0) + state.noise(0.5)

    # ── Stage 3: Ultrafiltration ──────────────────────────────────
    mv301 = regs[21]
    mv302 = regs[22]
    mv303 = regs[23]
    p301  = regs[24]

    # UF output = Stage 1 output × 0.95 (5% loss to backwash/drain)
    if mv301 == 1 and p301 == 2 and state.lit101 > 50:
        state.fit301       = outflow_1 * 0.95 + state.noise(1)
        state.uf_filter_age += dt / 3600.0
        fouling            = 1.0 + 0.03 * state.uf_filter_age
        state.dpit301      = 40.0 * fouling + state.noise(0.3)
    else:
        state.fit301  = max(0.0, state.fit301  - 10.0 * dt)
        state.dpit301 = max(5.0, state.dpit301 -  1.0 * dt)

    if mv303 == 1:
        state.uf_filter_age         = max(0.0, state.uf_filter_age - 0.5 * dt)
        state.last_active_component = "MV303"
    if mv302 == 1:
        state.last_active_component = "MV302"

    # ── Stage 4: RO Feed Tank ─────────────────────────────────────
    p401 = regs[33]

    inflow_4  = state.fit301    # UF permeate → RO feed tank
    outflow_4 = 0.0
    if p401 == 2 and state.lit401 > 30:
        # Match outflow to inflow — no accumulation
        outflow_4 = inflow_4 * 0.98 + state.noise(0.5)

    # Tank area = 0.3 m² → dh = delta_Q * dt / (0.3 * 3600) = delta_Q * dt / 1080
    state.lit401 = max(0.0, min(800.0,
        state.lit401 + (inflow_4 - outflow_4) * dt / 1080.0))
    state.fit401     = outflow_4
    state.ait402_cond = 0.5 + 0.001 * state.membrane_hours + state.noise(0.005)

    # ── Stage 5: Reverse Osmosis ──────────────────────────────────
    p501 = regs[44]

    if p501 == 2 and state.lit401 > 30:
        recovery      = max(0.5, 0.75 - 0.0005 * state.membrane_hours)
        state.fit501  = state.fit401 * recovery + state.noise(0.3)
        state.membrane_hours += dt / 3600.0
    else:
        state.fit501 = max(0.0, state.fit501 - 5.0 * dt)

    state.ait501 = 6.8  + 0.002 * state.membrane_hours + state.noise(0.003)
    state.ait502 = 50.0 - 0.2  * state.membrane_hours + state.noise(0.3)

    # ── Stage 6: Distribution ─────────────────────────────────────
    mv601 = regs[51]
    p601  = regs[52]

    inflow_6  = state.fit501    # RO permeate → clean water tank
    outflow_6 = 0.0
    if mv601 == 1 and p601 == 2 and state.lit601 > 20:
        # CRITICAL: outflow strictly limited to what RO produces
        # This prevents LIT601 from draining when RO output is low
        outflow_6 = min(inflow_6 * 0.90, 50.0) + state.noise(0.5)
        outflow_6 = max(0.0, outflow_6)

    # Tank area = 0.8 m² → dh = delta_Q * dt / (0.8 * 3600) = delta_Q * dt / 2880
    state.lit601 = max(0.0, min(1200.0,
        state.lit601 + (inflow_6 - outflow_6) * dt / 2880.0))
    state.fit601 = outflow_6

    # ── System flags ──────────────────────────────────────────────
    attack_flag         = regs[62] if len(regs) > 62 else 0
    state.attack_active = (attack_flag != 0)

    return state

def writeback(client: ModbusClient, state: PlantState):
    """Write evolved continuous sensor values back as discretized registers."""
    try:
        writes = {
            0:  int(state.lit101),
            3:  int(state.fit101),
            10: int(state.ait201 * 100),
            11: int(state.ait202),
            20: int(state.dpit301 * 10),
            25: int(state.fit301 * 10),
            30: int(state.lit401),
            32: 1 if state.ait402_cond > 1.0 else 0,
            34: int(state.fit401 * 10),
            40: int(state.ait501 * 100),
            41: int(state.ait502),
            43: int(state.fit501 * 10),
            50: int(state.lit601),
            53: int(state.fit601 * 10),
        }
        for addr, val in writes.items():
            val = max(0, min(65535, val))
            client.write_register(addr, val)
    except Exception as e:
        log.warning(f"Writeback failed: {e}")


# ---------------------------------------------------------------------------
# Twin Report Generator
# ---------------------------------------------------------------------------
def build_sensor_vector(state: PlantState) -> list:
    """
    Build the 71-dimensional feature vector matching the AI model's
    expected sensor input configuration.

    Layout (indices 0-70):
      0-9:   Stage 1 sensors + derivatives
      10-19: Stage 2 sensors + derivatives
      20-29: Stage 3 sensors + derivatives
      30-39: Stage 4 sensors + derivatives
      40-49: Stage 5 sensors + derivatives
      50-59: Stage 6 sensors + derivatives
      60-70: Cross-stage correlations + system-level features
    """
    vec = [0.0] * 71

    # Stage 1
    vec[0] = state.lit101
    vec[1] = state.fit101
    vec[2] = 1.0 if state.fit101 > 100 else 0.0  # pump active indicator
    vec[3] = state.lit101 / 1000.0                 # normalized level

    # Stage 2
    vec[10] = state.ait201
    vec[11] = state.ait202
    vec[12] = abs(state.ait201 - 7.0)             # pH deviation
    vec[13] = 1.0 if abs(state.ait201 - 7.0) > 0.5 else 0.0

    # Stage 3
    vec[20] = state.dpit301
    vec[21] = state.fit301
    vec[22] = state.uf_filter_age
    vec[23] = state.dpit301 / max(state.fit301, 1.0)  # resistance ratio
    vec[24] = 1.0 if state.dpit301 > 80 else 0.0      # fouling alert

    # Stage 4
    vec[30] = state.lit401
    vec[31] = state.ait402_cond
    vec[32] = state.fit401
    vec[33] = state.lit401 / 800.0                     # normalized level
    vec[34] = 1.0 if state.ait402_cond > 1.0 else 0.0 # conductivity alert

    # Stage 5
    vec[40] = state.ait501
    vec[41] = state.ait502
    vec[42] = state.fit501
    vec[43] = state.membrane_hours
    vec[44] = state.fit501 / max(state.fit401, 1.0)    # recovery ratio

    # Stage 6
    vec[50] = state.lit601
    vec[51] = state.fit601
    vec[52] = state.lit601 / 1200.0                    # normalized level
    vec[53] = 1.0 if state.lit601 < 200 else 0.0       # low-level alert

    # Cross-stage features
    vec[60] = state.fit101 - state.fit601              # system mass imbalance
    vec[61] = state.lit101 + state.lit401 + state.lit601  # total water inventory
    vec[62] = state.ait201 - state.ait501              # pH differential across plant
    vec[63] = 1.0 if state.attack_active else 0.0      # attack flag from register
    vec[64] = float(state.tick_count)                   # simulation time proxy
    vec[65] = state.dpit301 * state.fit301 / 10000.0   # UF power proxy
    vec[66] = state.ait202 * state.ait201 / 1000.0     # ORP-pH interaction
    vec[67] = max(0, state.lit101 - state.lit601)      # head differential
    vec[68] = state.fit301 - state.fit501              # UF-RO flow gap
    vec[69] = state.ait402_cond * state.membrane_hours  # degradation product
    vec[70] = state.uf_filter_age * state.membrane_hours  # dual aging feature

    return vec


def export_twin_report(state: PlantState):
    """Write the unified digital twin report to JSON."""
    report = {
        "timestamp": time.time(),
        "readable_time": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
        "tick": state.tick_count,

        # Full 71-dim sensor vector for the AI engine
        "sensors": build_sensor_vector(state),

        # Human-readable state breakdown for the dashboard
        "stages": {
            "stage_1": {
                "LIT101_level_mm": round(state.lit101, 2),
                "FIT101_flow_Lh": round(state.fit101, 2),
            },
            "stage_2": {
                "AIT201_pH": round(state.ait201, 3),
                "AIT202_ORP_mV": round(state.ait202, 1),
            },
            "stage_3": {
                "DPIT301_kPa": round(state.dpit301, 2),
                "FIT301_flow_Lh": round(state.fit301, 2),
                "filter_age_hrs": round(state.uf_filter_age, 3),
            },
            "stage_4": {
                "LIT401_level_mm": round(state.lit401, 2),
                "AIT402_conductivity_mScm": round(state.ait402_cond, 3),
                "FIT401_flow_Lh": round(state.fit401, 2),
            },
            "stage_5": {
                "AIT501_pH": round(state.ait501, 3),
                "AIT502_ORP_mV": round(state.ait502, 1),
                "FIT501_flow_Lh": round(state.fit501, 2),
                "membrane_hours": round(state.membrane_hours, 3),
            },
            "stage_6": {
                "LIT601_level_mm": round(state.lit601, 2),
                "FIT601_flow_Lh": round(state.fit601, 2),
            },
        },

        # Component tracking for root-cause localization
        "last_active_component": state.last_active_component,
        "attack_active": state.attack_active,

        # Physical asset inventory for dashboard
        "physical_assets": {
            "tanks": ["T101", "T301", "T401", "T601"],
            "pumps": ["P101", "P201", "P202", "P301", "P401", "P501", "P601"],
            "valves": ["MV101", "MV301", "MV302", "MV303", "MV601"],
            "analyzers": ["AIT201", "AIT202", "AIT401", "AIT402",
                          "AIT501", "AIT502", "AIT503"],
            "flow_meters": ["FIT101", "FIT301", "FIT401", "FIT501", "FIT601"],
            "level_sensors": ["LIT101", "LIT401", "LIT601"],
            "pressure_sensors": ["DPIT301"],
        },
    }

    try:
        tmp = TWIN_REPORT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, TWIN_REPORT_PATH)
    except Exception as e:
        log.warning(f"Twin report export failed: {e}")


# ---------------------------------------------------------------------------
# Main Simulation Loop
# ---------------------------------------------------------------------------
def main():
    state = PlantState()
    client = ModbusClient(MODBUS_HOST, MODBUS_PORT)

    # Wait for Modbus server to come online
    log.info("Waiting for Modbus server...")
    while True:
        try:
            client.connect()
            log.info("Connected to Modbus server.")
            break
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(1.0)

    log.info(f"Physics simulation running at {1.0/TICK_INTERVAL:.1f} Hz")
    log.info(f"Twin report → {TWIN_REPORT_PATH}")

    reconnect_delay = 1.0

    while True:
        try:
            # Read all 100 registers
            regs = client.read_registers(0, 70)

            # Evolve physics
            state = evolve(state, regs, TICK_INTERVAL)

            # Write sensor readings back to Modbus
            writeback(client, state)

            # Export twin report for AI engine
            export_twin_report(state)

            reconnect_delay = 1.0  # reset on success

            if state.tick_count % 20 == 0:
                log.info(
                    f"Tick {state.tick_count:>6d} | "
                    f"L1={state.lit101:6.1f} pH={state.ait201:.2f} "
                    f"ΔP={state.dpit301:5.1f} L4={state.lit401:6.1f} "
                    f"RO={state.fit501:5.1f} L6={state.lit601:6.1f}"
                )

        except (ConnectionResetError, BrokenPipeError, socket.timeout, OSError) as e:
            log.warning(f"Modbus connection lost: {e}. Reconnecting in {reconnect_delay:.0f}s...")
            client.close()
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)
            try:
                client.connect()
                log.info("Reconnected to Modbus server.")
            except Exception:
                pass
            continue

        time.sleep(TICK_INTERVAL)


if __name__ == "__main__":
    main()
