#!/usr/bin/env python3
"""
TrustGate Enhanced — Modbus TCP Server
Intel Cup ESDC 2025 | IIT Gandhinagar

Simulates a Modbus TCP slave holding registers for an industrial
water treatment / distribution plant. Register map mirrors the
SWaT (Secure Water Treatment) testbed topology with 6 process stages.

Holding Registers (Function Code 0x03 / 0x06 / 0x10):
  0-9    : Stage 1 — Raw Water Intake
  10-19  : Stage 2 — Pre-Treatment
  20-29  : Stage 3 — Ultrafiltration (UF)
  30-39  : Stage 4 — De-Chlorination (RO Feed)
  40-49  : Stage 5 — Reverse Osmosis
  50-59  : Stage 6 — Backwash / Distribution
  60-69  : System-wide status flags
  70-79  : Reserved for attack injection markers

Listens on 0.0.0.0:5020 (non-privileged port to avoid sudo).
"""

import threading
import time
import logging
import struct
import socket
import json
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODBUS_HOST = "0.0.0.0"
MODBUS_PORT = 5020
NUM_REGISTERS = 100
REGISTER_LOG_PATH = "/tmp/modbus_register_state.json"

# Modbus TCP constants
MBAP_HEADER_LEN = 7
FC_READ_HOLDING = 0x03
FC_WRITE_SINGLE = 0x06
FC_WRITE_MULTIPLE = 0x10

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] MODBUS  | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("modbus_server")

# ---------------------------------------------------------------------------
# Shared Register File (thread-safe)
# ---------------------------------------------------------------------------
class RegisterFile:
    """Thread-safe Modbus holding register bank."""

    def __init__(self, size: int = NUM_REGISTERS):
        self._lock = threading.Lock()
        self._regs = [0] * size
        self._write_log = []  # audit trail for the dashboard

    # -- bulk read --------------------------------------------------------
    def read(self, start: int, count: int) -> list:
        with self._lock:
            end = min(start + count, len(self._regs))
            return self._regs[start:end]

    # -- single write -----------------------------------------------------
    def write_single(self, addr: int, value: int, source: str = "local"):
        with self._lock:
            if 0 <= addr < len(self._regs):
                old = self._regs[addr]
                self._regs[addr] = value & 0xFFFF
                if old != value:
                    entry = {
                        "t": time.time(),
                        "addr": addr,
                        "old": old,
                        "new": value & 0xFFFF,
                        "src": source,
                    }
                    self._write_log.append(entry)
                    # keep last 200 entries
                    if len(self._write_log) > 200:
                        self._write_log = self._write_log[-200:]

    # -- bulk write -------------------------------------------------------
    def write_multiple(self, start: int, values: list, source: str = "local"):
        for i, v in enumerate(values):
            self.write_single(start + i, v, source=source)

    # -- snapshot for JSON export -----------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "registers": list(self._regs),
                "last_writes": list(self._write_log[-20:]),
                "timestamp": time.time(),
            }


# Global register file instance
REGS = RegisterFile()

# ---------------------------------------------------------------------------
# Modbus TCP Frame Handling
# ---------------------------------------------------------------------------
def build_exception(tid, uid, fc, exc_code):
    """Build a Modbus exception response."""
    pdu = bytes([fc | 0x80, exc_code])
    length = len(pdu) + 1
    mbap = struct.pack(">HHHB", tid, 0, length, uid)
    return mbap + pdu


def handle_read_holding(tid, uid, pdu_body):
    """FC 0x03 — Read Holding Registers."""
    if len(pdu_body) < 4:
        return build_exception(tid, uid, FC_READ_HOLDING, 0x03)

    start = struct.unpack(">H", pdu_body[0:2])[0]
    count = struct.unpack(">H", pdu_body[2:4])[0]

    if count < 1 or count > 125:
        return build_exception(tid, uid, FC_READ_HOLDING, 0x03)
    if start + count > NUM_REGISTERS:
        return build_exception(tid, uid, FC_READ_HOLDING, 0x02)

    values = REGS.read(start, count)
    byte_count = count * 2
    resp_pdu = bytes([FC_READ_HOLDING, byte_count])
    for v in values:
        resp_pdu += struct.pack(">H", v)

    length = len(resp_pdu) + 1
    mbap = struct.pack(">HHHB", tid, 0, length, uid)
    return mbap + resp_pdu


def handle_write_single(tid, uid, pdu_body, client_addr):
    """FC 0x06 — Write Single Register."""
    if len(pdu_body) < 4:
        return build_exception(tid, uid, FC_WRITE_SINGLE, 0x03)

    addr = struct.unpack(">H", pdu_body[0:2])[0]
    value = struct.unpack(">H", pdu_body[2:4])[0]

    if addr >= NUM_REGISTERS:
        return build_exception(tid, uid, FC_WRITE_SINGLE, 0x02)

    REGS.write_single(addr, value, source=f"tcp:{client_addr[0]}:{client_addr[1]}")
    log.info(f"WRITE reg[{addr}] = {value}  from {client_addr}")

    # Echo back (standard Modbus response)
    resp_pdu = bytes([FC_WRITE_SINGLE]) + pdu_body[:4]
    length = len(resp_pdu) + 1
    mbap = struct.pack(">HHHB", tid, 0, length, uid)
    return mbap + resp_pdu


def handle_write_multiple(tid, uid, pdu_body, client_addr):
    """FC 0x10 — Write Multiple Registers."""
    if len(pdu_body) < 5:
        return build_exception(tid, uid, FC_WRITE_MULTIPLE, 0x03)

    start = struct.unpack(">H", pdu_body[0:2])[0]
    count = struct.unpack(">H", pdu_body[2:4])[0]
    byte_count = pdu_body[4]

    if len(pdu_body) < 5 + byte_count:
        return build_exception(tid, uid, FC_WRITE_MULTIPLE, 0x03)
    if count < 1 or count > 123:
        return build_exception(tid, uid, FC_WRITE_MULTIPLE, 0x03)
    if start + count > NUM_REGISTERS:
        return build_exception(tid, uid, FC_WRITE_MULTIPLE, 0x02)

    values = []
    for i in range(count):
        v = struct.unpack(">H", pdu_body[5 + i * 2 : 7 + i * 2])[0]
        values.append(v)

    REGS.write_multiple(start, values, source=f"tcp:{client_addr[0]}:{client_addr[1]}")
    log.info(f"WRITE_MULTI reg[{start}..{start+count-1}]  from {client_addr}")

    resp_pdu = bytes([FC_WRITE_MULTIPLE]) + pdu_body[:4]
    length = len(resp_pdu) + 1
    mbap = struct.pack(">HHHB", tid, 0, length, uid)
    return mbap + resp_pdu


def process_frame(frame: bytes, client_addr) -> bytes:
    """Parse one Modbus TCP ADU and return the response frame."""
    if len(frame) < MBAP_HEADER_LEN + 1:
        return b""

    tid = struct.unpack(">H", frame[0:2])[0]
    proto = struct.unpack(">H", frame[2:4])[0]
    length = struct.unpack(">H", frame[4:6])[0]
    uid = frame[6]
    fc = frame[7]
    pdu_body = frame[8:]

    if proto != 0:
        return b""

    if fc == FC_READ_HOLDING:
        return handle_read_holding(tid, uid, pdu_body)
    elif fc == FC_WRITE_SINGLE:
        return handle_write_single(tid, uid, pdu_body, client_addr)
    elif fc == FC_WRITE_MULTIPLE:
        return handle_write_multiple(tid, uid, pdu_body, client_addr)
    else:
        return build_exception(tid, uid, fc, 0x01)


# ---------------------------------------------------------------------------
# Client Connection Handler
# ---------------------------------------------------------------------------
def client_handler(conn: socket.socket, addr):
    """Handle a single Modbus TCP client connection."""
    log.info(f"Client connected: {addr}")
    conn.settimeout(30.0)
    buf = b""

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buf += data

            # Process complete frames in buffer
            while len(buf) >= MBAP_HEADER_LEN:
                if len(buf) < 6:
                    break
                expected_len = struct.unpack(">H", buf[4:6])[0] + 6
                if len(buf) < expected_len:
                    break

                frame = buf[:expected_len]
                buf = buf[expected_len:]

                response = process_frame(frame, addr)
                if response:
                    conn.sendall(response)
    except (ConnectionResetError, socket.timeout, BrokenPipeError):
        pass
    finally:
        conn.close()
        log.info(f"Client disconnected: {addr}")


# ---------------------------------------------------------------------------
# Register State Exporter (for dashboard consumption)
# ---------------------------------------------------------------------------
def register_exporter():
    """Periodically dump register state to JSON for the dashboard."""
    while True:
        try:
            snap = REGS.snapshot()
            tmp = REGISTER_LOG_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, REGISTER_LOG_PATH)
        except Exception:
            pass
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Seed Default Register Values
# ---------------------------------------------------------------------------
def seed_registers():
    """Load initial register values representing normal plant startup state."""
    defaults = {
        # Stage 1 — Raw Water Intake
        0: 500,   # LIT101 — Raw water tank level (mm, scaled x10)
        1: 1,     # MV101  — Inlet valve (1=open, 0=closed)
        2: 2,     # P101   — Pump status (2=running, 1=standby, 0=off)
        3: 250,   # FIT101 — Flow rate (L/h, scaled x10)

        # Stage 2 — Pre-Treatment
        10: 700,  # AIT201 — pH (scaled x100, so 700 = pH 7.00)
        11: 300,  # AIT202 — ORP (mV)
        12: 1,    # P201   — Dosing pump
        13: 1,    # P202   — Dosing pump backup

        # Stage 3 — UF
        20: 400,  # DPIT301 — Differential pressure (Pa, scaled x10)
        21: 1,    # MV301  — UF inlet valve
        22: 0,    # MV302  — UF drain valve (normally closed)
        23: 0,    # MV303  — UF backwash valve
        24: 2,    # P301   — UF feed pump
        25: 800,  # FIT301 — UF permeate flow

        # Stage 4 — De-Chlorination / RO Feed
        30: 250,  # LIT401 — RO feed tank level
        31: 1,    # AIT401 — Hardness (safe threshold)
        32: 0,    # AIT402 — Conductivity anomaly flag
        33: 2,    # P401   — RO booster pump
        34: 450,  # FIT401 — RO feed flow

        # Stage 5 — Reverse Osmosis
        40: 150,  # AIT501 — Permeate pH (scaled x100)
        41: 50,   # AIT502 — Permeate ORP
        42: 1,    # AIT503 — Permeate conductivity flag
        43: 300,  # FIT501 — RO permeate flow
        44: 2,    # P501   — High-pressure pump
        45: 0,    # P502   — Bypass valve

        # Stage 6 — Distribution
        50: 800,  # LIT601 — Clean water tank level
        51: 1,    # MV601  — Distribution valve
        52: 2,    # P601   — Distribution pump
        53: 500,  # FIT601 — Distribution flow

        # System flags
        60: 1,    # System operational mode (1=auto, 0=manual)
        61: 0,    # Global alarm status
        62: 0,    # Attack injection active flag (for testing)
    }

    for addr, val in defaults.items():
        REGS.write_single(addr, val, source="init")

    log.info(f"Seeded {len(defaults)} default register values.")


# ---------------------------------------------------------------------------
# Main Server Loop
# ---------------------------------------------------------------------------
def main():
    seed_registers()

    # Start register state exporter thread
    exp_thread = threading.Thread(target=register_exporter, daemon=True)
    exp_thread.start()

    # Bind TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((MODBUS_HOST, MODBUS_PORT))
    server.listen(5)

    log.info(f"Modbus TCP server listening on {MODBUS_HOST}:{MODBUS_PORT}")
    log.info(f"Register state exported to {REGISTER_LOG_PATH}")

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=client_handler, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log.info("Server shutdown requested.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
