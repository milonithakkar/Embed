
#!/usr/bin/env python3
"""
TrustGate Enhanced — Plan B Real-Time Network Feature Extractor
Intel Cup ESDC 2025 | IIT Gandhinagar

Board path : /opt/trustgate/ai_engine/network_extractor.py
Requires   : sudo python3 network_extractor.py
Writes     : /tmp/network_features.json  (consumed by inference.py)

Architecture:
  - Dual-source packet capture: physical eth0 + loopback lo
  - Raw MBAP binary frame parsing (not JSON log reading)
  - True wire-level feature extraction matching training data

Modbus TCP MBAP Header (7 bytes):
  [0-1] Transaction ID   (uint16 big-endian)
  [2-3] Protocol ID      (uint16 = 0x0000 for Modbus)
  [4-5] Length           (uint16 — number of following bytes)
  [6]   Unit ID          (uint8)
  [7]   Function Code    (uint8)
  [8+]  Data

Function Codes:
  0x01  Read Coils              → read
  0x02  Read Discrete Inputs    → read
  0x03  Read Holding Registers  → read  ← physics_sim uses this
  0x04  Read Input Registers    → read
  0x05  Write Single Coil       → write
  0x06  Write Single Register   → write ← physics_sim uses this
  0x0F  Write Multiple Coils    → write
  0x10  Write Multiple Regs     → write ← emergency shutdown uses this

19 Features (exact network_cols order from A12_windowed_v3_fixed.npz):
  [0]  packet_count
  [1]  total_bytes
  [2]  valve_write_count
  [3]  pump_write_count
  [4]  valve_read_count
  [5]  pump_read_count
  [6]  level_access_count
  [7]  dosing_access_count
  [8]  state_poll_count
  [9]  valve_write_deviation
  [10] pump_write_deviation
  [11] novel_tag_count
  [12] tag_entropy
  [13] burst_component_max
  [14] rare_access_score
  [15] unique_tags_this_sec
  [16] unique_src_ips
  [17] reset_count
  [18] permissive_count
"""

import os
import sys
import json
import time
import math
import struct
import socket
import logging
import threading
import pickle
from collections import defaultdict, deque, Counter

# ---------------------------------------------------------------------------
# Require root — raw socket capture needs it
# ---------------------------------------------------------------------------
if os.geteuid() != 0:
    print("ERROR: network_extractor.py must run as root.")
    print("       sudo python3 /opt/trustgate/ai_engine/network_extractor.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NETWORK_SCALER_PATH = "/opt/trustgate/model/network_scaler.pkl"
OUTPUT_PATH      = "/tmp/network_features.json"
MODBUS_PORT      = 5020
WINDOW_SECONDS   = 1.0     # rolling feature window (matches training)
TICK_INTERVAL    = 0.5     # output update frequency
BASELINE_WINDOW  = 300     # seconds to build baseline (5 minutes)

# Physical interface — auto-detected, overridable
# Will try eth0, eth1, enp*, ens* in order
# Replace the candidates list with the exact interface name
PHYSICAL_IFACE_CANDIDATES = [
    "enp3s0",   # confirmed UP with 169.254.195.2
    "enp1s0",
    "enp2s0",
    "enp4s0",
]
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] NETCAP  | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("network_extractor")

# ---------------------------------------------------------------------------
# Register Address → Component Tag Mapping
# Mirrors modbus_server.py register layout exactly
# ---------------------------------------------------------------------------
REGISTER_TO_TAG = {
    # Stage 1
    0:  "LIT101",  1:  "MV101",  2:  "P101",   3:  "FIT101",
    # Stage 2
    10: "AIT201",  11: "AIT202", 12: "P201",   13: "P202",
    # Stage 3
    20: "DPIT301", 21: "MV301",  22: "MV302",  23: "MV303",
    24: "P301",    25: "FIT301",
    # Stage 4
    30: "LIT401",  31: "AIT401", 32: "AIT402", 33: "P401",
    34: "FIT401",
    # Stage 5
    40: "AIT501",  41: "AIT502", 42: "AIT503", 43: "FIT501",
    44: "P501",    45: "P502",
    # Stage 6
    50: "LIT601",  51: "MV601",  52: "P601",   53: "FIT601",
    # System
    60: "SYS_MODE", 61: "ALARM", 62: "ATK_FLAG",
}

# Component classification sets
VALVES = {
    'MV101','MV201','MV301','MV302','MV303','MV304',
    'MV501','MV502','MV503','MV504','MV601','P502',
}
PUMPS = {
    'P101','P102','P201','P202','P203','P204','P205',
    'P206','P207','P208','P301','P302','P401','P402',
    'P403','P404','P501','P601','P602','P603',
}
LEVEL_SENSORS   = {'LIT101','LIT301','LIT401','LIT601','LIT602'}
DOSING_SENSORS  = {
    'AIT201','AIT202','AIT203','AIT301','AIT302','AIT303',
    'AIT401','AIT402','AIT501','AIT502','AIT503','AIT504',
}
FLOW_SENSORS    = {'FIT101','FIT201','FIT301','FIT401','FIT501','FIT601'}
PRESSURE_SENSORS= {'DPIT301','PIT501'}
SYSTEM_REGS     = {'SYS_MODE','ALARM','ATK_FLAG'}

# Read function codes
FC_READ  = {0x01, 0x02, 0x03, 0x04}
# Write function codes
FC_WRITE = {0x05, 0x06, 0x0F, 0x10}


def classify_access(tag: str, is_write: bool) -> str:
    """
    Map (tag, is_write) → behavioral category.
    Mirrors categorize_tag() from extract_smart_features.py.
    """
    if tag in VALVES:
        return 'valve_write' if is_write else 'valve_read'
    if tag in PUMPS:
        return 'pump_write'  if is_write else 'pump_read'
    if tag in LEVEL_SENSORS:
        return 'level_access'
    if tag in DOSING_SENSORS:
        return 'dosing_access'
    if tag in FLOW_SENSORS or tag in PRESSURE_SENSORS:
        return 'state_poll'
    if tag in ('ALARM', 'ATK_FLAG'):
        return 'reset_signal'
    if tag == 'SYS_MODE':
        return 'permissive_signal'
    return 'other'


# ---------------------------------------------------------------------------
# Parsed Modbus Event
# ---------------------------------------------------------------------------
class ModbusEvent:
    __slots__ = ('t', 'src_ip', 'dst_ip', 'fc',
                 'start_reg', 'count', 'is_write',
                 'payload_len', 'tags')

    def __init__(self, t, src_ip, dst_ip, fc,
                 start_reg, count, payload_len):
        self.t           = t
        self.src_ip      = src_ip
        self.dst_ip      = dst_ip
        self.fc          = fc
        self.start_reg   = start_reg
        self.count       = count
        self.is_write    = fc in FC_WRITE
        self.payload_len = payload_len

        # Resolve register range → tag names
        self.tags = []
        for addr in range(start_reg, start_reg + max(count, 1)):
            tag = REGISTER_TO_TAG.get(addr)
            if tag:
                self.tags.append(tag)
        if not self.tags:
            self.tags = [f"REG{start_reg}"]


# ---------------------------------------------------------------------------
# MBAP Frame Parser
# ---------------------------------------------------------------------------
def parse_mbap(payload: bytes, src_ip: str,
               dst_ip: str) -> ModbusEvent | None:
    """
    Parse a raw TCP payload as a Modbus TCP MBAP frame.
    Returns a ModbusEvent or None if not valid Modbus.

    MBAP structure:
      Bytes 0-1: Transaction ID
      Bytes 2-3: Protocol ID (must be 0x0000)
      Bytes 4-5: Length
      Byte  6:   Unit ID
      Byte  7:   Function Code
      Bytes 8+:  PDU data
    """
    if len(payload) < 8:
        return None

    try:
        trans_id  = struct.unpack('>H', payload[0:2])[0]
        proto_id  = struct.unpack('>H', payload[2:4])[0]
        length    = struct.unpack('>H', payload[4:6])[0]
        unit_id   = payload[6]
        fc        = payload[7]
    except struct.error:
        return None

    # Must be Modbus TCP
    if proto_id != 0x0000:
        return None

    # Must be a known function code
    if fc not in (FC_READ | FC_WRITE):
        return None

    # Parse register address from PDU
    start_reg = 0
    count     = 1

    pdu = payload[8:]   # everything after Unit ID + FC

    if fc in FC_READ and len(pdu) >= 4:
        # FC 0x01/0x02/0x03/0x04 request:
        # [0-1] Starting Address
        # [2-3] Quantity
        start_reg = struct.unpack('>H', pdu[0:2])[0]
        count     = struct.unpack('>H', pdu[2:4])[0]

    elif fc in (0x05, 0x06) and len(pdu) >= 4:
        # FC 0x05/0x06 single write:
        # [0-1] Register Address
        # [2-3] Value
        start_reg = struct.unpack('>H', pdu[0:2])[0]
        count     = 1

    elif fc in (0x0F, 0x10) and len(pdu) >= 4:
        # FC 0x0F/0x10 multiple write:
        # [0-1] Starting Address
        # [2-3] Quantity
        start_reg = struct.unpack('>H', pdu[0:2])[0]
        count     = struct.unpack('>H', pdu[2:4])[0]

    return ModbusEvent(
        t           = time.time(),
        src_ip      = src_ip,
        dst_ip      = dst_ip,
        fc          = fc,
        start_reg   = start_reg,
        count       = count,
        payload_len = len(payload),
    )


# ---------------------------------------------------------------------------
# Raw Socket Packet Sniffer
# One thread per interface
# ---------------------------------------------------------------------------
class RawSocketSniffer(threading.Thread):
    """
    Captures TCP packets on a given interface using a raw socket.
    Filters for port 5020 (Modbus TCP) in software.
    No scapy dependency — uses AF_PACKET raw sockets.
    Requires root.
    """

    ETH_P_IP = 0x0800
    ETH_HDR  = 14       # standard Ethernet II header bytes

    def __init__(self, iface: str, event_queue: deque,
                 queue_lock: threading.Lock):
        super().__init__(daemon=True)
        self.iface       = iface
        self.queue       = event_queue
        self.lock        = queue_lock
        self._running    = False
        self._sock       = None
        self.packets_captured = 0
        self.name        = f"sniffer-{iface}"

    def _open_socket(self) -> bool:
        try:
            # AF_PACKET = raw Ethernet frames
            # SOCK_RAW  = full packet including headers
            # ETH_P_IP  = only IP packets
            self._sock = socket.socket(
                socket.AF_PACKET,
                socket.SOCK_RAW,
                socket.htons(0x0003),
            )
            self._sock.bind((self.iface, 0))
            self._sock.settimeout(1.0)

            # Put interface in promiscuous mode
            import fcntl
            SIOCGIFFLAGS = 0x8913
            SIOCSIFFLAGS = 0x8914
            IFF_PROMISC  = 0x100

            ifreq = struct.pack('16sh', self.iface.encode(), 0)
            flags = struct.unpack('16sh',
                fcntl.ioctl(self._sock, SIOCGIFFLAGS, ifreq)
            )[1]
            ifreq = struct.pack('16sh',
                self.iface.encode(), flags | IFF_PROMISC
            )
            fcntl.ioctl(self._sock, SIOCSIFFLAGS, ifreq)

            log.info(f"  Sniffer bound to {self.iface} "
                     f"(promiscuous mode)")
            return True

        except OSError as e:
            log.warning(f"  Cannot bind to {self.iface}: {e}")
            return False

    def run(self):
        if not self._open_socket():
            return

        self._running = True
        buf = b""

        while self._running:
            try:
                raw, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            # Parse Ethernet header
            if len(raw) < self.ETH_HDR + 20:
                continue

            eth_proto = struct.unpack('>H', raw[12:14])[0]
            if eth_proto != self.ETH_P_IP:
                continue

            # Parse IP header
            ip_hdr    = raw[self.ETH_HDR:]
            ip_ihl    = (ip_hdr[0] & 0x0F) * 4
            ip_proto  = ip_hdr[9]

            if ip_proto != 6:   # TCP only
                continue

            src_ip = socket.inet_ntoa(ip_hdr[12:16])
            dst_ip = socket.inet_ntoa(ip_hdr[16:20])

            # Parse TCP header
            tcp_hdr     = ip_hdr[ip_ihl:]
            if len(tcp_hdr) < 20:
                continue

            src_port  = struct.unpack('>H', tcp_hdr[0:2])[0]
            dst_port  = struct.unpack('>H', tcp_hdr[2:4])[0]

            # Filter for Modbus port
            if MODBUS_PORT not in (src_port, dst_port):
                continue

            tcp_offset = ((tcp_hdr[12] >> 4) & 0xF) * 4
            tcp_payload = tcp_hdr[tcp_offset:]

            if not tcp_payload:
                continue

            # Parse MBAP
            event = parse_mbap(tcp_payload, src_ip, dst_ip)
            if event is None:
                continue

            self.packets_captured += 1
            with self.lock:
                self.queue.append(event)

        if self._sock:
            self._sock.close()

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Loopback Sniffer (lo interface special handling)
# Loopback has a 4-byte family header instead of 14-byte Ethernet
# ---------------------------------------------------------------------------
class LoopbackSniffer(threading.Thread):
    """
    Raw socket sniffer for the loopback (lo) interface.

    On this board (Ubuntu 22.04, kernel 6.8):
      tcpdump reports lo link-type EN10MB (Ethernet)
    This means loopback uses standard 14-byte Ethernet headers,
    same as physical interfaces — NOT the 4-byte null header
    that older kernels use.
    """

    ETH_P_IP = 0x0800

    def __init__(self, event_queue: deque,
                 queue_lock: threading.Lock):
        super().__init__(daemon=True)
        self.queue    = event_queue
        self.lock     = queue_lock
        self._running = False
        self._sock    = None
        self.packets_captured = 0
        self.name     = "sniffer-lo"

    def _open_socket(self) -> bool:
        try:
            self._sock = socket.socket(
                socket.AF_PACKET,
                socket.SOCK_RAW,
                socket.htons(0x0003),
            )
            self._sock.bind(("lo", 0))
            self._sock.settimeout(1.0)
            log.info("  Loopback sniffer bound to lo (EN10MB mode)")
            return True
        except OSError as e:
            log.warning(f"  Cannot bind to lo: {e}")
            return False

    def run(self):
        if not self._open_socket():
            return

        self._running = True
        ETH_HDR = 14   # standard Ethernet II header

        while self._running:
            try:
                raw, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(raw) < ETH_HDR + 20:
                continue

            # Same parsing as RawSocketSniffer
            eth_proto = struct.unpack('>H', raw[12:14])[0]
            if eth_proto != self.ETH_P_IP:
                continue

            ip_hdr   = raw[ETH_HDR:]
            ip_ihl   = (ip_hdr[0] & 0x0F) * 4
            ip_proto = ip_hdr[9]

            if ip_proto != 6:
                continue

            src_ip = socket.inet_ntoa(ip_hdr[12:16])
            dst_ip = socket.inet_ntoa(ip_hdr[16:20])

            tcp_hdr  = ip_hdr[ip_ihl:]
            if len(tcp_hdr) < 20:
                continue

            src_port = struct.unpack('>H', tcp_hdr[0:2])[0]
            dst_port = struct.unpack('>H', tcp_hdr[2:4])[0]

            if MODBUS_PORT not in (src_port, dst_port):
                continue

            tcp_offset  = ((tcp_hdr[12] >> 4) & 0xF) * 4
            tcp_payload = tcp_hdr[tcp_offset:]

            if not tcp_payload:
                continue

            event = parse_mbap(tcp_payload, src_ip, dst_ip)
            if event is None:
                continue

            self.packets_captured += 1
            with self.lock:
                self.queue.append(event)

        if self._sock:
            self._sock.close()

    def stop(self):
        self._running = False

# ---------------------------------------------------------------------------
# Auto-detect Physical Interface
# ---------------------------------------------------------------------------
def detect_physical_interface() -> str | None:
    """Find the first available physical network interface."""
    try:
        with open('/proc/net/if_inet6') as f:
            pass
    except FileNotFoundError:
        pass

    # Read from /proc/net/dev
    try:
        with open('/proc/net/dev') as f:
            lines = f.readlines()[2:]   # skip header rows
        interfaces = [l.split(':')[0].strip() for l in lines]
        interfaces = [i for i in interfaces if i and i != 'lo']

        # Prefer candidates in order
        for cand in PHYSICAL_IFACE_CANDIDATES:
            if cand in interfaces:
                return cand

        # Return first non-loopback interface found
        return interfaces[0] if interfaces else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Baseline Tracker
# ---------------------------------------------------------------------------
class BaselineTracker:
    """
    Builds a rolling baseline of normal Modbus access rates
    from the first BASELINE_WINDOW seconds of observation.
    Falls back to SWaT-derived priors immediately.
    """

    # Priors from A12 normal period (4-hour baseline analysis)
    PRIORS = {
        'valve_write':  0.30,
        'pump_write':   0.50,
        'valve_read':   2.00,
        'pump_read':    2.00,
        'level_access': 1.50,
        'dosing_access':1.20,
        'state_poll':   3.00,
    }

    TAG_PRIORS = {
        'LIT101': 2.0, 'LIT401': 2.0, 'LIT601': 2.0,
        'MV101':  0.1, 'MV301':  0.1, 'MV601':  0.1,
        'P101':   0.5, 'P301':   0.5, 'P401':   0.5,
        'P501':   0.5, 'P601':   0.5,
        'FIT101': 1.0, 'FIT301': 1.0, 'FIT401': 1.0,
        'FIT501': 1.0, 'FIT601': 1.0,
        'AIT201': 1.0, 'AIT202': 1.0, 'AIT402': 1.0,
        'AIT501': 1.0, 'AIT502': 1.0,
        'DPIT301':1.0,
    }

    def __init__(self):
        self._cat_acc  = defaultdict(float)
        self._tag_acc  = defaultdict(float)
        self._secs     = 0
        self._done     = False
        self._rates    = dict(self.PRIORS)
        self._tag_rates= dict(self.TAG_PRIORS)
        self._known    = set(REGISTER_TO_TAG.values())

    def update(self, cat_counts: dict, tag_counts: dict):
        if self._done:
            return
        self._secs += 1
        for c, n in cat_counts.items():
            self._cat_acc[c] += n
        for t, n in tag_counts.items():
            self._tag_acc[t] += n
        if self._secs >= BASELINE_WINDOW:
            self._done = True
            for c, total in self._cat_acc.items():
                self._rates[c] = total / self._secs
            for t, total in self._tag_acc.items():
                self._tag_rates[t] = total / self._secs
            log.info(
                f"Baseline locked after {self._secs}s | "
                f"rates={dict(list(self._rates.items())[:4])}"
            )

    @property
    def rates(self):     return self._rates
    @property
    def tag_rates(self): return self._tag_rates
    @property
    def known(self):     return self._known

class NetworkScaler:
    """
    Loads and applies the RobustScaler fitted on A12 raw network
    features. Ensures the 19-dim vector fed to inference.py is in
    the exact value range the model was trained on.
    """

    def __init__(self, scaler_path: str):
        self._scaler        = None
        self._feature_names = []
        self._load(scaler_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            log.warning(
                f"Network scaler not found at {path} — "
                f"features will be fed UNSCALED (model may output garbage)"
            )
            return

        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)

            # Handle both dict wrapper and raw scaler
            if isinstance(obj, dict):
                self._scaler        = obj['scaler']
                self._feature_names = obj.get('feature_names', [])
            else:
                self._scaler        = obj
                self._feature_names = []

            log.info(
                f"Network scaler loaded — "
                f"expects {self._scaler.n_features_in_} features"
            )
        except Exception as e:
            log.error(f"Failed to load network scaler: {e}")

    def transform(self, features: list) -> list:
        """Scale a raw 19-dim feature list."""
        if self._scaler is None:
            return features   # passthrough if scaler not loaded

        try:
            import numpy as np
            arr    = np.array(features, dtype=np.float32).reshape(1, -1)
            scaled = self._scaler.transform(arr)
            return scaled[0].tolist()
        except Exception as e:
            log.warning(f"Scaling failed: {e} — using raw features")
            return features

    @property
    def loaded(self) -> bool:
        return self._scaler is not None
# ---------------------------------------------------------------------------
# Rolling Event Window
# ---------------------------------------------------------------------------
class EventWindow:
    """Maintains events within the last WINDOW_SECONDS."""

    def __init__(self, window: float = WINDOW_SECONDS):
        self._window = window
        self._events = deque()

    def add(self, event: ModbusEvent):
        self._events.append(event)

    def prune(self):
        cutoff = time.time() - self._window
        while self._events and self._events[0].t < cutoff:
            self._events.popleft()

    def get(self) -> list:
        self.prune()
        return list(self._events)


# ---------------------------------------------------------------------------
# Feature Computer
# ---------------------------------------------------------------------------
class FeatureComputer:

    def __init__(self):
        self._baseline      = BaselineTracker()
        self._window        = EventWindow()
        self._cumul_packets = 0
        self._cumul_bytes   = 0
        self._scaler        = NetworkScaler(NETWORK_SCALER_PATH)  # ADD THIS
    def feed(self, events: list):
        """Ingest a batch of new events."""
        for ev in events:
            self._window.add(ev)
            self._cumul_packets += 1
            self._cumul_bytes   += ev.payload_len

    def compute(self) -> list:
        """Compute the 19-dim feature vector."""
        window_events = self._window.get()

        # ── Aggregate window statistics ───────────────────────────
        cat_counts       = Counter()
        tag_counts       = Counter()
        src_ips          = set()
        component_writes = Counter()

        for ev in window_events:
            for tag in ev.tags:
                cat = classify_access(tag, ev.is_write)
                cat_counts[cat]   += 1
                tag_counts[tag]   += 1
                if ev.is_write:
                    component_writes[tag] += 1
            src_ips.add(ev.src_ip)

        # ── Update baseline ───────────────────────────────────────
        self._baseline.update(dict(cat_counts), dict(tag_counts))
        br  = self._baseline.rates
        btr = self._baseline.tag_rates
        kt  = self._baseline.known

        # ── Feature 9-10: Category deviations ────────────────────
        valve_w = float(cat_counts.get('valve_write', 0))
        pump_w  = float(cat_counts.get('pump_write',  0))
        vb      = br.get('valve_write', 0.1)
        pb      = br.get('pump_write',  0.1)
        valve_dev = (valve_w - vb) / max(vb, 0.1)
        pump_dev  = (pump_w  - pb) / max(pb,  0.1)

        # ── Feature 11: Novel tag count ───────────────────────────
        unique_tags = set(tag_counts.keys())
        novel_count = float(len(unique_tags - kt))

        # ── Feature 12: Tag entropy ───────────────────────────────
        n_total = sum(tag_counts.values())
        if n_total > 0:
            probs   = [c / n_total for c in tag_counts.values()]
            entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
        else:
            entropy = 0.0

        # ── Feature 13: Burst component max ──────────────────────
        burst_max = float(
            component_writes.most_common(1)[0][1]
            if component_writes else 0
        )

        # ── Feature 14: Rare access score ────────────────────────
        rare_score = 0.0
        for tag, count in tag_counts.items():
            baseline_rate = btr.get(tag, 0.01)
            if count > baseline_rate * 3:
                rare_score += float(count - baseline_rate)

        # ── Feature 17-18: Reset / permissive counts ──────────────
        reset_count      = float(cat_counts.get('reset_signal',      0))
        permissive_count = float(cat_counts.get('permissive_signal', 0))

        # ── Assemble 19-dim vector in exact training order ────────
        features = [
            float(len(window_events)),                              # [0] per-window count
            float(sum(e.payload_len for e in window_events)),       # [1] per-window bytes
            valve_w,                                         # [2]
            pump_w,                                          # [3]
            float(cat_counts.get('valve_read',   0)),       # [4]
            float(cat_counts.get('pump_read',    0)),       # [5]
            float(cat_counts.get('level_access', 0)),       # [6]
            float(cat_counts.get('dosing_access',0)),       # [7]
            float(cat_counts.get('state_poll',   0)),       # [8]
            valve_dev,                                       # [9]
            pump_dev,                                        # [10]
            novel_count,                                     # [11]
            entropy,                                         # [12]
            burst_max,                                       # [13]
            rare_score,                                      # [14]
            float(len(unique_tags)),                         # [15]
            float(len(src_ips)),                             # [16]
            reset_count,                                     # [17]
            permissive_count,                                # [18]
        ]
        scaled_features = self._scaler.transform(features)

        return scaled_features, dict(cat_counts), len(window_events)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("TrustGate Plan B — Real-Time Network Feature Extractor")
    log.info("=" * 60)

    # ── Detect physical interface ─────────────────────────────────
    phys_iface = detect_physical_interface()
    if phys_iface:
        log.info(f"Physical interface detected: {phys_iface}")
    else:
        log.warning("No physical interface found — loopback only")

    # ── Shared event queue ────────────────────────────────────────
    event_queue = deque(maxlen=10000)
    queue_lock  = threading.Lock()

    # ── Start sniffers ────────────────────────────────────────────
    sniffers = []

    # Loopback sniffer — always start (captures physics_sim traffic)
    lo_sniffer = LoopbackSniffer(event_queue, queue_lock)
    lo_sniffer.start()
    sniffers.append(lo_sniffer)
    log.info("Loopback sniffer started (127.0.0.1:5020)")

    # Physical interface sniffer — start if interface found
    if phys_iface:
        phys_sniffer = RawSocketSniffer(
            phys_iface, event_queue, queue_lock
        )
        phys_sniffer.start()
        sniffers.append(phys_sniffer)
        log.info(f"Physical sniffer started ({phys_iface}:5020)")

    # ── Feature computer ──────────────────────────────────────────
    computer = FeatureComputer()

    log.info(f"Writing features to: {OUTPUT_PATH}")
    log.info(f"Window: {WINDOW_SECONDS}s | Tick: {TICK_INTERVAL}s")
    log.info(
        f"Baseline: establishing over first "
        f"{BASELINE_WINDOW}s..."
    )

    cycle        = 0
    total_events = 0

    while True:
        try:
            # ── Drain event queue into feature computer ───────────
            with queue_lock:
                new_events = list(event_queue)
                event_queue.clear()

            if new_events:
                computer.feed(new_events)
                total_events += len(new_events)

            # ── Compute features ──────────────────────────────────
            features, cat_counts, window_size = computer.compute()

            # ── Build output payload ──────────────────────────────
            payload = {
                "features": features,
                "feature_names": [
                    "packet_count",        "total_bytes",
                    "valve_write_count",   "pump_write_count",
                    "valve_read_count",    "pump_read_count",
                    "level_access_count",  "dosing_access_count",
                    "state_poll_count",
                    "valve_write_deviation","pump_write_deviation",
                    "novel_tag_count",     "tag_entropy",
                    "burst_component_max", "rare_access_score",
                    "unique_tags_this_sec","unique_src_ips",
                    "reset_count",         "permissive_count",
                ],
                "dim":            len(features),
                "window_events":  window_size,
                "total_events":   total_events,
                "cat_counts":     cat_counts,
                "sniffers": {
                    s.name: s.packets_captured
                    for s in sniffers
                },
                "timestamp":      time.time(),
                "cycle":          cycle,
            }

            tmp = OUTPUT_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, OUTPUT_PATH)

            cycle += 1

            # ── Periodic log ──────────────────────────────────────
            if cycle % 20 == 0:
                sniffer_stats = " | ".join(
                    f"{s.name}:{s.packets_captured}"
                    for s in sniffers
                )
                log.info(
                    f"Cycle {cycle:>5d} | "
                    f"events={total_events} | "
                    f"window={window_size} | "
                    f"v_w={features[2]:.0f} "
                    f"p_w={features[3]:.0f} | "
                    f"ent={features[12]:.3f} | "
                    f"dev_v={features[9]:+.2f} | "
                    f"{sniffer_stats}"
                )

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(TICK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Network extractor stopped.")
