import json
import time
import os
from pymodbus.client import ModbusTcpClient

# Calibrated bounds from live plant readings:
#   FIT301 raw=2356, FIT401 raw=2308, FIT501 raw=1731, FIT601 raw=499
WATCH_LIST = [
    {"name":"LIT101 Raw Water Tank",   "address":0,  "safe_min":200,  "safe_max":950,  "unit":"mm",       "scale":1.0,  "category":"level"},
    {"name":"LIT401 RO Feed Tank",     "address":30, "safe_min":100,  "safe_max":780,  "unit":"mm",       "scale":1.0,  "category":"level"},
    {"name":"LIT601 Clean Water Tank", "address":50, "safe_min":100,  "safe_max":1150, "unit":"mm",       "scale":1.0,  "category":"level"},
    {"name":"FIT101 Intake Flow",      "address":3,  "safe_min":0,    "safe_max":400,  "unit":"Lh",       "scale":1.0,  "category":"flow"},
    {"name":"FIT301 UF Permeate",      "address":25, "safe_min":0,    "safe_max":3000, "unit":"Lhx10",    "scale":0.1,  "category":"flow"},
    {"name":"FIT401 RO Feed Flow",     "address":34, "safe_min":0,    "safe_max":3000, "unit":"Lhx10",    "scale":0.1,  "category":"flow"},
    {"name":"FIT501 RO Permeate",      "address":43, "safe_min":0,    "safe_max":2500, "unit":"Lhx10",    "scale":0.1,  "category":"flow"},
    {"name":"FIT601 Distribution",     "address":53, "safe_min":0,    "safe_max":1000, "unit":"Lhx10",    "scale":0.1,  "category":"flow"},
    {"name":"AIT201 pH",               "address":10, "safe_min":600,  "safe_max":850,  "unit":"pHx100",   "scale":0.01, "category":"chemistry"},
    {"name":"DPIT301 UF Pressure",     "address":20, "safe_min":300,  "safe_max":1500, "unit":"kPax10",   "scale":0.1,  "category":"pressure"},
    {"name":"ATK_FLAG Attack Register","address":62, "safe_min":0,    "safe_max":0,    "unit":"flag",     "scale":1.0,  "category":"security"},
]

MODBUS_HOST = "127.0.0.1"
MODBUS_PORT = 5020
OUTPUT_PATH = "/tmp/ics_alerts.json"

def run_monitor():
    client    = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    connected = False
    cycle     = 0

    print("=" * 55)
    print("  ICS Threshold Monitor — TrustGate Enhanced")
    print(f"  Watching {len(WATCH_LIST)} registers")
    print("=" * 55)

    while True:
        if not connected:
            try:
                client.connect()
                connected = True
                print("[MONITOR] Connected to Modbus server")
            except Exception as e:
                print(f"[MONITOR] Connection failed: {e}")
                time.sleep(2)
                continue

        try:
            result = client.read_holding_registers(address=0, count=63)
            if result.isError():
                connected = False
                time.sleep(1)
                continue
            regs = result.registers
        except Exception as e:
            connected = False
            time.sleep(1)
            continue

        alerts     = []
        violations = []

        for watch in WATCH_LIST:
            addr    = watch["address"]
            raw_val = regs[addr] if addr < len(regs) else 0
            is_viol = raw_val < watch["safe_min"] or raw_val > watch["safe_max"]

            alert = {
                "sensor":       watch["name"],
                "address":      addr,
                "raw_value":    raw_val,
                "scaled_value": round(raw_val * watch["scale"], 3),
                "unit":         watch["unit"],
                "safe_min":     watch["safe_min"],
                "safe_max":     watch["safe_max"],
                "is_violation": is_viol,
                "category":     watch["category"],
                "message":      f"VIOLATION: {raw_val} outside [{watch['safe_min']},{watch['safe_max']}]" if is_viol else "Normal"
            }
            alerts.append(alert)
            if is_viol:
                violations.append(alert)
                print(f"[ALERT] X {watch['name']} = {raw_val} (safe:{watch['safe_min']}-{watch['safe_max']})")

        output = {
            "timestamp":       time.time(),
            "cycle":           cycle,
            "alerts":          alerts,
            "violation_count": len(violations),
            "all_clear":       len(violations) == 0,
        }

        tmp = OUTPUT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(output, f, indent=2)
        os.replace(tmp, OUTPUT_PATH)
        cycle += 1

        if cycle % 10 == 0:
            status = "ALL CLEAR" if not violations else f"{len(violations)} VIOLATIONS"
            print(f"[MONITOR] {status} | LIT101={regs[0]} LIT401={regs[30]} LIT601={regs[50]} | FIT401={regs[34]} FIT501={regs[43]}")

        time.sleep(1)

if __name__ == "__main__":
    run_monitor()
