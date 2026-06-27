import socket
import struct
import threading
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ModbusServer")

ICS_ALERTS_PATH = "/tmp/ics_alerts.json"

registers = [0] * 200

WATCH_REGISTERS = {
    1:  {"name": "Pump P101",                  "safe_min": 0,   "safe_max": 1,    "addr": "40001"},
    2:  {"name": "Tank T101 Level",            "safe_min": 500, "safe_max": 800,  "addr": "40002"},
    3:  {"name": "Chemical Dosing Valve MV201","safe_min": 0,   "safe_max": 1,    "addr": "40003"},
    4:  {"name": "Pump P205",                  "safe_min": 0,   "safe_max": 1,    "addr": "40004"},
    5:  {"name": "Pump P301",                  "safe_min": 0,   "safe_max": 1,    "addr": "40005"},
    6:  {"name": "Tank T301 Level",            "safe_min": 500, "safe_max": 1200, "addr": "40006"},
    7:  {"name": "Pressure Relief Valve PRV1", "safe_min": 1,   "safe_max": 1,    "addr": "40007"},
    8:  {"name": "Pump P501",                  "safe_min": 0,   "safe_max": 1,    "addr": "40008"},
    9:  {"name": "Tank T601 Level",            "safe_min": 500, "safe_max": 1000, "addr": "40009"},
    10: {"name": "Emergency Shutoff EV2",      "safe_min": 1,   "safe_max": 1,    "addr": "40010"},
}

def write_alert(reg, val, source_ip, is_violation):
    try:
        watch = WATCH_REGISTERS.get(reg, {})
        alert = {
            "timestamp": datetime.now().isoformat(),
            "source_ip": source_ip,
            "register": watch.get("addr", str(reg)),
            "component": watch.get("name", f"Register {reg}"),
            "value": val,
            "is_write": True,
            "is_violation": is_violation
        }
        with open(ICS_ALERTS_PATH, "a") as f:
            f.write(json.dumps(alert) + "\n")
        if is_violation:
            log.critical("ALERT: %s = %d from %s", watch.get("name", reg), val, source_ip)
    except Exception as e:
        log.error("Alert write error: %s", e)

def handle_client(conn, addr):
    source_ip = addr[0]
    log.info("Client connected: %s", source_ip)
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            if len(data) < 8:
                continue

            tid = data[0:2]
            pid = data[2:4]
            uid = data[6]
            fc  = data[7]

            if fc == 6 and len(data) >= 12:
                reg = struct.unpack(">H", data[8:10])[0]
                val = struct.unpack(">H", data[10:12])[0]
                registers[reg] = val
                log.info("WRITE reg=%d val=%d from %s", reg, val, source_ip)

                # Check if violation and write alert
                if reg in WATCH_REGISTERS:
                    w = WATCH_REGISTERS[reg]
                    is_violation = not (w["safe_min"] <= val <= w["safe_max"])
                    write_alert(reg, val, source_ip, is_violation)

                conn.send(data)

            elif fc == 3 and len(data) >= 12:
                reg   = struct.unpack(">H", data[8:10])[0]
                count = struct.unpack(">H", data[10:12])[0]
                vals  = registers[reg:reg+count]
                body  = struct.pack(">BB", uid, fc) + bytes([count*2])
                for v in vals:
                    body += struct.pack(">H", v)
                length = struct.pack(">H", len(body))
                conn.send(tid + pid + length + body)

        except Exception as e:
            log.error("Error: %s", e)
            break
    conn.close()

def run():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 502))
    s.listen(5)
    log.info("Modbus server listening on 127.0.0.1:502")
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    run()
