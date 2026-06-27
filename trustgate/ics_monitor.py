import json
import time
import logging
import threading
from datetime import datetime
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.device import ModbusDeviceIdentification
import socket
import struct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ics_monitor.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ICSMonitor")

ICS_ALERTS_PATH  = "/tmp/ics_alerts.json"
MODBUS_HOST      = "127.0.0.1"
MODBUS_PORT      = 502

# Registers that are dangerous if written with unexpected values
WATCH_REGISTERS = {
    40001: {"name": "Pump P101",                "safe_min": 0, "safe_max": 1},
    40002: {"name": "Tank T101 Level",          "safe_min": 500, "safe_max": 800},
    40003: {"name": "Chemical Dosing Valve MV201", "safe_min": 0, "safe_max": 1},
    40004: {"name": "Pump P205",                "safe_min": 0, "safe_max": 1},
    40005: {"name": "Pump P301",                "safe_min": 0, "safe_max": 1},
    40006: {"name": "Tank T301 Level",          "safe_min": 500, "safe_max": 1200},
    40007: {"name": "Pressure Relief Valve PRV1","safe_min": 1, "safe_max": 1},
    40008: {"name": "Pump P501",                "safe_min": 0, "safe_max": 1},
    40009: {"name": "Tank T601 Level",          "safe_min": 500, "safe_max": 1000},
    40010: {"name": "Emergency Shutoff EV2",    "safe_min": 1, "safe_max": 1},
}


class ICSMonitor:

    def __init__(self):
        self.alert_count = 0
        self.packet_count = 0
        log.info("ICS Monitor initialized — watching %d registers", len(WATCH_REGISTERS))

    def analyze_modbus_write(self, register, value, source_ip="unknown"):
        """
        Analyze a Modbus write command.
        Returns an alert dict if suspicious, None if normal.
        """
        self.packet_count += 1
        reg_num = register

        # Check if this register is in our watchlist
        if reg_num not in WATCH_REGISTERS:
            return None

        watch = WATCH_REGISTERS[reg_num]
        is_write = True
        is_violation = not (watch["safe_min"] <= value <= watch["safe_max"])

        # Build the alert
        alert = {
            "timestamp": datetime.now().isoformat(),
            "source_ip": source_ip,
            "register": str(40000 + reg_num) if reg_num < 1000 else str(reg_num),
            "register_offset": reg_num,
            "component": watch["name"],
            "value": value,
            "safe_min": watch["safe_min"],
            "safe_max": watch["safe_max"],
            "is_write": is_write,
            "is_violation": is_violation,
            "severity": self.get_severity(reg_num, value, is_violation)
        }

        if is_violation:
            self.alert_count += 1
            log.critical(
                "SUSPICIOUS WRITE: Register %s (%s) = %s from %s | Severity: %s",
                alert["register"],
                watch["name"],
                value,
                source_ip,
                alert["severity"]
            )
        else:
            log.info(
                "Normal write: Register %s (%s) = %s",
                alert["register"],
                watch["name"],
                value
            )

        self.write_alert(alert)
        return alert

    def get_severity(self, reg_num, value, is_violation):
        if not is_violation:
            return "NONE"
        # Chemical valve and high pressure pump are most dangerous
        if reg_num in [3, 40003]:   # Chemical dosing valve
            return "CATASTROPHIC"
        if reg_num in [7, 40007, 8, 40008]:  # Pressure relief + RO pump
            return "CRITICAL"
        if reg_num in [2, 40002, 6, 40006]:  # Tank levels
            return "HIGH"
        return "MEDIUM"

    def write_alert(self, alert):
        try:
            with open(ICS_ALERTS_PATH, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            log.error("Failed to write alert: %s", e)

    def monitor_raw_tcp(self, host=MODBUS_HOST, port=MODBUS_PORT):
        """
        Listen for raw Modbus TCP packets and parse write commands.
        This is what runs when the switch is connected and traffic is mirrored.
        """
        log.info("Starting raw TCP monitor on %s:%d", host, port)
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(5)
        log.info("Listening for Modbus connections...")

        while True:
            try:
                conn, addr = server_sock.accept()
                source_ip = addr[0]
                log.info("Connection from %s", source_ip)
                threading.Thread(
                    target=self.handle_connection,
                    args=(conn, source_ip),
                    daemon=True
                ).start()
            except Exception as e:
                log.error("Accept error: %s", e)

    def handle_connection(self, conn, source_ip):
        """Parse Modbus TCP frames from a connection"""
        buffer = b""
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data

                # Modbus TCP frame: 6 byte header + PDU
                while len(buffer) >= 6:
                    # Parse MBAP header
                    transaction_id = struct.unpack(">H", buffer[0:2])[0]
                    protocol_id    = struct.unpack(">H", buffer[2:4])[0]
                    length         = struct.unpack(">H", buffer[4:6])[0]

                    if len(buffer) < 6 + length:
                        break  # Wait for more data

                    frame = buffer[6:6 + length]
                    buffer = buffer[6 + length:]

                    if len(frame) < 2:
                        continue

                    unit_id      = frame[0]
                    function_code = frame[1]

                    # Function code 6 = Write Single Register
                    # Function code 16 = Write Multiple Registers
                    if function_code == 6 and len(frame) >= 6:
                        register = struct.unpack(">H", frame[2:4])[0] + 1
                        value    = struct.unpack(">H", frame[4:6])[0]
                        self.analyze_modbus_write(register, value, source_ip)

                    elif function_code == 16 and len(frame) >= 7:
                        start_reg  = struct.unpack(">H", frame[2:4])[0] + 1
                        reg_count  = struct.unpack(">H", frame[4:6])[0]
                        byte_count = frame[6]
                        for i in range(reg_count):
                            offset = 7 + i * 2
                            if offset + 2 <= len(frame):
                                value = struct.unpack(">H", frame[offset:offset+2])[0]
                                self.analyze_modbus_write(start_reg + i, value, source_ip)

            except Exception as e:
                log.error("Connection handler error: %s", e)
                break
        conn.close()


if __name__ == "__main__":
    monitor = ICSMonitor()

    # Test analyze function directly
    print("\n--- Test 1: Normal pump command ---")
    alert = monitor.analyze_modbus_write(1, 1, "192.168.1.100")
    print(f"Violation: {alert['is_violation']} | Severity: {alert['severity']}")

    print("\n--- Test 2: Oldsmar attack on chemical valve ---")
    alert = monitor.analyze_modbus_write(3, 9999, "192.168.1.101")
    print(f"Violation: {alert['is_violation']} | Severity: {alert['severity']}")
    print(f"Component: {alert['component']}")

    print("\n--- Test 3: PRV1 closure attack ---")
    alert = monitor.analyze_modbus_write(7, 0, "192.168.1.101")
    print(f"Violation: {alert['is_violation']} | Severity: {alert['severity']}")

    print("\n--- Test 4: Normal tank level ---")
    alert = monitor.analyze_modbus_write(2, 650, "192.168.1.100")
    print(f"Violation: {alert['is_violation']} | Severity: {alert['severity']}")

    print(f"\nTotal packets analyzed: {monitor.packet_count}")
    print(f"Total alerts generated: {monitor.alert_count}")
    print(f"Alerts written to: {ICS_ALERTS_PATH}")
