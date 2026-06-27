import json
import time
import random
import logging
from datetime import datetime
from pymodbus.client import ModbusTcpClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("ModbusSimulator")

MODBUS_HOST = "127.0.0.1"
MODBUS_PORT = 502

# Normal operating commands (safe traffic)
NORMAL_TRAFFIC = [
    {"register": 1,  "value": 1,   "desc": "Pump P101 ON"},
    {"register": 2,  "value": 650, "desc": "Tank T101 level 650mm"},
    {"register": 3,  "value": 0,   "desc": "Chemical valve closed"},
    {"register": 4,  "value": 1,   "desc": "Chemical pump ON"},
    {"register": 5,  "value": 1,   "desc": "UF pump ON"},
    {"register": 6,  "value": 800, "desc": "Tank T301 level 800mm"},
    {"register": 7,  "value": 1,   "desc": "PRV1 relief valve OPEN"},
    {"register": 8,  "value": 1,   "desc": "RO pump ON"},
    {"register": 9,  "value": 750, "desc": "Clean tank level 750mm"},
    {"register": 10, "value": 1,   "desc": "Emergency shutoff OPEN"},
]

# Attack sequences
ATTACK_SCENARIOS = {
    "oldsmar": [
        {"register": 3, "value": 9999, "desc": "ATTACK: Chemical valve forced open"},
        {"register": 4, "value": 1,    "desc": "ATTACK: Chemical pump forced ON"},
    ],
    "pressure": [
        {"register": 7, "value": 0, "desc": "ATTACK: PRV1 relief valve CLOSED"},
        {"register": 8, "value": 1, "desc": "ATTACK: RO pump kept running"},
    ],
    "drain": [
        {"register": 9, "value": 1, "desc": "ATTACK: Clean tank drain spoofed"},
    ]
}


class ModbusSimulator:

    def __init__(self):
        self.client = None
        self.connected = False

    def connect(self):
        self.client = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
        self.connected = self.client.connect()
        if self.connected:
            log.info("Connected to Modbus server at %s:%d", MODBUS_HOST, MODBUS_PORT)
        else:
            log.error("Failed to connect to Modbus server")
        return self.connected

    def send_normal_traffic(self, cycles=3):
        log.info("Sending normal traffic (%d cycles)...", cycles)
        for i in range(cycles):
            for cmd in NORMAL_TRAFFIC:
                # Add small random variation to sensor values
                value = cmd["value"]
                if cmd["register"] in [2, 6, 9]:
                    value += random.randint(-10, 10)
                result = self.client.write_register(cmd["register"], value)
                if not result.isError():
                    log.info("NORMAL: %s = %d", cmd["desc"], value)
                time.sleep(0.2)
            time.sleep(1)

    def send_attack(self, scenario_name):
        if scenario_name not in ATTACK_SCENARIOS:
            log.error("Unknown scenario: %s", scenario_name)
            return
        log.critical("=== LAUNCHING ATTACK: %s ===", scenario_name.upper())
        for cmd in ATTACK_SCENARIOS[scenario_name]:
            result = self.client.write_register(cmd["register"], cmd["value"])
            if not result.isError():
                log.critical("SENT: %s", cmd["desc"])
            time.sleep(0.5)

    def run_demo_sequence(self):
        """
        Full demo sequence:
        1. Normal traffic for 10 seconds
        2. Oldsmar attack
        3. Normal traffic for 10 seconds
        4. Pressure attack
        5. Normal traffic
        """
        log.info("Starting demo sequence...")

        log.info("Phase 1: Normal operations")
        self.send_normal_traffic(cycles=2)

        log.info("Phase 2: Oldsmar chemical attack")
        self.send_attack("oldsmar")
        time.sleep(5)

        log.info("Phase 3: Back to normal")
        self.send_normal_traffic(cycles=2)

        log.info("Phase 4: Pressure rupture attack")
        self.send_attack("pressure")
        time.sleep(5)

        log.info("Phase 5: Normal operations restored")
        self.send_normal_traffic(cycles=2)

        log.info("Demo sequence complete")

    def disconnect(self):
        if self.client:
            self.client.close()
            log.info("Disconnected")


if __name__ == "__main__":
    sim = ModbusSimulator()
    if sim.connect():
        sim.run_demo_sequence()
        sim.disconnect()
    else:
        log.error("Could not connect — is the Modbus server running?")
        log.info("Start it with: python3 -c \"from pymodbus.server import StartTcpServer; from pymodbus.datastore import *; store = ModbusSlaveContext(hr=ModbusSequentialDataBlock(0, [0]*100)); StartTcpServer(ModbusServerContext(store), address=(\'127.0.0.1\', 502))\"")
