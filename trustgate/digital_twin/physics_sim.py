import json
import time
import logging
from datetime import datetime
import sys
sys.path.insert(0, '/opt/trustgate/digital_twin')
from twin_engine import DigitalTwin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/tmp/physics_sim.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('PhysicsSim')

TWIN_REPORT_PATH = '/tmp/twin_report.json'
ICS_ALERTS_PATH  = '/tmp/ics_alerts.json'
CYCLE_SECONDS    = 5

NATURAL_DRIFT = {
    "40002": (500, 800,  5),
    "40006": (500, 1200, 8),
    "40009": (500, 1000, 6),
}


class PhysicsSimulator:

    def __init__(self):
        self.twin = DigitalTwin()
        self.cycle = 0
        self.attack_active = False
        # Skip all existing alerts on startup — only process NEW ones
        try:
            with open("/tmp/ics_alerts.json") as f:
                existing = [l.strip() for l in f.readlines() if l.strip()]
            self.last_alert_line = len(existing)
            log.info("Skipping %d existing alerts on startup", len(existing))
        except FileNotFoundError:
            self.last_alert_line = 0
        log.info("Physics simulator ready")

    def apply_natural_drift(self):
        import random
        for reg, (mn, mx, drift) in NATURAL_DRIFT.items():
            current = self.twin.state[reg]
            change = random.randint(-drift, drift)
            new_val = max(mn - 50, min(mx + 50, current + change))
            self.twin.state[reg] = new_val

    def ingest_ics_alerts(self):
        try:
            with open(ICS_ALERTS_PATH) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if not lines:
                return
            if len(lines) <= self.last_alert_line:
                return
            new_lines = lines[self.last_alert_line:]
            self.last_alert_line = len(lines)
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    alert = json.loads(line)
                except:
                    continue
                if not alert.get("is_write", False):
                    continue
                if not alert.get("is_violation", False):
                    continue
                reg = str(alert.get("register", ""))
                val = alert.get("value", 0)
                if reg and reg in self.twin.assets:
                    log.info("Ingesting violation: register %s = %s", reg, val)
                    report = self.twin.apply_command(reg, val)
                    if report["is_violation"]:
                        self.attack_active = True
                        log.critical(
                            "ATTACK CONFIRMED: %s — %ds to damage",
                            report["component"],
                            report["time_to_failure_seconds"]
                        )
                        attack_report = self.twin.get_current_state_report()
                        attack_report["cycle"] = self.cycle
                        attack_report["simulator"] = "TrustGate PhysicsSim v1.0"
                        self.write_report(attack_report)
                        import time as t
                        t.sleep(15)
                        safe_val = self.twin.assets[reg]["safe_range"]["min"]
                        self.twin.state[reg] = safe_val
                        self.attack_active = False
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass
        except Exception as e:
            log.error("ICS ingest error: %s", e)

    def write_report(self, state_report):
        try:
            with open(TWIN_REPORT_PATH, "w") as f:
                json.dump(state_report, f, indent=2)
        except Exception as e:
            log.error("Failed to write twin report: %s", e)

    def run(self):
        log.info("Physics simulation loop starting — cycle every %ds", CYCLE_SECONDS)
        while True:
            self.cycle += 1
            log.info("--- Cycle %d ---", self.cycle)
            self.apply_natural_drift()
            self.ingest_ics_alerts()
            state_report = self.twin.get_current_state_report()
            state_report["cycle"] = self.cycle
            state_report["simulator"] = "TrustGate PhysicsSim v1.0"
            self.write_report(state_report)
            status = state_report["overall_status"]
            violations = len(state_report["active_violations"])
            if violations:
                critical = state_report["most_critical"]
                log.critical(
                    "STATUS: %s | Violations: %d | Most critical: %s (%ds)",
                    status,
                    violations,
                    critical["component"],
                    critical["time_to_failure_seconds"]
                )
            else:
                log.info("STATUS: NORMAL | All registers in safe range")
            time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    sim = PhysicsSimulator()
    sim.run()
