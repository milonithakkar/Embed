import json
import logging
from datetime import datetime

log = logging.getLogger("Remediation")

REMEDIATION_ACTIONS = {
    "Oldsmar Chemical Attack": {
        "priority": 1,
        "steps": [
            "Close EV2 emergency shutoff valve at (6.1m, 4.5m) — Register 40010 = 0",
            "Stop chemical pump P205 at (8.0m, 3.5m) — Register 40004 = 0",
            "Alert operator: NaOCl overdose risk — evacuate distribution zone",
            "Do NOT restore MV201 until manual inspection at (8.7m, 2.3m)"
        ],
        "automated_writes": [
            {"register": "40010", "value": 0},
            {"register": "40004", "value": 0}
        ],
        "time_critical_seconds": 23
    },
    "RO Pressure Rupture": {
        "priority": 1,
        "steps": [
            "Emergency stop P501 pump at (11.0m, 5.0m) — Register 40008 = 0",
            "Emergency stop P502 pump — Register 40008 = 0",
            "Open PRV1 relief valve at (12.1m, 6.8m) — Register 40007 = 1",
            "Alert operator: High pressure zone — do not enter P5 stage"
        ],
        "automated_writes": [
            {"register": "40008", "value": 0},
            {"register": "40007", "value": 1}
        ],
        "time_critical_seconds": 8
    },
    "Tank Overflow Attack": {
        "priority": 2,
        "steps": [
            "Reduce intake pump P101 speed at (1.5m, 5.0m) — Register 40001 = 0",
            "Open drain valve at T101 tank",
            "Monitor T101 level sensor at (3.0m, 5.0m) — Register 40002"
        ],
        "automated_writes": [
            {"register": "40001", "value": 0}
        ],
        "time_critical_seconds": 90
    },
    "STATISTICAL_ANOMALY": {
        "priority": 3,
        "steps": [
            "Flag register for manual inspection",
            "Cross-check with physical sensor reading",
            "If confirmed — isolate affected stage"
        ],
        "automated_writes": [],
        "time_critical_seconds": 180
    },
    "RAPID_CHANGE": {
        "priority": 3,
        "steps": [
            "Block source IP from Modbus access",
            "Log all commands from source for forensic analysis",
            "Alert operator of possible replay attack"
        ],
        "automated_writes": [],
        "time_critical_seconds": 120
    }
}


class RemediationEngine:

    def __init__(self):
        self.remediation_log = []
        log.info("Remediation engine initialized with %d action plans", len(REMEDIATION_ACTIONS))

    def get_action_plan(self, threat):
        """Get remediation plan for a detected threat"""
        threat_name = threat.get("name") or threat.get("type", "UNKNOWN")

        if threat_name not in REMEDIATION_ACTIONS:
            return {
                "threat": threat_name,
                "priority": 99,
                "steps": ["No automated remediation available — manual inspection required"],
                "automated_writes": [],
                "time_critical_seconds": None
            }

        plan = REMEDIATION_ACTIONS[threat_name].copy()
        plan["threat"] = threat_name
        plan["confidence"] = threat.get("confidence", 0)
        plan["timestamp"] = datetime.now().isoformat()

        log.warning(
            "REMEDIATION PLAN: %s | Priority %d | Time critical: %ss",
            threat_name,
            plan["priority"],
            plan["time_critical_seconds"]
        )
        for i, step in enumerate(plan["steps"], 1):
            log.warning("  Step %d: %s", i, step)

        self.remediation_log.append(plan)
        return plan

    def write_remediation_report(self, plan, path="/tmp/remediation_report.json"):
        try:
            with open(path, "w") as f:
                json.dump(plan, f, indent=2)
        except Exception as e:
            log.error("Failed to write remediation report: %s", e)

    def get_log(self):
        return self.remediation_log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = RemediationEngine()

    print("\n--- Test 1: Oldsmar remediation ---")
    threat = {"name": "Oldsmar Chemical Attack", "confidence": 0.99}
    plan = engine.get_action_plan(threat)
    print(f"Priority: {plan['priority']}")
    print(f"Time critical: {plan['time_critical_seconds']}s")
    print(f"Automated writes: {plan['automated_writes']}")
    print(f"Steps: {len(plan['steps'])}")

    print("\n--- Test 2: Pressure rupture remediation ---")
    threat = {"name": "RO Pressure Rupture", "confidence": 0.99}
    plan = engine.get_action_plan(threat)
    print(f"Priority: {plan['priority']}")
    print(f"Time critical: {plan['time_critical_seconds']}s")
    print(f"Automated writes: {plan['automated_writes']}")

    print("\n--- Test 3: Statistical anomaly remediation ---")
    threat = {"type": "STATISTICAL_ANOMALY", "confidence": 0.75}
    plan = engine.get_action_plan(threat)
    print(f"Priority: {plan['priority']}")
    print(f"Steps: {plan['steps']}")
