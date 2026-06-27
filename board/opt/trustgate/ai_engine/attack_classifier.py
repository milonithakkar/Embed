import json
import logging
import numpy as np
from datetime import datetime
from collections import deque

log = logging.getLogger("AttackClassifier")

# How many cycles of history to keep for pattern detection
HISTORY_LENGTH = 20

# Register groups that are dangerous together
COMPOUND_RULES = [
    {
        "name": "Oldsmar Chemical Attack",
        "description": "Chemical valve opened with pump running",
        "registers": ["40003", "40004"],
        "conditions": [
            {"register": "40003", "operator": "gt", "threshold": 1},
            {"register": "40004", "operator": "eq", "value": 1},
        ],
        "severity": "CATASTROPHIC",
        "confidence_boost": 0.4
    },
    {
        "name": "RO Pressure Rupture",
        "description": "Relief valve closed while high pressure pump running",
        "registers": ["40007", "40008"],
        "conditions": [
            {"register": "40007", "operator": "eq", "value": 0},
            {"register": "40008", "operator": "eq", "value": 1},
        ],
        "severity": "CRITICAL",
        "confidence_boost": 0.5
    },
    {
        "name": "Tank Overflow Attack",
        "description": "Intake pump running with outlet valve closed",
        "registers": ["40001", "40002"],
        "conditions": [
            {"register": "40001", "operator": "eq", "value": 1},
            {"register": "40002", "operator": "gt", "threshold": 850},
        ],
        "severity": "HIGH",
        "confidence_boost": 0.3
    }
]

# Anomaly thresholds — how much a value can deviate from its rolling mean
ANOMALY_THRESHOLDS = {
    "40002": 100,   # Tank T101 — flag if jumps more than 100mm suddenly
    "40006": 150,   # Tank T301
    "40009": 120,   # Tank T601
}


class AttackClassifier:

    def __init__(self):
        # Rolling history of register values: {register: deque of values}
        self.history = {str(i): deque(maxlen=HISTORY_LENGTH) for i in range(40001, 40011)}
        self.alert_history = deque(maxlen=50)
        self.total_classifications = 0
        self.total_attacks_detected = 0
        log.info("Attack classifier initialized")

    def update_history(self, plant_state):
        """Feed current plant state into rolling history"""
        for reg, val in plant_state.items():
            if reg in self.history:
                self.history[reg].append(val)

    def check_single_register_anomaly(self, register, value):
        """
        Statistical anomaly detection — flag sudden large changes
        even if value is technically within safe range.
        Catches slow/stealthy attacks.
        """
        if register not in ANOMALY_THRESHOLDS:
            return None
        if len(self.history[register]) < 5:
            return None

        hist = list(self.history[register])
        mean = sum(hist) / len(hist)
        deviation = abs(value - mean)
        threshold = ANOMALY_THRESHOLDS[register]

        if deviation > threshold:
            return {
                "type": "STATISTICAL_ANOMALY",
                "register": register,
                "value": value,
                "mean": round(mean, 2),
                "deviation": round(deviation, 2),
                "threshold": threshold,
                "confidence": min(0.95, 0.5 + (deviation / threshold) * 0.3),
                "message": f"Register {register} deviated {deviation:.0f} units from mean {mean:.0f}"
            }
        return None

    def check_compound_attack(self, plant_state):
        """
        Check if current state matches a known multi-register attack pattern.
        More sophisticated than single-register checks.
        """
        detections = []

        for rule in COMPOUND_RULES:
            matched = 0
            total = len(rule["conditions"])

            for condition in rule["conditions"]:
                reg = condition["register"]
                current_val = plant_state.get(reg, 0)

                if condition["operator"] == "gt":
                    if current_val > condition["threshold"]:
                        matched += 1
                elif condition["operator"] == "lt":
                    if current_val < condition["threshold"]:
                        matched += 1
                elif condition["operator"] == "eq":
                    if current_val == condition["value"]:
                        matched += 1

            confidence = matched / total
            if confidence >= 1.0:
                confidence += rule["confidence_boost"] * confidence
                confidence = min(0.99, confidence)
                detections.append({
                    "type": "COMPOUND_ATTACK",
                    "name": rule["name"],
                    "description": rule["description"],
                    "severity": rule["severity"],
                    "confidence": round(confidence, 2),
                    "matched_conditions": matched,
                    "total_conditions": total,
                    "message": f"{rule['name']} detected with {confidence*100:.0f}% confidence"
                })

        return detections

    def check_rapid_change(self, register, value):
        """
        Detect rapid sequential changes to the same register.
        Legitimate SCADA systems change registers slowly.
        """
        if len(self.history[register]) < 3:
            return None

        recent = list(self.history[register])[-3:]
        changes = [abs(recent[i+1] - recent[i]) for i in range(len(recent)-1)]
        avg_change = sum(changes) / len(changes)

        if avg_change > 500:
            return {
                "type": "RAPID_CHANGE",
                "register": register,
                "value": value,
                "avg_change_rate": round(avg_change, 2),
                "confidence": min(0.90, 0.6 + avg_change / 5000),
                "message": f"Rapid changes detected on register {register} — avg change {avg_change:.0f}/cycle"
            }
        return None

    def classify(self, plant_state, active_violations):
        """
        Main classification function.
        Takes current plant state and returns full threat assessment.
        """
        self.total_classifications += 1
        self.update_history(plant_state)

        findings = []

        # 1. Check compound attack patterns
        compound = self.check_compound_attack(plant_state)
        findings.extend(compound)

        # 2. Check statistical anomalies on sensor registers
        for reg, val in plant_state.items():
            anomaly = self.check_single_register_anomaly(reg, val)
            if anomaly:
                findings.append(anomaly)

        # 3. Check rapid change patterns
        for reg, val in plant_state.items():
            if reg in self.history:
                rapid = self.check_rapid_change(reg, val)
                if rapid:
                    findings.append(rapid)

        # 4. Boost confidence if rule-based violations also present
        if active_violations and findings:
            for f in findings:
                f["confidence"] = min(0.99, f["confidence"] + 0.15)
                f["rule_based_confirmed"] = True

        # Build final assessment
        if findings:
            self.total_attacks_detected += 1
            top_threat = max(findings, key=lambda x: x["confidence"])
            assessment = {
                "timestamp": datetime.now().isoformat(),
                "threat_detected": True,
                "threat_count": len(findings),
                "top_threat": top_threat,
                "all_threats": findings,
                "overall_confidence": top_threat["confidence"],
                "recommendation": self.get_recommendation(top_threat)
            }
        else:
            assessment = {
                "timestamp": datetime.now().isoformat(),
                "threat_detected": False,
                "threat_count": 0,
                "top_threat": None,
                "all_threats": [],
                "overall_confidence": 0.0,
                "recommendation": "NONE — system operating normally"
            }

        self.alert_history.append(assessment)
        return assessment

    def get_recommendation(self, threat):
        """Map threat type to actionable remediation step"""
        if threat.get("name") == "Oldsmar Chemical Attack":
            return "IMMEDIATE: Close EV2 emergency valve at (6.1m, 4.5m)"
        elif threat.get("name") == "RO Pressure Rupture":
            return "IMMEDIATE: Emergency stop P501 pump at (11.0m, 5.0m)"
        elif threat.get("name") == "Tank Overflow Attack":
            return "URGENT: Reduce intake pump P101 speed at (1.5m, 5.0m)"
        elif threat.get("type") == "STATISTICAL_ANOMALY":
            return f"INVESTIGATE: Unusual readings on {threat['register']} — possible sensor spoofing"
        elif threat.get("type") == "RAPID_CHANGE":
            return f"INVESTIGATE: Rapid commands on {threat['register']} — possible replay attack"
        return "MONITOR: Suspicious activity detected"

    def get_stats(self):
        return {
            "total_classifications": self.total_classifications,
            "total_attacks_detected": self.total_attacks_detected,
            "detection_rate": round(
                self.total_attacks_detected / max(1, self.total_classifications) * 100, 2
            )
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    classifier = AttackClassifier()

    print("\n--- Test 1: Normal state ---")
    normal_state = {
        "40001": 1, "40002": 650, "40003": 0, "40004": 1,
        "40005": 1, "40006": 800, "40007": 1, "40008": 1,
        "40009": 750, "40010": 1
    }
    # Feed some history first
    for _ in range(10):
        classifier.update_history(normal_state)

    result = classifier.classify(normal_state, [])
    print(f"Threat detected: {result['threat_detected']}")
    print(f"Confidence: {result['overall_confidence']}")

    print("\n--- Test 2: Oldsmar attack state ---")
    attack_state = normal_state.copy()
    attack_state["40003"] = 9999
    result = classifier.classify(attack_state, [{"register": "40003"}])
    print(f"Threat detected: {result['threat_detected']}")
    print(f"Top threat: {result['top_threat']['name'] if result['top_threat'] else None}")
    print(f"Confidence: {result['overall_confidence']}")
    print(f"Recommendation: {result['recommendation']}")

    print("\n--- Test 3: Pressure rupture attack ---")
    pressure_state = normal_state.copy()
    pressure_state["40007"] = 0
    result = classifier.classify(pressure_state, [{"register": "40007"}])
    print(f"Threat detected: {result['threat_detected']}")
    print(f"Top threat: {result['top_threat']['name'] if result['top_threat'] else None}")
    print(f"Confidence: {result['overall_confidence']}")
    print(f"Recommendation: {result['recommendation']}")

    print("\n--- Test 4: Stealthy tank drain (statistical anomaly) ---")
    stealthy_state = normal_state.copy()
    stealthy_state["40009"] = 200   # sudden drop — below normal but not zero
    result = classifier.classify(stealthy_state, [])
    print(f"Threat detected: {result['threat_detected']}")
    print(f"Type: {result['top_threat']['type'] if result['top_threat'] else None}")
    print(f"Confidence: {result['overall_confidence']}")

    print(f"\nClassifier stats: {classifier.get_stats()}")
