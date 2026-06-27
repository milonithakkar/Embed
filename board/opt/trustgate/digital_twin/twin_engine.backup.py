import json
import time
import logging
from datetime import datetime

log = logging.getLogger('DigitalTwin')

class DigitalTwin:

    def __init__(self, asset_map_path='/opt/trustgate/digital_twin/physical_assets.json'):
        self.asset_map_path = asset_map_path
        self.assets = {}
        self.attack_scenarios = {}
        self.pipe_connections = []

        # Live plant state — current register values
        self.state = {
            "40001": 1,    # P101 pump ON
            "40002": 650,  # T101 level 650mm (safe: 500-800)
            "40003": 0,    # MV201 chemical valve CLOSED (normal)
            "40004": 1,    # P205 chemical pump ON
            "40005": 1,    # P301 UF pump ON
            "40006": 800,  # T301 level 800mm (safe: 500-1200)
            "40007": 1,    # PRV1 relief valve OPEN (must stay open)
            "40008": 1,    # P501 RO pump ON
            "40009": 750,  # T601 level 750mm (safe: 500-1000)
            "40010": 1     # EV2 emergency shutoff OPEN
        }

        self.violation_log = []
        self.load_asset_map()
        log.info("Digital Twin initialized with %d assets", len(self.assets))

    def load_asset_map(self):
        with open(self.asset_map_path) as f:
            data = json.load(f)
        self.assets = data['physical_assets']
        self.attack_scenarios = data['attack_scenarios']
        self.pipe_connections = data['pipe_connections']
        log.info("Asset map loaded: %d components", len(self.assets))

    def apply_command(self, register, new_value):
        """
        Core function — apply a Modbus write command to the twin.
        Returns a full impact report.
        """
        register = str(register)

        if register not in self.assets:
            return {
                'status': 'UNKNOWN_REGISTER',
                'register': register,
                'message': f'Register {register} not in asset map'
            }

        asset = self.assets[register]
        old_value = self.state.get(register)
        self.state[register] = new_value

        # Check if value is outside safe range
        safe_min = asset['safe_range']['min']
        safe_max = asset['safe_range']['max']
        is_violation = not (safe_min <= new_value <= safe_max)

        # Calculate time to physical damage
        time_to_failure = asset['time_to_failure_seconds'] if is_violation else None

        # Build impact report
        report = {
            'timestamp': datetime.now().isoformat(),
            'register': register,
            'component': asset['component'],
            'stage': asset['stage'],
            'coordinates': asset['coordinates'],
            'old_value': old_value,
            'new_value': new_value,
            'safe_range': asset['safe_range'],
            'is_violation': is_violation,
            'severity': asset['severity'] if is_violation else 'NONE',
            'affected_components': asset['affects'],
            'failure_consequence': asset['failure_consequence'] if is_violation else None,
            'attack_effect': asset['attack_effect'] if is_violation else None,
            'time_to_failure_seconds': time_to_failure,
            'remediation': asset['remediation'] if is_violation else None
        }

        if is_violation:
            self.violation_log.append(report)
            log.critical(
                "PHYSICAL VIOLATION: %s at (%.1f, %.1f) — %s — damage in %ds",
                asset['component'],
                asset['coordinates']['x'],
                asset['coordinates']['y'],
                asset['failure_consequence'],
                time_to_failure
            )

        return report

    def check_compound_attack(self):
        """
        Check if current state matches a known multi-register attack pattern.
        e.g. PRV1 closed AND P501 running = pipe rupture attack
        """
        detections = []

        for scenario_id, scenario in self.attack_scenarios.items():
            matched_registers = []

            for reg in scenario['target_registers']:
                current_val = self.state.get(reg)
                if current_val is not None:
                    asset = self.assets[reg]
                    safe_min = asset['safe_range']['min']
                    safe_max = asset['safe_range']['max']
                    if not (safe_min <= current_val <= safe_max):
                        matched_registers.append(reg)

            if matched_registers:
                detections.append({
                    'scenario': scenario_id,
                    'name': scenario['name'],
                    'description': scenario['description'],
                    'matched_registers': matched_registers,
                    'detection_window_seconds': scenario['detection_window_seconds'],
                    'confidence': len(matched_registers) / len(scenario['target_registers'])
                })

        return detections

    def get_current_state_report(self):
        """Full snapshot of plant state — written to twin_report.json every cycle"""
        violations = []
        for reg, val in self.state.items():
            if reg in self.assets:
                asset = self.assets[reg]
                safe_min = asset['safe_range']['min']
                safe_max = asset['safe_range']['max']
                if not (safe_min <= val <= safe_max):
                    violations.append({
                        'register': reg,
                        'component': asset['component'],
                        'coordinates': asset['coordinates'],
                        'severity': asset['severity'],
                        'time_to_failure_seconds': asset['time_to_failure_seconds']
                    })

        compound = self.check_compound_attack()

        return {
            'timestamp': datetime.now().isoformat(),
            'plant_state': self.state.copy(),
            'active_violations': violations,
            'compound_attack_detections': compound,
            'overall_status': 'ATTACK' if violations else 'NORMAL',
            'most_critical': min(
                violations,
                key=lambda x: x['time_to_failure_seconds']
            ) if violations else None
        }


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')

    twin = DigitalTwin()

    print("\n--- Test 1: Normal command (pump on) ---")
    r = twin.apply_command('40001', 1)
    print(f"Violation: {r['is_violation']} | Severity: {r['severity']}")

    print("\n--- Test 2: Oldsmar attack (chemical valve forced open) ---")
    r = twin.apply_command('40003', 9999)
    print(f"Violation: {r['is_violation']}")
    print(f"Component: {r['component']} at {r['coordinates']}")
    print(f"Consequence: {r['failure_consequence']}")
    print(f"Time to damage: {r['time_to_failure_seconds']} seconds")
    print(f"Remediation: {r['remediation']}")

    print("\n--- Test 3: Pressure attack (close relief valve) ---")
    r = twin.apply_command('40007', 0)
    print(f"Violation: {r['is_violation']}")
    print(f"Time to pipe rupture: {r['time_to_failure_seconds']} seconds")

    print("\n--- Test 4: Compound attack detection ---")
    detections = twin.check_compound_attack()
    for d in detections:
        print(f"DETECTED: {d['name']} — confidence {d['confidence']*100:.0f}%")

    print("\n--- Full state report ---")
    report = twin.get_current_state_report()
    print(f"Overall status: {report['overall_status']}")
    print(f"Active violations: {len(report['active_violations'])}")
    if report['most_critical']:
        print(f"Most critical: {report['most_critical']['component']} "
              f"— {report['most_critical']['time_to_failure_seconds']}s to damage")
