import subprocess
import json
import time
import logging
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/opt/trustgate/tpm_monitor.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('TPMMonitor')

GOLDEN_PCR_FILE = '/opt/trustgate/golden_pcr_values.txt'
KNOWN_HASHES_FILE = '/opt/trustgate/known_good_hashes.txt'
REPORT_FILE = '/tmp/tpm_report.json'
CHECK_INTERVAL = 60
CRITICAL_PCRS = ['0', '4', '7', '8', '9']

def read_pcrs():
    result = subprocess.run(
        ['tpm2_pcrread', 'sha256'],
        capture_output=True, text=True
    )
    pcrs = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        if '0x' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                pcr_num = parts[0].strip()
                pcr_val = parts[1].strip()
                pcrs[pcr_num] = pcr_val
    return pcrs

def load_golden_pcrs():
    golden = {}
    try:
        with open(GOLDEN_PCR_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if '0x' in line and ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        pcr_num = parts[0].strip()
                        pcr_val = parts[1].strip()
                        golden[pcr_num] = pcr_val
    except Exception as e:
        log.error(f'Failed to load golden PCRs: {e}')
    return golden

def check_pcr_integrity(golden, current):
    violations = []
    pcr_descriptions = {
        '0': 'BIOS firmware tampered — possible firmware rootkit',
        '4': 'Bootloader tampered — possible bootkit',
        '7': 'Secure Boot state changed',
        '8': 'GRUB configuration modified',
        '9': 'Kernel image tampered — CRITICAL'
    }
    for pcr in CRITICAL_PCRS:
        if pcr in golden and pcr in current:
            if golden[pcr] != current[pcr]:
                violations.append({
                    'type': 'PCR_VIOLATION',
                    'pcr': pcr,
                    'severity': 'CRITICAL',
                    'expected': golden[pcr],
                    'actual': current[pcr],
                    'description': pcr_descriptions.get(pcr, f'PCR {pcr} changed')
                })
    return violations

def load_known_hashes():
    known = {}
    try:
        with open(KNOWN_HASHES_FILE) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    known[parts[1]] = parts[0]
        log.info(f'Loaded {len(known)} known good hashes')
    except Exception as e:
        log.error(f'Failed to load hashes: {e}')
    return known

def check_ima_integrity(known_hashes):
    violations = []
    try:
        with open('/sys/kernel/security/ima/ascii_runtime_measurements') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    file_hash = parts[3]
                    filename = parts[4]
                    if filename.startswith(('/bin/', '/sbin/',
                                           '/usr/bin/', '/usr/sbin/')):
                        if filename in known_hashes:
                            expected = f'sha256:{known_hashes[filename]}'
                            if file_hash != expected:
                                violations.append({
                                    'type': 'IMA_VIOLATION',
                                    'severity': 'HIGH',
                                    'file': filename,
                                    'description': f'Modified binary detected: {filename}'
                                })
                        else:
                            violations.append({
                                'type': 'UNKNOWN_BINARY',
                                'severity': 'MEDIUM',
                                'file': filename,
                                'description': f'Unknown binary executed: {filename}'
                            })
    except Exception as e:
        log.error(f'IMA check failed: {e}')
    return violations
  
def trigger_led_alert():
    try:
        gpio_path = '/sys/class/gpio/gpio3/value'
        if os.path.exists(gpio_path):
            for _ in range(5):
                with open(gpio_path, 'w') as f:
                    f.write('1')
                time.sleep(0.3)
                with open(gpio_path, 'w') as f:
                    f.write('0')
                time.sleep(0.3)
    except Exception as e:
        log.error(f'LED alert failed: {e}')

def save_report(report):
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)

def run():
    log.info('TrustGate TPM Monitor starting...')
    golden = load_golden_pcrs()
    log.info(f'Loaded {len(golden)} golden PCR values')
    
    # Load once at startup — not every 60 seconds
    known_hashes = load_known_hashes()

    while True:
        try:
            current_pcrs = read_pcrs()
            pcr_violations = check_pcr_integrity(golden, current_pcrs)
            ima_violations = check_ima_integrity(known_hashes)  # pass in
            all_violations = pcr_violations + ima_violations

            report = {
                'timestamp': datetime.now().isoformat(),
                'system_trusted': len(all_violations) == 0,
                'violation_count': len(all_violations),
                'violations': all_violations,
                'pcr_snapshot': current_pcrs
            }

            save_report(report)

            if all_violations:
                log.critical(
                    f'INTEGRITY VIOLATION: {len(all_violations)} issues found'
                )
                for v in all_violations:
                    log.critical(
                        f"  → [{v['severity']}] {v['description']}"
                    )
                trigger_led_alert()
            else:
                log.info('System integrity: VERIFIED ✅')

        except Exception as e:
            log.error(f'Monitor error: {e}')

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    run()
