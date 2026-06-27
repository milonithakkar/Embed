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

GOLDEN_PCR_FILE  = '/opt/trustgate/golden_pcr_values.txt'
KNOWN_HASHES_FILE = '/opt/trustgate/known_good_hashes_ima.txt'
REPORT_FILE = '/tmp/tpm_report.json'
CHECK_INTERVAL = 60
CRITICAL_PCRS = ['0', '4', '7', '8', '9']
EXCLUDE_FILES = [
    '/usr/bin/tpm2',
    '/usr/bin/sh',
    '/usr/bin/dash',
    '/usr/bin/mount',
    '/usr/bin/umount',
    '/usr/bin/plymouth',
    '/usr/sbin/blkid',
    '/usr/sbin/fsck',
    '/usr/sbin/e2fsck',
    '/usr/sbin/lvm',
    '/usr/bin/kmod',
    '/usr/bin/udevadm',
    '/usr/bin/systemd-detect-virt',
]




def verify_from_package_db(filename):
    """Verify file using debsums - handle /bin vs /usr/bin paths"""
    import os
    
    try:
        # Try multiple path variants (/bin vs /usr/bin)
        paths_to_try = [filename]
        
        if filename.startswith('/usr/bin/'):
            paths_to_try.append(filename.replace('/usr/bin/', '/bin/'))
        elif filename.startswith('/bin/'):
            paths_to_try.append(filename.replace('/bin/', '/usr/bin/'))
        elif filename.startswith('/usr/sbin/'):
            paths_to_try.append(filename.replace('/usr/sbin/', '/sbin/'))
        elif filename.startswith('/sbin/'):
            paths_to_try.append(filename.replace('/sbin/', '/usr/sbin/'))
        
        package = None
        found_path = None
        
        # Try all path variants
        for try_path in paths_to_try:
            result = subprocess.run(
                ['dpkg', '-S', try_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if ':' in output:
                    package = output.split(':')[0].strip()
                    found_path = try_path
                    log.debug(f'{filename} found in dpkg as {found_path}, package: {package}')
                    break
        
        if not package:
            log.debug(f'{filename} not owned by any package (tried {paths_to_try})')
            return None
        
        # Handle diversions
        if 'diversion' in package.lower():
            parts = output.split()
            if len(parts) >= 3:
                package = parts[2]
        
        # Verify package integrity with debsums
        result = subprocess.run(
            ['debsums', '-s', package],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            log.info(f'{filename} verified OK via package {package}')
            return "PACKAGE_OK"
        else:
            log.error(f'{filename} FAILED debsums check in package {package}')
            return "PACKAGE_TAMPERED"
            
    except subprocess.TimeoutExpired:
        log.error(f'Timeout verifying {filename}')
        return None
    except FileNotFoundError as e:
        log.warning(f'Command not found: {e.filename}')
        return None
    except Exception as e:
        log.error(f'Package verification error for {filename}: {e}')
        import traceback
        log.error(traceback.format_exc())
        return None


def log_unknown_for_approval(filename, file_hash):
    """Log unknown binary for manual review"""
    approval_queue = '/opt/trustgate/pending_approval.txt'
    try:
        if os.path.exists(approval_queue):
            with open(approval_queue) as f:
                if filename in f.read():
                    return

        with open(approval_queue, 'a') as f:
            f.write(f'{datetime.now().isoformat()} | {filename} | {file_hash}\n')

        log.warning(f'PENDING APPROVAL: {filename}')
        log.warning(f'To approve: sudo /opt/trustgate/approve_binary.sh {filename}')

    except Exception as e:
        log.error(f'Failed to log for approval: {e}')





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
					if len(parts) >=2:
						pcr_num = parts[0].strip()
						pcr_val = parts[1].strip()
						golden[pcr_num] = pcr_val
	except Exception as e:
		log.error(f'Failed to load golden PCRs: {e}')
	return golden

def load_known_hashes():
	known = {}
	try:
		with open(KNOWN_HASHES_FILE) as f:
			for line in f:
				parts = line.strip().split(None,1)
				if len(parts)==2:
					known[parts[1]] = parts[0]
		log.info(f'Loaded {len(known)} known good hashes')
	
	except Exception as e:
		log.error(f'Failed to load hashes: {e}')
	return known

def check_pcr_integrity(golden, current):
	violations = []
	pcr_descriptions = {
		'0': 'BIOS firmware tampered - possible firmware rootkit',
		'4': 'Bootloader tampered - possible bootkit',
		'7': 'Secure Boot state changed',
		'8': 'GRUB configuration modified',
		'9': 'Kernel image tampered - CRITICAL'
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
def check_ima_integrity(known_hashes):
        violations = []
        last_measurement = {}
        try:
                with open('/sys/kernel/security/ima/ascii_runtime_measurements') as f:
  	                for line in f:
       	                        parts = line.strip().split()
       	                        if len(parts) >= 5:
                                        filename = parts[4]  # ← NEW: assign to variable
                                        # Skip files that no longer exist on disk (deleted files)
                                        if not os.path.exists(filename):
                                                log.debug(f'Skipping deleted file: {filename}')
                                                continue
                                        last_measurement[filename] = parts[3] 

        except Exception as e:
                log.error(f'IMA read failed: {e}')
                return violations

        for filename, file_hash in last_measurement.items():
                if filename.startswith(('/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/')):
                        if filename in EXCLUDE_FILES:
                                continue

                        # Strip sha256: prefix for comparison
                        clean_hash = file_hash.replace('sha256:', '')

                        if filename in known_hashes:
                                if clean_hash != known_hashes[filename]:
                                        violations.append({
                                                'type': 'IMA_VIOLATION',
                                                'severity': 'HIGH',
                                                'file': filename,
                                                'description': f'Modified binary detected: {filename}'
                                        })
                        else:
                                pkg_status = verify_from_package_db(filename)

                                if pkg_status == "PACKAGE_OK":
                                        log.info(f'{filename} verified by package manager')
                                        log_unknown_for_approval(filename, clean_hash)
                                        violations.append({
                                                'type': 'UNKNOWN_BINARY',
                                                'severity': 'MEDIUM',
                                                'file': filename,
                                                'description': f'New package binary (verified): {filename} - pending approval'
                                        })

                                elif pkg_status == "PACKAGE_TAMPERED":
                                        violations.append({
                                                'type': 'PACKAGE_TAMPERED',
                                                'severity': 'CRITICAL',
                                                'file': filename,
                                                'description': f'Package tampered: {filename}'
                                        })

                                else:
                                        log_unknown_for_approval(filename, clean_hash)
                                        violations.append({
                                                'type': 'UNKNOWN_BINARY',
                                                'severity': 'CRITICAL',
                                                'file': filename,
                                                'description': f'Unknown binary (not in package DB): {filename}'
                                        })
        return violations
def check_ima_integrity(known_hashes):

	violations=[]
	last_measurement = {}
	try:
		with open('/sys/kernel/security/ima/ascii_runtime_measurements') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) >= 5:
					last_measurement[parts[4]] = parts[3]

	except Exception as e:
		log.error(f'IMA read failed: {e}')
		return violations
	for filename, file_hash in last_measurement.items():
		if filename.startswith(('/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/')):
			if filename in EXCLUDE_FILES:
				continue
			if filename in known_hashes:
				ima_hash= file_hash.replace('sha256:', '')
				if ima_hash != known_hashes[filename]:
					violations.append({
						'type': 'IMA_VIOLATION',
						'severity':'HIGH',
						'file':filename,
						'description':f'Modified binary detected: {filename}'
					})
			else:
                            # Unknown binary - check package manager
                            pkg_status = verify_from_package_db(filename)
                            
                            if pkg_status == "PACKAGE_OK":
                                # Verified by debsums - safe but needs approval
                                log.info(f'{filename} verified by package manager')
                                log_unknown_for_approval(filename, file_hash)
                                violations.append({
                                    'type': 'UNKNOWN_BINARY',
                                    'severity': 'MEDIUM',
                                    'file': filename,
                                    'description': f'New package binary (verified): {filename} - pending approval'
                                })
                                
                            elif pkg_status == "PACKAGE_TAMPERED":
                                # Package checksum failed - CRITICAL
                                violations.append({
                                    'type': 'PACKAGE_TAMPERED',
                                    'severity': 'CRITICAL',
                                    'file': filename,
                                    'description': f'Package tampered: {filename}'
                                })
                                
                            else:
                                # Not from package manager - CRITICAL
                                log_unknown_for_approval(filename, file_hash)
                                violations.append({
                                    'type': 'UNKNOWN_BINARY',
                                    'severity': 'CRITICAL',
                                    'file': filename,
                                    'description': f'Unknown binary (not in package DB): {filename}'
                                })
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
	known_hashes = load_known_hashes()	
	while True:
		try:
			current_pcrs = read_pcrs()
			pcr_violations = check_pcr_integrity(golden, current_pcrs)
			ima_violations = check_ima_integrity(known_hashes)
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
				log.critical(f'INTEGRITY VIOLATION: {len(all_violations)} issues found')
				for v in all_violations:
					log.critical(f" -> [{v['severity']}] {v['description']}")
				trigger_led_alert()

			else:
				log.info('System integrity: VERIFIED')
		
		except Exception as e:
			log.error(f'Monitor error: {e}')
		
		time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
	run()
 
