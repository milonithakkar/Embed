# find_cip_fields.py
# Lists all CIP-related fields available in your tshark

import subprocess

TSHARK = r'D:\Program Files\Wireshark\tshark.exe'

print("Searching for CIP fields in your tshark version...")
print("="*60)

# Get all fields with 'cip' in the name
result = subprocess.run(
    [TSHARK, '-G', 'fields'],
    capture_output=True,
    text=True,
    encoding='utf-8'
)

if result.returncode != 0:
    print(f"ERROR: {result.stderr[:500]}")
    exit(1)

# Parse output — format is tab-separated
cip_fields = []
for line in result.stdout.split('\n'):
    parts = line.split('\t')
    if len(parts) >= 3:
        field_type = parts[0]
        field_name = parts[2]
        if field_name.startswith('cip.') or field_name == 'cip':
            cip_fields.append((field_name, parts[1]))

print(f"\nFound {len(cip_fields)} CIP-related fields")
print(f"\nKey fields we need:")
print(f"{'Field Name':<35} {'Description'}")
print(f"{'-'*35} {'-'*40}")

# Look for our specific fields
keywords = ['service', 'sc', 'path', 'tag', 'req', 'resp', 'data']
for field_name, desc in cip_fields:
    if any(kw in field_name.lower() for kw in keywords):
        print(f"{field_name:<35} {desc[:40]}")