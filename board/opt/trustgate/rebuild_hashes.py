known={}
with open('/sys/kernel/security/ima/ascii_runtime_measurements') as f:
	for line in f:
		parts = line.strip().split()
		if len(parts)>=5:
			file_hash=parts[3]
			filename=parts[4]
			known[filename]=file_hash
with open('/opt/trustgate/known_good_hashes_ima.txt','w') as f:
	for filename, filehash in known.items():
		f.write(f'{filehash} {filename}\n')

print(f'Captured {len(known)} IMA-format hashes')
