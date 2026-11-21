"""Stream the pinned public LTAFDB waveforms and verify publisher SHA256s."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request

P = Path('/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference')
BASE = 'https://physionet.org/files/ltafdb/1.0.0/'


def main():
    expected = {}
    for line in (P/'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1);expected[name.lstrip('*')] = digest
    records = (P/'RECORDS').read_bytes()
    assert hashlib.sha256(records).hexdigest() == expected['RECORDS']
    names = [rid+'.dat' for rid in records.decode().split()]
    assert len(names) == 84 and all(name in expected for name in names)

    def download(name):
        target = P/name;partial = P/(name+'.partial')
        assert not partial.exists(), 'Inspect previous partial download before retry'
        digest = hashlib.sha256();size = 0
        if target.exists():
            with target.open('rb') as source:
                while chunk := source.read(1024*1024): digest.update(chunk);size += len(chunk)
        else:
            with urllib.request.urlopen(BASE+name, timeout=60) as source, partial.open('xb') as output:
                while chunk := source.read(1024*1024):
                    output.write(chunk);digest.update(chunk);size += len(chunk)
                output.flush();os.fsync(output.fileno())
            assert digest.hexdigest() == expected[name], name
            partial.rename(target)
        assert digest.hexdigest() == expected[name], name
        print(name, size, 'SHA256 verified', flush=True)
        return {'file': name, 'bytes': size, 'sha256': digest.hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, names))
    receipt = {'complete': True, 'release': '1.0.0', 'source': BASE, 'files': files,
               'total_bytes': sum(f['bytes'] for f in files),
               'scope': 'Publisher-checksummed waveform download; clinical quality and model evaluation not established.'}
    (P/'waveform_download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('COMPLETE_LTAFDB_WAVEFORMS_VERIFIED', flush=True)


if __name__ == '__main__': main()
