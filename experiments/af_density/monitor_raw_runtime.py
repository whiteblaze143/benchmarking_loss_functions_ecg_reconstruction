"""Bounded live telemetry for the declared raw-model process; no restarts."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from waveform_protocol import W


def main():
    root = W/'raw_mask_resnet_v1'
    pid = json.loads((root/'controller_launch.json').read_text())['pid']
    status_file = Path(f'/proc/{pid}/status')
    output = root/'live_runtime.jsonl'
    assert not output.exists(), 'Preserve existing telemetry'
    with output.open('w', buffering=1) as stream:
        while status_file.exists():
            status = status_file.read_text()
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'train_raw_waveform.py --train' not in command or '\nState:\tZ' in status:
                break
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            process = subprocess.run(['ps', '-p', str(pid), '-o', 'pid=,etime=,pcpu=,rss=,nlwp=,stat='],
                                     capture_output=True, text=True, check=True).stdout.strip()
            record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'pid': pid,
                      'gpu_columns': 'timestamp,memory_used_MiB,memory_total_MiB,gpu_util_percent,memory_util_percent,power_W',
                      'gpu_csv': gpu, 'process_columns': 'pid,elapsed,cpu_percent,rss_KiB,threads,state',
                      'process_values': process}
            stream.write(json.dumps(record)+'\n')
            time.sleep(15)
    (root/'telemetry_completion.json').write_text(json.dumps({'pid': pid, 'status': 'declared_process_terminal_or_missing',
          'time_utc': datetime.now(timezone.utc).isoformat()})+'\n')


if __name__ == '__main__':
    main()
