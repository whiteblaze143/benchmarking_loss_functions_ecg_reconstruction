"""One-shot gated handoff to already frozen full-cohort development training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction')
W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
PYTHON = '/home/mithunmanivannan/.venv/bin/python'
O = W/'set_models_v1/queue_controller'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_status(value):
    temporary = O/'status.json.partial';temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(O/'status.json')


def main(feature_pid):
    O.mkdir(exist_ok=False)
    protocol = W/'set_models_v1/protocol.json';phase_protocol = W/'phase_features_v1/protocol.json'
    receipt = {'controller_source_sha256': sha(__file__), 'pid': os.getpid(), 'feature_pid': feature_pid,
        'training_protocol_sha256': sha(protocol), 'phase_protocol_sha256': sha(phase_protocol),
        'kernel_receipt_sha256': sha(W/'phase_kernel_v1/receipt.json'),
        'environment_spec_sha256': sha(REPO/'.aris/compute/af-set-gpu-env-spec.json'),
        'mode_order': ['categorical', 'q_ablated', 'hybrid'],
        'scope': 'No test inference; no retries, caps, hyperparameter changes or artifact overwrite. Wait for identified feature process, complete cache, full independent serialization check and free GPU before training.'}
    (O/'launch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic();completed = []
    try:
        while True:
            command_path = Path(f'/proc/{feature_pid}/cmdline')
            command = command_path.read_bytes() if command_path.exists() else b''
            status = json.loads((W/'phase_features_v1/status.json').read_text())
            if status['status'] == 'failed': raise RuntimeError('Phase producer failed; preserve attempt')
            if b'generate_phase_features.py' not in command:
                assert status.get('complete') and status['files'] == 289 and status['windows'] == 802491, 'Producer stopped without complete cache'
                break
            assert b'--run' in command, 'Unexpected producer process'
            save_status({'status': 'waiting_for_phase_cache', 'feature_pid': feature_pid,
                         'verified_live_now': True, 'elapsed_seconds': time.monotonic()-started})
            time.sleep(10)
        assert sha(protocol) == receipt['training_protocol_sha256'] and sha(phase_protocol) == receipt['phase_protocol_sha256']
        assert sha(W/'phase_kernel_v1/receipt.json') == receipt['kernel_receipt_sha256']
        environment = os.environ.copy()
        for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: environment[name] = '1'
        library_root = Path('/home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia')
        libraries = ':'.join(str(p) for p in sorted(library_root.glob('*/lib')))
        environment['LD_LIBRARY_PATH'] = '/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:'+libraries
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        save_status({'status': 'independent_full_cache_verification'})
        verification = subprocess.run([PYTHON, 'experiments/af_density/verify_phase_cache_snapshot.py'],
            cwd=REPO, env=environment, capture_output=True, text=True, check=True)
        (O/'full_cache_verification.log').write_text(verification.stdout+verification.stderr)
        checked = json.loads(verification.stdout.strip().splitlines()[-1])
        assert checked['all_pass'] and checked['checked_files'] == 289 and checked['checked_windows'] == 802491
        for mode in receipt['mode_order']:
            assert not (W/'set_models_v1'/mode).exists(), 'Prior model attempt requires inspection, not overwrite'
            free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, check=True).stdout.strip().splitlines()
            assert len(free) == 1 and int(free[0]) >= 14*1024+512, 'Insufficient free GPU; stop without retry/cap'
            command = [PYTHON, 'experiments/af_density/train_set_waveform.py', '--train', mode]
            save_status({'status': 'training', 'mode': mode, 'command': command, 'completed_modes': completed})
            with (O/(mode+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                (O/(mode+'_launch.json')).write_text(json.dumps({'pid': child.pid, 'command': command,
                    'training_protocol_sha256': sha(protocol)}, indent=2)+'\n')
                code = child.wait()
            assert code == 0, f'{mode} exited {code}; no retry'
            lock_path = W/'set_models_v1'/mode/'evaluation_lock.json';lock = json.loads(lock_path.read_text())
            assert lock['complete_train_validation'] and not lock['test_inference_performed']
            completed.append({'mode': mode, 'evaluation_lock_sha256': sha(lock_path)})
        save_status({'status': 'complete', 'complete': True, 'completed_modes': completed,
                     'seconds': time.monotonic()-started, 'test_inference_performed': False})
    except Exception as error:
        save_status({'status': 'failed', 'error': str(error), 'completed_modes': completed})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--feature-pid', type=int, required=True)
    main(parser.parse_args().feature_pid)
