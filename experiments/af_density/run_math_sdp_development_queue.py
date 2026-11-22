"""Execute the registered corrected family once, stopping on any failed mode."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path('/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction')
W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
ROOT = W/'set_models_math_sdp_v2'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    accepted = json.loads((ROOT/'witness_acceptance.json').read_text())
    assert accepted['all_pass'] and accepted['protocol_sha256'] == sha(ROOT/'protocol.json')
    assert not (W/'evaluation_protocol_v1/heldout_release.json').exists()
    completed_cache = json.loads((W/'phase_features_v1/status.json').read_text())
    assert completed_cache['complete'] and completed_cache['files'] == 289 and completed_cache['windows'] == 802491
    output = ROOT/'queue';output.mkdir(exist_ok=False)
    launch = {'pid': os.getpid(), 'source_sha256': sha(__file__),
        'protocol_sha256': sha(ROOT/'protocol.json'), 'acceptance_sha256': sha(ROOT/'witness_acceptance.json'),
        'environment_spec_sha256': sha(REPO/'.aris/compute/af-set-math-sdp-gpu-env-spec.json'),
        'modes': ['categorical', 'q_ablated', 'hybrid'], 'test_inference': False,
        'scope': 'One registered technical-correction attempt per mode; no retries, caps, architecture/optimizer/seed changes or selection between versions.'}
    (output/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    environment = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: environment[key] = '1'
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    environment['LD_LIBRARY_PATH'] = '/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:'+':'.join(str(p) for p in sorted(Path('/home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia').glob('*/lib')))
    completed = [];started = time.monotonic()
    try:
        for mode in launch['modes']:
            assert sha(ROOT/'protocol.json') == launch['protocol_sha256']
            assert not (ROOT/mode).exists()
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            assert free >= 14*1024+512, 'Insufficient GPU memory; stop without reducing cohort or retrying'
            command = ['/home/mithunmanivannan/.venv/bin/python', 'experiments/af_density/train_set_math_sdp_v2.py', mode]
            with (output/f'{mode}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                (output/'status.json').write_text(json.dumps({'status': 'running', 'mode': mode, 'child_pid': child.pid, 'completed': completed})+'\n')
                code = child.wait()
            assert code == 0, f'{mode} exited{code}; preserve failed attempt, no retry'
            lock = json.loads((ROOT/mode/'evaluation_lock.json').read_text())
            assert lock['complete_train_validation'] and not lock['test_inference_performed']
            assert lock['protocol_sha256'] == launch['protocol_sha256']
            completed.append({'mode': mode, 'lock_sha256': sha(ROOT/mode/'evaluation_lock.json')})
        (output/'status.json').write_text(json.dumps({'status': 'complete', 'complete': True,
            'completed': completed, 'seconds': time.monotonic()-started})+'\n')
    except BaseException as error:
        (output/'status.json').write_text(json.dumps({'status': 'failed', 'error': str(error), 'completed': completed})+'\n')
        raise


if __name__ == '__main__': main()
