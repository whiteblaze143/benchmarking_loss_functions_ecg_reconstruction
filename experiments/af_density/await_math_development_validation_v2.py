"""Validate each completed corrected model once; never enable held-out inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path('/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction')
W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
ROOT = W/'set_models_math_sdp_v2'
O = ROOT/'development_validation_queue'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def status(value):
    temporary = O/'status.json.partial';temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(O/'status.json')


def main():
    O.mkdir(exist_ok=False)
    training_pid = json.loads((ROOT/'queue_launch.json').read_text())['pid']
    names = ['await_math_development_validation_v2.py', 'verify_set_math_validation_v2.py',
        'evaluate_set_math_validation_timelines_v2.py', 'summarize_math_validation_v2.py',
        'compare_set_math_validation_endpoints_v2.py', 'paired_endpoint_comparison.py',
        'assess_joint_endpoint_support.py', 'report_validation_calibration_v2.py',
        'summarize_validation_timelines.py']
    sources = {str(REPO/'experiments/af_density'/name): sha(REPO/'experiments/af_density'/name) for name in names}
    inputs = {name: sha(W/name) for name in ['set_models_math_sdp_v2/protocol.json',
        'set_models_math_sdp_v2/validation_reader_contract.json', 'evaluation_protocol_v1/protocol.json',
        'evaluation_protocol_v1/math_sdp_technical_amendment_v1.json', 'evaluation_protocol_v1/decision_hierarchy_proposal_v1.json']}
    (O/'launch.json').write_text(json.dumps({'pid': os.getpid(), 'training_queue_pid': training_pid,
        'source_hashes': sources, 'input_hashes': inputs, 'test_inference': False,
        'scope': 'Wait for actual model locks; complete validation replay, frozen timelines, descriptive summaries and paired development comparisons only. No retry, model changes or test release.'}, indent=2)+'\n')
    environment = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: environment[key] = '1'
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    environment['LD_LIBRARY_PATH'] = '/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/runtime_cudnn93/nvidia/cudnn/lib:'+':'.join(str(p) for p in sorted(Path('/home/mithunmanivannan/.venv/lib/python3.12/site-packages/nvidia').glob('*/lib')))
    completed = []
    try:
        for mode in ['categorical', 'q_ablated', 'hybrid']:
            lock_path = ROOT/mode/'evaluation_lock.json'
            while not lock_path.exists():
                command = Path(f'/proc/{training_pid}/cmdline')
                assert command.exists() and b'run_math_sdp_development_queue.py' in command.read_bytes(), 'Training queue stopped before required model lock'
                queue = json.loads((ROOT/'queue/status.json').read_text())
                assert queue['status'] != 'failed', queue
                status({'status': 'waiting_for_completed_model', 'mode': mode, 'training_queue_pid': training_pid,
                        'training_queue_live_now': True, 'completed': completed})
                time.sleep(10)
            lock = json.loads(lock_path.read_text())
            assert lock['complete_train_validation'] and not lock['test_inference_performed']
            assert lock['protocol_sha256'] == inputs['set_models_math_sdp_v2/protocol.json']
            commands = [
                ['verify_set_math_validation_v2.py', '--mode', mode],
                ['evaluate_set_math_validation_timelines_v2.py', '--mode', mode],
                ['summarize_math_validation_v2.py', '--mode', mode],
                ['compare_set_math_validation_endpoints_v2.py', '--set-models', *completed, mode]]
            for step, args in enumerate(commands):
                for path, digest in sources.items(): assert sha(path) == digest
                for path, digest in inputs.items(): assert sha(W/path) == digest
                assert not (W/'evaluation_protocol_v1/heldout_release.json').exists()
                if step == 0:
                    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                    assert free >= 6*1024, 'Insufficient free memory for full replay; stop, no cap/retry'
                with (O/f'{mode}_step{step}.log').open('x') as stream:
                    child = subprocess.Popen(['/home/mithunmanivannan/.venv/bin/python', 'experiments/af_density/'+args[0], *args[1:]],
                        cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    status({'status': 'running_validation', 'mode': mode, 'step': args[0], 'child_pid': child.pid, 'completed': completed})
                    code = child.wait()
                assert code == 0, f'{mode}/{args[0]} exited{code}; preserve attempt, no retry'
            completed.append(mode)
        status({'status': 'complete', 'complete': True, 'completed': completed, 'test_inference': False})
    except BaseException as error:
        status({'status': 'failed', 'error': str(error), 'completed': completed, 'test_inference': False})
        raise


if __name__ == '__main__': main()
