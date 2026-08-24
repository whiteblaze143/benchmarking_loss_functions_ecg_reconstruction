# Wavelet SSL RDB queue runbook

The durable supervisor is tmux session `wavelet_ssl_after_spatial`. It holds the
three-architecture queue lock and leaves
`refine-logs/queue_3arch/WAVELET_PRIORITY_BARRIER.json` in place until the RDB
wavelet sweep has completed and passed validation. The order is therefore
spatial queue -> 10-job GPU preflight -> 94-job full sweep -> 3-architecture
queue. A crash or failed validation retains the barrier and fails closed.

## Observe

```bash
tmux has-session -t wavelet_ssl_after_spatial
tail -f refine-logs/wavelet_ssl_smokes/supervisor.log
cat refine-logs/wavelet_ssl_smokes/supervisor_state.json
cat refine-logs/wavelet_ssl_smokes/preflight/manifest.state.json
cat refine-logs/wavelet_ssl_smokes/full/manifest.state.json
sqlite3 refine-logs/wavelet_ssl_smokes/full/queue.sqlite \
  'select status,count(*) from jobs group by status;'
```

The SQLite databases are canonical; JSON state files are atomic monitoring
exports. Each successful run is revalidated against its immutable command,
code/data fingerprints, completion contract, and checkpoint inventory.

Start or restart the supervisor only when the named session is absent:

```bash
tmux has-session -t wavelet_ssl_after_spatial 2>/dev/null || \
tmux new-session -d -s wavelet_ssl_after_spatial \
  "bash -lc 'set -o pipefail; cd /home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction; source /home/mithunmanivannan/.venv/bin/activate; python scripts/run_wavelet_ssl_after_spatial.py 2>&1 | tee -a refine-logs/wavelet_ssl_smokes/supervisor.log'"
```

## Pause and resume

To stop before claiming wavelet work, create
`refine-logs/wavelet_ssl_smokes/STOP_SUPERVISOR`. To pause an active queue after
the current job, first create `STOP` in either the `preflight` or `full`
directory. Let the worker observe it at the next job boundary; do not kill the
child first. For an urgent stop, create `STOP` first and then send `SIGTERM` to
the supervisor, which forwards termination to the child process group and
records an interrupted queue state. The barrier remains present. After the
process has exited, remove the applicable stop file and use the exact guarded
tmux command above; SQLite reconciliation resumes only incomplete jobs from
their rolling checkpoint.

Do not delete the priority barrier merely to recover a failed wavelet job. A
deliberate decision to abandon this experiment and resume the three-architecture
queue requires separately confirming no wavelet worker is alive, then removing
the barrier and starting `scripts/run_3arch_queue.py`.

## Capacity and integrity gates

The supervisor waits for spatial terminal state, exact process quiescence, an
idle GPU, at least 8 GiB free disk, and at least 5 GiB available RAM. It verifies
the PTB-XL and RDB tensor content identities before GPU work. The representative
preflight must remain below 36 GiB peak allocation and must not increase total
volatile ECC errors. Any failure retains the barrier and records the reason in
`supervisor_state.json`.
