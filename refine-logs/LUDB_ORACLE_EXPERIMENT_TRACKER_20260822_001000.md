# LUDB Oracle Experiment Tracker

| Run ID | Purpose | Scope | Priority | Status | Evidence |
|---|---|---|---|---|---|
| ORACLE-DATA-01 | Complete LUDB inventory | 200 records / 2,400 streams | MUST | PASS | record 1 restored from PhysioNet 1.0.1 and WFDB-readable |
| ORACLE-METRIC-01 | Exact-index landmark fidelity | all mapped P/QRS/T landmarks | MUST | IN PROGRESS | implementation and deterministic tests |
| ORACLE-METRIC-02 | Wave area/amplitude/shape | complete labeled intervals | MUST | IN PROGRESS | implementation and deterministic tests |
| ORACLE-METRIC-03 | ST/J fidelity | QRS offset through following T onset | MUST | IN PROGRESS | implementation and deterministic tests |
| ORACLE-SMOKE-01 | One-record/model smoke | primary and control lead roles | MUST | TODO | database row/value audit |
| ORACLE-FULL-01 | First full checkpoint | 200 LUDB records | MUST | TODO | integrity, row counts, storage, paired summaries |
| ORACLE-DAEMON-01 | Resumable CPU daemon | all eligible ECG-AIM checkpoints | MUST | TODO | detached tmux and live DB audit |
| DETECTOR-SECONDARY-01 | Automatic findability | frozen external segmenter, 150 ms | NICE | TODO | independent project; NeuroKit not accepted as reference |

