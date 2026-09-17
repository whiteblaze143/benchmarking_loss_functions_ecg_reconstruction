# Paper 08 quick inductive-bias checks

These tests isolate architectural claims before any clinical training run. They
are engineering and synthetic falsification checks, not scientific results.

They verify:

1. local cyclic attention has no CLS-mediated global bypass;
2. static routing is input-independent while retaining matched Q/K machinery;
3. phase-agnostic attention is permutation invariant;
4. dynamic routing changes with the record;
5. phase scrambling is deterministic per record and does not shuffle labels;
6. the current equivalence-token production path is still absent and must not
   be inferred from attention-only tests.

Run from the experiment root:

```bash
PYTHONPATH=src /home/mithunmanivannan/.venv/bin/python -m pytest -q \
  scripts/paper08/inductive_bias/test_quick_biases.py
```

The read-only fold-8 representation audit writes its explicitly non-scientific
diagnostic record alongside these tests:

```bash
PYTHONPATH=src /home/mithunmanivannan/.venv/bin/python \
  scripts/paper08/inductive_bias/audit_quick_biases.py \
  --representations /data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper02_kernel_mean/development_representations \
  --output scripts/paper08/inductive_bias/quick_bias_audit.json
```

`pilot_artifact/` is an end-to-end 256-record engineering pilot. Its manifest
must remain `status: pilot_only` with `audit.passed: false`; it cannot satisfy
the production trainer's artifact gate.
