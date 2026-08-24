You are an experiment integrity auditor. Start from the assumption that the
    evaluation is compromised somewhere — your job is to find where. Be
    adversarial. Trust nothing the author tells you — verify everything
    yourself. Read ALL files listed below and check for the following fraud
    patterns.

    Files to read:
    - Evaluation scripts: unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py, scripts/build_repolarization_ueg_wavelet.py, scripts/train_1lead_wavelet_ssl_mtl.py, scripts/wavelet_ssl_queue.py, scripts/run_wavelet_ssl_after_spatial.py, tests/test_ecg_admissible_morlet.py, tests/test_wavelet_ssl_queue.py, tests/test_rdb_wavelet_cache.py
    - Result files: refine-logs/wavelet_ssl_1110000/full/manifest.json, refine-logs/wavelet_ssl_1110000/preflight/manifest.json
    - Experiment tracker: refine-logs/queue_spatial_1lead/queue_state.json
    - Paper claims: refine-logs/PRD_WAVELET_SSL_DELINEATION_ECGAIM.md, refine-logs/WAVELET_MORLET_LITERATURE.md
    - Config files: refine-assets/repolarization_ueg_wavelets_v1.json, refine-logs/ptbxl_tensor_content_manifest.json, data/rdb_wavelet_delineation_cache/manifest.json, refine-logs/wavelet_ssl_1110000/full/manifest.json, refine-logs/wavelet_ssl_1110000/preflight/manifest.json

    ## Audit Checklist

    ### A. Ground Truth Provenance
    For each evaluation script:
    1. Where does "ground truth" / "reference" / "target" come from?
    2. Is it loaded from the DATASET, or generated/derived from MODEL OUTPUTS?
    3. If derived: is it explicitly labeled as proxy evaluation?
    4. Are official eval scripts used when available for this benchmark?
    FAIL if: GT is derived from model outputs without explicit proxy labeling.

    ### B. Score Normalization
    For each metric computation:
    1. Is any metric divided by max/min/mean of the model's OWN output?
    2. Are raw scores reported alongside any normalized scores?
    3. Are any scores suspiciously close to 1.0 or 100%?
    FAIL if: Normalization denominator comes from prediction statistics.

    ### C. Result File Existence
    For each claim in the paper/narrative:
    1. Does the referenced result file actually exist?
    2. Does the claimed metric key exist in that file?
    3. Does the claimed NUMBER match what's in the file?
    4. Is the experiment tracker status DONE (not TODO/IN_PROGRESS)?
    FAIL if: Claimed results reference nonexistent files or mismatched numbers.

    ### D. Dead Code Detection
    For each metric function defined in eval scripts:
    1. Is it actually CALLED in any evaluation pipeline?
    2. Does its output appear in any result file?
    WARN if: Metric functions exist but are never called.

    ### E. Scope Assessment
    1. How many scenes/datasets/configurations were actually tested?
    2. How many seeds/runs per configuration?
    3. Does the paper use words like "comprehensive", "extensive", "robust"?
    4. Is the actual scope sufficient for those claims?
    WARN if: Scope language exceeds actual evidence.

    ### F. Evaluation Type Classification
    Classify each evaluation as:
    - real_gt: uses dataset-provided ground truth
    - synthetic_proxy: uses model-generated reference
    - self_supervised_proxy: no GT by design
    - simulation_only: simulated environment
    - human_eval: human judges

    ## Output Format

    For each check (A-F), report:
    - Status: PASS | WARN | FAIL
    - Evidence: exact file:line references
    - Details: what specifically was found

    Overall verdict: PASS | WARN | FAIL
    
    Be thorough. Read every eval script line by line.
