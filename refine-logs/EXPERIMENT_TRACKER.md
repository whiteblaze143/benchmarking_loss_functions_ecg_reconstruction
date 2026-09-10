# Experiment Tracker: LCT-MTL Multi-Institutional Validation

| Run ID | Milestone | Purpose | System / Variant | Split / Cohort | Metrics | Priority | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R001** | M0 | AWS S3 Permission Check | CLI Probe | `bdsp-credentialed-ac` | STS / S3 LS status | MUST | PENDING_IAM | Waiting for S3 policy attachment in console |
| **R002** | M0 | ArXiv Preprint Ingestion | `curl` fetch | `arXiv:2410.04133` | PDF Integrity (34 pgs) | MUST | COMPLETED | Stored in `/data/mithunmanivannan/papers/` |
| **R003** | M0 | HEEDB Census Extraction | `heedb_mgh_euh_census.py` | MGH & EUH Cohorts | 10-point census table | MUST | COMPLETED | Report & JSON written to `/data/mithunmanivannan/manifests/` |
| **R004** | M1 | PTB-XL Test Benchmark | LCT-MTL-384 | PTB-XL Fold 10 | $r$, $P_{05}$, RMSE | MUST | READY | Baseline champion validation |
| **R005** | M2 | Tabular Metadata Sync | `sync_heedb_metadata.sh` | BDSP S3 Metadata | CSV sync status | MUST | QUEUED | Zero-waveform download safety gate |
| **R006** | M2 | Manifest Export | Manifest Generator | MGH Paired AF Cohort | $N=70,900$ pairs | MUST | QUEUED | $7\text{--}45$d, $60\text{--}120$d, $150\text{--}210$d windows |
| **R007** | M2 | Manifest Export | Manifest Generator | EUH Paired AF Cohort | $N=49,250$ pairs | MUST | QUEUED | $7\text{--}45$d, $60\text{--}120$d, $150\text{--}210$d windows |
| **R008** | M3 | Targeted Waveform Download | Parallel Downloader | MGH & EUH Paired Set | Total GB $\le 21\,\text{GB}$ | MUST | QUEUED | Consumes only $4.1\%$ of NFS quota |
| **R009** | M4 | AF&rarr;AF Preservation Eval | LCT-MTL-384 | Combined AF&rarr;AF ($N=72,100$) | Fibrillatory Band Power | MUST | QUEUED | Test absence of pseudo-sinus regression |
| **R010** | M4 | AF&rarr;SR Recovery Eval | LCT-MTL-384 | Combined AF&rarr;SR ($N=48,050$) | P-wave Amp, PR Interval | MUST | QUEUED | Test clean emergence without AF hallucination |
| **R011** | M4 | Cross-Site Generalization | LCT-MTL-384 | MGH vs EUH Breakdown | Inter-site $\Delta r$, AUROC | MUST | QUEUED | Boston vs Atlanta external validation |
