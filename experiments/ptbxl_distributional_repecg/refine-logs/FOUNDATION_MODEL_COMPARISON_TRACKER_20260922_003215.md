# Foundation-Model Comparison Tracker

| Run ID | Milestone | System / variant | Task | Priority | Status | Notes |
|---|---|---|---|---|---|---|
| FM-00 | Admission | ECGFounder 1L + 12L | Archive checksum and strict model state load | MUST | QUEUED | New immutable artifact directory only. |
| FM-01 | Main | ECGFounder 12L | EchoNext 5,442 records, 12 endpoints, KFold seed 42 | MUST | QUEUED | Full-12 context; not a subset claim. |
| FM-02 | Main | ECGFounder 1L / Lead I | Same EchoNext probe | MUST | QUEUED | True one-channel author architecture; no padding. |
| FM-03 | Admission | ECG-FM archived artifact | Payload/provenance audit | MUST | QUEUED | Audit only, never an evaluation. |
| FM-04 | Admission | HuBERT-ECG large | Pinned official snapshot acquisition | MUST | QUEUED | Evaluation requires a later extractor gate. |
| FM-05 | Excluded | CSFM, CLEF, multi-lead subsets | N/A | N/A | EXCLUDED | No real eligible artifact/interface. |

**Dependency:** strict Braid completion marker, no related GPU process, and zero GPU compute occupancy.
