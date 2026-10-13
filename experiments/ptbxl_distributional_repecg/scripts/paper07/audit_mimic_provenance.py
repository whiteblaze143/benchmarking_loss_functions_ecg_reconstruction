#!/usr/bin/env python3
"""MIMIC-III-Ext-PPG Waveform and Representation Provenance Audit.

Computes exact filtering funnel counts across every stage:
  1. All Raw Metadata Rhythm Segments
  2. Filtered to SR and AF Rhythms
  3. Lead-II ECG Present in Record (ecg_avail)
  4. Signal Quality Threshold Passed (ecg_good / both_good_sqi)
  5. Collapsed Unique Events ((subject_id, event_id))
  6. Chronological Consecutive Event Pairs
  7. Transition Cells (SR->SR, SR->AF, AF->SR, AF->AF)
  8. ICU Stay Consistency (Confounding Audit)
"""

import json
from pathlib import Path
import polars as pl

metadata_path = Path("/data/mithunmanivannan/datasets/mimic3_ext_ppg/metadata.csv")
event_table_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_m1_event_transition_audit/event_table.parquet")
transition_pairs_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_m1_event_transition_audit/transition_pairs.parquet")
audit_results_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_m1_event_transition_audit/results.json")

with open(audit_results_path) as f:
    audit_json = json.load(f)

# Load parquet tables
events = pl.read_parquet(event_table_path)
pairs = pl.read_parquet(transition_pairs_path)

provenance = {
    "total_raw_segments": audit_json["dataset_metadata"]["total_raw_segments"],
    "total_sr_af_segments": audit_json["dataset_metadata"]["total_sr_af_segments"],
    "sr_segments": audit_json["dataset_metadata"]["sr_segments"],
    "af_segments": audit_json["dataset_metadata"]["af_segments"],
    
    # Event Level
    "total_unique_events": len(events),
    "unique_patients_with_events": events["subject_id"].n_unique(),
    "events_sr": int((events["event_rhythm"] == "SR").sum()),
    "events_af": int((events["event_rhythm"] == "AF").sum()),
    "events_with_lead_ii_ecg": int(events["ecg_available"].sum()),
    "events_with_high_quality_ecg": int(events["ecg_high_quality"].sum()),
    "events_with_both_ecg_ppg_high_quality": int(events["both_high_quality"].sum()),
    
    # Transition Pair Level
    "total_consecutive_pairs": len(pairs),
    "unique_patients_with_pairs": pairs["subject_id"].n_unique(),
    "pairs_sr_to_sr": int((pairs["cell"] == "SR->SR").sum()),
    "pairs_sr_to_af": int((pairs["cell"] == "SR->AF").sum()),
    "pairs_af_to_sr": int((pairs["cell"] == "AF->SR").sum()),
    "pairs_af_to_af": int((pairs["cell"] == "AF->AF").sum()),
    
    # Simultaneous Lead II and SQI in transitions
    "sr_to_af_both_ecg_avail": int(pairs.filter(pl.col("cell") == "SR->AF")["both_ecg_avail"].sum()),
    "sr_to_af_both_good_sqi": int(pairs.filter(pl.col("cell") == "SR->AF")["both_good_sqi"].sum()),
    "sr_to_sr_both_ecg_avail": int(pairs.filter(pl.col("cell") == "SR->SR")["both_ecg_avail"].sum()),
    "sr_to_sr_both_good_sqi": int(pairs.filter(pl.col("cell") == "SR->SR")["both_good_sqi"].sum()),
    
    # ICU Stay Consistency
    "same_icustay_sr_to_af": int(pairs.filter(pl.col("cell") == "SR->AF")["same_icustay"].sum()),
    "same_icustay_sr_to_sr": int(pairs.filter(pl.col("cell") == "SR->SR")["same_icustay"].sum()),
}

print(json.dumps(provenance, indent=2))

output_path = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/mimic_provenance_audit.json")
with open(output_path, "w") as f:
    json.dump(provenance, f, indent=2)
print(f"Saved provenance JSON to {output_path}")
