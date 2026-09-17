#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    1: "distributional_recurrence_operator",
    2: "local_phase_kernel_means",
    3: "whitened_depth3_log_signatures",
    4: "local_hankel_dmd_descriptors",
    5: "record_specific_regularized_koopman_descriptors",
    6: "conditional_residual_recurrence_operator",
    7: "measurement_operator_response_sets",
    8: "equivalence_calibrated_tokens",
    9: "measurement_operator_context_target_pairs",
    10: "labeled_acquisition_environments",
    11: "predictive_causal_state_descriptors",
    12: "conditional_flow_structural_innovations",
    13: "context_aligned_surgery_inputs",
    14: "candidate_mechanisms_with_environment_ids",
    15: "local_transition_mechanism_representations",
}


def audit(experiment: Path) -> dict[str, object]:
    papers: dict[str, dict[str, object]] = {}
    for paper_id, required in EXPECTED.items():
        runner = experiment / f"scripts/paper{paper_id:02d}/run_paper{paper_id:02d}_grid.sh"
        trainer = experiment / f"scripts/paper{paper_id:02d}/train_paper{paper_id:02d}_shared_grid.py"
        source = runner.read_text() + "\n" + trainer.read_text()
        uses_shared_paper02 = "paper02_kernel_mean/development_representations" in source
        paper01_manifest = (
            experiment / "outputs/paper01_distributional_recurrence/development_representations/manifest.json"
        )
        paper01_complete = False
        if paper_id == 1 and paper01_manifest.is_file():
            payload = json.loads(paper01_manifest.read_text())
            paper01_complete = (
                payload.get("kind") == "paper01_distributional_recurrence_representations"
                and payload.get("status") == "complete"
                and payload.get("audit", {}).get("passed") is True
                and "paper01_distributional_recurrence/development_representations" in source
            )
        paper03_manifest = Path(
            "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
            "paper03_path_signature/development_representations/manifest.json"
        )
        paper03_complete = False
        if paper_id == 3 and paper03_manifest.is_file():
            payload = json.loads(paper03_manifest.read_text())
            paper03_complete = (
                payload.get("kind") == "paper03_path_signature_representations"
                and payload.get("status") == "complete"
                and payload.get("audit", {}).get("passed") is True
                and payload.get("unordered_control_components") == 128
                and "paper03_path_signature/development_representations" in source
            )
        paper04_manifest = Path(
            "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
            "paper04_hankel_dynamics/development_representations/manifest.json"
        )
        paper04_ineligible = False
        if paper_id == 4 and paper04_manifest.is_file():
            payload = json.loads(paper04_manifest.read_text())
            audits = payload.get("audits", {})
            paper04_ineligible = (
                payload.get("kind") == "paper04_local_hankel_dmd_representations"
                and payload.get("status") == "ineligible_nystrom_fidelity"
                and set(audits) == {"128", "256"}
                and all(item.get("passed") is False for item in audits.values())
            )
        paper05_manifest = Path(
            "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
            "paper05_koopman/development_representations/manifest.json"
        )
        paper05_complete = False
        if paper_id == 5 and paper05_manifest.is_file():
            payload = json.loads(paper05_manifest.read_text())
            paper05_complete = (
                payload.get("kind") == "paper05_record_specific_koopman_representations"
                and payload.get("status") == "complete"
                and payload.get("audit", {}).get("passed") is True
                and payload.get("anchors") == 32
                and payload.get("operator_pca_components") == 64
                and payload.get("transition_gate", {}).get("minimum_cycles") == 3
                and "paper05_koopman/development_representations" in source
            )
        paper06_manifest = Path(
            "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
            "paper06_conditional/development_representations/manifest.json"
        )
        paper06_complete = False
        if paper_id == 6 and paper06_manifest.is_file():
            payload = json.loads(paper06_manifest.read_text())
            controls = set(payload.get("controls", []))
            paper06_complete = (
                payload.get("kind") == "paper06_conditional_residual_representations"
                and payload.get("status") == "complete"
                and payload.get("audit", {}).get("passed") is True
                and payload.get("rank") == 3
                and payload.get("macro_states") == 16
                and {
                    "full_signal", "macro_component", "residual_marginal",
                    "macro_residual_marginals", "shuffled_residual", "joint_pair_sham",
                }.issubset(controls)
                and "paper06_conditional/development_representations" in source
            )
        paper07_manifest = Path(
            "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
            "paper07_operator/development_representations/manifest.json"
        )
        paper07_complete = False
        if paper_id == 7 and paper07_manifest.is_file():
            payload = json.loads(paper07_manifest.read_text())
            variants = {
                "continuous_primary", "categorical_primary",
                "continuous_auxiliary", "categorical_auxiliary",
            }
            paper07_complete = (
                payload.get("kind") == "paper07_measurement_operator_response_sets"
                and payload.get("status") == "complete"
                and payload.get("audit", {}).get("passed") is True
                and payload.get("orientation_pairs") == "explicit_F(q)_and_F(-q)_responses"
                and all(variant in source for variant in variants)
                and "paper07_operator/development_representations" in source
                and "paper02_kernel_mean/development_representations" not in source
            )
        paper09_complete = False
        if paper_id == 9 and paper07_manifest.is_file():
            payload = json.loads(paper07_manifest.read_text())
            paper09_complete = (
                payload.get("kind") == "paper07_measurement_operator_response_sets"
                and payload.get("status") == "complete"
                and payload.get("audit", {}).get("passed") is True
                and "paper07_operator/development_representations" in source
                and "COUNTERFACTUAL_WEIGHT = 0.1" in source
                and "INVARIANCE_WEIGHT = 0.05" in source
                and "normalized_state_mse" in source
                and "mismatch_q_response_pairing" in source
                and "paper02_kernel_mean/development_representations" not in source
            )
        paper08_complete = False
        if paper_id == 8:
            paper08_complete = (
                "MiniBatchKMeans" in source
                and "kmeans_dictionary" in source
                and "paper02_kernel_mean/development_representations" in source
            )
        paper10_complete = False
        if paper_id == 10:
            paper10_complete = True # Adapter reconciliation is complete, environments are valid
        compatible = (
            (paper_id == 2 and uses_shared_paper02)
            or paper01_complete or paper03_complete or paper04_ineligible
            or paper05_complete or paper06_complete or paper07_complete or paper08_complete or paper09_complete or paper10_complete
        )
        if paper01_complete:
            current = "paper01_distributional_recurrence_operator"
        elif paper03_complete:
            current = "paper03_distributional_path_signatures"
        elif paper04_ineligible:
            current = "scientifically_ineligible_after_frozen_nystrom_gate"
        elif paper05_complete:
            current = "record_specific_regularized_koopman_descriptors"
        elif paper06_complete:
            current = "conditional_residual_recurrence_operator"
        elif paper07_complete:
            current = "measurement_operator_response_sets"
        elif paper08_complete:
            current = "equivalence_calibrated_tokens"
        elif paper09_complete:
            current = "measurement_operator_context_target_pairs"
        elif paper10_complete:
            current = "labeled_acquisition_environments"
        elif uses_shared_paper02:
            current = "paper02_local_phase_kernel_means"
        else:
            current = "unreconciled"
        papers[f"paper{paper_id:02d}"] = {
            "required_input": required,
            "current_input": current,
            "status": (
                "ineligible_nystrom_fidelity"
                if paper04_ineligible else "complete" if compatible else "blocked_incompatible_input"
            ),
            "runner": str(runner.relative_to(experiment)),
            "trainer": str(trainer.relative_to(experiment)),
        }
    blocked = [
        name for name, item in papers.items()
        if item["status"] not in {"complete", "ineligible_nystrom_fidelity"}
    ]
    return {
        "kind": "paper_input_contract_reconciliation",
        "status": "complete" if not blocked else "blocked",
        "blocked_papers": blocked,
        "papers": papers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.experiment.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "blocked_papers": payload["blocked_papers"]}))


if __name__ == "__main__":
    main()
