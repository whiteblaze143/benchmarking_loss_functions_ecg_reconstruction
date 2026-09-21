from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class ExperimentVariant:
    representation: str = "full"
    mechanism: str = "full"
    head: str = "native"
    capacity: str = "native"
    control: str = "none"
    use_mmd: bool = False

PAPER01_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "mean_distance_recurrence": ExperimentVariant(mechanism="mean_distance"),
    "phase_content_permuted": ExperimentVariant(mechanism="phase_content_permutation"),
    "cyclic_relabel_sham": ExperimentVariant(control="cyclic_relabel_sham"),
}

PAPER02_VARIANTS = {
    "full": ExperimentVariant(),
    "global_kme": ExperimentVariant(representation="kernel", mechanism="global_kme"),
    "no_circular": ExperimentVariant(mechanism="no_circular"),
    "moments_circular": ExperimentVariant(representation="moments"),
    "moments_noncircular": ExperimentVariant(representation="moments", mechanism="no_circular"),
    "gaussian_surrogate": ExperimentVariant(representation="gaussian"),
    "linear_kernel": ExperimentVariant(representation="linear"),
    "phase_kme_linear_probe": ExperimentVariant(head="phase_kme_linear"),
    "global_kme_linear_probe": ExperimentVariant(representation="kernel", mechanism="global_kme", head="global_kme_linear"),
    "linear_probe": ExperimentVariant(head="linear"),
}

PAPER03_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "phase_signature_linear_probe": ExperimentVariant(head="phase_kme_linear"),
    "order_scrambled": ExperimentVariant(representation="order_scrambled"),
    "time_reversed": ExperimentVariant(representation="time_reversed"),
    "monotone_warp_sham": ExperimentVariant(representation="monotone_warp_sham"),
    "unordered_kme": ExperimentVariant(representation="unordered_kme"),
}

PAPER04_VARIANTS = {
    "full": ExperimentVariant(),
    "paper02_phasecnn": ExperimentVariant(mechanism="paper02_phasecnn"),
    "flat_phase_mlp": ExperimentVariant(head="flat_phase_mlp"),
    "phase_aware_linear_probe": ExperimentVariant(head="phase_aware_linear"),
    "linear_probe": ExperimentVariant(head="linear"),
    "lag1": ExperimentVariant(mechanism="lag1"),
    "lag2": ExperimentVariant(mechanism="lag2"),
    "open_chain": ExperimentVariant(mechanism="open_chain"),
    "time_shuffled": ExperimentVariant(mechanism="time_shuffle_local"),
    "time_reversed": ExperimentVariant(mechanism="time_reversed"),
    "operator_summary_probe": ExperimentVariant(head="operator_summary_probe"),
    "ridge_strength_sensitivity": ExperimentVariant(mechanism="ridge_strength_sensitivity"),
}

PAPER05_VARIANTS = {
    "full": ExperimentVariant(),
    "full_dual_koopman": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "phase_aware_linear_probe": ExperimentVariant(head="phase_aware_linear"),
    "occupancy_only": ExperimentVariant(representation="occupancy_only", mechanism="occupancy_only"),
    "phase_koopman_only": ExperimentVariant(mechanism="phase_koopman_only"),
    "cycle_koopman_only": ExperimentVariant(mechanism="cycle_koopman_only"),
    "linear_state_dmd": ExperimentVariant(mechanism="linear_state_dmd"),
    "soft_markov": ExperimentVariant(mechanism="soft_markov"),
    "phase_order_shuffled": ExperimentVariant(mechanism="phase_order_shuffled"),
    "beat_order_shuffled": ExperimentVariant(mechanism="beat_order_shuffled"),
    "global_shuffled": ExperimentVariant(representation="chronology_shuffled", mechanism="global_shuffled"),
    "chronology_shuffled": ExperimentVariant(representation="chronology_shuffled", mechanism="chronology_shuffle"),
    "operator_only": ExperimentVariant(mechanism="operator_only"),
    "spectral_only": ExperimentVariant(mechanism="spectral_only"),
    "identity_order_sham": ExperimentVariant(representation="identity_order_sham", control="identity_order_sham"),
}

PAPER06_VARIANTS = {
    # Primary matched comparisons and scientific hierarchy
    "macro_plus_conditional_residual": ExperimentVariant(representation="full", mechanism="anchor_conditioned_mmd"),
    "macro_only": ExperimentVariant(representation="macro_component"),
    "residual_marginal_only": ExperimentVariant(representation="residual_marginal"),
    "macro_plus_marginal_residual": ExperimentVariant(representation="macro_residual_marginals"),
    "conditional_residual_only": ExperimentVariant(representation="full"),
    "vcg_plus_conditional_residual": ExperimentVariant(representation="macro_residual_marginals", control="vcg_control"),
    "phase_shuffled_within_macro": ExperimentVariant(representation="shuffled_residual", mechanism="within_macro_phase_shuffle"),
    "macro_overlap_only": ExperimentVariant(representation="macro_component", mechanism="overlap_channel"),
    "all_pairs": ExperimentVariant(representation="full", mechanism="all_pairs_mask"),
    "raw_affinity": ExperimentVariant(representation="full", mechanism="unnormalized_affinity"),
    "linear_probe": ExperimentVariant(head="linear"),
    "phase_aware_linear_probe": ExperimentVariant(head="phase_aware_linear"),
    # Backward compatibility aliases
    "full": ExperimentVariant(),
    "full_signal": ExperimentVariant(representation="full_signal"),
    "macro_component": ExperimentVariant(representation="macro_component"),
    "residual_marginal": ExperimentVariant(representation="residual_marginal"),
    "macro_residual_marginals": ExperimentVariant(representation="macro_residual_marginals"),
    "shuffled_residual": ExperimentVariant(representation="shuffled_residual", mechanism="shuffle_residual_given_z"),
    "joint_pair_sham": ExperimentVariant(representation="joint_pair_sham", control="joint_pair_sham"),
}


PAPER07_VARIANTS = {
    # Primary benchmark hierarchy
    "projective_continuous": ExperimentVariant(mechanism="projective_continuous"),
    "projective_continuous_aux": ExperimentVariant(mechanism="projective_continuous_aux"),
    "continuous_mlp": ExperimentVariant(mechanism="continuous_mlp"),
    "q_ablated_set": ExperimentVariant(mechanism="q_ablated_set"),
    "categorical_ids": ExperimentVariant(mechanism="categorical_ids"),
    "nearest_known_operator": ExperimentVariant(mechanism="nearest_known_operator"),
    "linear_q_encoder": ExperimentVariant(mechanism="linear_q_encoder"),
    "GraphECG_geometry": ExperimentVariant(mechanism="GraphECG_geometry", control="graphecg_baseline"),
    "analytic_pinv": ExperimentVariant(mechanism="analytic_pinv", control="pinv_oracle"),
    "LMMSE_operator": ExperimentVariant(mechanism="LMMSE_operator", control="lmmse_baseline"),
    # Backward compatibility aliases
    "continuous_primary": ExperimentVariant(mechanism="continuous_primary"),
    "categorical_primary": ExperimentVariant(mechanism="categorical_primary"),
    "continuous_auxiliary": ExperimentVariant(mechanism="continuous_auxiliary"),
    "categorical_auxiliary": ExperimentVariant(mechanism="categorical_auxiliary"),
}


PAPER08_VARIANTS = {
    "global_dynamic": ExperimentVariant(mechanism="global"),
    "local_banded": ExperimentVariant(mechanism="local_banded"),
    "static_attention": ExperimentVariant(mechanism="static_attention"),
    "uniform_attention": ExperimentVariant(mechanism="uniform_attention"),
    "phase_agnostic_set": ExperimentVariant(mechanism="phase_agnostic"),
    "scrambled_phases": ExperimentVariant(mechanism="scrambled_phases"),
    "linear_probe": ExperimentVariant(head="linear"),
    "kmeans_tokens": ExperimentVariant(mechanism="kmeans_dictionary"),
    "cnn_matched_control": ExperimentVariant(mechanism="cnn_matched"),
}


PAPER09_VARIANTS = {
    "full": ExperimentVariant(),
    "diagnosis_only": ExperimentVariant(mechanism="diagnosis_only"),
    "mismatched_q": ExperimentVariant(control="mismatch_q_waveform")
}

PAPER10_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "erm": ExperimentVariant(mechanism="erm"),
    "irm": ExperimentVariant(mechanism="irm"),
    "coral": ExperimentVariant(mechanism="coral"),
    "causirl": ExperimentVariant(mechanism="causirl")
}

PAPER11_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "shuffled_futures": ExperimentVariant(mechanism="shuffle_history_futures")
}

PAPER12_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "unconditional_z": ExperimentVariant(mechanism="unconditional_state")
}

PAPER13_VARIANTS = {
    "full": ExperimentVariant(),
    "linear_probe": ExperimentVariant(head="linear"),
    "no_propagation": ExperimentVariant(mechanism="no_structural_propagation"),
    "wrong_phase_surgery": ExperimentVariant(control="wrong_phase_surgery")
}

PAPER14_VARIANTS = {
    # Unambiguous V2 variants (ERM-AUG and MMD candidates)
    "erm": ExperimentVariant(mechanism="erm", use_mmd=False),
    "mmd": ExperimentVariant(mechanism="mmd", use_mmd=True),
    "mmd_lambda_0.1": ExperimentVariant(mechanism="mmd", use_mmd=True),
    "mmd_lambda_0.3": ExperimentVariant(mechanism="mmd", use_mmd=True),
    "mmd_lambda_1.0": ExperimentVariant(mechanism="mmd", use_mmd=True),
    "mmd_lambda_3.0": ExperimentVariant(mechanism="mmd", use_mmd=True),
    "mmd_lambda_10.0": ExperimentVariant(mechanism="mmd", use_mmd=True),
    # Backward compatibility mappings
    "full": ExperimentVariant(mechanism="mmd", use_mmd=True),
    "linear_probe": ExperimentVariant(head="linear", mechanism="erm", use_mmd=False),
    "mmd_penalty_only": ExperimentVariant(mechanism="mmd_penalty_only", use_mmd=True),
}

PAPER15_VARIANTS = {
    "full": ExperimentVariant(mechanism="modular"),
    "linear_probe": ExperimentVariant(head="linear"),
    "capacity_matched_shared": ExperimentVariant(mechanism="shared_capacity_matched", capacity="matched"),
    "shared_same_width": ExperimentVariant(mechanism="shared_same_width"),
    "shared_phase_conditioned": ExperimentVariant(mechanism="shared_phase_conditioned"),
    "linear_modular": ExperimentVariant(mechanism="linear_modular"),
}

def get_variants_for_paper(paper_id: int) -> Dict[str, ExperimentVariant]:
    name = f"PAPER{paper_id:02d}_VARIANTS"
    return globals()[name]
